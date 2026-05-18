# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
import traceback
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import VLLM_SLEEP_LEVEL
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from verl.workers.config import RolloutConfig
from verl.single_controller.base import Worker

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout_dypo(Worker):

    def __init__(self, model_path: str, config: RolloutConfig, tokenizer, model_hf_config, world_size, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = config.get("engine_kwargs", {}).get("vllm", {}) or {}

        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        compilation_config = {}

        cudagraph_capture_sizes = config.get("cudagraph_capture_sizes")
        # enforce_eager must be False to use cudagraph
        if not config.enforce_eager and cudagraph_capture_sizes:
            if isinstance(cudagraph_capture_sizes, ListConfig):
                compilation_config["compilation_config"] = CompilationConfig(
                    level=CompilationLevel.PIECEWISE, cudagraph_capture_sizes=cudagraph_capture_sizes
                )

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            max_num_seqs=config.max_num_seqs,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **compilation_config,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=VLLM_SLEEP_LEVEL)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)) and k != "seed":
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id


        # TODO: support FSDP hybrid shard for larger model
        infer_tp = tensor_parallel_size
        dp = world_size // infer_tp
        assert world_size % infer_tp == 0
        from torch.distributed.device_mesh import init_device_mesh
        device_mesh = init_device_mesh(
            "cuda", mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        is_collect = device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            mesh_name="rollout",
            dp_rank=device_mesh["dp"].get_local_rank(),
            is_collect=is_collect
        )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
        """Generate sequences using vLLM engine with retry logic for failures.

        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids and meta_info.
            max_retries (int, optional): Maximum number of retries on failure. Defaults to 1e9.
            **kwargs: Additional sampling parameters to override defaults.

        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Generated response token ids
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence

        Raises:
            RuntimeError: If generation fails after max_retries attempts.
        """
        max_retries = int(max_retries)
        for attempt in range(max_retries):
            try:
                # Rebuild vLLM cache engine if configured
                if self.config.free_cache_engine:
                    self.inference_engine.init_cache_engine()

                # Extract input tensors from prompt batch
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                eos_token_id = prompts.meta_info['eos_token_id']
                batch_size = idx.size(0)

                # Pre-process input token ids
                idx_list = [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ]

                # Configure sampling parameters
                do_sample = prompts.meta_info.get('do_sample', True)
                if not do_sample:
                    kwargs = {
                        'best_of': 1,
                        'top_p': 1.0,
                        'top_k': -1,
                        'min_p': 0.0,
                        'temperature': 0,
                        'n': 1
                    }
                if prompts.meta_info.get('val_temperature', None):
                    kwargs['temperature'] = prompts.meta_info['val_temperature']
                if prompts.meta_info.get('val_top_p', None):
                    kwargs['top_p'] = prompts.meta_info['val_top_p']
                # Generate sequences
                with self.update_sampling_params(**kwargs):
                    output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=idx_list,
                        use_tqdm=False)

                # Process outputs
                response = output[0].to(idx.device)
                log_probs = output[1].to(idx.device)

                # Pad sequences if needed
                if response.shape[1] < self.config.response_length:
                    response = pad_sequence_to_length(
                        response, self.config.response_length, self.pad_token_id)
                    log_probs = pad_sequence_to_length(
                        log_probs, self.config.response_length, self.pad_token_id)

                # Handle multiple samples per prompt
                if self.config.n > 1 and do_sample:
                    idx = idx.repeat_interleave(self.config.n, dim=0)
                    attention_mask = attention_mask.repeat_interleave(
                        self.config.n, dim=0)
                    position_ids = position_ids.repeat_interleave(
                        self.config.n, dim=0)
                    batch_size = batch_size * self.config.n

                # Concatenate prompt and response
                seq = torch.cat([idx, response], dim=-1)

                # Create position IDs and attention mask for full sequence
                response_length = response.size(1)
                delta_position_id = torch.arange(
                    1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).repeat(
                    batch_size, 1)

                response_position_ids = position_ids[:, -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids],
                                       dim=-1)
                response_attention_mask = get_eos_mask(
                    response_id=response,
                    eos_token=eos_token_id,
                    dtype=attention_mask.dtype)
                attention_mask = torch.cat(
                    (attention_mask, response_attention_mask), dim=-1)

                # Construct output batch
                batch = TensorDict(
                    {
                        'prompts': idx,
                        'responses': response,
                        'input_ids': seq,
                        'attention_mask': attention_mask,
                        'position_ids': position_ids
                    },
                    batch_size=batch_size)

                # Free cache if configured
                if self.config.free_cache_engine:
                    self.inference_engine.free_cache_engine()

                return DataProto(batch=batch)

            except Exception as e:
                traceback.print_exc()
                print("Restarting vLLM due to error: ", e)
                print("Retrying...")

                # Clean up and restart engine
                torch.cuda.empty_cache()
                if hasattr(self.inference_engine, 'free_cache_engine'):
                    self.inference_engine.free_cache_engine()
                del self.inference_engine

                # Reinitialize engine with same parameters
                self.inference_engine = LLM(
                    self.actor_module,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.model_hf_config,
                    tensor_parallel_size=self.tensor_parallel_size,
                    dtype=self.config.dtype,
                    enforce_eager=self.config.enforce_eager,
                    gpu_memory_utilization=self.config.gpu_memory_utilization,
                    skip_tokenizer_init=False,
                    max_model_len=self.config.prompt_length +
                    self.config.response_length,
                    load_format=self.config.load_format)
                print("vLLM is ready to roll!")

                if attempt < max_retries - 1:
                    continue

        raise RuntimeError(
            f"Failed to generate sequences after {max_retries} attempts")
