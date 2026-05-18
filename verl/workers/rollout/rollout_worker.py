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


import datetime
import logging
import os
from typing import Any, Optional
import torch
import traceback
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
)
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config.model import HFModelConfig
from verl.workers.config.rollout import RolloutConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RolloutWorker(Worker):
    def __init__(self, config: RolloutConfig, model_config: HFModelConfig) -> None:
        super().__init__()
        import torch.distributed
        from torch.distributed.device_mesh import init_device_mesh

        self.device_name = get_device_name()

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{self.device_name}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        self.config = config
        self.model_config = model_config

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            self.device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )

        rollout_name = self.config.name

        if rollout_name == "hf":
            self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)
        else:
            is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )

        # build rollout engine here
        if self.config.name == "vllm":
            from verl.workers.rollout.vllm_rollout import vLLMRollout

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            lora_kwargs = (
                {"lora_kwargs": {"enable_lora": True, "max_loras": 1, "max_lora_rank": self.model_config.lora_rank}}
                if self.model_config.lora_rank > 0
                else {}
            )
            from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

            vllm_rollout_cls = vLLMRollout if self.config.mode == "sync" else vLLMAsyncRollout
            self.rollout = vllm_rollout_cls(
                model_path=self.model_config.local_path,
                config=self.config,
                tokenizer=self.model_config.tokenizer,
                model_hf_config=self.model_config.hf_config,
                device_mesh=rollout_device_mesh,
                trust_remote_code=self.model_config.trust_remote_code,
                **lora_kwargs,
            )
        elif self.config.name == "sglang":
            from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout

            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76

            log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
            self.rollout = SGLangRollout(
                actor_module=self.model_config.local_path,
                config=self.config,
                processing_class=self.model_config.get_processor(),
                model_hf_config=self.model_config.hf_config,
                trust_remote_code=self.model_config.trust_remote_code,
            )
            log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)
        else:
            raise ValueError(f"Unknown rollout name: {self.config.name}")


    @torch.no_grad()
    def generate_off_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
        """Generate sequences using off-policy data without vLLM sampling.
        
        Args:
            prompts (DataProto): Input prompts containing batch data with input_ids, attention_mask,
                position_ids, tgt_input_ids and meta_info.
            max_retries (int, optional): Not used in this function but kept for consistency.
            **kwargs: Additional parameters (not used in off-policy generation).
            
        Returns:
            DataProto: Generated sequences containing:
                - prompts: Original input token ids
                - responses: Target response token ids (from tgt_input_ids)
                - input_ids: Concatenated prompt and response tokens
                - attention_mask: Attention mask for full sequence
                - position_ids: Position ids for full sequence
                - tgt_input_ids: Original target input ids
                - prefix_mask: Mask indicating all tokens are from off-policy data
        """
        # Extract input tensors from prompt batch
        idx = prompts.batch['input_ids']
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        eos_token_id = prompts.meta_info['eos_token_id']
        tgt_input_ids = prompts.batch['tgt_input_ids']  # [bsz, tgt_len]
        
        batch_size = idx.size(0)
        
        # Process target input ids - add eos token if needed
        tgt_list = [
            _pre_process_inputs_right_pad(self.pad_token_id, tgt_input_ids[i]) for i in range(batch_size)
        ]
        tgt_list = [
            tgt_list[i] + [self.tokenizer.eos_token_id,] if len(tgt_list[i]) > 0 else tgt_list[i]
            for i in range(batch_size)
        ]
        
        # For off-policy data, prefix_ratio is always 1.0 (use all target data)
        # No repetition needed for off-policy data
        prefix_ratios = [1.0] * len(tgt_list)
        
        # Use entire target as response (prefix_ratio = 1.0)
        response_list = tgt_list
        
        # Prepare response tensor
        resp_max_len = max([len(resp) for resp in response_list]) if response_list else 0
        response = torch.ones(len(response_list), max(resp_max_len, self.config.response_length)).fill_(self.pad_token_id)
        
        # Fill response tensor and create prefix mask
        prefix_mask = torch.zeros([len(response_list), self.config.response_length], dtype=torch.bool).to(idx.device)
        
        for i in range(len(response_list)):
            resp_len = min(len(response_list[i]), self.config.response_length)
            if resp_len > 0:
                response[i][:resp_len] = torch.tensor(response_list[i][:resp_len])
                # All tokens are from off-policy data (prefix)
                prefix_mask[i, :resp_len] = True
        
        response = response.to(idx.device)[:, :self.config.response_length].to(idx.dtype)
        
        # No repetition for off-policy data - keep original batch_size
        # Concatenate prompt and response
        seq = torch.cat([idx, response], dim=-1)
        
        # Create position IDs and attention mask for full sequence
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        
        response_attention_mask = get_eos_mask(
            response_id=response,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        # Construct output batch
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'tgt_input_ids': tgt_input_ids,
                'prefix_mask': prefix_mask,
            },
            batch_size=batch_size)
        
        meta_info = {
            'prefix_ratios': prefix_ratios,
        }
        
        return DataProto(batch=batch, meta_info=meta_info)
    
    @torch.no_grad()
    def generate_on_sequences(self, prompts: DataProto, max_retries: int = 1e9, **kwargs) -> DataProto:
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
        # if on_num is None:
        #     on_num = self.config.n
        on_num = 1
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

                # we use repeat to get n generations for each prompt
                # Pre-process input token ids
                batch_size = idx.size(0)
                idx_list = [
                    _pre_process_inputs(self.pad_token_id, idx[i])
                    for i in range(batch_size)
                ]
                # repeat idx_list to get n generations for each prompt
                do_sample = prompts.meta_info.get('do_sample', True)
                if do_sample:
                    idx_list = sum([[idx_list[i]] for i in range(len(idx_list))], [])
                
                prefix_ratios = None

                # Configure sampling parameters
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

                # we use n=1 because we have repeated the idx_list to get n generations for each prompt
                kwargs['n'] = 1

                # Generate sequences
                with self.update_sampling_params(**kwargs):
                    output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=self.sampling_params,
                        prompt_token_ids=idx_list,
                        use_tqdm=False)

                # Process outputs
                response = output[0].to(idx.device)
                
                prefix_mask = torch.zeros([batch_size, self.config.response_length], dtype=torch.bool).to(idx.device)
                
                # Pad sequences if needed
                if response.shape[1] < self.config.response_length:
                    response = pad_sequence_to_length(
                        response, self.config.response_length, self.pad_token_id)

                # Handle multiple samples per prompt
                if on_num > 1 and do_sample:
                    idx = idx.repeat_interleave(on_num, dim=0)
                    prefix_mask = prefix_mask.repeat_interleave(on_num, dim=0)
                    
                    tgt_input_ids = None
                    attention_mask = attention_mask.repeat_interleave(
                        on_num, dim=0)
                    position_ids = position_ids.repeat_interleave(
                        on_num, dim=0)
                    batch_size = batch_size * on_num

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
                        'position_ids': position_ids,
                    },
                    batch_size=batch_size)
                
                if prefix_mask.shape[0] > 0:
                    batch['prefix_mask'] = prefix_mask

                # Free cache if configured
                if self.config.free_cache_engine:
                    self.inference_engine.free_cache_engine()

                if prefix_ratios is not None:
                    meta_info = {
                        'prefix_ratios': prefix_ratios,
                    }
                    return DataProto(batch=batch, meta_info=meta_info)
                else:
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

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="infer"))
    def generate_sequences(self, prompts: DataProto):
        """Given a batch of prompts, return a batch of responses. Internally, it can use"""
        meta_info = {
            "eos_token_id": self.model_config.generation_config.eos_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.eos_token_id,
            "pad_token_id": self.model_config.generation_config.pad_token_id
            if self.model_config.generation_config is not None
            else self.model_config.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        output = self.rollout.generate_sequences(prompts=prompts)
        return output

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def execute_method(self, method: str | bytes, *args, **kwargs):
        """Called by ExternalRayDistributedExecutor collective_rpc."""
        return self.rollout._execute_method(method, *args, **kwargs)

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        if self.config.free_cache_engine:
            await self.rollout.wake_up()
        # return something to block the caller
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        if self.config.free_cache_engine:
            await self.rollout.sleep()
        # return something to block the caller
        return True
