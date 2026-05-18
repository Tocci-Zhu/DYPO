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
import numpy as np
import time
import datetime

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from verl.single_controller.base.decorator import register, Dispatch
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length

from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
)
from verl.single_controller.base.worker import Worker
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics

import logging
import os
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'INFO'))

# from pprint import pprint

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _pre_process_inputs_right_pad(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    if len(non_pad_index) == 0:
        return []
    else:
        token_ids = prompt_token_ids[:non_pad_index[-1][0]+1].tolist()
    return token_ids

from verl.workers.rollout.vllm_rollout import vLLMRollout_dypo

class dypo_RolloutWorker(vLLMRollout_dypo):
    def __init__(self, model_path, config, tokenizer, model_hf_config, world_size, **kwargs):
        super().__init__(model_path, config, tokenizer, model_hf_config, world_size, **kwargs)
        self.prefix_strategy = self.config.get('prefix_strategy', 'random')
        
        self.prefix_steps = self.config.get('prefix_steps', 300)
        self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
        if self.prefix_strategy == 'linear':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            pass
        elif self.prefix_strategy == 'linear_max':
            self.prefix_ratio_windows = [(0, i*self.prefix_linear_max_ratio/10) for i in range(10, 0, -1)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'linear_variance':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            self.prefix_lienar_max_var = self.config.get('prefix_lienar_max_var', 0.1)
        elif self.prefix_strategy == 'reverse_linear':
            # self.prefix_linear_max_ratio = self.config.get('prefix_linear_max_ratio', 0.8)
            self.prefix_ratio_windows = [(0, (i+1)*self.prefix_linear_max_ratio/10) for i in range(10)]
            self.prefix_step_windows = [(i*self.prefix_steps/10, (i+1)*self.prefix_steps/10) for i in range(10)]
        elif self.prefix_strategy == 'fixed':
            assert self.config.prefix_share_across_samples == False, "Fixed strategy could not work with prefix_share_across_samples=True ! "
            # self.prefix_fixed_num = self.config.get('prefix_fixed_num', 2)
            n_prefix = self.config.n_prefix if self.config.n_prefix != -1 else self.config.n
            ratio_step = (self.config.max_prefix_ratio - self.config.min_prefix_ratio) / (n_prefix-1)
            self.prefix_fix_ratios = [self.config.min_prefix_ratio + i*ratio_step for i in range(n_prefix)]

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
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        self.max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)
        self.load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format
        self.trust_remote_code = kwargs.get("trust_remote_code", False)
        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        # TODO: support FSDP hybrid shard for larger model
        # infer_tp = self.config.tensor_model_parallel_size
        # dp = world_size // infer_tp
        # assert world_size % infer_tp == 0, (
        #     f"rollout world_size: {world_size} is not divisible by infer_tp: {infer_tp}"
        # )
        # rollout_device_mesh = init_device_mesh(
        #     self.device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        # )

        # rollout_name = self.config.name

        # if rollout_name == "hf":
        #     self._register_dispatch_collect_info(mesh_name = "rollout", dp_rank=self.rank, is_collect=True)
        # else:
        #     is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        #     self._register_dispatch_collect_info(
        #         mesh_name = "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        #     )


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
        on_num = 1
        max_retries = int(max_retries)
        for attempt in range(max_retries):
            try:
                # Rebuild vLLM cache engine if configured
                if self.config.free_cache_engine and hasattr(self.inference_engine, 'init_cache_engine'):

                    self.inference_engine.init_cache_engine()
                    
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                eos_token_id = prompts.meta_info['eos_token_id']
                # we use repeat to get n generations for each prompt
                # Pre-process input token ids
                batch_size = idx.size(0)

                non_tensor_batch = prompts.non_tensor_batch
                if "raw_prompt_ids" not in non_tensor_batch:
                    non_tensor_batch["raw_prompt_ids"] = np.array(
                        [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
                    )

                if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                    print(f"❌ [generate_on_sequences] vLLM sharding manager working abnormally: batch_size={batch_size}, raw_prompt_ids_len={len(non_tensor_batch['raw_prompt_ids'])}")
                    raise RuntimeError("vllm sharding manager is not work properly.")

                if "multi_modal_data" in non_tensor_batch:
                    vllm_inputs = []
                    for raw_prompt_ids, multi_modal_data in zip(
                        non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
                    ):
                        vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
                    print(f"✅ [generate_on_sequences] Multi-modal input construction completed, count: {len(vllm_inputs)}")
                else:
                    vllm_inputs = [
                        {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                    ]


                for input_data in vllm_inputs:
                    # Ensure token IDs are lists or numpy arrays
                    if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                        raise TypeError(
                            f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                        )

                    input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

                # repeat idx_list to get n generations for each prompt
                do_sample = prompts.meta_info.get('do_sample', True)
                is_validate = prompts.meta_info.get("validate", False)

                if not do_sample:
                    kwargs = {
                        "best_of": 1,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "temperature": 0,
                        "n": 1,  # if greedy, only 1 response
                    }
                elif is_validate:
                    # TODO: try **
                    kwargs = {
                        "top_k": self.config.val_kwargs.top_k,
                        "top_p": self.config.val_kwargs.top_p,
                        "temperature": self.config.val_kwargs.temperature,
                        "n": 1,  # if validate, already repeat in ray_trainer
                    }

                lora_requests = None
                if self.lora_kwargs:
                    lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                    if len(lora_int_ids) > 0:
                        lora_int_id = lora_int_ids[0]
                        lora_requests = [
                            LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                        ] * batch_size
                
                with self.update_sampling_params(**kwargs):
                    start_time = time.time()
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                    generation_time = time.time() - start_time
                    print(f"✅ [generate_on_sequences] vLLM generation completed, time taken: {generation_time:.2f}s")

                    # TODO(sgm): disable logprob when recompute_log_prob is enable
                    # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
                    response = []
                    rollout_log_probs = []
                    output_count = 0
                    for output in outputs:
                        for sample_id in range(len(output.outputs)):
                            response_ids = output.outputs[sample_id].token_ids
                            response.append(response_ids)
                            output_count += 1
                            if self.config.calculate_log_probs:
                                curr_log_prob = []
                                for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                    curr_log_prob.append(logprob[response_ids[i]].logprob)
                                rollout_log_probs.append(curr_log_prob)
                    
                    response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                        idx.device
                    )
                    # print(f"📊 [generate_on_sequences] Response shape: {response.shape}")
                    
                    if self.config.calculate_log_probs:
                        # print("📊 [generate_on_sequences] Processing log probabilities...")
                        rollout_log_probs = pad_2d_list_to_length(
                            rollout_log_probs, -1, max_length=self.config.response_length
                        ).to(idx.device)
                        rollout_log_probs = rollout_log_probs.to(torch.float32)
                        # print(f"📊 [generate_on_sequences] log概率形状: {rollout_log_probs.shape}")

                    seq = torch.cat([idx, response], dim=-1)

                response_length = response.size(1)
                delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
                if position_ids.dim() == 3:  # qwen2vl mrope
                    delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

                # TODO(sgm): fix position_ids on right_pad
                # prompt: left pad + response: right pad
                # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
                # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
                response_position_ids = position_ids[..., -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                response_attention_mask = get_response_mask(
                    response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

                # all the tp ranks should contain the same data here. data in all ranks are valid
                # print("🔧 [generate_on_sequences] 构建最终输出批次...")
                batch = TensorDict(
                    {
                        "prompts": idx,
                        "responses": response,
                        "input_ids": seq,  # here input_ids become the whole sentences
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    },
                    batch_size=batch_size,
                )
                if self.config.calculate_log_probs:
                    # we will recompute old log prob with actor
                    print("📊 [generate_on_sequences] Adding rollout log probabilities...")
                    batch["rollout_log_probs"] = rollout_log_probs

                return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            
            except Exception as e:
                print(f"❌ [generate_on_sequences] Generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    print(f"🔄 [generate_on_sequences] Preparing to retry...")
                    time.sleep(1)  # 等待1秒后重试
                else:
                    print(f"❌ [generate_on_sequences] All retries failed, throwing exception")
                    raise


    
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
        
        result = DataProto(batch=batch, meta_info=meta_info)

        return result
    
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
                if self.config.free_cache_engine and hasattr(self.inference_engine, 'init_cache_engine'):

                    self.inference_engine.init_cache_engine()
                    
                idx = prompts.batch['input_ids']
                attention_mask = prompts.batch['attention_mask']
                position_ids = prompts.batch['position_ids']
                eos_token_id = prompts.meta_info['eos_token_id']
                # we use repeat to get n generations for each prompt
                # Pre-process input token ids
                batch_size = idx.size(0)

                non_tensor_batch = prompts.non_tensor_batch
                if "raw_prompt_ids" not in non_tensor_batch:
                    non_tensor_batch["raw_prompt_ids"] = np.array(
                        [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
                    )

                if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
                    print(f"❌ [generate_on_sequences] vLLM sharding manager working abnormally: batch_size={batch_size}, raw_prompt_ids_len={len(non_tensor_batch['raw_prompt_ids'])}")
                    raise RuntimeError("vllm sharding manager is not work properly.")

                if "multi_modal_data" in non_tensor_batch:
                    vllm_inputs = []
                    for raw_prompt_ids, multi_modal_data in zip(
                        non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
                    ):
                        vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
                    print(f"✅ [generate_on_sequences] Multi-modal input construction completed, count: {len(vllm_inputs)}")
                else:
                    vllm_inputs = [
                        {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
                    ]


                for input_data in vllm_inputs:
                    # Ensure token IDs are lists or numpy arrays
                    if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                        raise TypeError(
                            f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                        )

                    input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

                # repeat idx_list to get n generations for each prompt
                do_sample = prompts.meta_info.get('do_sample', True)
                is_validate = prompts.meta_info.get("validate", False)

                if not do_sample:
                    kwargs = {
                        "best_of": 1,
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "temperature": 0,
                        "n": 1,  # if greedy, only 1 response
                    }
                elif is_validate:
                    # TODO: try **
                    kwargs = {
                        "top_k": self.config.val_kwargs.top_k,
                        "top_p": self.config.val_kwargs.top_p,
                        "temperature": self.config.val_kwargs.temperature,
                        "n": 1,  # if validate, already repeat in ray_trainer
                    }

                lora_requests = None
                if self.lora_kwargs:
                    lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
                    if len(lora_int_ids) > 0:
                        lora_int_id = lora_int_ids[0]
                        lora_requests = [
                            LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                        ] * batch_size
                
                with self.update_sampling_params(**kwargs):
                    start_time = time.time()
                    outputs = self.inference_engine.generate(
                        prompts=vllm_inputs,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        lora_request=lora_requests,
                        use_tqdm=False,
                    )
                    generation_time = time.time() - start_time
                    print(f"✅ [generate_on_sequences] vLLM generation completed, time taken: {generation_time:.2f}s")

                    # TODO(sgm): disable logprob when recompute_log_prob is enable
                    # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
                    response = []
                    rollout_log_probs = []
                    output_count = 0
                    for output in outputs:
                        for sample_id in range(len(output.outputs)):
                            response_ids = output.outputs[sample_id].token_ids
                            response.append(response_ids)
                            output_count += 1
                            if self.config.calculate_log_probs:
                                curr_log_prob = []
                                for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                                    curr_log_prob.append(logprob[response_ids[i]].logprob)
                                rollout_log_probs.append(curr_log_prob)
                    
                    response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                        idx.device
                    )
                    # print(f"📊 [generate_on_sequences] Response shape: {response.shape}")
                    
                    if self.config.calculate_log_probs:
                        # print("📊 [generate_on_sequences] Processing log probabilities...")
                        rollout_log_probs = pad_2d_list_to_length(
                            rollout_log_probs, -1, max_length=self.config.response_length
                        ).to(idx.device)
                        rollout_log_probs = rollout_log_probs.to(torch.float32)
                        # print(f"📊 [generate_on_sequences] log概率形状: {rollout_log_probs.shape}")

                    seq = torch.cat([idx, response], dim=-1)

                response_length = response.size(1)
                delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
                delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
                if position_ids.dim() == 3:  # qwen2vl mrope
                    delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

                # TODO(sgm): fix position_ids on right_pad
                # prompt: left pad + response: right pad
                # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
                # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
                response_position_ids = position_ids[..., -1:] + delta_position_id
                position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
                response_attention_mask = get_response_mask(
                    response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
                )
                attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

                # all the tp ranks should contain the same data here. data in all ranks are valid
                # print("🔧 [generate_on_sequences] 构建最终输出批次...")
                batch = TensorDict(
                    {
                        "prompts": idx,
                        "responses": response,
                        "input_ids": seq,  # here input_ids become the whole sentences
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                    },
                    batch_size=batch_size,
                )
                if self.config.calculate_log_probs:
                    # we will recompute old log prob with actor
                    print("📊 [generate_on_sequences] Adding rollout log probabilities...")
                    batch["rollout_log_probs"] = rollout_log_probs

                return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            
            except Exception as e:
                print(f"❌ [generate_on_sequences] Generation failed (attempt {attempt + 1}/{max_retries}): {e}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    print(f"🔄 [generate_on_sequences] Preparing to retry...")
                    time.sleep(1)  # 等待1秒后重试
                else:
                    print(f"❌ [generate_on_sequences] All retries failed, throwing exception")
                    raise


    

def unit_test():
    batch = DataProto.from_single_dict({
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
        'tgt_input_ids': torch.tensor([[1, 2, 3, 4, 5]])
    })
    idx = batch.batch['input_ids']
    tgt_input_ids = batch.batch['tgt_input_ids']
    
    batch_size = tgt_input_ids.size(0)
    
    # idx_list = [1, 2, 3, 4, 5]
    idx_list = [
        _pre_process_inputs(1, idx[i])
        for i in range(batch_size)
    ]

    idx_list = sum([[idx_list[i]] * 2 for i in range(len(idx_list))], [])

    tgt_input_ids = batch.batch['tgt_input_ids']  # [bsz, tgt_len]

    tgt_list = [
        _pre_process_inputs(1, tgt_input_ids[i])
        for i in range(batch_size)
    ]
    
    tgt_list = sum([[tgt_list[i]] * 2 for i in range(len(tgt_list))], [])
    
    import random
    prefix_ratios = [random.randint(0, 100)/100 for _ in range(len(tgt_list))]
    prefix_list = [tgt_list[i][:int(len(tgt_list[i]) * prefix_ratios[i])] for i in range(len(tgt_list))]
    idx_list = [idx_list[i] + prefix_list[i] for i in range(len(idx_list))]
    print(idx_list)
    print(tgt_list)
    print(prefix_list)

if __name__ == "__main__":
    unit_test()