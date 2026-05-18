# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, compute_sft_pure_loss
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available


def compute_kl_divergence_loss(log_prob, ref_log_prob, response_mask, loss_agg_mode='mean'):
    """
    Compute KL divergence loss between current policy and reference policy.
    
    Args:
        log_prob: Current policy log probabilities [bsz, seq_len]
        ref_log_prob: Reference policy log probabilities [bsz, seq_len]
        response_mask: Mask for valid tokens [bsz, seq_len]
        loss_agg_mode: How to aggregate the loss ('mean', 'sum', etc.)
    
    Returns:
        KL divergence loss (scalar)
    """
    # KL(π_current || π_ref) = E[log π_current - log π_ref]
    kl_div = log_prob - ref_log_prob
    kl_loss = agg_loss(loss_mat=kl_div, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return kl_loss
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches, prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor_combined_loss"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor_combined_loss(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        # Safely use torch.compile with exception handling
        if self.config.get("use_torch_compile", True):
            try:
                self.compute_entropy_from_logits = torch.compile(entropy_from_logits, dynamic=True)
            except Exception as e:
                # If torch.compile fails, fall back to original function
                print(f"Warning: torch.compile failed, falling back to original function: {e}")
                self.compute_entropy_from_logits = entropy_from_logits
        else:
            self.compute_entropy_from_logits = entropy_from_logits
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = self.compute_entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            base_max_token_len = data.meta_info["max_token_len"]
            max_token_len = base_max_token_len * self.ulysses_sequence_parallel_size
            
            # Check actual sequence length and truncate if necessary
            actual_max_seq_len = max([data.batch['input_ids'][i].size(0) for i in range(data.batch['input_ids'].size(0))])
            
            if max_token_len < actual_max_seq_len:
                print(f"[Policy Update Debug] WARNING: max_token_len ({max_token_len}) < actual_max_seq_len ({actual_max_seq_len})")
                print(f"[Policy Update Debug] Truncating sequences to max_token_len: {max_token_len}")
                # Truncate sequences instead of adjusting max_token_len
                for i in range(data.batch['input_ids'].size(0)):
                    if data.batch['input_ids'][i].size(0) > max_token_len:
                        data.batch['input_ids'][i] = data.batch['input_ids'][i][:max_token_len]
                        if 'attention_mask' in data.batch:
                            data.batch['attention_mask'][i] = data.batch['attention_mask'][i][:max_token_len]
                        if 'position_ids' in data.batch:
                            data.batch['position_ids'][i] = data.batch['position_ids'][i][:max_token_len]
            
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)
            # 过滤掉空的micro batches（特别是最后一个可能为空的batch）
            micro_batches = [mb for mb in micro_batches if len(mb) > 0]

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        
        # 保留 whether_pad 作为 mask，不要物理删除样本
        # 这样可以确保所有 rank 的 batch size 一致，避免 NCCL 通信问题
        has_padding = 'whether_pad' in data.batch
        
        if self.config.use_sft:
            select_keys = [
                "responses",
                "input_ids",
                "attention_mask",
                "position_ids",
                "old_log_probs",
                "advantages",
                'prefix_mask'
            ]

            select_keys.append('contrastive_loss')
            select_keys.append('contrastive_mask')
        else:
            select_keys = [
                "responses",
                "input_ids",
                "attention_mask",
                "position_ids",
                "old_log_probs",
                "advantages",
            ]

        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        
        # 保留 whether_pad 字段用于 mask
        if has_padding:
            select_keys.append("whether_pad")

        batch = data.select(batch_keys=select_keys).batch
        
        if torch.distributed.get_rank() == 0:
            if has_padding:
                n_valid = (~batch['whether_pad']).sum().item()
                n_total = batch['whether_pad'].size(0)
                print(f"[Actor] Batch size: {n_total} (valid: {n_valid}, padding: {n_total - n_valid})")
            else:
                print(f"[Actor] Batch size: {batch.batch_size[0]} (no padding)")

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = batch.split(self.config.ppo_mini_batch_size)
        
        if torch.distributed.get_rank() == 0:
            print(f"[Actor] Number of mini_batches: {len(mini_batches)}, ppo_epochs: {self.config.ppo_epochs}")

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1
        
        metrics = {}
        for epoch_idx in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    base_max_token_len = self.config.ppo_max_token_len_per_gpu
                    max_token_len = base_max_token_len * self.ulysses_sequence_parallel_size
                    
                    # Check actual sequence length and truncate if necessary
                    actual_max_seq_len = max([mini_batch['input_ids'][i].size(0) for i in range(mini_batch['input_ids'].size(0))])
                    print(f"[Policy Update Debug] actual_max_seq_len: {actual_max_seq_len}")
                    
                    if max_token_len < actual_max_seq_len:
                        print(f"[Policy Update Debug] WARNING: max_token_len ({max_token_len}) < actual_max_seq_len ({actual_max_seq_len})")
                        print(f"[Policy Update Debug] Truncating sequences to max_token_len: {max_token_len}")
                        # Truncate sequences instead of adjusting max_token_len
                        for i in range(mini_batch['input_ids'].size(0)):
                            if mini_batch['input_ids'][i].size(0) > max_token_len:
                                mini_batch['input_ids'][i] = mini_batch['input_ids'][i][:max_token_len]
                                if 'attention_mask' in mini_batch:
                                    mini_batch['attention_mask'][i] = mini_batch['attention_mask'][i][:max_token_len]
                                if 'position_ids' in mini_batch:
                                    mini_batch['position_ids'][i] = mini_batch['position_ids'][i][:max_token_len]
                    
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    # 过滤掉空的micro batches（特别是最后一个可能为空的batch）
                    micro_batches = [mb for mb in micro_batches if len(mb) > 0]

                self.actor_optimizer.zero_grad()

                # Get entropy coefficient from meta_info if available, otherwise use config default
                entropy_coeff = data.meta_info.get("entropy_coeff", self.config.get("entropy_coeff", 0.01))
                if torch.distributed.get_rank() == 0 and batch_idx == 0 and epoch_idx == 0:
                    print(f"[Actor] Using entropy coefficient: {entropy_coeff}")

                for micro_batch in micro_batches:
                    data = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    
                    # 应用 whether_pad mask：将 padding 样本的 response_mask 全部置为 0
                    # 这样在计算 loss 时，padding 样本不会贡献任何梯度
                    if has_padding and 'whether_pad' in data:
                        sample_mask = (~data['whether_pad']).float().unsqueeze(1)  # (bsz, 1)
                        response_mask = response_mask * sample_mask  # (bsz, response_length)
                    
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = responses.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        data, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if self.config.offline_loss_type == "sft":
                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = data['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        
                        on_policy_mask = ~off_policy_mask
                        on_policy_logprob = log_prob[on_policy_mask]
                        on_policy_old_logprob = old_log_prob[on_policy_mask]
                        
                        # assert self.config.algorithm.adv_estimator == 'grpo_split'
                        # The on-policy advantages should not be computed together with the off-policy rewards
                        on_policy_advantages = advantages[on_policy_mask]
                        on_policy_eos_mask = response_mask[on_policy_mask]
                        
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=on_policy_old_logprob, log_prob=on_policy_logprob,
                            advantages=on_policy_advantages,
                            eos_mask=on_policy_eos_mask,
                            cliprange=clip_ratio,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )

                        if torch.isnan(sft_loss):
                            print('sft_loss is nan, skipping sft_loss')
                        else:
                            pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss
                            print("pg_loss is: {pg_loss}" )

                            
                    else:
                        # Non-SFT branch: standard PPO logic
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = data["old_log_probs"]
                            
                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)
                        pg_loss, pg_clipfrac, ppo_kl, _ = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                        )

                        # Handle entropy for non-SFT branch
                        if entropy_coeff != 0:
                            entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            metrics['actor/entropy_coeff'] = entropy_coeff
                            print(f"[SFT Debug] entropy_loss: {entropy_loss.item()}")
                        else:
                            policy_loss = pg_loss

                    # add contrastive learning loss
                    # Only add if we already have a valid policy_loss OR contrastive loss itself is valid
                    if 'contrastive_loss' in data and 'contrastive_mask' in data:
                        contrastive_loss_values = data['contrastive_loss']  # [bsz]
                        contrastive_mask = data['contrastive_mask']  # [bsz]
                        
                        # 应用 whether_pad mask：排除 padding 样本
                        if has_padding and 'whether_pad' in data:
                            sample_valid_mask = ~data['whether_pad']  # [bsz]
                            contrastive_mask = contrastive_mask & sample_valid_mask
                        
                        # 如果有 on_policy_mask，只在 on-policy 样本上应用对比学习
                        if on_policy_mask is not None:
                            contrastive_mask = contrastive_mask & on_policy_mask
                            # print(f"[Contrastive Debug] Applying contrastive loss only on on-policy samples")
                        
                        # only compute loss for samples with contrastive learning (on-policy with mixed success/fail)
                        if contrastive_mask.any():
                            # compute mean contrastive loss
                            contrastive_loss_mean = contrastive_loss_values[contrastive_mask].mean()
                            
                            # get coefficient from config
                            contrastive_coef = self.config.get('contrastive_loss_coef', 0.1)
                            
                            # add to total loss (initialize if needed)
                            weighted_contrastive_loss = contrastive_loss_mean * contrastive_coef
                            if policy_loss is None or not policy_loss.requires_grad:
                                # Contrastive loss can serve as the sole loss if no other valid loss
                                policy_loss = weighted_contrastive_loss
                                print(f"[Loss Fusion] Using contrastive loss as sole loss: {weighted_contrastive_loss.item():.6f}")
                            else:
                                policy_loss = policy_loss + weighted_contrastive_loss
                                print(f"[Loss Fusion] Adding contrastive_loss: {contrastive_loss_mean.item():.6f}, "
                                      f"weighted: {weighted_contrastive_loss.item():.6f} (coef: {contrastive_coef})")
                            
                            # record metrics
                            metrics['actor/contrastive_loss'] = contrastive_loss_mean.detach().item()
                            metrics['actor/contrastive_samples'] = contrastive_mask.sum().item()
                            metrics['actor/contrastive_ratio'] = contrastive_mask.float().mean().item()


                    if self.config.use_kl_loss:
                        ref_log_prob = data["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        
                        weighted_kl_loss = kl_loss * self.config.kl_loss_coef
                        
                        # Add KL loss (initialize if needed)
                        if policy_loss is None or not policy_loss.requires_grad:
                            policy_loss = weighted_kl_loss
                            print(f"[Loss Fusion] Using KL loss as sole loss: {weighted_kl_loss.item():.6f}")
                        else:
                            policy_loss = policy_loss + weighted_kl_loss
                        
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor

                    # Only backward if loss has gradients (i.e., we have valid data)
                    if loss.requires_grad:
                        loss.backward()
                    else:
                        print('[Loss Fusion] WARNING: Skipping backward due to no valid loss with gradients')
                    
                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        
        self.actor_optimizer.zero_grad()
        return metrics
