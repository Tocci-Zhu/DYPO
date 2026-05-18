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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches
import verl.utils.torch_functional as verl_F

__all__ = ['MIXDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor

from tensordict import TensorDict

class MIXDataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_adaptive_temperature = self.config.use_adaptive_temperature
        self.adaptive_temperature_target_entropy = self.config.adaptive_temperature_target_entropy
        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            from torch import optim
            self.alpha_optimizer = optim.AdamW([self.log_alpha],
                                          lr=self.config.alpha_lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=1e-2)
        else:
            self.alpha_optimizer = None
            
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        if 'whether_pad' in data.batch:
            whether_keep = (~data.batch['whether_pad']).tolist()
        else:
            whether_keep = None

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'prefix_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.config.offline_loss_type == "off_policy" and self.config.off_policy_loss_impl == 'seq':
            select_keys.append('on_logprobs_mean')
            select_keys.append('on_logprobs_std')
        if self.config.offline_loss_type == "off_policy" and self.config.use_off_policy_probs:
            select_keys.append('target_probs')
        use_sft = False
        if self.config.offline_loss_type == "off_sft":
            if 'whether_off' in data.batch:
                select_keys.append('whether_off')
            else:
                use_sft = True
        if self.config.offline_loss_type == "srft":
            select_keys.append('token_level_scores')
        # add contrastive learning keys (always present, but may be zeros if not enabled)
        select_keys.append('contrastive_loss')
        select_keys.append('contrastive_mask')

        batch = data.select(batch_keys=select_keys).batch

        if whether_keep is not None:
            filtered_dict = {}
            for key in batch.keys():
                if key != 'batch_size':
                    filtered_dict[key] = batch[key][whether_keep]
            
            batch = TensorDict(filtered_dict, batch_size=[sum(whether_keep)])
            del filtered_dict

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for data in micro_batches:
                    print("MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    if self.config.offline_loss_type == "sft" or use_sft:
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

                    elif self.config.offline_loss_type == "off_policy":
                        from .mix_core_alg import compute_token_on_off_policy_loss
                        loss_fn = compute_token_on_off_policy_loss

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            clip_upper_bound=self.config.clip_upper_bound,
                            prefix_mask=data['prefix_mask'],
                            off_cliprange=self.config.off_policy_cliprange,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)

                    elif self.config.offline_loss_type == "switch_off_sft":
                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = data['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        
                        from .mix_core_alg import compute_token_on_off_policy_loss_weight
                        loss_fn = compute_token_on_off_policy_loss_weight

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            clip_upper_bound=self.config.clip_upper_bound,
                            prefix_mask=data['prefix_mask'],
                            off_cliprange=self.config.off_policy_cliprange,
                            off_loss_coef=self.config.off_loss_coef,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        if torch.isnan(sft_loss):
                            print('sft_loss is nan, skipping sft_loss')
                        else:
                            pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)

                    elif self.config.offline_loss_type == "off_sft":
                        from .mix_core_alg import compute_sft_pure_loss
                        offline_mask = data['prefix_mask'].any(-1) # [No]
                        off_rl_mask = data['whether_off']

                        sft_mask = offline_mask * (~off_rl_mask)

                        sft_logprob = log_prob[sft_mask]
                        sft_eos_mask = response_mask[sft_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=sft_logprob,
                                                        eos_mask=sft_eos_mask)
                        

                        from .mix_core_alg import compute_token_on_off_policy_mask_loss
                        loss_fn = compute_token_on_off_policy_mask_loss

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            clip_upper_bound=self.config.clip_upper_bound,
                            prefix_mask=data['prefix_mask'],
                            off_cliprange=self.config.off_policy_cliprange,
                            off_rl_mask = off_rl_mask,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        if torch.isnan(sft_loss):
                            print('sft_loss is nan, skipping sft_loss')
                        else:
                            pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss

                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)

                    elif self.config.offline_loss_type == "srft":
                        H_coef = verl_F.masked_mean(entropy, response_mask, axis=-1)
                        H_coef = H_coef.detach()
                        sft_coef = 0.5 * torch.exp(-1 * H_coef)
                        on_coef = 0.1 * torch.exp(H_coef)

                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = data['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob * (sft_coef.view(-1, 1))
                        off_policy_logprob = off_policy_logprob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]

                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        

                        from .mix_core_alg import compute_token_on_off_policy_srft_loss
                        loss_fn = compute_token_on_off_policy_srft_loss

                        token_level_scores = data['token_level_scores']
                        correct_answer_mask = token_level_scores.sum(-1) == 1

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            clip_upper_bound=self.config.clip_upper_bound,
                            prefix_mask=data['prefix_mask'],
                            off_cliprange=self.config.off_policy_cliprange,
                            on_coef=on_coef,
                            correct_answer_mask=correct_answer_mask,
                            srft_type=self.config.srft_type,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        if torch.isnan(sft_loss):
                            print('sft_loss is nan, skipping sft_loss')
                        else:
                            pg_loss = sft_loss + pg_loss

                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                            'srft/H': H_coef.mean().item(),
                            'srft/sft_coef': sft_coef.mean().item(),
                            'srft/on_coef': on_coef.mean().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)
                    
                    else:
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob, log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                eos_mask=response_mask,
                                                                                cliprange=clip_ratio,
                                                                                loss_remove_token_mean=self.config.loss_remove_token_mean,
                                                                                loss_remove_clip=self.config.loss_remove_clip)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    if self.config.use_adaptive_temperature:
                        if self.config.use_adaptive_temperature_fixed is False:
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = self.log_alpha.exp()
                            if self.config.adaptive_temperature_clip > 0:
                                entropy_coeff = torch.clamp(entropy_coeff, max=self.config.adaptive_temperature_clip)
                            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * entropy_coeff
                            alpha_loss = alpha_loss / self.gradient_accumulation
                            alpha_loss.backward()
                            
                            policy_loss = pg_loss - entropy_loss * entropy_coeff.detach().item()
                            metrics['actor/alpha_loss'] = alpha_loss.detach().item()
                            metrics['actor/entropy_coeff'] = entropy_coeff.detach().item()
                            metrics['actor/log_alpha'] = self.log_alpha.detach().item()
                        else: # fixed strategy for entropy coeff
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            # cur_entropy = verl_F.masked_mean(entropy, response_mask)
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            metrics['actor/entropy_coeff'] = entropy_coeff
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef
                    
                    # add contrastive learning loss
                    if 'contrastive_loss' in data and 'contrastive_mask' in data:
                        contrastive_loss_values = data['contrastive_loss']  # [bsz]
                        contrastive_mask = data['contrastive_mask']  # [bsz]
                        
                        # only compute loss for samples with contrastive learning (on-policy with mixed success/fail)
                        if contrastive_mask.any():
                            # compute mean contrastive loss
                            contrastive_loss_mean = contrastive_loss_values[contrastive_mask].mean()
                            
                            # get coefficient from config
                            contrastive_coef = self.config.get('contrastive_loss_coef', 0.1)
                            
                            # add to total loss
                            policy_loss = policy_loss + contrastive_loss_mean * contrastive_coef
                            
                            # record metrics
                            metrics['actor/contrastive_loss'] = contrastive_loss_mean.detach().item()
                            metrics['actor/contrastive_coef'] = contrastive_coef
                            metrics['actor/contrastive_samples'] = contrastive_mask.sum().item()
                            metrics['actor/contrastive_ratio'] = contrastive_mask.float().mean().item()
                    
                    if self.config.use_ppo_kl_loss:
                        policy_loss = policy_loss + ppo_kl.abs() * self.config.ppo_kl_loss_coef
                        metrics['actor/ppo_kl_loss'] = ppo_kl.abs().detach().item()
                        
                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/loss': loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        elif self.config.get("max_grad_norm", None) and grad_norm > self.config.max_grad_norm:
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is too large: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        return grad_norm
