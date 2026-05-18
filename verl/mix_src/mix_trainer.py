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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
import psutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from collections import defaultdict, Counter

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import torch

from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer, 
    Role, 
    ResourcePoolManager, 
    WorkerType, 
    _timer, 
    # compute_data_metrics, 
    compute_timing_metrics, 
    dataprotoitem_to_dataproto, 
    # compute_advantage, 
    reduce_metrics
)
from verl.utils.torch_functional import masked_mean

from tensordict import TensorDict
import math

import gc

def check_memory_usage(stage=""):
    """Monitor memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    print(f"[{stage}] Memory usage: {memory_gb:.2f} GB")

        
def memory_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# directly copied from verl/trainer/ppo/ray_trainer.py
def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, grpo_use_std=True):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_balance':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_balance_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_split':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        prefix_mask = data.batch['prefix_mask']
        on_policy_mask = ~prefix_mask.any(-1)
        from .mix_core_alg import compute_grpo_outcome_advantage_split
        advantages, returns = compute_grpo_outcome_advantage_split(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
            on_policy_mask=on_policy_mask,
            use_std=grpo_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        
    elif adv_estimator == 'reinforce':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                             eos_mask=response_mask,
                                                                             index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data

class MIXRayPPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_balance':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'grpo_split':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        from .rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(
            config=self.config,
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True, return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
            filter_targets=self.config.data.get('filter_targets', False),
            suffix_prompt=self.config.data.get('suffix_prompt', ''),
            sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)
        
        self.val_dataset = RLHFDataset(
            config=self.config,
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get('return_raw_chat', False),
            suffix_prompt=self.config.data.get('suffix_prompt', ''),
            truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def select_on_off_ada_balance(self, on_solve_num: int):
        if self.config.trainer.unify_strategy == 'switch':
            on_add_num = 0
            if on_solve_num <= self.config.trainer.switch_gate:
                on_remove_num = 8
                off_add_num = 1
            else:
                on_remove_num = 0
                off_add_num = 0

            return on_remove_num, on_add_num, off_add_num

        if self.config.trainer.unify_strategy == 'soft':
            on_remove_num = 0
            on_add_num = 0
            off_add_num = 1

            return on_remove_num, on_add_num, off_add_num
    
    def compute_contrastive_loss(self, batch: DataProto, success_value: float, fail_value: float):
        """
        计算对比学习loss，针对能有几率回答对的prompt的on-policy样本
        使用DPO风格的pairwise ranking loss：让成功样本的log_prob高于失败样本
        
        核心思想：
        - 对于每个prompt，如果on-policy样本"有几率回答对"（既有成功也有失败）
        - 计算该prompt的对比学习loss（所有success-fail pairs的平均）
        - 将该loss赋给该prompt的所有on-policy样本（广播相同的值）
        - Off-policy样本保持为0
        - 成功的responses作为preferred (y_w)，失败的responses作为rejected (y_l)  
        - Loss = -log(sigmoid(beta * (log_prob(y_w) - log_prob(y_l))))
        
        Returns:
            batch: 更新后的batch，包含：
                - batch.batch['contrastive_loss']: [bsz] 每个样本的对比loss，on-policy有值，off-policy为0
                - batch.batch['contrastive_mask']: [bsz] 标记哪些on-policy样本来自"有几率回答对"的prompt
                - batch.non_tensor_batch['contrastive_uid']: [bsz] 每个样本对应的uid（有对比学习的样本）
                - batch.meta_info['contrastive_uid_loss']: dict[uid -> float] 每个prompt的对比loss（简洁版，用于日志/采样/课程学习）
                - batch.meta_info['contrastive_prompt_stats']: dict[uid -> stats] 每个prompt的详细统计
                - batch.meta_info['contrastive_prompt_weighted_loss']: dict[uid -> float] 每个uid对应的加权loss
        """
        uids = np.array(batch.non_tensor_batch['uid'])
        unique_uids = np.unique(uids)
        
        reward_tensor = batch.batch['token_level_scores']
        prefix_mask = batch.batch['prefix_mask']
        old_log_probs = batch.batch['old_log_probs']
        responses = batch.batch['responses']
        attention_mask = batch.batch['attention_mask']
        
        device = old_log_probs.device
        response_length = responses.size(-1)
        response_mask = attention_mask[:, -response_length:]
        batch_size = responses.size(0)
        
        # 初始化，注意device
        contrastive_loss = torch.zeros(batch_size, dtype=torch.float32, device=device)
        contrastive_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 用于存储每个样本对应的uid（用于后续加权）
        contrastive_uids = np.array([''] * batch_size, dtype=object)  # 存储每个样本的uid
        
        # 用于存储每个prompt的对比loss统计和加权loss
        prompt_contrastive_stats = {}
        prompt_weighted_loss = {}  # uid -> 加权后的loss值
        prompt_uid_loss = {}  # uid -> prompt的对比学习loss（简洁版，用于日志/采样/课程学习）
        
        beta = self.config.trainer.get('contrastive_beta', 0.1)
        loss_on_failures = self.config.trainer.get('contrastive_loss_on_failures', False)
        
        for uid in unique_uids:
            # uid是字符串，使用numpy比较
            uid_mask = torch.from_numpy(uids == uid).to(device)  # [bsz] 该prompt的所有样本
            if not uid_mask.any():
                continue
                
            uid_rewards = reward_tensor[uid_mask].sum(-1)  # [n]
            uid_prefix_masks = prefix_mask[uid_mask]
            is_on_policy = ~uid_prefix_masks.any(-1)  # [n]
            
            if not is_on_policy.any():
                continue
                
            on_policy_rewards = uid_rewards[is_on_policy]
            success_mask = (on_policy_rewards == success_value)
            fail_mask = (on_policy_rewards == fail_value)
            
            # 只处理"有几率回答对"的情况（既有成功也有失败）
            if not (success_mask.any() and fail_mask.any()):
                prompt_uid_loss[str(uid)] = 0
                continue
                
            # 获取索引
            indices = torch.where(uid_mask)[0]  # 该prompt的所有样本索引
            on_policy_indices = indices[is_on_policy]
            success_indices = on_policy_indices[success_mask]
            fail_indices = on_policy_indices[fail_mask]
            
            # 计算序列log probs (只计算response部分)
            success_logp = (old_log_probs[success_indices] * response_mask[success_indices]).sum(-1)  # [S]
            fail_logp = (old_log_probs[fail_indices] * response_mask[fail_indices]).sum(-1)        # [F]
            
            # 向量化的Pairwise DPO loss: [S, F]
            diff = success_logp.unsqueeze(1) - fail_logp.unsqueeze(0)  # [S, F]
            logits = beta * diff
            pairwise_loss = -torch.nn.functional.logsigmoid(logits)    # [S, F]
            
            # 计算该prompt的平均对比学习loss
            prompt_avg_loss = pairwise_loss.mean().item()
            
            # 计算加权loss（根据成功/失败比例进行加权）
            # 可以根据实际需求调整加权策略
            n_success = success_mask.sum().item()
            n_fail = fail_mask.sum().item()
            total_on = n_success + n_fail
            
            # 加权策略：更难的问题（success/fail比例接近0.5）权重更高
            # weight = 4 * (n_success / total) * (n_fail / total)  # 最大值为1（当比例为0.5:0.5时）
            balance_weight = 4.0 * (n_success / total_on) * (n_fail / total_on)
            weighted_loss = prompt_avg_loss * balance_weight
            
            # 也可以用其他加权策略，这里提供可配置选项
            if self.config.trainer.get('contrastive_weight_type', 'balance') == 'uniform':
                weighted_loss = prompt_avg_loss
            elif self.config.trainer.get('contrastive_weight_type', 'balance') == 'inverse_success_rate':
                # 成功率越低，权重越高
                success_rate = n_success / total_on
                weighted_loss = prompt_avg_loss * (1.0 - success_rate)
            
            # 只将该prompt的loss赋给on-policy样本，off-policy保持为0
            contrastive_loss[on_policy_indices] = prompt_avg_loss
            contrastive_mask[on_policy_indices] = True
            
            # 记录每个on-policy样本对应的uid
            uid_indices_np = on_policy_indices.cpu().numpy()
            contrastive_uids[uid_indices_np] = str(uid)
            
            # 记录该uid的loss（简洁版）
            prompt_uid_loss[str(uid)] = prompt_avg_loss
            
            # 记录该uid的加权loss
            prompt_weighted_loss[str(uid)] = weighted_loss
            
            # 记录该prompt的统计信息
            prompt_contrastive_stats[str(uid)] = {
                'n_success': n_success,
                'n_fail': n_fail,
                'n_samples': uid_mask.sum().item(),  # 该prompt的总样本数
                'n_on_policy': is_on_policy.sum().item(),
                'success_loss_mean': pairwise_loss.mean(dim=1).mean().item(),  # 成功样本的平均loss
                'success_loss_std': pairwise_loss.mean(dim=1).std().item() if n_success > 1 else 0.0,
                'fail_loss_mean': pairwise_loss.mean(dim=0).mean().item(),  # 失败样本的平均loss
                'fail_loss_std': pairwise_loss.mean(dim=0).std().item() if n_fail > 1 else 0.0,
                'pairwise_loss_mean': prompt_avg_loss,  # 所有pairs的平均loss
                'pairwise_loss_max': pairwise_loss.max().item(),
                'pairwise_loss_min': pairwise_loss.min().item(),
                'balance_weight': balance_weight,  # 平衡权重
                'weighted_loss': weighted_loss,  # 加权后的loss
            }

        # 将loss和mask存入batch
        batch.batch['contrastive_loss'] = contrastive_loss
        batch.batch['contrastive_mask'] = contrastive_mask

        # 将每个样本对应的uid存入non_tensor_batch
        batch.non_tensor_batch['contrastive_uid'] = contrastive_uids
        
        # 将prompt级别的统计和loss存入meta_info
        batch.meta_info['contrastive_uid_loss'] = prompt_uid_loss  # 简洁版: uid -> loss
        batch.meta_info['contrastive_prompt_stats'] = prompt_contrastive_stats  # 详细统计
        batch.meta_info['contrastive_prompt_weighted_loss'] = prompt_weighted_loss  # 加权loss

        # 打印统计信息
        if contrastive_mask.any():
            avg_loss = contrastive_loss[contrastive_mask].mean().item()
            n_samples = contrastive_mask.sum().item()
            n_prompts = len(prompt_contrastive_stats)
            
            # 计算加权loss的统计
            if prompt_weighted_loss:
                weighted_losses = list(prompt_weighted_loss.values())
                avg_weighted_loss = np.mean(weighted_losses)
                print(f"DPO-style contrastive loss: {avg_loss:.4f} (weighted: {avg_weighted_loss:.4f}) over {n_samples} samples from {n_prompts} prompts")
            else:
                print(f"DPO-style contrastive loss: {avg_loss:.4f} over {n_samples} samples from {n_prompts} prompts")
        
        return batch
            

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        n_samples = self.config.actor_rollout_ref.rollout.n
        if self.config.data.get('add_tgt_with_acc', False):
            n_samples = n_samples - 1 # if filter tgt with acc, we either use tgt or on policy samples.

        if self.config.trainer.remove_sfted_data:
            sfted_data_item_list = []
        
        for _ in range(self.config.trainer.total_epochs):

            if self.config.trainer.remove_sfted_data:
                if len(sfted_data_item_list) > 0:
                    self.train_dataset.remove_data(sfted_data_item_list)
                    
                    # Reconstruct train_dataloader
                    from torch.utils.data import DataLoader, SequentialSampler
                    from verl.utils.dataset.rl_dataset import collate_fn
                    
                    if self.config.data.shuffle:
                        from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
                        sampler = ResumableRandomSampler(data_source=self.train_dataset)
                    else:
                        sampler = SequentialSampler(data_source=self.train_dataset)
                    
                    self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                                       batch_size=self.config.data.train_batch_size,
                                                       drop_last=True,
                                                       collate_fn=collate_fn,
                                                       sampler=sampler)
                    
                sfted_data_item_list = []
                memory_cleanup()
            
            for batch_dict in self.train_dataloader:
                check_memory_usage("batch_start")
                
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                metrics = {}
                timing_raw = {}

                if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                    # Before popping, copy the required data first
                    batch.batch['raw_input_ids'] = batch.batch['input_ids'].clone()
                    batch.batch['raw_attention_mask'] = batch.batch['attention_mask'].clone()
                    batch.batch['raw_position_ids'] = batch.batch['position_ids'].clone()
                    
                    # pop those keys for generation
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                else:
                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                gen_batch.meta_info['global_steps'] = self.global_steps

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                            gen_batch_output = self.actor_rollout_wg.generate_on_sequences(gen_batch, on_num=self.config.actor_rollout_ref.rollout.n_verify)
                        else:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # This code matches a prompt ID with its N responses.
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    
                    if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_verify, interleave=True)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                
                    batch = batch.union(gen_batch_output)
                    
                    if self.config.trainer.add_full_target_when_none:
                        pass

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores using reward model and/or reward function
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        reward_tensor = self.reward_fn(batch) # [bsz, l], only the last valid token has reward

                        batch.batch['token_level_scores'] = reward_tensor
                        
                        # Rejection sampling based on rewards
                        # Group rewards by uid
                        uids = batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        # valid_mask = torch.ones(len(uids), dtype=torch.bool)
                        
                        if self.config.data.reward_impl_version == 0:
                            fail_value = 0
                            success_value = 1
                            format_value = -1 # not defined.
                        elif self.config.data.reward_impl_version == 1:
                            fail_value = -0.5
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 2:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 3:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 4:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 5:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        elif self.config.data.reward_impl_version == 6:
                            fail_value = 0
                            success_value = 1
                            format_value = -1
                        else:
                            raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                        
                        solve_none = 0
                        solve_all = 0
                        solve_none_format = 0
                        for uid in unique_uids:
                            uid_mask = uids == uid
                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                            
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_rewards == fail_value).all():
                                solve_none += 1
                            elif (uid_rewards == success_value).all():
                                solve_all += 1
                            elif (uid_rewards == format_value).all():
                                solve_none_format += 1

                        # Log to metrics
                        metrics['batch/solve_none'] = solve_none
                        metrics['batch/solve_none_format'] = solve_none_format
                        metrics['batch/solve_all'] = solve_all

                        # add more metrics
                        metrics['batch/solved'] = (reward_tensor.sum(-1) == success_value).sum().item() / len(uids)
                        metrics['batch/failed'] = (reward_tensor.sum(-1) == fail_value).sum().item() / len(uids)

                        
                        if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                            # Collect all uid information that needs on-policy data generation
                            all_on_batches = []
                            uid_balance = {}
                            uid_raw_data = {}
                            
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                uid_rewards = reward_tensor[uid_mask].sum(-1)
                                # Count on_solve_num for this uid
                                on_solve_num = (uid_rewards == success_value).sum().item()
                                on_remove_num, on_add_num, off_add_num = self.select_on_off_ada_balance(on_solve_num)

                                uid_balance[uid] = (on_remove_num, on_add_num, off_add_num)

                                uid_indices = np.where(uid_mask)[0]
                                first_idx = uid_indices[0]
                                uid_raw_data[uid] = {
                                    'input_ids': batch.batch['raw_input_ids'][first_idx:first_idx+1],
                                    'attention_mask': batch.batch['raw_attention_mask'][first_idx:first_idx+1],
                                    'position_ids': batch.batch['raw_position_ids'][first_idx:first_idx+1],
                                }


                                if on_add_num != 0:                                
                                    # Extract data for this prompt and repeat on_add_num times
                                    prompt_data = {}
                                    prompt_data['input_ids'] = batch.batch['raw_input_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                    prompt_data['attention_mask'] = batch.batch['raw_attention_mask'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                    prompt_data['position_ids'] = batch.batch['raw_position_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                    prompt_data['tgt_input_ids'] = batch.batch['tgt_input_ids'][first_idx:first_idx+1].repeat(on_add_num, 1)
                                    
                                    # Extract non_tensor_batch and meta_info from original batch
                                    new_non_tensor_batch = {}
                                    for key, value in batch.non_tensor_batch.items():
                                        if key == 'uid':
                                            new_non_tensor_batch[key] = np.array([uid] * on_add_num, dtype=object)
                                        else:
                                            # Copy the first sample's value to all new samples
                                            new_non_tensor_batch[key] = np.array([value[first_idx]] * on_add_num, dtype=value.dtype)
                                    
                                    # Create DataProto object
                                    on_batch = DataProto(
                                        batch=TensorDict(prompt_data, batch_size=[on_add_num]),
                                        non_tensor_batch=new_non_tensor_batch,
                                    )
                                    
                                    all_on_batches.append(on_batch)
                                    # Immediately clean up temporary data
                                    del prompt_data, new_non_tensor_batch
                            
                            # Clean up original data
                            gc.collect()
        
                            
                            # If there's data to generate, process it uniformly
                            if all_on_batches:
                                # Merge all on-policy data
                                merged_on_batch_dict = {}
                                merged_on_non_tensor_dict = {}
                                
                                # Merge tensor data
                                for key in all_on_batches[0].batch.keys():
                                    if key != 'batch_size':
                                        merged_on_batch_dict[key] = torch.cat([b.batch[key] for b in all_on_batches], dim=0)
                                
                                # Merge non_tensor data
                                for key in all_on_batches[0].non_tensor_batch.keys():
                                    merged_on_non_tensor_dict[key] = np.concatenate([b.non_tensor_batch[key] for b in all_on_batches], axis=0)
                                
                                total_on_batch_size = sum(b.batch.batch_size[0] for b in all_on_batches)
                                merged_on_batch_dict = TensorDict(merged_on_batch_dict, batch_size=[total_on_batch_size])
                                
                                combined_on_batch = DataProto(
                                    batch=merged_on_batch_dict,
                                    non_tensor_batch=merged_on_non_tensor_dict,
                                    meta_info={'global_steps': self.global_steps}
                                )

                                on_gen_batch = combined_on_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                                on_gen_batch.meta_info['global_steps'] = self.global_steps
                                
                                # Check if batch size is a multiple of 8, pad if not
                                original_batch_size = on_gen_batch.batch.batch_size[0]
                                remainder = original_batch_size % 8
                                if remainder != 0:
                                    padding_size = 8 - remainder
                                    
                                    # Pad the three types of data by repeating the last sample
                                    padded_batch_dict = {}
                                    for key in ['input_ids', 'attention_mask', 'position_ids']:
                                        last_sample = on_gen_batch.batch[key][-1:].repeat(padding_size, 1)
                                        padded_batch_dict[key] = torch.cat([on_gen_batch.batch[key], last_sample], dim=0)
                                    
                                    # Create padded batch
                                    padded_batch_size = original_batch_size + padding_size
                                    padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                                    
                                    on_gen_batch = DataProto(
                                        batch=padded_batch_dict,
                                        meta_info=on_gen_batch.meta_info
                                    )
                                    del padded_batch_dict
                                
                                # Uniformly generate on-policy sequences
                                on_gen_output = self.actor_rollout_wg.generate_on_sequences(on_gen_batch, 1)
                                
                                # If padding was done before, now need to remove padded data
                                if remainder != 0:
                                    # Remove padded data, keep only original data
                                    filtered_output_dict = {}
                                    for key in on_gen_output.batch.keys():
                                        if key != 'batch_size':
                                            filtered_output_dict[key] = on_gen_output.batch[key][:original_batch_size]
                                    
                                    filtered_output_dict = TensorDict(filtered_output_dict, batch_size=[original_batch_size])
                                    
                                    on_gen_output = DataProto(
                                        batch=filtered_output_dict,
                                        meta_info=on_gen_output.meta_info
                                    )
                                    del filtered_output_dict
                                
                                combined_on_batch = combined_on_batch.union(on_gen_output)

                                # Calculate rewards
                                on_reward_tensor = self.reward_fn(combined_on_batch)
                                combined_on_batch.batch['token_level_scores'] = on_reward_tensor

                                batch.pop(batch_keys=['raw_input_ids', 'raw_attention_mask', 'raw_position_ids'])
                                
                                # Merge into original batch
                                merged_batch_dict = {}
                                merged_non_tensor_dict = {}
                                
                                # Merge tensor data
                                for key in batch.batch.keys():
                                    if key != 'batch_size':
                                        merged_batch_dict[key] = torch.cat([batch.batch[key], combined_on_batch.batch[key]], dim=0)
                                
                                # Merge non_tensor data
                                for key in batch.non_tensor_batch.keys():
                                    merged_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], combined_on_batch.non_tensor_batch[key]], axis=0)
                                
                                new_batch_size = batch.batch.batch_size[0] + combined_on_batch.batch.batch_size[0]
                                merged_batch_dict = TensorDict(merged_batch_dict, batch_size=[new_batch_size])
                                
                                batch = DataProto(
                                    batch=merged_batch_dict,
                                    non_tensor_batch=merged_non_tensor_dict,
                                    meta_info=batch.meta_info.copy()
                                )

                                del combined_on_batch, on_gen_output, merged_on_batch_dict, merged_on_non_tensor_dict
                                gc.collect()
                            
                            else:
                                batch.pop(batch_keys=['raw_input_ids', 'raw_attention_mask', 'raw_position_ids'])
                            

                            all_off_batches = []
                            for uid in unique_uids:
                                off_add_num = uid_balance[uid][2]
                                
                                if off_add_num != 0:
                                    whether_off = False
                                    if off_add_num < 0:
                                        off_add_num = -1 * off_add_num
                                        whether_off = True
                                        
                                    uid_mask = uids == uid
                                    uid_indices = np.where(uid_mask)[0]
                                    first_idx = uid_indices[0]

                                    prompt_data = {}
                                    prompt_data['input_ids'] = uid_raw_data[uid]['input_ids']
                                    prompt_data['attention_mask'] = uid_raw_data[uid]['attention_mask']
                                    prompt_data['position_ids'] = uid_raw_data[uid]['position_ids']
                                    prompt_data['tgt_input_ids'] = batch.batch['tgt_input_ids'][first_idx:first_idx+1]

                                    other_off_add_num = off_add_num - 1
                                    if other_off_add_num > 0:
                                        other_off_data = self.train_dataloader.dataset.random_get(num=other_off_add_num)
                                        prompt_data['input_ids'] = torch.cat([prompt_data['input_ids'], other_off_data['input_ids']], dim=0)
                                        prompt_data['attention_mask'] = torch.cat([prompt_data['attention_mask'], other_off_data['attention_mask']], dim=0)
                                        prompt_data['position_ids'] = torch.cat([prompt_data['position_ids'], other_off_data['position_ids']], dim=0)
                                        prompt_data['tgt_input_ids'] = torch.cat([prompt_data['tgt_input_ids'], other_off_data['tgt_input_ids']], dim=0)

                                    # Extract non_tensor_batch and meta_info from original batch
                                    new_non_tensor_batch = {}
                                    for key, value in batch.non_tensor_batch.items():
                                        if key == 'uid':
                                            new_non_tensor_batch[key] = np.array([uid] * off_add_num, dtype=object)
                                        else:
                                            # Copy the first sample's value to all new samples
                                            new_non_tensor_batch[key] = np.array([value[first_idx]] * off_add_num, dtype=value.dtype)

                                    if whether_off:
                                        prompt_data['whether_off'] = torch.tensor([True] * off_add_num, dtype=torch.bool)
                                    else:
                                        prompt_data['whether_off'] = torch.tensor([False] * off_add_num, dtype=torch.bool)
                                    
                                    # Create DataProto object
                                    off_batch = DataProto(
                                        batch=TensorDict(prompt_data, batch_size=[off_add_num]),
                                        non_tensor_batch=new_non_tensor_batch,
                                    )
                                    
                                    all_off_batches.append(off_batch)

                            # If there's data to generate, process it uniformly
                            if all_off_batches:
                                # Merge all on-policy data
                                merged_off_batch_dict = {}
                                merged_off_non_tensor_dict = {}
                                
                                # Merge tensor data
                                for key in all_off_batches[0].batch.keys():
                                    if key != 'batch_size':
                                        merged_off_batch_dict[key] = torch.cat([b.batch[key] for b in all_off_batches], dim=0)
                                
                                # Merge non_tensor data
                                for key in all_off_batches[0].non_tensor_batch.keys():
                                    merged_off_non_tensor_dict[key] = np.concatenate([b.non_tensor_batch[key] for b in all_off_batches], axis=0)
                                
                                total_off_batch_size = sum(b.batch.batch_size[0] for b in all_off_batches)
                                merged_off_batch_dict = TensorDict(merged_off_batch_dict, batch_size=[total_off_batch_size])
                                
                                combined_off_batch = DataProto(
                                    batch=merged_off_batch_dict,
                                    non_tensor_batch=merged_off_non_tensor_dict,
                                    meta_info={'global_steps': self.global_steps}
                                )

                                off_gen_batch = combined_off_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                                off_gen_batch.meta_info['global_steps'] = self.global_steps
                                
                                # Check if batch size is a multiple of 8, pad if not
                                original_batch_size = off_gen_batch.batch.batch_size[0]
                                remainder = original_batch_size % 8
                                if remainder != 0:
                                    padding_size = 8 - remainder
                                    
                                    # Pad the four types of data by repeating the last sample
                                    padded_batch_dict = {}
                                    for key in ['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids']:
                                        last_sample = off_gen_batch.batch[key][-1:].repeat(padding_size, 1)
                                        padded_batch_dict[key] = torch.cat([off_gen_batch.batch[key], last_sample], dim=0)
                                    
                                    # Create padded batch
                                    padded_batch_size = original_batch_size + padding_size
                                    padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                                    
                                    off_gen_batch = DataProto(
                                        batch=padded_batch_dict,
                                        meta_info=off_gen_batch.meta_info
                                    )
                                    del padded_batch_dict
                                
                                # Uniformly generate off-policy sequences
                                off_gen_output = self.actor_rollout_wg.generate_off_sequences(off_gen_batch)
                                
                                # If padding was done before, now need to remove padded data
                                if remainder != 0:
                                    # Remove padded data, keep only original data
                                    filtered_output_dict = {}
                                    for key in off_gen_output.batch.keys():
                                        if key != 'batch_size':
                                            filtered_output_dict[key] = off_gen_output.batch[key][:original_batch_size]
                                    
                                    filtered_output_dict = TensorDict(filtered_output_dict, batch_size=[original_batch_size])
                                    
                                    off_gen_output = DataProto(
                                        batch=filtered_output_dict,
                                        meta_info=off_gen_output.meta_info
                                    )
                                    del filtered_output_dict
                                
                                combined_off_batch = combined_off_batch.union(off_gen_output)

                                # Calculate rewards
                                off_reward_tensor = self.reward_fn(combined_off_batch)
                                combined_off_batch.batch['token_level_scores'] = off_reward_tensor
                                
                                # Merge into original batch
                                if 'whether_off' in batch.batch:
                                    batch.batch['whether_off'] = torch.tensor([False] * batch.batch.batch_size[0], dtype=torch.bool)

                                merged_batch_dict = {}
                                merged_non_tensor_dict = {}
                                
                                # Merge tensor data
                                for key in batch.batch.keys():
                                    if key != 'batch_size':
                                        merged_batch_dict[key] = torch.cat([batch.batch[key], combined_off_batch.batch[key]], dim=0)
                                
                                # Merge non_tensor data
                                for key in batch.non_tensor_batch.keys():
                                    merged_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], combined_off_batch.non_tensor_batch[key]], axis=0)
                                
                                new_batch_size = batch.batch.batch_size[0] + combined_off_batch.batch.batch_size[0]
                                merged_batch_dict = TensorDict(merged_batch_dict, batch_size=[new_batch_size])
                                
                                batch = DataProto(
                                    batch=merged_batch_dict,
                                    non_tensor_batch=merged_non_tensor_dict,
                                    meta_info=batch.meta_info.copy()
                                )

                                del combined_off_batch, off_gen_output, merged_off_batch_dict, merged_off_non_tensor_dict
                                gc.collect()


                            if self.config.trainer.remove_on or self.config.trainer.unify_strategy == 'switch':
                                if self.config.trainer.unify_strategy != 'switch':
                                    # Calculate the amount of data to remove
                                    remove_count = self.config.actor_rollout_ref.rollout.n_verify * self.config.data.train_batch_size
                                    
                                    # Remove tensor data from batch
                                    filtered_batch_dict = {}
                                    for key in batch.batch.keys():
                                        if key != 'batch_size':
                                            filtered_batch_dict[key] = batch.batch[key][remove_count:]
                                    
                                    # Remove data from non_tensor_batch
                                    filtered_non_tensor_dict = {}
                                    for key in batch.non_tensor_batch.keys():
                                        filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][remove_count:]
                                    
                                    # Calculate new batch size
                                    new_batch_size = batch.batch.batch_size[0] - remove_count
                                    filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])
                                    
                                    # Rebuild batch
                                    batch = DataProto(
                                        batch=filtered_batch_dict,
                                        non_tensor_batch=filtered_non_tensor_dict,
                                        meta_info=batch.meta_info.copy()
                                    )
                                    # Immediately clean up temporary data
                                    del filtered_batch_dict, filtered_non_tensor_dict
                                    gc.collect()
                                else:
                                    keep_mask = torch.ones(batch.batch.batch_size[0], dtype=torch.bool)

                                    for uid in unique_uids:
                                        on_remove_num = uid_balance[uid][0]
                                        if on_remove_num != 0:
                                            uid_mask = uids == uid
                                            uid_indices = np.where(uid_mask)[0]

                                            if self.config.trainer.remove_sfted_data:
                                                sfted_data_item = batch.non_tensor_batch['item'][uid_indices[0]]
                                                sfted_data_item_list.append(sfted_data_item)
                                                print('Removing SFTed data:', sfted_data_item)
                                                print(sfted_data_item_list)

                                            # Get prefix_mask for all data corresponding to this uid
                                            uid_prefix_masks = batch.batch['prefix_mask'][uid_indices]
                                            # Determine if it's on-policy data (on-policy data has no True in prefix_mask)
                                            is_on_policy = ~uid_prefix_masks.any(-1)
                                            on_policy_indices = uid_indices[is_on_policy.cpu().numpy()]

                                            keep_mask[on_policy_indices] = False
                                            
                                    # Remove tensor data from batch
                                    filtered_batch_dict = {}
                                    for key in batch.batch.keys():
                                        if key != 'batch_size':
                                            filtered_batch_dict[key] = batch.batch[key][keep_mask]
                                    
                                    # Remove data from non_tensor_batch
                                    filtered_non_tensor_dict = {}
                                    for key in batch.non_tensor_batch.keys():
                                        filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][keep_mask.cpu().numpy()]
                                    
                                    # Calculate new batch size
                                    new_batch_size = keep_mask.sum().item()
                                    filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])
                                    
                                    # Rebuild batch
                                    batch = DataProto(
                                        batch=filtered_batch_dict,
                                        non_tensor_batch=filtered_non_tensor_dict,
                                        meta_info=batch.meta_info.copy()
                                    )
                                    # Immediately clean up temporary data  
                                    del filtered_batch_dict, filtered_non_tensor_dict, keep_mask
                                    gc.collect()


                        else:
                            # add on-policy metrics
                            prefix_mask = batch.batch['prefix_mask']
                            off_policy_mask = prefix_mask.any(-1)
                            on_policy_mask = ~off_policy_mask
                            metrics['batch/on_solved'] = (reward_tensor[on_policy_mask].sum(-1) == success_value).sum().item() / (on_policy_mask.sum().item() + 1e-6)
                            metrics['batch/off_solved'] = (reward_tensor[off_policy_mask].sum(-1) == success_value).sum().item() / (off_policy_mask.sum().item() + 1e-6)

                        if self.config.trainer.unify_strategy == 'soft':
                            on_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])
                            off_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])
                            sft_coef_list = torch.tensor([0.] * batch.batch.batch_size[0])

                            if self.config.trainer.soft_type == 1:
                                coef_dict = {
                                    0: (1., 1., 1.), # Should not occur, can be ignored
                                    1: (0., 1., 1.),
                                    2: (0.125, 1., 0.5),
                                    3: (0.25, 1., 0.25),
                                    4: (0.5, 1., 0.125),
                                    5: (1., 1., 0.),
                                    6: (1., 1., 0.),
                                    7: (1., 1., 0.),
                                    8: (1., 1., 0.),
                                }
                            elif self.config.trainer.soft_type == 2:
                                coef_dict = {
                                    0: (1., 1., 1.), # Should not occur, can be ignored
                                    1: (0., 0., 1.),
                                    2: (0.125, 0., 0.5),
                                    3: (0.25, 0., 0.25),
                                    4: (0.5, 0., 0.125),
                                    5: (1., 0., 0.),
                                    6: (1., 0., 0.),
                                    7: (1., 0., 0.),
                                    8: (1., 0., 0.),
                                }
                            else:
                                coef_dict = {
                                    0: (1., 1., 1.), # Should not occur, can be ignored
                                    1: (0., 0., 1.),
                                    2: (0.125, 0.5, 0.5),
                                    3: (0.25, 1., 0.25),
                                    4: (0.5, 0.5, 0.125),
                                    5: (1., 0.25, 0.),
                                    6: (1., 0.125, 0.),
                                    7: (1., 0., 0.),
                                    8: (1., 0., 0.),
                                }
                            
                            uids = batch.non_tensor_batch['uid']
                            unique_uids = np.unique(uids)
                            for uid in unique_uids:
                                uid_mask = uids == uid
                                reward_tensor = batch.batch['token_level_scores']
                                uid_rewards = reward_tensor[uid_mask].sum(-1)
                                on_solve_num = (uid_rewards == success_value).sum().item()

                                on_coef, off_coef, sft_coef = coef_dict[on_solve_num]
                                on_coef_list[uid_mask] = on_coef
                                off_coef_list[uid_mask] = off_coef
                                sft_coef_list[uid_mask] = sft_coef

                            batch.batch['on_coef'] = on_coef_list
                            batch.batch['off_coef'] = off_coef_list
                            batch.batch['sft_coef'] = sft_coef_list      
                        
                        # Check if batch size is a multiple of 8, pad if not
                        original_batch_size = batch.batch.batch_size[0]
                        remainder = original_batch_size % 8
                        if remainder != 0:
                            padding_size = 8 - remainder
                            
                            # Pad tensor data in batch.batch by repeating the last sample
                            padded_batch_dict = {}
                            for key in batch.batch.keys():
                                if key in ['on_coef', 'off_coef', 'sft_coef']:
                                    last_sample = batch.batch[key][-1:].repeat(padding_size)
                                    padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                                    continue
                                if key != 'batch_size':
                                    last_sample = batch.batch[key][-1:].repeat(padding_size, 1)
                                    padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                            
                            # Pad data in batch.non_tensor_batch
                            padded_non_tensor_dict = {}
                            for key in batch.non_tensor_batch.keys():
                                last_sample = np.array([batch.non_tensor_batch[key][-1]] * padding_size, dtype=batch.non_tensor_batch[key].dtype)
                                padded_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], last_sample], axis=0)
                            
                            # Create padded batch
                            padded_batch_size = original_batch_size + padding_size
                            padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                            
                            batch = DataProto(
                                batch=padded_batch_dict,
                                non_tensor_batch=padded_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )
                            # Immediately clean up temporary data
                            del padded_batch_dict, padded_non_tensor_dict
                            gc.collect()
                        
                        # recompute old_log_probs
                        with _timer('old_log_prob', timing_raw):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with _timer('ref', timing_raw):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                batch = batch.union(ref_log_prob)

                        # compute rewards with KL penalty if needed

                        # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                        # where it is subtracted directly from the policy loss

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']
                        
                        # If padding was done before, now need to remove padded data
                        if remainder != 0:
                            # Remove padded data, keep only original data
                            filtered_batch_dict = {}
                            for key in batch.batch.keys():
                                if key != 'batch_size':
                                    filtered_batch_dict[key] = batch.batch[key][:original_batch_size]
                            
                            # Remove data from non_tensor_batch
                            filtered_non_tensor_dict = {}
                            for key in batch.non_tensor_batch.keys():
                                filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][:original_batch_size]
                            
                            filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[original_batch_size])
                            
                            batch = DataProto(
                                batch=filtered_batch_dict,
                                non_tensor_batch=filtered_non_tensor_dict,
                                meta_info=batch.meta_info.copy()
                            )
                            # Immediately clean up temporary data
                            del filtered_batch_dict, filtered_non_tensor_dict
                            gc.collect()

                        # NOTE: the advantages are the same for all tokens in the response
                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  grpo_use_std=self.config.algorithm.grpo_use_std)
                        
                        # compute contrastive learning loss for on-policy samples with mixed results
                        if self.config.trainer.get('use_contrastive_loss', False):
                            batch = self.compute_contrastive_loss(batch, success_value=success_value, fail_value=fail_value)
                        else:
                            # initialize contrastive loss tensors as zeros even when not enabled
                            # to ensure batch.batch always has these keys for consistency
                            device = batch.batch['advantages'].device
                            batch_size = batch.batch.batch_size[0]
                            batch.batch['contrastive_loss'] = torch.zeros(batch_size, dtype=torch.float32, device=device)
                            batch.batch['contrastive_mask'] = torch.zeros(batch_size, dtype=torch.bool, device=device)
                            
                        # compute alpha and beta for prefix reward weighting
                        prefix_mask = batch.batch['prefix_mask']
                        advantages = batch.batch['advantages']
                        assert prefix_mask.shape == advantages.shape
                        
                        alpha_weight = prefix_mask.float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_alpha
                        beta_weight = (~prefix_mask).float() * self.config.actor_rollout_ref.rollout.prefix_reward_weight_beta
                        prefix_weight = alpha_weight + beta_weight
                        batch.batch['advantages'] = prefix_weight * advantages
                        
                        if self.config.data.get('disable_truncation_advantage', False):
                            responses = batch.batch['responses']
                            responses_mask = responses != self.tokenizer.pad_token_id
                            response_length = responses_mask.sum(-1) # [bsz]
                            max_len = self.config.data.max_response_length
                            has_truncated = response_length >= max_len
                            no_eos = ~((responses == self.tokenizer.eos_token_id).any(-1))
                            truncated_mask = has_truncated & no_eos
                            batch.batch['advantages'][truncated_mask] = 0

                        if self.config.actor_rollout_ref.actor.get('use_sft_prefix_reward', False):
                            assert self.config.actor_rollout_ref.rollout.n_prefix == -1
                            reward_weight = self.config.actor_rollout_ref.actor.get('sft_prefix_reward_weight', 1.0)
                            batch.batch['advantages'][prefix_mask] = reward_weight / n_samples
                    
                    if self.config.trainer.debug is True:
                        breakpoint()
                    
                    # Check if batch size is a multiple of 8, pad if not
                    batch.batch['whether_pad'] = torch.tensor([False] * batch.batch.batch_size[0], dtype=torch.bool)
                    original_batch_size = batch.batch.batch_size[0]
                    remainder = original_batch_size % 8
                    if remainder != 0:
                        padding_size = 8 - remainder
                        
                        # Pad tensor data in batch.batch by repeating the last sample
                        padded_batch_dict = {}
                        for key in batch.batch.keys():
                            if key != 'batch_size':
                                if key == 'whether_pad':
                                    last_sample = torch.tensor([True] * padding_size, dtype=torch.bool)
                                    padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                                elif key in ['contrastive_loss', 'contrastive_mask']:
                                    # For contrastive loss fields, pad with zeros/False (these keys always exist)
                                    if key == 'contrastive_loss':
                                        last_sample = torch.zeros(padding_size, dtype=torch.float32, device=batch.batch[key].device)
                                    else:  # contrastive_mask
                                        last_sample = torch.zeros(padding_size, dtype=torch.bool, device=batch.batch[key].device)
                                    padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                                else:
                                    last_sample = batch.batch[key][-1:].repeat(padding_size, 1)
                                    padded_batch_dict[key] = torch.cat([batch.batch[key], last_sample], dim=0)
                        
                        # Pad data in batch.non_tensor_batch
                        padded_non_tensor_dict = {}
                        for key in batch.non_tensor_batch.keys():
                            last_sample = np.array([batch.non_tensor_batch[key][-1]] * padding_size, dtype=batch.non_tensor_batch[key].dtype)
                            padded_non_tensor_dict[key] = np.concatenate([batch.non_tensor_batch[key], last_sample], axis=0)
                        
                        # Create padded batch
                        padded_batch_size = original_batch_size + padding_size
                        padded_batch_dict = TensorDict(padded_batch_dict, batch_size=[padded_batch_size])
                        
                        batch = DataProto(
                            batch=padded_batch_dict,
                            non_tensor_batch=padded_non_tensor_dict,
                            meta_info=batch.meta_info.copy()
                        )
                        # Immediately clean up temporary data
                        del padded_batch_dict, padded_non_tensor_dict
                        gc.collect()
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        if 'avg_score' not in val_metrics:
                            val_metrics['avg_score'] = np.mean([val_metrics[key] for key in val_metrics if key.startswith('val/test_score/')])
                        metrics.update(val_metrics)
                        self.maybe_save_best_hf(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics_ours(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # add contrastive loss metrics if available
                if 'contrastive_loss' in batch.batch and 'contrastive_mask' in batch.batch:
                    contrastive_mask = batch.batch['contrastive_mask']
                    if contrastive_mask.any():
                        contrastive_loss = batch.batch['contrastive_loss'][contrastive_mask]
                        metrics['contrastive/loss_mean'] = contrastive_loss.mean().item()
                        metrics['contrastive/loss_max'] = contrastive_loss.max().item()
                        metrics['contrastive/loss_min'] = contrastive_loss.min().item()
                        metrics['contrastive/loss_std'] = contrastive_loss.std().item()
                        metrics['contrastive/n_samples'] = contrastive_mask.sum().item()
                        metrics['contrastive/sample_ratio'] = contrastive_mask.sum().item() / contrastive_mask.shape[0]
                        
                        # 添加prompt级别的统计
                        if 'contrastive_prompt_stats' in batch.meta_info:
                            prompt_stats = batch.meta_info['contrastive_prompt_stats']
                            metrics['contrastive/n_prompts'] = len(prompt_stats)
                            
                            # 聚合prompt级别的统计
                            if prompt_stats:
                                avg_n_success = np.mean([stats['n_success'] for stats in prompt_stats.values()])
                                avg_n_fail = np.mean([stats['n_fail'] for stats in prompt_stats.values()])
                                avg_pairwise_loss = np.mean([stats['pairwise_loss_mean'] for stats in prompt_stats.values()])
                                avg_balance_weight = np.mean([stats['balance_weight'] for stats in prompt_stats.values()])
                                
                                metrics['contrastive/prompt_avg_n_success'] = avg_n_success
                                metrics['contrastive/prompt_avg_n_fail'] = avg_n_fail
                                metrics['contrastive/prompt_pairwise_loss_mean'] = avg_pairwise_loss
                                metrics['contrastive/prompt_avg_balance_weight'] = avg_balance_weight
                        
                        # 添加加权loss的统计
                        if 'contrastive_prompt_weighted_loss' in batch.meta_info:
                            weighted_losses = batch.meta_info['contrastive_prompt_weighted_loss']
                            if weighted_losses:
                                weighted_loss_values = list(weighted_losses.values())
                                metrics['contrastive/weighted_loss_mean'] = np.mean(weighted_loss_values)
                                metrics['contrastive/weighted_loss_max'] = np.max(weighted_loss_values)
                                metrics['contrastive/weighted_loss_min'] = np.min(weighted_loss_values)
                                metrics['contrastive/weighted_loss_std'] = np.std(weighted_loss_values)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                # Force memory cleanup
                memory_cleanup()
                check_memory_usage("batch_end")
                
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    return

    def maybe_save_best_hf(self, val_metrics: dict):
        import json
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'best',
                                        f'actor')
        
        os.makedirs(actor_local_path, exist_ok=True)
        if os.path.exists(f'{actor_local_path}/metrics.json'):
            with open(f'{actor_local_path}/metrics.json', 'r') as f:
                metrics = json.load(f)
            best_score = metrics['best_avg_score']
        else:
            print('Find no current best saved. Best score is set to -inf')
            best_score = -float('inf')
        
        cur_score = val_metrics['avg_score']
        
        if cur_score > best_score:
            print(f'Saving best checkpoint with score {cur_score} at {actor_local_path}')
            best_score = cur_score
            self.actor_rollout_wg.save_checkpoint_hf(actor_local_path)
            with open(f'{actor_local_path}/metrics.json', 'w') as f:
                f.write(json.dumps({'best_avg_score': best_score, 'global_step': self.global_steps})+'\n')
        
def compute_data_metrics_ours(batch, use_critic=True):
    # TODO: add response length
    whether_keep = ~batch.batch['whether_pad']
    # Remove tensor data from batch
    filtered_batch_dict = {}
    for key in batch.batch.keys():
        if key != 'batch_size':
            filtered_batch_dict[key] = batch.batch[key][whether_keep]
    
    # Remove data from non_tensor_batch
    filtered_non_tensor_dict = {}
    for key in batch.non_tensor_batch.keys():
        filtered_non_tensor_dict[key] = batch.non_tensor_batch[key][whether_keep.cpu().numpy()]
    
    # Calculate new batch size
    new_batch_size = whether_keep.sum().item()
    filtered_batch_dict = TensorDict(filtered_batch_dict, batch_size=[new_batch_size])
    
    # Rebuild batch
    batch = DataProto(
        batch=filtered_batch_dict,
        non_tensor_batch=filtered_non_tensor_dict,
        meta_info=batch.meta_info.copy()
    )
    del filtered_batch_dict
    del filtered_non_tensor_dict
    gc.collect()

    
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    from verl.trainer.ppo.ray_trainer import _compute_response_info
    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    # compute on/off policy stats
    off_policy_mask = batch.batch['prefix_mask'].any(-1) # [bsz, ]
    on_policy_mask = ~off_policy_mask
    off_response_length = response_length[off_policy_mask]
    on_response_length = response_length[on_policy_mask]
    
    off_example_ratio = off_policy_mask.sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())

    off_sequence_score = sequence_score[off_policy_mask]
    on_sequence_score = sequence_score[on_policy_mask]

    # on/off prompt score
    # batch_size = batch.batch.batch_size[0] / n_samples
    # on_prompt_score, off_prompt_score = [], []
    # sequence_score = sequence_score.reshape(batch_size, n_samples, sequence_score.shape[-1]) # [bsz, n, l]
    # for i in range(batch_size):
    #     on_prompt_score.append(sequence_score[i][on_policy_mask[i]].mean())
    #     off_prompt_score.append(sequence_score[i][off_policy_mask[i]].mean())

    # on_prompt_score = torch.cat(on_prompt_score, dim=0)
    # off_prompt_score = torch.cat(off_prompt_score, dim=0)

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # on/off policy response length
        'on_off_metrics/on_response_length_mean':
            torch.mean(on_response_length).detach().item(),
        'on_off_metrics/off_response_length_mean':
            torch.mean(off_response_length).detach().item(),
        'on_off_metrics/on_score':
            torch.mean(on_sequence_score).detach().item(),
        'on_off_metrics/off_score':
            torch.mean(off_sequence_score).detach().item(),
        # 'on_off_metrics/on_prompt_score':
        #     torch.mean(on_prompt_score).detach().item(),
        # 'on_off_metrics/off_prompt_score':
        #     torch.mean(off_prompt_score).detach().item(),
        'on_off_metrics/off_example_ratio':
            off_example_ratio,
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    if 'whether_off' in batch.batch:
        on_data_ratio = on_policy_mask.sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        off_data_ratio = batch.batch['whether_off'].sum().item() / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        sft_data_ratio = (off_policy_mask.sum().item() - batch.batch['whether_off'].sum().item()) / (off_policy_mask.sum().item() + on_policy_mask.sum().item())
        metrics['uni/on_data_ratio'] = on_data_ratio
        metrics['uni/off_data_ratio'] = off_data_ratio
        metrics['uni/sft_data_ratio'] = sft_data_ratio
    return metrics
