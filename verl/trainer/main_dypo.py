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
DYPO (Dynamic Policy Optimization) main entry point.

DYPO dynamically classifies training samples into:
  - Hard samples (all rollouts fail) -> SFT training
  - Easy samples (all rollouts succeed) -> filtered out
  - Partial samples (mixed success/failure) -> RL training (GRPO)

This module provides the reward manager and Hydra-based launcher.
"""

import hydra
import os
import ray
import numpy as np
import torch
from typing import List, Union

from verl import DataProto
from verl.trainer.ppo.ray_trainer_dypo import RayPPOTrainer, Role
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.reward_score import gsm8k, math


def _select_rm_score_fn(data_source, reward_impl_version):
    """Select reward scoring function based on data source and version.

    Users can extend this function to add custom reward functions for new data sources.
    """
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math.compute_score
    else:
        if reward_impl_version == 0:
            from deepscaler.rewards.math_reward import deepscaler_reward_fn
            return deepscaler_reward_fn
        elif reward_impl_version == 1:
            from verl.mix_src.reward_with_format import deepscaler_reward_fn_impl1
            return deepscaler_reward_fn_impl1
        elif reward_impl_version == 2:
            from deepscaler.rewards.math_reward import deepscaler_reward_fn, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
            def deepscaler_reward_fn_nothink(solution_str, ground_truth, enable_llm=False):
                solution_str = f"{THOUGHT_DELIMITER_START}\n{THOUGHT_DELIMITER_END}\n{solution_str}"
                return deepscaler_reward_fn(solution_str, ground_truth, enable_llm)
            return deepscaler_reward_fn_nothink
        elif reward_impl_version == 3:
            from verl.mix_src.math_verify_reward import reward_fn_math_verify
            return reward_fn_math_verify
        elif reward_impl_version == 4:
            from verl.mix_src.math_verify_reward import reward_fn_math_verify_no_think
            return reward_fn_math_verify_no_think
        elif reward_impl_version == 5:
            from verl.mix_src.prime_math import compute_score
            return compute_score
        elif reward_impl_version == 6:
            from verl.mix_src.entropy_math import compute_score
            return compute_score
        else:
            raise NotImplementedError(f"reward_impl_version={reward_impl_version} is not supported")


class RewardManager():
    """Reward manager for DYPO training.

    Supports multiple reward implementations selectable via `reward_impl_version`:
      - 0: deepscaler (default)
      - 1: deepscaler with format reward
      - 2: deepscaler no-think
      - 3: math-verify
      - 4: math-verify no-think
      - 5: prime math
      - 6: entropy math
    """

    def __init__(self, tokenizer, num_examine, reward_impl_version, format_penalty_coef=0.0) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_impl_version = reward_impl_version
        self.format_penalty_coef = format_penalty_coef

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        def process_item(args):
            i, data_item = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = valid_response_ids
            sequences_str = self.tokenizer.decode(sequences)

            if self.reward_impl_version != 4:
                try:
                    from deepscaler.globals import THOUGHT_DELIMITER_START
                    sequences_str = THOUGHT_DELIMITER_START + '\n' + sequences_str
                except ImportError:
                    sequences_str = '<think>\n' + sequences_str

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source, reward_impl_version=self.reward_impl_version)
            score_result = compute_score_fn(sequences_str, ground_truth)

            if isinstance(score_result, dict):
                score = score_result.get('total', 0.0)
            else:
                score = score_result

            format_penalty = 0.0
            if self.format_penalty_coef > 0:
                try:
                    from deepscaler.globals import THOUGHT_DELIMITER_START
                except ImportError:
                    THOUGHT_DELIMITER_START = '<think>'
                think_start_pos = sequences_str.find(THOUGHT_DELIMITER_START)
                if think_start_pos > 0:
                    garbage_ratio = think_start_pos / len(sequences_str)
                    format_penalty = -self.format_penalty_coef * garbage_ratio
                elif think_start_pos == -1:
                    format_penalty = -self.format_penalty_coef

            final_score = score + format_penalty
            return i, final_score, valid_response_length

        args = [(i, data[i]) for i in range(len(data))]
        results = [process_item(arg) for arg in args]

        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


@hydra.main(config_path="config", config_name="ppo_trainer_dypo", version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
            }},
            num_cpus=config.ray_init.num_cpus,
        )

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )

        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
            assert config.critic.strategy in ["fsdp", "fsdp2"]
            from verl.single_controller.ray import RayWorkerGroup
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = RayWorkerGroup
        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

            actor_rollout_cls = (
                AsyncActorRolloutRefWorker
                if config.actor_rollout_ref.rollout.mode == "async"
                else ActorRolloutRefWorker
            )
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(actor_rollout_cls),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        if config.reward_model.enable:
            if config.reward_model.strategy in ["fsdp", "fsdp2"]:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        format_penalty_coef = config.data.get('format_penalty_coef', 0.0)
        reward_fn = RewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            reward_impl_version=config.data.reward_impl_version,
            format_penalty_coef=format_penalty_coef,
        )
        val_reward_fn = RewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            reward_impl_version=config.data.reward_impl_version,
            format_penalty_coef=format_penalty_coef,
        )
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            device_name=config.trainer.device,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
