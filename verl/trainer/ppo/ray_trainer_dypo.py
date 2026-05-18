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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import gc
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type
import math

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server_gspo import AsyncLLMServerManager

WorkerType = Type[Worker]


def memory_cleanup():
    """Force memory cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        if (vocab_size is None or vocab_size <= 0) and hasattr(self.tokenizer, "__len__"):
            try:
                vocab_size = len(self.tokenizer)
            except Exception:
                vocab_size = None
        self.log_vocab_size = math.log(vocab_size) if vocab_size and vocab_size > 0 else None

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        # Initialize buffers for sample management with disk offloading
        import tempfile
        import time
        temp_base_dir = os.path.join(config.trainer.default_local_dir, "tmp")
        os.makedirs(temp_base_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.temp_sample_dir = os.path.join(temp_base_dir, f"dypo_samples_{timestamp}_{os.getpid()}")
        os.makedirs(self.temp_sample_dir, exist_ok=True)
        
        self.hard_samples_buffer = []  # Store file paths instead of actual data
        self.rl_samples_buffer = []  # Store file paths instead of actual data
        self.sft_batch_size = config.data.train_batch_size
        self.sample_counter = 0  # Counter for unique sample filenames
        print(f"[Sample Management] Initialized disk-based sample buffers")
        print(f"[Sample Management] Temporary directory: {self.temp_sample_dir}")
        print(f"[Sample Management] SFT batch size: {self.sft_batch_size}")
        
        
    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

        print("[validate_config] All configuration checks passed successfully!")


    def _create_dataloader(self):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from torch.utils.data import DataLoader, SequentialSampler
        from verl.utils.dataset.rl_dataset_dypo import RLHFDataset, collate_fn
        from verl.utils.dataset.rl_dataset_with_target import RLHFDatasetWithTarget
        self.train_dataset = RLHFDatasetWithTarget(
            config=self.config,
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=False, return_raw_chat=self.config.data.get('return_raw_chat', False),
            truncation='error',
            max_target_length=self.config.actor_rollout_ref.rollout.max_prefix_len,
            filter_targets=self.config.data.get('filter_targets', False),
            suffix_prompt=self.config.data.get('suffix_prompt', ''),
            sample_target_ratio=self.config.data.get('sample_target_ratio', 1.0))

        print(f'[Dataset Info] Train dataset size after filtering: {len(self.train_dataset)}')
        
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            from verl.mix_src.rl_dataset_with_target import ResumableRandomSampler
            sampler = ResumableRandomSampler(data_source=self.train_dataset)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           num_workers=8,
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

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

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
            
            # Set SFT learning rate if not already configured
            if not hasattr(self.config.actor_rollout_ref.actor.optim, 'sft_lr'):
                # Default SFT learning rate to be 10x the RL learning rate for better convergence on hard samples
                rl_lr = self.config.actor_rollout_ref.actor.optim.lr
                sft_lr = rl_lr * 10.0  # 10x learning rate for SFT
                self.config.actor_rollout_ref.actor.optim.sft_lr = sft_lr
                print(f"[Config] Setting SFT learning rate to {sft_lr} (10x RL learning rate {rl_lr})")


    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            # if "multi_modal_data" in test_batch.non_tensor_batch:
            #     non_tensor_batch_keys_to_pop.append("multi_modal_data")
            # if "raw_prompt" in test_batch.non_tensor_batch:
            #     non_tensor_batch_keys_to_pop.append("raw_prompt")
            # if "tools_kwargs" in test_batch.non_tensor_batch:
            #     non_tensor_batch_keys_to_pop.append("tools_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                self.async_rollout_manager.sleep()

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            print(input_texts, output_texts)
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        print(val_data_dir)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (var_name == core_var) and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"]) and (f"@{n_max}" in metric_name):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        weighted_mean = float(np.mean(sample_scores))
        metric_dict["val-core/mean@1"] = weighted_mean

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        # BaseCheckpointManager.local_mkdir(local_global_step_folder)
        os.makedirs(local_global_step_folder, exist_ok=True)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)


    def perform_incremental_sft_update(self, timing_raw):
        """
        Perform a single SFT update on exactly sft_batch_size samples from buffer
        This is called when buffer reaches the required size
        Samples are loaded from disk to save CPU memory
        """
        if len(self.hard_samples_buffer) < self.sft_batch_size:
            print(f"[SFT Update] Insufficient samples ({len(self.hard_samples_buffer)} < {self.sft_batch_size}), skipping")
            return {}
        
        print(f"\n[SFT Incremental Update] Processing {self.sft_batch_size} hard samples from disk")
        
        # Take file paths from buffer
        sample_files = self.hard_samples_buffer[:self.sft_batch_size]
        
        # Load samples from disk
        import pickle
        resolved_samples = []
        for sample_file in sample_files:
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
                resolved_samples.append(sample_data)
        
        print(f"[DEBUG] Loaded {len(resolved_samples)} samples from disk.")
        
        # Merge tensor data
        sft_batch_dict = {}
        for key in resolved_samples[0]['batch'].keys():
            if key != 'batch_size':
                # Ensure all tensors are actual tensors before concatenating
                tensors_to_cat = []
                for s in resolved_samples:
                    tensor = s['batch'][key]
                    
                    # Handle Ray ObjectRef (should be rare after initial resolution)
                    if isinstance(tensor, ray.ObjectRef):
                        tensor = ray.get(tensor)
                    
                    # Add tensor to concatenation list
                    tensors_to_cat.append(tensor)
                sft_batch_dict[key] = torch.cat(tensors_to_cat, dim=0)

        sft_batch_dict = TensorDict(sft_batch_dict, batch_size=[self.sft_batch_size])
        
        # Compute global_token_num (required by fsdp_workers)
        if 'attention_mask' in sft_batch_dict:
            global_token_num = torch.sum(sft_batch_dict['attention_mask'], dim=-1).tolist()
        else:
            global_token_num = [(ids != 0).sum().item() for ids in sft_batch_dict['input_ids']]
        
        sft_batch = DataProto(
            batch=sft_batch_dict,
            meta_info={
                'global_steps': self.global_steps,
                'is_pure_sft': True,
                'temperature': self.config.actor_rollout_ref.rollout.temperature,
                'micro_batch_size': self.config.actor_rollout_ref.actor.ppo_mini_batch_size,
                'use_dynamic_bsz': self.config.actor_rollout_ref.actor.use_dynamic_bsz,
                'global_token_num': global_token_num,
            }
        )
        
        # Add placeholder fields for compatibility
        sft_batch.batch['old_log_probs'] = torch.zeros_like(sft_batch.batch['responses'], dtype=torch.float32)
        sft_batch.batch['advantages'] = torch.zeros_like(sft_batch.batch['responses'], dtype=torch.float32)
        
        # Execute actor update in pure SFT mode
        with _timer("sft_update_actor", timing_raw):
            print("Input device:", sft_batch.batch['input_ids'].device)
            actor_output = self.actor_rollout_wg.update_actor(sft_batch)
        
        # Collect metrics
        batch_metrics = reduce_metrics(actor_output.meta_info.get("metrics", {}))
        sft_metrics = {f"sft/{k}": v for k, v in batch_metrics.items()}
        sft_metrics['sft/num_samples'] = self.sft_batch_size
        
        print(f"[SFT Incremental Update] Completed. Processed {self.sft_batch_size} samples")
        
        # Delete temporary files from disk
        for sample_file in sample_files:
            try:
                os.remove(sample_file)
            except Exception as e:
                print(f"[Warning] Failed to remove temp file {sample_file}: {e}")
        
        # Cleanup memory
        del sft_batch, sft_batch_dict, actor_output, sample_files, resolved_samples
        memory_cleanup()

        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete

        return sft_metrics

    def perform_incremental_rl_update(self, timing_raw):
        """
        Perform a single RL update on available samples from buffer
        This is called when buffer reaches the minimum required size
        Samples are loaded from disk to save CPU memory
        """
        dp_size = self.actor_rollout_wg.world_size
        n_rollouts = self.config.actor_rollout_ref.rollout.n
        min_rl_samples = dp_size * n_rollouts
        
        if len(self.rl_samples_buffer) < min_rl_samples:
            print(f"[RL Update] Insufficient samples ({len(self.rl_samples_buffer)} < {min_rl_samples}), skipping")
            return {}
        
        print(f"\n[RL Incremental Update] Processing {len(self.rl_samples_buffer)} RL samples from disk")
        
        # Process all available samples
        num_samples = len(self.rl_samples_buffer)
        num_trainable = (num_samples // dp_size) * dp_size
        
        if num_trainable == 0:
            print(f"[RL Update] No trainable samples after dp_size alignment")
            return {}
        
        # Ensure we have enough samples for n_rollouts grouping
        num_groups = num_trainable // n_rollouts  
        num_trainable = num_groups * n_rollouts  
        
        if num_trainable == 0:
            print(f"[RL Update] No trainable samples after n_rollouts grouping")
            return {}
                
        sample_files = self.rl_samples_buffer[:num_trainable]
        
        # Load samples from disk
        import pickle
        resolved_samples = []
        for sample_file in sample_files:
            with open(sample_file, 'rb') as f:
                sample_data = pickle.load(f)
                resolved_samples.append(sample_data)
        
        print(f"[DEBUG] Loaded {len(resolved_samples)} RL samples from disk.")
        
        index = np.repeat(np.arange(num_groups), n_rollouts)  # e.g., [0,0,0,0,1,1,1,1,...]
        
        merged_batch_dict = {}
        for key in resolved_samples[0]['batch'].keys():
            if key != 'batch_size':
                # Ensure all tensors are actual tensors before concatenating
                tensors_to_cat = []
                for s in resolved_samples:
                    tensor = s['batch'][key]
                    # Handle Ray ObjectRef (should be rare after initial resolution)
                    if isinstance(tensor, ray.ObjectRef):
                        tensor = ray.get(tensor)
                    # Add tensor to concatenation list
                    tensors_to_cat.append(tensor)
                merged_batch_dict[key] = torch.cat(tensors_to_cat, dim=0)
        
        merged_non_tensor_dict = {}
        for key in resolved_samples[0]['non_tensor_batch'].keys():
            # Ensure all arrays are actual arrays before concatenating
            arrays_to_cat = []
            for s in resolved_samples:
                array = s['non_tensor_batch'][key]
                # Handle Ray ObjectRef (should be rare after initial resolution)
                if isinstance(array, ray.ObjectRef):
                    array = ray.get(array)
                # Add array to concatenation list
                arrays_to_cat.append(array)
            merged_non_tensor_dict[key] = np.concatenate(arrays_to_cat, axis=0)
        
        merged_non_tensor_dict['grpo_group_index'] = index
        
        merged_batch_dict = TensorDict(merged_batch_dict, batch_size=[num_trainable])
        batch = DataProto(
            batch=merged_batch_dict,
            non_tensor_batch=merged_non_tensor_dict,
            meta_info={'global_steps': self.global_steps}
        )
        
        print(f"[RL Incremental Update] Computing log probs and advantages...")
        
        # Process the batch
        rl_metrics = self._process_rl_batch(batch, timing_raw)
        
        # Delete temporary files from disk
        for sample_file in sample_files:
            try:
                os.remove(sample_file)
            except Exception as e:
                print(f"[Warning] Failed to remove temp file {sample_file}: {e}")
        
        # Cleanup memory
        del batch, merged_batch_dict, merged_non_tensor_dict, sample_files, resolved_samples
        memory_cleanup()
        
        print(f"[RL Incremental Update] Completed. Processed {num_trainable} samples")
        
        return rl_metrics

    def _process_rl_batch(self, batch: DataProto, timing_raw: dict) -> dict:
        """
        Process a single batch for RL update.
        Returns the metrics for this batch.
        """
        rl_metrics = {}
        
        # Compute old_log_probs
        with _timer("rl_old_log_prob", timing_raw):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["attention_mask"][:, -batch.batch["responses"].size(1):]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
            rl_metrics["rl/entropy_loss"] = entropy_loss.detach().item()
            total_tokens = response_masks.sum().clamp(min=1).item()
            rl_metrics["rl/entropy_loss_token_mean"] = rl_metrics["rl/entropy_loss"] / total_tokens
            if self.log_vocab_size:
                rl_metrics["rl/entropy_loss_token_mean_norm01"] = min(
                    rl_metrics["rl/entropy_loss_token_mean"] / self.log_vocab_size, 1.0
                )
            
            # Compute detailed entropy statistics
            masked_entropys = entropys * response_masks
            valid_entropys = masked_entropys[response_masks.bool()]
            if valid_entropys.numel() > 0:
                entropy_mean = valid_entropys.mean().item()
                if self.log_vocab_size:
                    rl_metrics["rl/entropy_mean"] = min(entropy_mean / self.log_vocab_size, 1.0)
                # Per-sequence entropy (sum over tokens, then statistics over sequences)
                seq_entropys = masked_entropys.sum(dim=-1)
                seq_lengths = response_masks.sum(dim=-1).clamp(min=1)
                seq_avg_entropys = seq_entropys / seq_lengths
                rl_metrics["rl/entropy_per_seq_mean"] = seq_avg_entropys.mean().item()
                if self.log_vocab_size:
                    rl_metrics["rl/entropy_per_seq_mean_norm"] = rl_metrics["rl/entropy_per_seq_mean"] / self.log_vocab_size
                    rl_metrics["rl/entropy_per_seq_mean_norm01"] = min(
                        rl_metrics["rl/entropy_per_seq_mean_norm"], 1.0
                    )
            
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)
        
        # Compute reference log_prob if needed
        if self.use_reference_policy:
            with _timer("rl_ref", timing_raw):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
        
        # Apply KL penalty if needed
        if self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty)
            rl_metrics.update({f"rl/{k}": v for k, v in kl_metrics.items()})
        else:
            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

        # Compute detailed token_level_rewards statistics
        rewards = batch.batch['token_level_rewards']
        response_length = batch.batch["responses"].size(1)
        response_masks = batch.batch["attention_mask"][:, -response_length:]
        masked_rewards = rewards * response_masks
        valid_rewards = masked_rewards[response_masks.bool()]
        if valid_rewards.numel() > 0:
            rl_metrics["rl/token_level_rewards_mean"] = valid_rewards.mean().item()
            rl_metrics["rl/token_level_rewards_std"] = valid_rewards.std().item() if valid_rewards.numel() > 1 else 0.0
            rl_metrics["rl/token_level_rewards_max"] = valid_rewards.max().item()
            rl_metrics["rl/token_level_rewards_min"] = valid_rewards.min().item()
            # Per-sequence rewards (sum over tokens, then statistics over sequences)
            seq_rewards = masked_rewards.sum(dim=-1)
            rl_metrics["rl/seq_rewards_mean"] = seq_rewards.mean().item()
            rl_metrics["rl/seq_rewards_std"] = seq_rewards.std().item() if seq_rewards.numel() > 1 else 0.0
            rl_metrics["rl/seq_rewards_max"] = seq_rewards.max().item()
            rl_metrics["rl/seq_rewards_min"] = seq_rewards.min().item()
            # Count positive/negative/zero rewards
            rl_metrics["rl/seq_rewards_positive_ratio"] = (seq_rewards > 0).float().mean().item()
            rl_metrics["rl/seq_rewards_negative_ratio"] = (seq_rewards < 0).float().mean().item()
            rl_metrics["rl/seq_rewards_zero_ratio"] = (seq_rewards == 0).float().mean().item()

        # Compute contrastive learning loss for RL samples
        if self.config.trainer.get('use_contrastive_loss', False):
            with _timer("rl_contrastive_loss", timing_raw):
                # Determine reward values for contrastive learning
                if self.config.data.reward_impl_version == 0:
                    fail_value = 0
                    success_value = 1
                elif self.config.data.reward_impl_version == 1:
                    fail_value = -0.5
                    success_value = 1
                elif self.config.data.reward_impl_version in [2, 3, 4, 5, 6, 7]:
                    fail_value = 0
                    success_value = 1
                else:
                    raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                
                batch = self.compute_contrastive_loss(batch, success_value, fail_value)
                
                # Extract contrastive learning metrics
                if 'contrastive_uid_loss' in batch.meta_info:
                    contrastive_losses = list(batch.meta_info['contrastive_uid_loss'].values())
                    if contrastive_losses:
                        rl_metrics["rl/contrastive_loss_mean"] = np.mean(contrastive_losses)
                        rl_metrics["rl/contrastive_loss_std"] = np.std(contrastive_losses)
                        rl_metrics["rl/contrastive_prompts_count"] = len(contrastive_losses)
                
                if 'contrastive_prompt_weighted_loss' in batch.meta_info:
                    weighted_losses = list(batch.meta_info['contrastive_prompt_weighted_loss'].values())
                    if weighted_losses:
                        rl_metrics["rl/contrastive_weighted_loss_mean"] = np.mean(weighted_losses)
                        rl_metrics["rl/contrastive_weighted_loss_std"] = np.std(weighted_losses)
        
        # Compute advantages
        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            grpo_use_std=self.config.algorithm.grpo_use_std
        )
        
        # Compute values if using critic
        if self.use_critic:
            with _timer('rl_values', timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)
        
        # Balance batch
        self._balance_batch(batch, metrics=rl_metrics)
        
        # Compute global_token_num
        batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
        
        # Update critic if used
        if self.use_critic:
            with _timer('rl_update_critic', timing_raw):
                critic_output = self.critic_wg.update_critic(batch)
            critic_metrics = reduce_metrics(critic_output.meta_info['metrics'])
            rl_metrics.update({f"rl/{k}": v for k, v in critic_metrics.items()})
        
        # Update actor
        with _timer("rl_update_actor", timing_raw):
            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
            entropy_coeff = self.config.actor_rollout_ref.actor.get("entropy_coeff", 0.01)
            batch.meta_info["entropy_coeff"] = entropy_coeff
            actor_output = self.actor_rollout_wg.update_actor(batch)
        
        actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        rl_metrics.update({f"rl/{k}": v for k, v in actor_metrics.items()})
        
        # Cleanup
        del batch, actor_output
        if self.use_critic:
            del critic_output
        
        return rl_metrics

    
    def compute_contrastive_loss(self, batch: DataProto, success_value: float, fail_value: float):
        """
        Compute contrastive learning loss for RL samples from prompts that have both success and failure.
        Uses DPO-style pairwise ranking loss: make successful samples have higher log_prob than failed samples.
        
        Core idea:
        - For each prompt UID in RL buffer, if both success and failure samples exist
        - Compute the contrastive learning loss for this prompt (average of all success-fail pairs)
        - Assign this loss to all samples of this prompt
        - Successful responses as preferred (y_w), failed responses as rejected (y_l)
        - Loss = -log(sigmoid(beta * (log_prob(y_w) - log_prob(y_l))))
        
        Returns:
            batch: Updated batch containing:
                - batch.batch['contrastive_loss']: [bsz] Contrastive loss for each sample
                - batch.batch['contrastive_mask']: [bsz] Mark which samples have contrastive loss
                - batch.non_tensor_batch['contrastive_uid']: [bsz] UID corresponding to each sample
                - batch.meta_info['contrastive_uid_loss']: dict[uid -> float] Contrastive loss for each prompt
                - batch.meta_info['contrastive_prompt_stats']: dict[uid -> stats] Detailed statistics for each prompt
                - batch.meta_info['contrastive_prompt_weighted_loss']: dict[uid -> float] Weighted loss for each uid
        """
        uids = np.array(batch.non_tensor_batch['uid'])
        unique_uids = np.unique(uids)
        
        reward_tensor = batch.batch['token_level_scores']
        old_log_probs = batch.batch['old_log_probs']
        responses = batch.batch['responses']
        attention_mask = batch.batch['attention_mask']
        
        device = old_log_probs.device
        response_length = responses.size(-1)
        response_mask = attention_mask[:, -response_length:]
        batch_size = responses.size(0)
        
        # Initialize
        contrastive_loss = torch.zeros(batch_size, dtype=torch.float32, device=device)
        contrastive_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Used to store the uid corresponding to each sample
        contrastive_uids = np.array([''] * batch_size, dtype=object)
        
        # Used to store contrastive loss statistics and weighted loss for each prompt
        prompt_contrastive_stats = {}
        prompt_weighted_loss = {}
        prompt_uid_loss = {}
        
        beta = self.config.trainer.get('contrastive_beta', 0.1)
        
        for uid in unique_uids:
            # Get all samples for this UID
            uid_mask = torch.from_numpy(uids == uid).to(device)
            if not uid_mask.any():
                continue
                
            uid_rewards = reward_tensor[uid_mask].sum(-1)  # [n]
            success_mask = (uid_rewards == success_value)
            fail_mask = (uid_rewards == fail_value)
            
            # Only handle cases that have both success and failure
            if not (success_mask.any() and fail_mask.any()):
                prompt_uid_loss[str(uid)] = 0
                continue
                
            # Get indices
            indices = torch.where(uid_mask)[0]
            success_indices = indices[success_mask]
            fail_indices = indices[fail_mask]
            
            # Compute sequence log probs (only response part)
            success_logp = (old_log_probs[success_indices] * response_mask[success_indices]).sum(-1)  # [S]
            fail_logp = (old_log_probs[fail_indices] * response_mask[fail_indices]).sum(-1)        # [F]
            
            # Vectorized Pairwise DPO loss: [S, F]
            diff = success_logp.unsqueeze(1) - fail_logp.unsqueeze(0)  # [S, F]
            logits = beta * diff
            pairwise_loss = -torch.nn.functional.logsigmoid(logits)    # [S, F]
            
            # Compute the average contrastive learning loss for this prompt
            prompt_avg_loss = pairwise_loss.mean().item()
            
            # Compute weighted loss (weighted based on success/failure ratio)
            n_success = success_mask.sum().item()
            n_fail = fail_mask.sum().item()
            total_samples = n_success + n_fail
            
            # Weighting strategy: harder problems (success/fail ratio close to 0.5) have higher weight
            balance_weight = 4.0 * (n_success / total_samples) * (n_fail / total_samples)
            weighted_loss = prompt_avg_loss * balance_weight
            
            # Can also use other weighting strategies
            if self.config.trainer.get('contrastive_weight_type', 'balance') == 'uniform':
                weighted_loss = prompt_avg_loss
            elif self.config.trainer.get('contrastive_weight_type', 'balance') == 'inverse_success_rate':
                # Lower success rate, higher weight
                success_rate = n_success / total_samples
                weighted_loss = prompt_avg_loss * (1.0 - success_rate)
            
            # Assign this prompt's loss to all samples of this UID
            contrastive_loss[indices] = prompt_avg_loss
            contrastive_mask[indices] = True
            
            # Record the uid corresponding to each sample
            uid_indices_np = indices.cpu().numpy()
            contrastive_uids[uid_indices_np] = str(uid)
            
            # Record the loss for this uid
            prompt_uid_loss[str(uid)] = prompt_avg_loss
            prompt_weighted_loss[str(uid)] = weighted_loss
            
            # Record statistics for this prompt
            prompt_contrastive_stats[str(uid)] = {
                'n_success': n_success,
                'n_fail': n_fail,
                'n_samples': uid_mask.sum().item(),
                'success_loss_mean': pairwise_loss.mean(dim=1).mean().item(),
                'success_loss_std': pairwise_loss.mean(dim=1).std().item() if n_success > 1 else 0.0,
                'fail_loss_mean': pairwise_loss.mean(dim=0).mean().item(),
                'fail_loss_std': pairwise_loss.mean(dim=0).std().item() if n_fail > 1 else 0.0,
                'pairwise_loss_mean': prompt_avg_loss,
                'pairwise_loss_max': pairwise_loss.max().item(),
                'pairwise_loss_min': pairwise_loss.min().item(),
                'balance_weight': balance_weight,
                'weighted_loss': weighted_loss,
            }

        # Store loss and mask in batch
        batch.batch['contrastive_loss'] = contrastive_loss
        batch.batch['contrastive_mask'] = contrastive_mask

        # Store the uid corresponding to each sample in non_tensor_batch
        batch.non_tensor_batch['contrastive_uid'] = contrastive_uids
        
        # Store prompt-level statistics and loss in meta_info
        batch.meta_info['contrastive_uid_loss'] = prompt_uid_loss
        batch.meta_info['contrastive_prompt_stats'] = prompt_contrastive_stats
        batch.meta_info['contrastive_prompt_weighted_loss'] = prompt_weighted_loss


        return batch
    
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()


        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        if self.config.trainer.remove_sfted_data:
            sfted_data_item_list = []
            
        for epoch in range(self.config.trainer.total_epochs):
            print(f"🔄 Starting epoch {epoch+1}/{self.config.trainer.total_epochs} training")
            
            if self.config.trainer.remove_sfted_data:
                if len(sfted_data_item_list) > 0:
                    print(f"🗑️  Removing {len(sfted_data_item_list)} SFT data items")
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
                    print(f"✅ Data loader reconstruction completed, batch count: {len(self.train_dataloader)}")
                    
                sfted_data_item_list = []
                memory_cleanup()

            print(f"📊 Data loader status: {len(self.train_dataloader)} batches")
            
            # ==================== Incremental Training Loop ====================
            print(f"\n{'='*80}")
            print(f"Incremental Training - Process samples and train immediately")
            print(f"{'='*80}\n")
            
            epoch_metrics = {}
            epoch_timing_raw = {}
            
            if True:  # Always process samples
                for batch_idx, batch_dict in enumerate(self.train_dataloader):
                    print(f"\n[Batch {batch_idx+1}/{len(self.train_dataloader)}] Processing batch")
                    
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    
                    # Prepare generation batch
                    if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                        batch.batch['raw_input_ids'] = batch.batch['input_ids'].clone()
                        batch.batch['raw_attention_mask'] = batch.batch['attention_mask'].clone()
                        batch.batch['raw_position_ids'] = batch.batch['position_ids'].clone()
                        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                    else:
                        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids', 'tgt_input_ids'])
                    gen_batch.meta_info['global_steps'] = self.global_steps
                    
                    # Generate sequences
                    with _timer("collection_gen", epoch_timing_raw):
                        gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    
                    # Assign UIDs
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    
                    # Repeat batch and union with generation output
                    if self.config.trainer.unify_strategy != 'no' and self.config.trainer.unify_strategy != 'soft':
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    else:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    
                    # Store ground truth for debugging before union
                    ground_truths = []
                    for i in range(len(batch)):
                        try:
                            gt = batch[i].non_tensor_batch['reward_model']['ground_truth']
                            ground_truths.append(gt)
                        except:
                            ground_truths.append("[not available]")

                    batch = batch.union(gen_batch_output)

                    # Compute rewards
                    reward_tensor = self.reward_fn(batch)
                    batch.batch['token_level_scores'] = reward_tensor

                    # Remove reward statistics debug prints - only keep output/ground_truth prints in custom_reward_functions.py
                    
                    # Determine reward values
                    if self.config.data.reward_impl_version == 0:
                        fail_value = 0
                        success_value = 1
                    elif self.config.data.reward_impl_version == 1:
                        fail_value = -0.5
                        success_value = 1
                    elif self.config.data.reward_impl_version in [2, 3, 4, 5, 6, 7]:
                        fail_value = 0
                        success_value = 1
                    else:
                        raise ValueError(f'Invalid reward implementation version: {self.config.data.reward_impl_version}')
                    
                    # Sample classification and immediate training
                    uids = batch.non_tensor_batch['uid']
                    unique_uids = np.unique(uids)
                    n_samples_per_uid = self.config.actor_rollout_ref.rollout.n
                    
                    n_hard_uids = 0
                    n_easy_uids = 0
                    n_partial_uids = 0
                    
                    for uid in unique_uids:
                        uid_mask = uids == uid
                        uid_rewards = reward_tensor[uid_mask].sum(-1)
                        on_solve_num = (uid_rewards == success_value).sum().item()

                        # Print hard/easy samples for debugging
                        if not hasattr(self, '_hard_sample_count'):
                            self._hard_sample_count = 0
                        if not hasattr(self, '_easy_sample_count'):
                            self._easy_sample_count = 0

                        if on_solve_num == 0 and self._hard_sample_count < 3:  # All rollouts failed
                            print(f"\n=== HARD Sample {self._hard_sample_count} (UID: {uid}, All {n_samples_per_uid} rollouts failed) ===")
                            # Print all rollouts for this hard sample
                            uid_indices = np.where(uid_mask)[0]
                            for i, idx in enumerate(uid_indices[:2]):  # Show first 2 rollouts
                                output_ids = batch.batch['responses'][idx]
                                output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                                print(f"Rollout {i}: {repr(output_text[:200])}...")
                            # Get ground truth from stored list
                            gt = ground_truths[uid_indices[0]] if uid_indices[0] < len(ground_truths) else "[not available]"
                            print(f"Ground Truth: {repr(gt[:200]) if gt != '[not available]' else gt}...")
                            self._hard_sample_count += 1

                        elif on_solve_num == n_samples_per_uid and self._easy_sample_count < 3:  # All rollouts succeeded
                            print(f"\n=== EASY Sample {self._easy_sample_count} (UID: {uid}, All {n_samples_per_uid} rollouts succeeded) ===")
                            # Print one successful rollout as example
                            uid_indices = np.where(uid_mask)[0]
                            first_idx = uid_indices[0]
                            output_ids = batch.batch['responses'][first_idx]
                            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                            print(f"Example Output: {repr(output_text[:200])}...")
                            # Get ground truth from stored list
                            gt = ground_truths[first_idx] if first_idx < len(ground_truths) else "[not available]"
                            print(f"Ground Truth: {repr(gt[:200]) if gt != '[not available]' else gt}...")
                            self._easy_sample_count += 1

                        if on_solve_num == 0:
                            # Hard sample: all failed -> SFT buffer
                            n_hard_uids += 1
                            uid_indices = np.where(uid_mask)[0]
                            first_idx = uid_indices[0]
                            
                            required_keys = ['prompts', 'tgt_input_ids', 'attention_mask', 'position_ids']
                            missing_keys = [k for k in required_keys if k not in batch.batch]
                            # print(missing_keys)
                            if not missing_keys:
                                # Ensure all tensors are actual tensors, not ObjectRefs
                                sample_batch = {}
                                for k, v in batch.batch.items():
                                    if k != 'batch_size':
                                        # Handle Ray ObjectRef first
                                        tensor = v[first_idx:first_idx+1]
                                        
                                        # Debug: Check if we have ObjectRef
                                        if isinstance(tensor, ray.ObjectRef):
                                            print(f"[DEBUG] Found ObjectRef for key '{k}' in hard sample storage")
                                            tensor = ray.get(tensor)
                                            print(f"[DEBUG] After ray.get for key '{k}', type: {type(tensor)}")
                                        
                                        # Additional check: if tensor is still a list of ObjectRefs
                                        if isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], ray.ObjectRef):
                                            print(f"[DEBUG] Key '{k}' is a list of ObjectRefs, resolving all...")
                                            tensor = [ray.get(obj) for obj in tensor]
                                            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0]
                                        
                                        # Clone and detach the tensor to ensure it's a real tensor without gradient computation graph
                                        if hasattr(tensor, 'clone'):
                                            sample_batch[k] = tensor.clone().detach()
                                        else:
                                            sample_batch[k] = tensor

                                sample_data = {
                                    'batch': sample_batch,
                                }
                                # Save to disk to avoid CPU memory accumulation
                                import pickle
                                sample_filename = os.path.join(self.temp_sample_dir, f"sft_sample_{self.sample_counter}.pkl")
                                self.sample_counter += 1
                                with open(sample_filename, 'wb') as f:
                                    pickle.dump(sample_data, f)
                                self.hard_samples_buffer.append(sample_filename)
                                print(f"  UID {uid}: Hard sample (0/{n_samples_per_uid}) -> SFT buffer (saved to disk)")
                            
                        elif on_solve_num == n_samples_per_uid:
                            # Easy sample: all succeeded -> filter out
                            n_easy_uids += 1
                            print(f"  UID {uid}: Easy sample ({n_samples_per_uid}/{n_samples_per_uid}) -> Filtered")
                            
                        else:
                            # Partial success: -> RL buffer
                            n_partial_uids += 1
                            uid_indices = np.where(uid_mask)[0]
                            
                            # Store all samples for this uid
                            for idx in uid_indices:
                                # Ensure all tensors are actual tensors, not ObjectRefs
                                sample_batch = {}
                                for k, v in batch.batch.items():
                                    if k != 'batch_size':
                                        # Handle Ray ObjectRef first
                                        tensor = v[idx:idx+1]
                                        
                                        # Debug: Check if we have ObjectRef
                                        if isinstance(tensor, ray.ObjectRef):
                                            print(f"[DEBUG] Found ObjectRef for key '{k}' in RL sample storage")
                                            tensor = ray.get(tensor)
                                            print(f"[DEBUG] After ray.get for key '{k}', type: {type(tensor)}")
                                        
                                        # Additional check: if tensor is still a list of ObjectRefs
                                        if isinstance(tensor, list) and len(tensor) > 0 and isinstance(tensor[0], ray.ObjectRef):
                                            print(f"[DEBUG] Key '{k}' is a list of ObjectRefs, resolving all...")
                                            tensor = [ray.get(obj) for obj in tensor]
                                            tensor = torch.stack(tensor) if len(tensor) > 1 else tensor[0]
                                        
                                        # Clone and detach the tensor to ensure it's a real tensor without gradient computation graph
                                        if hasattr(tensor, 'clone'):
                                            sample_batch[k] = tensor.clone().detach()
                                        else:
                                            sample_batch[k] = tensor

                                sample_non_tensor_batch = {}
                                for k, v in batch.non_tensor_batch.items():
                                    if hasattr(v, 'copy'):
                                        sample_non_tensor_batch[k] = v[idx:idx+1].copy()
                                    else:
                                        sample_non_tensor_batch[k] = v[idx:idx+1]
                                
                                sample_data = {
                                    'batch': sample_batch,
                                    'non_tensor_batch': sample_non_tensor_batch
                                }
                                # Save to disk to avoid CPU memory accumulation
                                import pickle
                                sample_filename = os.path.join(self.temp_sample_dir, f"rl_sample_{self.sample_counter}.pkl")
                                self.sample_counter += 1
                                with open(sample_filename, 'wb') as f:
                                    pickle.dump(sample_data, f)
                                self.rl_samples_buffer.append(sample_filename)
                            print(f"  UID {uid}: Partial success ({on_solve_num}/{n_samples_per_uid}) -> RL buffer (saved to disk)")
                    
                    print(f"[Batch {batch_idx+1} Summary] Hard: {n_hard_uids}, Easy: {n_easy_uids}, Partial: {n_partial_uids}")
                    print(f"[Current Buffers] SFT: {len(self.hard_samples_buffer)}, RL: {len(self.rl_samples_buffer)}")
                    
                    # ==================== Immediate Training Logic ====================
                    batch_metrics = {}
                    
                    # Check if SFT buffer is ready for training
                    if len(self.hard_samples_buffer) >= self.sft_batch_size:
                        print(f"\n[SFT Update] Buffer ready ({len(self.hard_samples_buffer)} >= {self.sft_batch_size}), performing SFT update")
                        sft_metrics = self.perform_incremental_sft_update(epoch_timing_raw)
                        batch_metrics.update(sft_metrics)
                        
                        # Remove trained samples from SFT buffer (keep only untrained ones)
                        self.hard_samples_buffer = self.hard_samples_buffer[self.sft_batch_size:]
                        print(f"[SFT Update] Completed, removed {self.sft_batch_size} trained samples, remaining: {len(self.hard_samples_buffer)}")
                    
                    # Check if RL buffer is ready for training
                    dp_size = self.actor_rollout_wg.world_size
                    n_rollouts = self.config.actor_rollout_ref.rollout.n
                    min_rl_samples = dp_size * n_rollouts
                    
                    if len(self.rl_samples_buffer) >= min_rl_samples:
                        print(f"\n[RL Update] Buffer ready ({len(self.rl_samples_buffer)} >= {min_rl_samples}), performing RL update")
                        rl_metrics = self.perform_incremental_rl_update(epoch_timing_raw)
                        batch_metrics.update(rl_metrics)
                        
                        # Calculate how many samples were actually trained
                        num_samples = len(self.rl_samples_buffer)
                        num_trainable = (num_samples // dp_size) * dp_size
                        num_groups = num_trainable // n_rollouts  
                        num_trained = num_groups * n_rollouts
                        
                        # Remove trained samples from RL buffer
                        self.rl_samples_buffer = self.rl_samples_buffer[num_trained:]
                        print(f"[RL Update] Completed, removed {num_trained} trained samples, remaining: {len(self.rl_samples_buffer)}")
                    
                    # Log batch metrics
                    if batch_metrics:
                        epoch_metrics.update(batch_metrics)
                        print(f"[Batch {batch_idx+1} Training] Metrics: {batch_metrics}")

                    # Continue with validation and checkpointing
                    is_last_step = self.global_steps >= self.total_training_steps
                    print(is_last_step, self.global_steps, self.total_training_steps)
                    # Save checkpoint
                    # if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                    if self.config.trainer.save_freq > 0 and (self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", epoch_timing_raw):
                            self._save_checkpoint()
                            
                    progress_bar.update(1)
                    self.global_steps += 1

                    # Log metrics
                    epoch_metrics.update({
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    })
                    logger.log(data=epoch_metrics, step=self.global_steps)
            

                    # Cleanup
                    del batch, gen_batch, gen_batch_output
                    memory_cleanup()

            # Epoch complete
            # Recalculate is_last_step after loop in case loop didn't execute
            is_last_step = self.global_steps >= self.total_training_steps
            
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1} Complete!")
            print(f"  Final SFT buffer size: {len(self.hard_samples_buffer)}")
            print(f"  Final RL buffer size: {len(self.rl_samples_buffer)}")
            print(f"{'='*80}\n")
            
            # Validate
            # if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
            #     with _timer("testing", epoch_timing_raw):
            #         val_metrics: dict = self._validate()
            #         if is_last_step:
            #             last_val_metrics = val_metrics
            #     epoch_metrics.update(val_metrics)


            if is_last_step:
                # pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                
                # Clean up temporary directory
                import shutil
                try:
                    shutil.rmtree(self.temp_sample_dir)
                    print(f"[Cleanup] Removed temporary directory: {self.temp_sample_dir}")
                except Exception as e:
                    print(f"[Warning] Failed to remove temp directory {self.temp_sample_dir}: {e}")
                
                if 'wandb' in logger.logger:
                    logger.logger['wandb'].finish()
                return
            
            
            # Keep buffers across epochs for accumulated training
            # Note: Buffers are preserved between epochs to allow cross-epoch training
            print(f"[Epoch {epoch+1}] Preserving {len(self.hard_samples_buffer)} SFT samples and {len(self.rl_samples_buffer)} RL samples for next epoch")
            memory_cleanup()
