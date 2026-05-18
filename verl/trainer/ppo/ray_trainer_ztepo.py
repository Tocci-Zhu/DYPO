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

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
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


def verify_math_answer_with_sympy(predicted_text: str, ground_truth_text: str) -> float:
    """使用SymPy验证数学答案的正确性。
    
    尝试将预测答案和ground truth转换为符号表达式并比较是否等价。
    适用于代数表达式、方程解、积分结果等。
    
    Args:
        predicted_text (str): 模型预测的答案文本。
        ground_truth_text (str): Ground truth答案文本。
    
    Returns:
        float: 符号验证分数，1.0表示完全正确，0.0表示错误，0.5表示部分匹配或无法验证。
    """
    try:
        from sympy import sympify, simplify, Eq, solve
        from sympy.parsing.latex import parse_latex
        import re
        
        # 提取数学表达式的辅助函数
        def extract_math_expression(text):
            """从文本中提取数学表达式"""
            # 尝试多种模式
            patterns = [
                r'\\boxed\{([^}]+)\}',  # LaTeX boxed
                r'\$([^\$]+)\$',         # LaTeX inline
                r'答案[是为：:]\s*([^\s，。,\.]+)',  # 中文答案模式
                r'answer[:\s]+([^\s,\.]+)',  # 英文答案模式
                r'x\s*=\s*([^\s,\.]+)',  # 方程解模式
                r'([+-]?\d+\.?\d*)',     # 纯数字
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # 如果没有匹配，返回原文本（去除空格）
            return text.strip()
        
        # 提取表达式
        pred_expr_str = extract_math_expression(predicted_text)
        gt_expr_str = extract_math_expression(ground_truth_text)
        
        # 简单的字符串比较（处理格式差异）
        pred_normalized = pred_expr_str.replace(' ', '').replace('**', '^').lower()
        gt_normalized = gt_expr_str.replace(' ', '').replace('**', '^').lower()
        
        if pred_normalized == gt_normalized:
            return 1.0
        
        # 尝试解析为SymPy表达式
        try:
            # 处理常见的LaTeX语法
            pred_expr_str = pred_expr_str.replace('\\frac', '').replace('\\', '')
            gt_expr_str = gt_expr_str.replace('\\frac', '').replace('\\', '')
            
            pred_expr = sympify(pred_expr_str)
            gt_expr = sympify(gt_expr_str)
            
            # 检查是否等价
            if pred_expr.equals(gt_expr):
                return 1.0
            
            # 尝试简化后比较
            diff = simplify(pred_expr - gt_expr)
            if diff == 0:
                return 1.0
            
            # 数值比较（对于数值答案）
            try:
                pred_val = float(pred_expr.evalf())
                gt_val = float(gt_expr.evalf())
                
                # 相对误差小于1%认为正确
                if abs(gt_val) > 1e-10:
                    rel_error = abs(pred_val - gt_val) / abs(gt_val)
                    if rel_error < 0.01:
                        return 1.0
                    elif rel_error < 0.1:
                        return 0.7  # 接近但不完全正确
                else:
                    # 绝对误差
                    if abs(pred_val - gt_val) < 1e-6:
                        return 1.0
            except:
                pass
            
            # 检查是否为相同类型的表达式但系数不同
            if pred_expr.func == gt_expr.func:
                return 0.3  # 结构相似
            
            return 0.0
            
        except:
            # 如果无法解析，尝试简单的数值比较
            try:
                pred_num = float(pred_expr_str)
                gt_num = float(gt_expr_str)
                
                if abs(gt_num) > 1e-10:
                    rel_error = abs(pred_num - gt_num) / abs(gt_num)
                    if rel_error < 0.01:
                        return 1.0
                    elif rel_error < 0.1:
                        return 0.7
                else:
                    if abs(pred_num - gt_num) < 1e-6:
                        return 1.0
            except:
                pass
            
            # 最后尝试字符串相似度（作为fallback）
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, pred_normalized, gt_normalized).ratio()
            return similarity * 0.5  # 字符串相似度打折扣
    
    except Exception as e:
        # 任何异常都返回无法验证
        return 0.5


def compute_contrastive_reward_shaping(batch: DataProto, tokenizer, reward_tensor: torch.Tensor, 
                                       temperature: float = 0.1, reward_scale: float = 0.5,
                                       use_semantic_similarity: bool = False,
                                       embedding_model=None,
                                       task_type: str = "general",
                                       use_symbolic_verification: bool = False,
                                       contrastive_loss_type: str = "infonce",
                                       triplet_margin: float = 0.2):
    """使用Contrastive Learning和Reward Shaping来引导模型学习。
    
    结合多种方法：
    1. Contrastive Learning: 对比同一prompt下的不同回答，拉近正确答案，推远错误答案
    2. Reward Shaping: 根据与ground truth的相似度给予梯度化的reward
    3. SFT Token Loss: 计算并保存token级别的匹配损失，可用作辅助损失
    4. Semantic Similarity: 使用embedding计算语义相似度，支持paraphrase
    5. Math-aware Verification: 对数学任务使用符号计算验证（SymPy）
    
    对于每个prompt的多个rollouts：
    - 计算每个回答与ground truth的相似度（token级别 + 语义级别 + 符号验证）
    - 根据任务类型动态调整各指标权重
    - 使用contrastive loss来对比不同回答
    - 根据相似度调整原始reward，给予部分奖励
    - 保存token级别的SFT损失和contrastive loss
    
    Args:
        batch (DataProto): 包含rollout数据的batch。
        tokenizer: Tokenizer用于编码ground truth responses。
        reward_tensor (torch.Tensor): 原始奖励张量，形状为 (batch_size, seq_len)。
        temperature (float): Contrastive learning的温度参数，默认0.1。
        reward_scale (float): Reward shaping的缩放因子，默认0.5。
        use_semantic_similarity (bool): 是否使用语义相似度，默认False。
        embedding_model: Sentence transformer模型，用于计算语义相似度。
        task_type (str): 任务类型，"general"或"math"，影响相似度权重分配。
        use_symbolic_verification (bool): 是否使用符号验证（针对数学任务）。
        contrastive_loss_type (str): 对比损失类型，"infonce", "triplet", 或"simple"。
        triplet_margin (float): Triplet loss的margin参数，默认0.2。
    
    Returns:
        torch.Tensor: 调整后的reward张量 (batch_size, seq_len)。
        torch.Tensor: Token级别的SFT损失张量 (batch_size, seq_len)。
        torch.Tensor: Contrastive loss张量 (batch_size,)。
        torch.Tensor: SFT loss标量，用于训练 (scalar)。
        torch.Tensor: Contrastive loss标量，用于训练 (scalar)。
        dict: 关于reward调整的指标。
    """
    import torch.nn.functional as F
    
    batch_size = reward_tensor.shape[0]
    device = reward_tensor.device
    
    # 根据任务类型设置相似度权重
    # Math任务：降低token权重，提高语义和符号验证权重
    # General任务：平衡token和语义权重
    if task_type == "math":
        # Math任务权重分配
        if use_symbolic_verification:
            # token 20% + semantic 30% + symbolic 40% + length 10%
            weight_token = 0.2
            weight_semantic = 0.3
            weight_symbolic = 0.4
            weight_length = 0.1
        elif use_semantic_similarity:
            # token 25% + semantic 60% + length 15%
            weight_token = 0.25
            weight_semantic = 0.60
            weight_symbolic = 0.0
            weight_length = 0.15
        else:
            # 只有token，但降低权重，增加长度
            weight_token = 0.6
            weight_semantic = 0.0
            weight_symbolic = 0.0
            weight_length = 0.4
    else:
        # General任务权重分配
        if use_semantic_similarity:
            # token 40% + semantic 50% + length 10%
            weight_token = 0.4
            weight_semantic = 0.5
            weight_symbolic = 0.0
            weight_length = 0.1
        else:
            # token 70% + length 30%
            weight_token = 0.7
            weight_semantic = 0.0
            weight_symbolic = 0.0
            weight_length = 0.3
    
    print(f"[相似度权重] 任务类型: {task_type}, Token: {weight_token:.0%}, Semantic: {weight_semantic:.0%}, "
          f"Symbolic: {weight_symbolic:.0%}, Length: {weight_length:.0%}")
    
    # 计算每个样本的总奖励（在序列长度上求和）
    total_rewards = reward_tensor.sum(dim=-1)  # (batch_size,)
    
    # 按照uid对样本进行分组（同一prompt的多个rollouts）
    uid_to_indices = defaultdict(list)
    uids = batch.non_tensor_batch["uid"]
    for idx, uid in enumerate(uids):
        uid_to_indices[uid].append(idx)
    
    # 初始化调整后的reward（从原始reward开始）
    shaped_reward_tensor = reward_tensor.clone()
    
    # 初始化相似度张量和对比损失
    similarity_scores = torch.zeros(batch_size, device=device)
    contrastive_losses = []
    
    # 初始化SFT token loss张量 (batch_size, seq_len)
    response_length = shaped_reward_tensor.size(1)
    sft_token_loss = torch.zeros(batch_size, response_length, device=device)
    
    # 初始化contrastive loss张量 (batch_size,)
    contrastive_loss_tensor = torch.zeros(batch_size, device=device)
    
    num_groups = len(uid_to_indices)
    num_groups_all_failed = 0  # 全部rollouts都失败的组数
    num_failed = 0  # 失败样本总数
    num_shaped = 0
    total_similarity = 0.0
    total_semantic_similarity = 0.0
    num_semantic_computed = 0
    total_symbolic_similarity = 0.0
    num_symbolic_computed = 0
    
    print(f"[对比学习+Reward调整] 处理 {num_groups} 个prompt组，共 {batch_size} 个样本")
    print(f"  - 策略: 仅对全部rollouts失败的组进行reward shaping")
    if use_semantic_similarity:
        if embedding_model is None:
            print("[语义相似度] 警告: 启用了语义相似度但未提供embedding_model，将只使用token相似度")
            use_semantic_similarity = False
        else:
            print(f"[语义相似度] 使用embedding模型计算语义相似度")
    
    # 检查是否存在reward_model字段
    if "reward_model" not in batch.non_tensor_batch:
        print("[对比学习+Reward调整] 警告: batch中未找到'reward_model'字段，跳过调整")
        sft_loss_scalar = torch.tensor(0.0, device=device)
        contrastive_loss_scalar = torch.tensor(0.0, device=device)
        metrics = {
            "contrastive/num_groups": num_groups,
            "contrastive/num_failed": 0,
            "contrastive/num_shaped": 0,
            "contrastive/avg_similarity": 0.0,
            "contrastive/avg_contrastive_loss": 0.0,
            "contrastive/loss_scalar": 0.0,
            "sft/loss_scalar": 0.0,
        }
        return shaped_reward_tensor, sft_token_loss, contrastive_loss_tensor, sft_loss_scalar, contrastive_loss_scalar, metrics
    
    # 对每个prompt组进行处理
    for uid, indices in uid_to_indices.items():
        if len(indices) == 0:
            continue
        
        # *** 关键修改：只处理所有rollouts都失败的组 ***
        # 检查该组内所有样本的reward
        group_rewards = total_rewards[indices]
        all_failed = torch.all(group_rewards <= 0).item()
        
        # 如果不是全部失败，跳过这个组（说明至少有一个正确答案）
        if not all_failed:
            continue
        
        # 统计全部失败的组和样本数
        num_groups_all_failed += 1
        num_failed += len(indices)
        
        # 获取ground truth（所有同uid的样本应该有相同的GT）
        try:
            reward_model_info = batch.non_tensor_batch["reward_model"][indices[0]]
            if not (isinstance(reward_model_info, dict) and "response" in reward_model_info):
                continue
            gt_response_text = reward_model_info["response"]
        except (KeyError, IndexError, TypeError):
            continue
        
        # Tokenize ground truth
        gt_response_tokens = tokenizer.encode(gt_response_text, add_special_tokens=False)
        gt_response_tensor = torch.tensor(gt_response_tokens, dtype=torch.long, device=device)
        gt_len = len(gt_response_tensor)
        
        # 存储该组内所有样本的相似度和文本
        group_similarities = []
        group_rewards = []
        group_response_texts = []
        
        # 如果使用语义相似度，先decode所有文本
        if use_semantic_similarity:
            for idx in indices:
                response = batch.batch["responses"][idx]
                response_mask = batch.batch["response_mask"][idx]
                valid_len = int(response_mask.sum().item())
                actual_response = response[:valid_len]
                response_text = tokenizer.decode(actual_response, skip_special_tokens=True)
                group_response_texts.append(response_text)
            
            # 批量计算语义相似度
            try:
                with torch.no_grad():
                    # 编码所有回答和GT
                    all_texts = group_response_texts + [gt_response_text]
                    embeddings = embedding_model.encode(all_texts, convert_to_tensor=True, device=device)
                    
                    # GT的embedding
                    gt_embedding = embeddings[-1:]  # (1, embed_dim)
                    # 回答的embeddings
                    response_embeddings = embeddings[:-1]  # (n, embed_dim)
                    
                    # 计算cosine相似度
                    semantic_similarities = F.cosine_similarity(response_embeddings, gt_embedding, dim=1)  # (n,)
                    # 将相似度从[-1, 1]映射到[0, 1]
                    semantic_similarities = (semantic_similarities + 1) / 2
                    semantic_similarities = semantic_similarities.cpu().tolist()
            except Exception as e:
                print(f"[语义相似度] 计算失败: {e}，回退到token相似度")
                semantic_similarities = None
        else:
            semantic_similarities = None
        
        # 计算该组内每个回答与GT的相似度
        for i, idx in enumerate(indices):
            response = batch.batch["responses"][idx]
            response_mask = batch.batch["response_mask"][idx]
            valid_len = int(response_mask.sum().item())
            actual_response = response[:valid_len]
            
            # 计算token级别的相似度（使用多种指标的组合）
            # 1. Token overlap
            min_len = min(valid_len, gt_len)
            max_len = max(valid_len, gt_len)
            
            if min_len == 0:
                token_match_score = 0.0
            else:
                # 前缀匹配
                gt_prefix = gt_response_tensor[:min_len]
                resp_prefix = actual_response[:min_len]
                matches = (gt_prefix == resp_prefix).float()
                token_match_score = matches.mean().item()
            
            # 2. 长度惩罚（避免过短或过长的回答）
            length_ratio = min_len / max(max_len, 1)
            
            # 3. 计算符号验证分数（如果启用）
            symbolic_sim = 0.0
            if use_symbolic_verification and task_type == "math":
                # 使用SymPy验证数学答案
                response_text = group_response_texts[i] if use_semantic_similarity else tokenizer.decode(actual_response, skip_special_tokens=True)
                symbolic_sim = verify_math_answer_with_sympy(response_text, gt_response_text)
                total_symbolic_similarity += symbolic_sim
                num_symbolic_computed += 1
            
            # 4. 综合相似度分数（根据任务类型和配置动态组合）
            if use_symbolic_verification and task_type == "math":
                # Math + 符号验证
                semantic_sim = semantic_similarities[i] if (use_semantic_similarity and semantic_similarities is not None) else 0.0
                similarity = (weight_token * token_match_score + 
                            weight_semantic * semantic_sim + 
                            weight_symbolic * symbolic_sim + 
                            weight_length * length_ratio)
                if use_semantic_similarity and semantic_similarities is not None:
                    total_semantic_similarity += semantic_sim
                    num_semantic_computed += 1
            elif use_semantic_similarity and semantic_similarities is not None:
                # 有语义相似度
                semantic_sim = semantic_similarities[i]
                similarity = (weight_token * token_match_score + 
                            weight_semantic * semantic_sim + 
                            weight_length * length_ratio)
                total_semantic_similarity += semantic_sim
                num_semantic_computed += 1
            else:
                # 只使用token相似度
                similarity = weight_token * token_match_score + weight_length * length_ratio
            
            group_similarities.append(similarity)
            group_rewards.append(total_rewards[idx].item())
            similarity_scores[idx] = similarity
            
            # 4. 计算token级别的SFT loss（token不匹配则为1，匹配则为0）
            # 对于错误的回答，计算每个token与GT的匹配情况
            # （注意：这里所有样本都是错误的，因为我们只处理全部失败的组）
            
            # 创建token级别的loss
            token_loss = torch.zeros(response_length, device=device)
            
            # 对有效token进行loss计算
            for t in range(min(valid_len, gt_len)):
                if actual_response[t] != gt_response_tensor[t]:
                    token_loss[t] = 1.0  # 不匹配的token，loss为1
                # 匹配的token，loss保持为0
            
            # 超出GT长度的token也算作loss
            if valid_len > gt_len:
                token_loss[gt_len:valid_len] = 1.0
            
            # 将token loss存储到对应位置
            sft_token_loss[idx] = token_loss
        
        # 如果该组有多个样本，计算contrastive loss
        if len(group_similarities) > 1:
            # 将相似度转换为tensor
            sim_tensor = torch.tensor(group_similarities, device=device)
            reward_tensor_group = torch.tensor(group_rewards, device=device)
            
            # 使用配置的contrastive loss类型
            if contrastive_loss_type == "infonce":
                # ===== InfoNCE Loss (标准对比学习损失) =====
                # 对于每个样本，将与GT最相似的作为正样本，其他作为负样本
                # L = -log(exp(sim_positive/τ) / Σ exp(sim_i/τ))
                
                # 找出该组中最相似GT的样本作为"伪正样本"
                best_idx_in_group = torch.argmax(sim_tensor).item()
                
                for i, idx in enumerate(indices):
                    # 当前样本的相似度（作为anchor）
                    anchor_sim = sim_tensor[i]
                    
                    # 构建正负样本对
                    # 正样本：与GT最相似的那个（或者相似度高于阈值的）
                    # 负样本：其他样本
                    
                    # 计算与所有其他样本的"关系分数"
                    # 使用相似度作为logits
                    logits = sim_tensor / temperature  # (n_samples,)
                    
                    # 对于当前anchor，正样本是相似度最高的
                    # 使用softmax来归一化
                    positive_mask = torch.zeros_like(logits, dtype=torch.bool)
                    positive_mask[best_idx_in_group] = True
                    
                    # InfoNCE loss: 
                    # 鼓励anchor接近正样本（高相似度样本），远离负样本（低相似度样本）
                    exp_logits = torch.exp(logits)
                    
                    # 正样本的概率
                    positive_prob = exp_logits[positive_mask].sum() / exp_logits.sum()
                    
                    # InfoNCE loss = -log(positive_prob)
                    infonce_loss = -torch.log(positive_prob + 1e-8)
                    
                    contrastive_losses.append(infonce_loss.item())
                    contrastive_loss_tensor[idx] = infonce_loss.detach()
                    
            elif contrastive_loss_type == "triplet":
                # ===== Triplet Loss =====
                # L = max(d(a,p) - d(a,n) + margin, 0)
                # 拉近anchor和positive，推远anchor和negative
                
                margin = triplet_margin
                
                # 按相似度排序，最相似的作为positive，最不相似的作为negative
                sorted_indices = torch.argsort(sim_tensor, descending=True)
                
                for i, idx in enumerate(indices):
                    anchor_sim = sim_tensor[i]
                    
                    # 找positive（相似度高的）和negative（相似度低的）
                    # Positive: 除了自己之外，相似度最高的
                    positive_candidates = sorted_indices[sorted_indices != i]
                    if len(positive_candidates) > 0:
                        positive_idx = positive_candidates[0].item()
                        positive_sim = sim_tensor[positive_idx]
                    else:
                        continue
                    
                    # Negative: 相似度最低的
                    negative_idx = sorted_indices[-1].item()
                    if negative_idx == i:
                        negative_idx = sorted_indices[-2].item()
                    negative_sim = sim_tensor[negative_idx]
                    
                    # Triplet loss: 
                    # anchor应该更接近positive而非negative
                    # 用相似度的相反数作为距离
                    dist_ap = 1.0 - positive_sim  # anchor到positive的距离
                    dist_an = 1.0 - negative_sim  # anchor到negative的距离
                    
                    triplet_loss = torch.clamp(dist_ap - dist_an + margin, min=0.0)
                    
                    contrastive_losses.append(triplet_loss.item())
                    contrastive_loss_tensor[idx] = triplet_loss
                    
            else:  # "simple" - 原来的简单方法（保留作为fallback）
                for i, idx in enumerate(indices):
                    anchor_sim = sim_tensor[i]
                    other_indices_in_group = list(range(len(indices)))
                    other_indices_in_group.remove(i)
                    other_sims = sim_tensor[other_indices_in_group]
                    
                    if len(other_sims) > 0:
                        sim_diff = torch.abs(anchor_sim - other_sims)
                        scaled_diff = sim_diff / temperature
                        weights = other_sims
                        weighted_diff = scaled_diff * (1.0 - weights)
                        contrastive_loss = torch.mean(weighted_diff)
                        contrastive_losses.append(contrastive_loss.item())
                        contrastive_loss_tensor[idx] = contrastive_loss.detach()
        
        # Reward Shaping: 根据相似度调整reward
        for i, idx in enumerate(indices):
            original_reward = total_rewards[idx].item()
            similarity = group_similarities[i]
            
            # 只对错误的回答进行reward shaping
            if original_reward <= 0:
                # 根据相似度给予部分reward
                # shaped_reward = original_reward + reward_scale * similarity
                # 使用token-level的reward shaping
                response_length = shaped_reward_tensor.size(1)
                
                # 给予与相似度成正比的bonus reward
                bonus_reward = reward_scale * similarity
                
                # 将bonus均匀分布到response的每个token上
                shaped_reward_tensor[idx] = reward_tensor[idx] + bonus_reward / response_length
                
                num_shaped += 1
                total_similarity += similarity
    
    # 计算统计指标
    avg_similarity = total_similarity / max(num_failed, 1)
    avg_semantic_similarity = total_semantic_similarity / max(num_semantic_computed, 1) if num_semantic_computed > 0 else 0.0
    avg_symbolic_similarity = total_symbolic_similarity / max(num_symbolic_computed, 1) if num_symbolic_computed > 0 else 0.0
    avg_contrastive_loss = np.mean(contrastive_losses) if contrastive_losses else 0.0
    
    # 计算reward的变化
    reward_change = (shaped_reward_tensor.sum(dim=-1) - total_rewards).abs().mean().item()
    
    # 计算SFT token loss的统计信息和聚合标量
    non_zero_sft_loss_mask = sft_token_loss > 0
    if non_zero_sft_loss_mask.any():
        avg_sft_token_loss = sft_token_loss[non_zero_sft_loss_mask].mean().item()
        num_mismatched_tokens = non_zero_sft_loss_mask.sum().item()
        
        # 聚合SFT loss为标量（用于训练）
        # 方法：对每个样本求平均，然后对batch求平均
        # 只计算有loss的样本（即全部失败组的样本）
        sample_has_loss = (sft_token_loss.sum(dim=1) > 0)  # (batch_size,)
        if sample_has_loss.any():
            # 对每个有loss的样本，计算其平均token loss
            sample_sft_losses = sft_token_loss[sample_has_loss].mean(dim=1)  # (n_failed_samples,)
            # 对所有失败样本求平均
            sft_loss_scalar = sample_sft_losses.mean()
        else:
            sft_loss_scalar = torch.tensor(0.0, device=device)
    else:
        avg_sft_token_loss = 0.0
        num_mismatched_tokens = 0
        sft_loss_scalar = torch.tensor(0.0, device=device)
    
    # 计算contrastive loss的统计信息和聚合标量
    non_zero_contrast_mask = contrastive_loss_tensor > 0
    if non_zero_contrast_mask.any():
        avg_contrastive_loss_tensor = contrastive_loss_tensor[non_zero_contrast_mask].mean().item()
        # 聚合为标量（用于训练）
        contrastive_loss_scalar = contrastive_loss_tensor[non_zero_contrast_mask].mean()
    else:
        avg_contrastive_loss_tensor = 0.0
        contrastive_loss_scalar = torch.tensor(0.0, device=device)
    
    metrics = {
        "contrastive/num_groups": num_groups,
        "contrastive/num_groups_all_failed": num_groups_all_failed,
        "contrastive/num_groups_with_success": num_groups - num_groups_all_failed,
        "contrastive/all_failed_rate": num_groups_all_failed / max(num_groups, 1),
        "contrastive/num_failed": num_failed,
        "contrastive/num_shaped": num_shaped,
        "contrastive/avg_similarity": avg_similarity,
        "contrastive/avg_contrastive_loss": avg_contrastive_loss,
        "contrastive/avg_contrastive_loss_tensor": avg_contrastive_loss_tensor,
        "contrastive/loss_scalar": contrastive_loss_scalar.item(),  # 聚合标量
        "contrastive/avg_reward_change": reward_change,
        "contrastive/min_similarity": similarity_scores[similarity_scores > 0].min().item() if (similarity_scores > 0).any() else 0.0,
        "contrastive/max_similarity": similarity_scores.max().item(),
        "contrastive/loss_type": contrastive_loss_type,
        "sft/avg_token_loss": avg_sft_token_loss,
        "sft/loss_scalar": sft_loss_scalar.item(),  # 聚合标量
        "sft/num_mismatched_tokens": num_mismatched_tokens,
        "sft/total_tokens": batch_size * response_length,
    }
    
    if use_semantic_similarity:
        metrics["semantic/avg_similarity"] = avg_semantic_similarity
        metrics["semantic/num_computed"] = num_semantic_computed
    
    if use_symbolic_verification:
        metrics["symbolic/avg_similarity"] = avg_symbolic_similarity
        metrics["symbolic/num_computed"] = num_symbolic_computed
        metrics["symbolic/task_type"] = task_type
    
    print(f"[对比学习+Reward调整] 处理结果:")
    print(f"  - 总组数: {num_groups}, 全部失败组数: {num_groups_all_failed} ({num_groups_all_failed/max(num_groups,1):.1%})")
    print(f"  - 至少一个成功的组: {num_groups - num_groups_all_failed} (跳过)")
    print(f"  - 调整样本数: {num_shaped} / {num_failed} 失败样本")
    print(f"  - 任务类型: {task_type}, 对比损失: {contrastive_loss_type}")
    print(f"  - 平均相似度: {avg_similarity:.4f}")
    if use_semantic_similarity:
        print(f"  - 平均语义相似度: {avg_semantic_similarity:.4f}")
    if use_symbolic_verification:
        print(f"  - 平均符号验证分数: {avg_symbolic_similarity:.4f}")
    print(f"  - 平均对比损失: {avg_contrastive_loss:.4f}")
    print(f"  - 平均reward变化: {reward_change:.4f}")
    print(f"  - SFT loss: token级别={avg_sft_token_loss:.4f}, 标量={sft_loss_scalar.item():.4f} ({num_mismatched_tokens} 不匹配tokens)")
    print(f"  - Contrastive loss标量: {contrastive_loss_scalar.item():.4f}")
    
    return shaped_reward_tensor, sft_token_loss, contrastive_loss_tensor, sft_loss_scalar, contrastive_loss_scalar, metrics


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

        # 初始化embedding model用于语义相似度计算（如果启用）
        self.embedding_model = None
        if self.config.trainer.get("use_semantic_similarity", False):
            try:
                embedding_model_name = self.config.trainer.get("embedding_model_name", "sentence-transformers/all-MiniLM-L6-v2")
                print(f"[语义相似度] 加载embedding模型: {embedding_model_name}")
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(embedding_model_name)
                # 将模型移到正确的设备
                if device_name == "cuda" and torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
                print(f"[语义相似度] Embedding模型加载成功")
            except Exception as e:
                print(f"[语义相似度] 警告: 加载embedding模型失败: {e}")
                print(f"[语义相似度] 将禁用语义相似度功能")
                self.embedding_model = None
        
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

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

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

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
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
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
        # val_data_dir = self.config.trainer.get("validation_data_dir", None)
        # print(val_data_dir)
        # if val_data_dir:
        #     self._dump_generations(
        #         inputs=sample_inputs,
        #         outputs=sample_outputs,
        #         scores=sample_scores,
        #         reward_extra_infos_dict=reward_extra_infos_dict,
        #         dump_path=val_data_dir,
        #     )

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

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                # repeat gen_batch to align with repeated responses in rollout
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            self.async_rollout_manager.wake_up()
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                            self.async_rollout_manager.sleep()

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # # === Self-Play: 选择每个 prompt 的最佳回答并回填到 prompt ===
                    # try:
                    #     # 标记回答区域
                    #     batch.batch["response_mask"] = compute_response_mask(batch)

                    #     # 计算 log_prob 并聚合得到回答段序列对数概率作为置信度
                    #     _log_prob_out = self.actor_rollout_wg.compute_log_prob(batch)
                    #     batch = batch.union(_log_prob_out)
                    #     seq_logprob = (batch.batch["log_probs"] * batch.batch["response_mask"]).sum(dim=-1)

                    #     # 将同一原始 prompt 的轨迹用 uid 分组
                    #     uids = batch.non_tensor_batch["uid"]
                    #     raw_prompts = batch.non_tensor_batch["raw_prompt_ids"]
                    #     uid2idxs = defaultdict(list)
                    #     for idx, uid in enumerate(uids):
                    #         uid2idxs[uid].append(idx)

                    #     # 提取最佳回答 token 并拼接回 prompt
                    #     input_ids = batch.batch["input_ids"]
                    #     resp_mask = batch.batch["response_mask"].bool()
                    #     aug_raw_prompts = list(raw_prompts)

                    #     for uid, idxs in uid2idxs.items():
                    #         # 选出该 uid 下 seq_logprob 最大的回答
                    #         best_idx_local = idxs[int(torch.argmax(seq_logprob[idxs]).item())]

                    #         # 提取最佳回答 token ids
                    #         best_resp_token_ids = input_ids[best_idx_local][resp_mask[best_idx_local]].tolist()

                    #         # 取该 uid 的基底 prompt（任取一条，内容相同）
                    #         base_prompt = raw_prompts[idxs[0]]
                    #         if hasattr(base_prompt, "tolist"):
                    #             base_prompt = base_prompt.tolist()

                    #         # 拼接回答到 prompt（若需要分隔符，可在两者间插入特殊 token）
                    #         new_prompt = base_prompt + best_resp_token_ids

                    #         # 将该 uid 下所有轨迹的 prompt 都替换为带最佳回答的版本
                    #         for j in idxs:
                    #             aug_raw_prompts[j] = np.array(new_prompt, dtype=np.int64)

                    #     # 覆盖回 batch
                    #     batch.non_tensor_batch["raw_prompt_ids"] = np.array(aug_raw_prompts, dtype=object)
                    # except Exception as _e:
                    #     print(f"[Self-Play augment prompt skipped] {type(_e).__name__}: {_e}")
                    # === Self-Play: 选择每个 prompt 的最佳回答并回填到 prompt ===

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # 使用Contrastive Learning + Reward Shaping来引导学习
                    with _timer("contrastive_reward_shaping", timing_raw):
                        if self.config.trainer.get("enable_contrastive_reward_shaping", False):
                            # 如果启用了异步reward计算，需要先等待
                            if self.config.reward_model.launch_reward_fn_async:
                                print("[对比学习+Reward调整] 等待异步reward计算完成...")
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            
                            # 获取配置参数
                            temperature = self.config.trainer.get("contrastive_temperature", 0.1)
                            base_reward_scale = self.config.trainer.get("contrastive_reward_scale", 0.5)
                            
                            # 动态reward_scale：根据训练进度调整
                            # 初期强（接近base_reward_scale），后期弱（接近0）
                            if self.config.trainer.get("use_dynamic_reward_scale", False):
                                # 计算训练进度 (0到1)
                                train_progress = self.global_steps / max(self.total_training_steps, 1)
                                # 使用余弦衰减
                                decay_factor = 0.5 * (1 + np.cos(np.pi * train_progress))
                                # 设置最小scale，避免完全归零
                                min_scale = self.config.trainer.get("min_reward_scale", 0.1)
                                reward_scale = max(min_scale, base_reward_scale * decay_factor)
                                metrics["contrastive/dynamic_reward_scale"] = reward_scale
                                metrics["contrastive/train_progress"] = train_progress
                                print(f"[动态Reward Scale] 训练进度: {train_progress:.2%}, Reward Scale: {reward_scale:.4f}")
                            else:
                                reward_scale = base_reward_scale
                            
                            # 语义相似度配置
                            use_semantic_similarity = self.config.trainer.get("use_semantic_similarity", False)
                            embedding_model = getattr(self, "embedding_model", None) if use_semantic_similarity else None
                            
                            # 任务类型和符号验证配置
                            task_type = self.config.trainer.get("task_type", "general")  # "general" or "math"
                            use_symbolic_verification = self.config.trainer.get("use_symbolic_verification", False)
                            
                            # 对比损失配置
                            contrastive_loss_type = self.config.trainer.get("contrastive_loss_type", "infonce")  # "infonce", "triplet", "simple"
                            triplet_margin = self.config.trainer.get("triplet_margin", 0.2)
                            
                            # 应用对比学习和reward shaping
                            shaped_reward_tensor, sft_token_loss, contrastive_loss_tensor, sft_loss_scalar, contrastive_loss_scalar, contrastive_metrics = compute_contrastive_reward_shaping(
                                batch=batch,
                                tokenizer=self.tokenizer,
                                reward_tensor=reward_tensor,
                                temperature=temperature,
                                reward_scale=reward_scale,
                                use_semantic_similarity=use_semantic_similarity,
                                embedding_model=self.embedding_model,
                                task_type=task_type,
                                use_symbolic_verification=use_symbolic_verification,
                                contrastive_loss_type=contrastive_loss_type,
                                triplet_margin=triplet_margin
                            )
                            
                            # 用调整后的reward替换原始reward
                            reward_tensor = shaped_reward_tensor
                            
                            # 保存损失到batch中
                            # Token级别的loss（用于分析或细粒度训练）
                            batch.batch["sft_token_loss"] = sft_token_loss  # (batch_size, seq_len)
                            batch.batch["contrastive_loss_tensor"] = contrastive_loss_tensor  # (batch_size,)
                            
                            # 标量loss（batch级别的聚合值，存储在meta_info中）
                            batch.meta_info["sft_loss_scalar"] = sft_loss_scalar  # scalar
                            batch.meta_info["contrastive_loss_scalar"] = contrastive_loss_scalar  # scalar
                            
                            metrics.update(contrastive_metrics)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        print(f"{list(reward_extra_infos_dict.keys())=}")
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            entropy_coeff = self.config.actor_rollout_ref.actor.get("entropy_coeff", 0.01)
                            batch.meta_info["entropy_coeff"] = entropy_coeff
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            print(batch.batch.keys())
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
