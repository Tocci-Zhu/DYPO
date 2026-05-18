#!/bin/bash
# DYPO (Dynamic Policy Optimization) training script for math reasoning
# Usage: bash examples/run_dypo_math.sh
#
# Before running:
#   1. Set ROOT to this repository's path
#   2. Prepare your training/validation data in parquet format (see data/README.md)
#   3. Download the base model (e.g., Qwen2.5-Math-7B)

set -x

export HYDRA_FULL_ERROR=1
export NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TORCH_COMPILE_DISABLE=1
export RAY_memory_usage_threshold=0.95

# ========== Configuration ==========
# Set to the root of this repository
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH=$ROOT:$PYTHONPATH

# Wandb project name (set WANDB_API_KEY in your environment)
export WANDB_PROJECT="dypo"

# DYPO strategy: "switch" enables dynamic sample routing (SFT for hard, RL for partial)
UNIFY_STRATEGY="switch"
OFFLINE_LOSS_TYPE="sft"
SWITCH_GATE=0
SWITCH_GATE_OFF=0
SFT_LOSS_COEF=1.0
MAX_GRAD_NORM=80.0
LR=5e-7

# Experiment naming
DATE=$(date +%m%d)
MODEL_NAME=Qwen2.5-Math-7B
EXP_NAME="${DATE}_dypo_${MODEL_NAME}_lr${LR}"

# ========== Paths (MODIFY THESE) ==========
MODEL_PATH=/path/to/Qwen2.5-Math-7B            # Base model path
TRAIN_DATA=/path/to/train.parquet               # Training data (parquet format)
VAL_DATA=/path/to/val.parquet                   # Validation data (parquet format)
SAVE_DIR=$ROOT/checkpoints/dypo/$EXP_NAME       # Checkpoint save directory

# ========== GPU Configuration ==========
N_GPUS_PER_NODE=8
N_NODES=1

# Loss mode: "gspo" for GSPO loss, "ppo" for standard PPO
LOSS_MODE=gspo
LOSS_AGG_MODE="seq-mean-token-mean"

# ==========================================

mkdir -p $SAVE_DIR

python3 -m verl.trainer.main_dypo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    +data.format_penalty_coef=0.5 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.sft=False \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    +actor_rollout_ref.actor.max_grad_norm=$MAX_GRAD_NORM \
    +actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE} \
    actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    +actor_rollout_ref.rollout.val_top_p=0.95 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=200 \
    trainer.test_freq=2000 \
    trainer.unify_strategy="$UNIFY_STRATEGY" \
    trainer.switch_gate="$SWITCH_GATE" \
    trainer.switch_gate_off=$SWITCH_GATE_OFF \
    trainer.remove_sfted_data=False \
    actor_rollout_ref.actor.offline_loss_type="$OFFLINE_LOSS_TYPE" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    actor_rollout_ref.actor.sft_loss_coef=$SFT_LOSS_COEF \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    data.truncation=right \
    data.filter_overlong_prompts=False \
    trainer.total_epochs=3 \
    trainer.default_local_dir=$SAVE_DIR
