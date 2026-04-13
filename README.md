# DYPO: Dynamic Policy Optimization for LLM Reasoning

This repository contains the implementation of **DYPO (Dynamic Policy Optimization)**, a reinforcement learning framework that dynamically routes training samples based on model capability to improve LLM reasoning.

## Overview

DYPO extends the standard GRPO/PPO training pipeline with a **dynamic sample routing** mechanism:

1. **Rollout**: Generate multiple responses for each prompt using the current policy
2. **Classify**: Based on rollout success rates, classify each prompt as:
   - **Hard** (all rollouts fail) → Train with **SFT** using ground-truth answers
   - **Easy** (all rollouts succeed) → **Filter out** (already mastered)
   - **Partial** (mixed success/failure) → Train with **RL** (GRPO)
3. **Train**: Apply the appropriate loss function for each sample type
   This approach ensures the model focuses its learning capacity on the most informative samples at each training step.

<p align="center">
  < img src="assets/workflow.png" alt="DYPO Workflow" width="800">
</p >
## Key Features
- **Dynamic Sample Routing**: Automatically classifies samples into hard/easy/partial based on online rollout performance
- **Hybrid SFT+RL Training**: Combines supervised fine-tuning on hard samples with reinforcement learning on partially-solved samples
- **Built on verl**: Leverages the [verl](https://github.com/volcengine/verl) framework for distributed training with Ray, FSDP, and vLLM
- **Multiple Reward Functions**: Supports math-verify, deepscaler, and custom reward implementations
- **Memory Efficient**: Disk-based sample buffer management for large-scale training
## Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/dypo.git
cd dypo
# Install dependencies
pip install -e ".[vllm,math]"
# Install additional dependencies
pip install deepscaler math-verify
```
### Requirements
- Python >= 3.10
- PyTorch >= 2.1
- vLLM >= 0.7.3
- Ray >= 2.41.0
- CUDA >= 12.1
## Quick Start
### 1. Prepare Data
Prepare your training data in Parquet format. See [data/README.md](data/README.md) for the required schema.
```python
import pandas as pd
data = [
    {
        "prompt": [{"role": "user", "content": "Solve: What is 17 * 23?"}],
        "data_source": "math_dapo",
        "reward_model": {"ground_truth": "391", "style": "rule"},
    },
    # ... more samples
]
df = pd.DataFrame(data)
df.to_parquet("data/train.parquet", index=False)
```
### 2. Configure and Run
Edit the paths in the example script and launch training:
```bash
# Edit examples/run_dypo_math.sh to set:
#   MODEL_PATH - path to your base model (e.g., Qwen2.5-Math-7B)
#   TRAIN_DATA - path to your training parquet
#   VAL_DATA   - path to your validation parquet
bash examples/run_dypo_math.sh
```
### 3. Key Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.unify_strategy` | `"switch"` | DYPO routing strategy (`"switch"` for dynamic, `"no"` for standard RL) |
| `trainer.switch_gate` | `0` | Minimum step before enabling sample routing |
| `actor_rollout_ref.rollout.n` | `8` | Number of rollouts per prompt |
| `actor_rollout_ref.actor.sft_loss_coef` | `1.0` | SFT loss coefficient for hard samples |
| `actor_rollout_ref.actor.offline_loss_type` | `"sft"` | Loss type for hard samples (`"sft"` or `"rl"`) |
| `data.format_penalty_coef` | `0.0` | Format penalty coefficient (penalizes malformed outputs) |
## Project Structure
```
dypo/
├── examples/
│   └── run_dypo_math.sh          # Example training script
├── data/
│   └── README.md                 # Data format documentation
├── verl/
│   ├── trainer/
│   │   ├── main_dypo.py          # DYPO entry point & reward manager
│   │   ├── config/
│   │   │   └── ppo_trainer_dypo.yaml  # Default DYPO config
│   │   └── ppo/
│   │       └── ray_trainer_dypo.py    # Core DYPO trainer (sample routing + training loop)
│   ├── utils/
│   │   ├── dataset/
│   │   │   ├── rl_dataset_dypo.py     # DYPO dataset implementation
│   │   │   └── rl_dataset_with_target.py  # Dataset with target responses
│   │   └── reward_score/              # Reward scoring functions
│   ├── workers/
│   │   ├── fsdp_workers.py            # FSDP distributed workers
│   │   ├── rollout/
│   │   │   └── dypo_rollout_worker.py # DYPO rollout worker
│   │   └── sharding_manager/
│   │       └── fsdp_vllm.py           # FSDP-vLLM sharding
│   └── mix_src/                       # Reward function implementations
│       ├── entropy_math.py
│       ├── math_verify_reward.py
│       └── reward_with_format.py
├── pyproject.toml
├── setup.py
└── README.md
```
## How DYPO Works
### Dynamic Sample Routing (`unify_strategy="switch"`)
```
For each training batch:
  1. Generate n rollouts per prompt
  2. Compute rewards for each rollout
  3. For each prompt, count successful rollouts:
     - 0 successes (Hard)  → Add to SFT buffer
     - n successes (Easy)  → Discard
     - 1..n-1 (Partial)   → Add to RL buffer
  4. When SFT buffer ≥ batch_size → Run SFT update
  5. When RL buffer ≥ min_samples → Run RL (GRPO) update
```
### Training Loss
- **SFT on Hard Samples**: Standard cross-entropy loss on ground-truth responses
- **RL on Partial Samples**: GRPO with advantage normalization across the rollout group
- **Optional Contrastive Loss**: DPO-style contrastive loss between successful and failed rollouts
## Citation
If you find this work useful, please cite:
```bibtex
@article{dypo2025,
  title={DYPO: Dynamic Policy Optimization for Large Language Model Reasoning},
  author={...},
  year={2025}
}
```
## Acknowledgments
This project is built on top of [verl](https://github.com/volcengine/verl) (Volcano Engine Reinforcement Learning for LLM). We thank the verl team for their excellent framework.
## License
This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
