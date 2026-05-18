# Data Format for DYPO Training

## Training Data

The training data should be in **Parquet** format with the following columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `prompt` | list[dict] | Yes | Chat-format prompt, e.g., `[{"role": "user", "content": "Solve: 2+3=?"}]` |
| `data_source` | str | Yes | Dataset identifier for reward function routing, e.g., `"math_dapo"`, `"openai/gsm8k"` |
| `reward_model` | dict | Yes | Must contain `ground_truth` key with the correct answer |

### Example Row

```json
{
  "prompt": [
    {"role": "user", "content": "What is the sum of all prime numbers less than 10?"}
  ],
  "data_source": "math_dapo",
  "reward_model": {
    "ground_truth": "17",
    "style": "rule"
  }
}
```

### Optional Columns (for DYPO with SFT)

When using `unify_strategy="switch"` with `model.sft=True`, the following column is also used:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `answer` | str | No | Reference answer for SFT on hard samples |

## Validation Data

Same format as training data, but typically a separate evaluation benchmark (e.g., AIME, MATH-500).

## Creating Data

You can use the following script to convert a JSON dataset to Parquet:

```python
import pandas as pd

data = [
    {
        "prompt": [{"role": "user", "content": "What is 2+3?"}],
        "data_source": "math_dapo",
        "reward_model": {"ground_truth": "5", "style": "rule"},
    },
]

df = pd.DataFrame(data)
df.to_parquet("train.parquet", index=False)
```

## Supported Data Sources

The reward function is selected based on `data_source`. Built-in support includes:

- `openai/gsm8k` - GSM8K math problems
- `lighteval/MATH` - MATH benchmark
- `math_dapo` / `aime*` - Competition math (uses math-verify)
- `numina_*` - Numina math datasets
- `codecontests`, `apps`, `codeforces`, `taco` - Code generation

For custom data sources, extend `_select_rm_score_fn` in `verl/trainer/main_dypo.py`.
