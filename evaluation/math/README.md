# Math Evaluation

Math benchmarks via [lm-evaluation-harness].

## Tasks (paper row)

```
arithmetic_2da
arithmetic_2ds
minerva_math_prealgebra
bigbench_simple_arithmetic_json_generate_until
mmlu_elementary_mathematics
```

## Quick run

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export PYTHONPATH="${LM_EVAL_REPO}:${PYTHONPATH}"

bash eval_math.sh
```

`eval_math.sh` honours these env vars:

- `MODEL_PATH` (default `/data/chenhang/models/SFT_math_Llama-3.2-1B`)
- `OUTPUT_DIR` (default `output/math`)
- `BATCH_SIZE`, `DEVICE`
- `TASKS` (override the task list)

[lm-evaluation-harness]: https://github.com/EleutherAI/lm-evaluation-harness
