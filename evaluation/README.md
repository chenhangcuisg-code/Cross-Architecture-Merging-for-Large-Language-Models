# Evaluation Utilities

This directory contains evaluation scripts for different tasks and benchmarks.

## Overview

The evaluation utilities support three main evaluation frameworks:

1. **lm-evaluation-harness** - For most standard NLP tasks
2. **MalayMMLU** - For Malay language evaluation
3. **Yue-Benchmark** - For Cantonese (CMMLU) evaluation

## Required Evaluation Libraries

### 1. lm-evaluation-harness

**Repository**: https://github.com/EleutherAI/lm-evaluation-harness

**Used for**: Medical, Thai, Finance, Indonesian tasks

**Installation**:
```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
```

**Usage**: The main script (`run_train_final.sh`) uses this library via the `run_lm_eval` function.

### 2. MalayMMLU

**Repository**: https://github.com/UMxYTL-AI-Labs/MalayMMLU

**Used for**: Malay language evaluation

**Installation**:
```bash
git clone https://github.com/UMxYTL-AI-Labs/MalayMMLU
cd MalayMMLU
# Follow the repository's installation instructions
```

**Usage**: The main script uses this library via the `run_malay_eval` function.

### 3. Yue-Benchmark

**Repository**: https://github.com/jiangjyjy/Yue-Benchmark

**Used for**: Cantonese (CMMLU) evaluation

**Installation**:
```bash
git clone https://github.com/jiangjyjy/Yue-Benchmark
cd Yue-Benchmark
# Follow the repository's installation instructions
```

**Usage**: This directory contains scripts that work with Yue-Benchmark:
- `generate_cmmlu_predictions.py` - Generates predictions for CMMLU tasks
- `evaluate_cmmlu_yue.py` - Evaluates CMMLU predictions

## Scripts in This Directory

### generate_cmmlu_predictions.py

Generates predictions for CMMLU (Cantonese) benchmark.

**Usage**:
```bash
python generate_cmmlu_predictions.py \
    --model_path <path_to_model> \
    --data_dir <path_to_yue_benchmark_data> \
    --output_dir <output_directory> \
    --num_shots 0 \
    --device cuda:0 \
    --dtype float16 \
    --max_new_tokens 512
```

**Arguments**:
- `--model_path`: Path to the trained model
- `--data_dir`: Path to Yue-Benchmark data directory (should contain JSON/JSONL files)
- `--output_dir`: Directory to save prediction files
- `--num_shots`: Number of few-shot examples (0 or 5)
- `--device`: Device to use (e.g., cuda:0)
- `--dtype`: Data type (float16 or float32)
- `--max_new_tokens`: Maximum number of tokens to generate

### evaluate_cmmlu_yue.py

Evaluates CMMLU predictions and generates accuracy reports.

**Usage**:
```bash
python evaluate_cmmlu_yue.py \
    --predictions_dir <directory_with_predictions> \
    --output_dir <output_directory>
```

**Arguments**:
- `--predictions_dir`: Directory containing prediction JSON files (organized by model)
- `--output_dir`: Directory to save evaluation results

**Output**:
- `cmmlu_yue_results.csv`: CSV file with accuracy per subject
- `cmmlu_yue_results.json`: Detailed JSON results with all metrics
- `cmmlu_yue_errors.json`: Error cases for analysis

## Integration with Main Script

The main reproduction script (`scripts/run_train_final.sh`) automatically uses these evaluation utilities. Set the following environment variables to specify library paths:

```bash
export LM_EVAL_REPO=/path/to/lm-evaluation-harness
export MALAY_REPO=/path/to/MalayMMLU
export YUE_BENCHMARK_ROOT=/path/to/Yue-Benchmark
```

Or modify the default paths in `run_train_final.sh` (they default to `$WORKSPACE_ROOT/<library_name>`).

## Notes

- All evaluation libraries should be cloned and set up before running the main script
- The scripts support both 0-shot and 5-shot evaluation
- Results are saved in structured formats (CSV and JSON) for easy analysis
- Error cases are preserved for debugging and analysis
