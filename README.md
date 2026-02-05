# Cross-Architecture Merging for Large Language Models

This repository contains the implementation of Cross-Architecture Merging for Large Language Models.

## Overview

The code implements a hierarchical optimal transport framework that enables knowledge transfer between models through:
1. **Activation Extraction**: Extracting activations from source and target models
2. **Transport Plan Computation**: Computing transport plans (P and Q matrices) using Sinkhorn algorithm
3. **Model Fusion**: Fusing knowledge from source model to target model using computed transport plans
4. **Training**: Fine-tuning the fused model with transport-based residual connections

## Structure

```
.
├── core/                    # Core algorithm implementations
│   ├── hot_transport.py     # Transport plan computation (Sinkhorn, correlation distance)
│   ├── generate_hot_residual.py  # Transport-based residual generation and model fusion
│   └── train_hot_residual_sft.py # Training script with transport-based residual support
├── datasets/                # Dataset loading utilities
│   ├── dataset_general_texts.py
│   └── dataset_gsm8k.py
├── scripts/                 # Scripts
│   ├── run_pipeline.sh      # Example end-to-end pipeline script
│   └── run_train_final.sh   # Full reproduction script (6 tasks: medical, thai, finance, cantonese, indonesian, malay)
├── evaluation/              # Evaluation utilities (code only, no large assets)
│   ├── generate_cmmlu_predictions.py  # CMMLU prediction generation
│   └── evaluate_cmmlu_yue.py           # CMMLU evaluation (Yue-MMLU)
├── requirements.txt         # Python dependencies
└── README.md

```

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.30.0
- datasets
- numpy
- scipy

## Full Reproduction (Paper Results)

To reproduce the paper results for the six tasks (medical, thai, finance, cantonese, indonesian, malay), use the script in `scripts/run_train_final.sh`. It uses **default paths** that you can override via environment variables:

- **WORKSPACE_ROOT**: Repository root (default: parent of `scripts/`). All relative paths are under this.
- **MODELS_ROOT**: Base/fused model directory (default: `$WORKSPACE_ROOT/models`).
- **TRANSPORT_RESULTS_ROOT**: Transport plan output directory (default: `$WORKSPACE_ROOT/transport_results`).
- **LM_EVAL_REPO**: Path to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (default: `$WORKSPACE_ROOT/lm-evaluation-harness`).
- **MALAY_REPO**: Path to [MalayMMLU](https://github.com/mesolitica/malay-dataset) evaluation repo (default: `$WORKSPACE_ROOT/MalayMMLU`).
- **YUE_BENCHMARK_ROOT**: Path to Yue-Benchmark (CMMLU data/scripts) (default: `$WORKSPACE_ROOT/Yue-Benchmark`).

Place `run_activs_and_hot.py` at `WORKSPACE_ROOT` (same repo or symlink). Then run from the repo root:

```bash
cd scripts && bash run_train_final.sh
```

Or with custom paths:

```bash
WORKSPACE_ROOT=/path/to/repo MODELS_ROOT=/path/to/models bash scripts/run_train_final.sh
```

## Usage

### Step 1: Extract Activations and Compute Transport Plans

```bash
python run_activs_and_hot.py \
    --model-a-path <source_model_path> \
    --model-b-path <target_model_path> \
    --data-subset <dataset_name> \
    --out-dir <transport_output_dir>
```

### Step 2: Fuse Models Using Transport Plans

```bash
python generate_hot_residual.py \
    --modelA_id <target_model_id> \
    --modelB_id <source_model_id> \
    --hot_dir <transport_output_dir> \
    --alpha <fusion_strength> \
    --output_dir <fused_model_dir>
```

### Step 3: Train with Transport-Based Residual

```bash
python train_hot_residual_sft.py \
    --model_dir <fused_model_dir> \
    --output_dir <training_output_dir> \
    --model_type <llama|qwen2|qwen2vl|tinyllava> \
    --training_scenario hot \
    --freeze_strategy frozen_hot
```

## Key Components

### Transport Plan Computation (`hot_transport.py`)

- `corr_distance_matrix`: Computes correlation distance between activations
- `sinkhorn_uniform_streaming`: Memory-efficient Sinkhorn algorithm for large matrices
- `compute_Q_and_layer_costs`: Computes inner-level transport plans Q
- `compute_P`: Computes outer-level layer coupling matrix P
- `reconstruct_X`: Reconstructs activations using transport plans

### Model Fusion (`generate_hot_residual.py`)

- `fuse_attention_only_from_hot_dir`: Fuses attention weights using transport plans
- `enable_hot_residual_for_model`: Enables transport-based residual connections in model
- Supports Q/K/V/O attention components with pre/post coupling

### Training (`train_hot_residual_sft.py`)

- Supports multiple model types: LLaMA, Qwen2, Qwen2-VL, TinyLLaVA
- Training scenarios: transport-based, no-transport (ablation)
- Freeze strategies: frozen_transport, frozen_base, none

## Supported Models

- **Text Models**: LLaMA-3, Qwen2, Qwen2.5

## Supported Datasets

- General text: C4 (multilingual), WikiText
- Domain-specific: Medical, Finance, GSM8K
- Multilingual: Indonesian, Malay, Thai, Cantonese

## Citation

