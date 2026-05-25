# Cross-Architecture Merging for Large Language Models

This repository contains the implementation of Cross-Architecture Merging for Large Language Models.

**Quick links**: [MODELS.md](MODELS.md) — every Hugging Face repo used (1B donor × 8B base, all 6 tasks) with their α / lr / dataset. [REPRODUCE.md](REPRODUCE.md) — end-to-end commands.

## Overview

1. **Activation Extraction**: Extracting activations from source and target models
2. **Transport Plan Computation**: Computing transport plans (P and Q matrices) using Sinkhorn algorithm
3. **Model Fusion**: Fusing knowledge from source model to target model using computed transport plans
4. **Training**: Fine-tuning the fused model with transport-based residual connections

## Structure

```
.
├── run_activs_and_hot.py    # Step 1: activation extraction + transport plan computation
├── generate_hot_residual.py # Step 2: model fusion (invoked from repo root by run_train_final.sh)
├── train_hot_residual_sft.py # Step 3: training script (invoked from repo root)
├── dataset_hot_texts.py     # Text dataset loading for run_activs_and_hot (malay, medical, gsm8k, etc.)
├── activs_llama3_modules.py # LLaMA-3 activation extraction
├── activs_qwen2_modules.py  # Qwen2 activation extraction
├── core/                    # Core algorithm implementations
│   ├── hot_transport.py     # Transport plan computation (Sinkhorn, correlation distance)
│   ├── hot_transport_chunk.py # Chunked stable HOT (compute_P_stable, used by run_activs_and_hot)
│   ├── generate_hot_residual.py  # Transport-based residual generation and model fusion
│   └── train_hot_residual_sft.py # Training script with transport-based residual support
├── data_loading/            # Dataset loading utilities (avoids shadowing HF 'datasets')
│   ├── dataset_general_texts.py
│   └── dataset_gsm8k.py
├── scripts/                 # Scripts
│   ├── download_models.py   # Download every HF model used by the paper (see MODELS.md)
│   ├── run_pipeline.sh      # Minimal end-to-end pipeline example (configurable via env vars)
│   └── run_train_final.sh   # Full reproduction script (6 tasks: medical, thai, finance, cantonese, indonesian, malay)
├── evaluation/              # Per-domain evaluation pipelines (code only, no large assets)
│   ├── master/              # Top-level orchestration (eval_all_tasks.sh, run_source8b_eval.sh)
│   ├── medical/             # MedQA + MMLU medical subsets (lm-evaluation-harness)
│   ├── thai/                # XCOPA / XQuAD / XNLI Thai (lm-evaluation-harness)
│   ├── finance/             # Finance lm-eval tasks + FinanceQA generative eval
│   ├── indonesian/          # Indonesian lm-eval + bundled IndoMMLU pipeline
│   ├── math/                # Arithmetic + Minerva math + MMLU elementary math
│   ├── cantonese/           # CMMLU / Yue-MMLU prediction + aggregation
│   ├── malay/               # Bundled MalayMMLU evaluator (code-only)
│   ├── general_ability/     # arc_easy / commonsense_qa / piqa / social_iqa / winogrande
│   └── utils/               # Result aggregation, baseline-vs-best comparison
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

Three-step reproduction. The per-task Hugging Face model IDs, α-fuse, α-train, lr and datasets are listed in [MODELS.md](MODELS.md); the end-to-end walk-through (including a single-task by-hand version) is in [REPRODUCE.md](REPRODUCE.md).

```bash
# 0. install
pip install -r requirements.txt

# 1. pull every model from HF (default: ./models)
python scripts/download_models.py

# 2. run all 6 paper tasks
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_train_final.sh
```

`run_train_final.sh` honours these env vars:

- **WORKSPACE_ROOT**: Repository root (default: parent of `scripts/`).
- **MODELS_ROOT**: Base/fused model directory (default: `$WORKSPACE_ROOT/models`, the destination of `download_models.py`).
- **HOT_RESULTS_ROOT**: Transport plan output directory (default: `$WORKSPACE_ROOT/transport_results`).
- **LM_EVAL_REPO**: Path to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (default: `$WORKSPACE_ROOT/lm-evaluation-harness`).
- **MALAY_REPO**: Path to [MalayMMLU](https://github.com/UMxYTL-AI-Labs/MalayMMLU) evaluation repo (default: `$WORKSPACE_ROOT/MalayMMLU`).
- **YUE_BENCHMARK_ROOT**: Path to [Yue-Benchmark](https://github.com/jiangjyjy/Yue-Benchmark) (CMMLU data/scripts) (default: `$WORKSPACE_ROOT/Yue-Benchmark`).

For a minimal demonstration with a single task (default: `malay`):

```bash
bash scripts/run_pipeline.sh
# or another task:
MODEL_A=PathFinderKR/Llama-3-1B-Medical-Instruct MODEL_B=unsloth/Llama-3.1-8B-Instruct \
DATA_SUBSET=medical ALPHA_FUSE=0.03 ALPHA_TRAIN=0.005 LR=3e-7 \
DATASET_TYPE=medical_llama3 bash scripts/run_pipeline.sh
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
- See [MODELS.md](MODELS.md) for the exact Hugging Face repo IDs used in the paper (1B donor × 8B base) and the paper-validated per-task α / lr.

## Supported Datasets

- General text: C4 (multilingual), WikiText
- Domain-specific: Medical, Finance, GSM8K
- Multilingual: Indonesian, Malay, Thai, Cantonese

## Citation

