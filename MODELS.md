# Models used in the paper

Every model that gets merged in this repo is publicly available on Hugging Face.
`scripts/download_models.py` downloads all of them in one go.

## Six paper tasks (run_train_final.sh)

For each task we fuse a **1B small "donor" model** (Model A — domain/language SFT'd) with the **shared 8B base** (Model B).

| Task | Model A (1B, fine-tuned) — HF repo | Model B (8B base) — HF repo | Fuse α | Train α | Train LR | Dataset |
|---|---|---|---:|---:|---:|---|
| **medical** | [`PathFinderKR/Llama-3-1B-Medical-Instruct`](https://huggingface.co/PathFinderKR/Llama-3-1B-Medical-Instruct) | [`unsloth/Llama-3.1-8B-Instruct`](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) | 0.03 | **0.005** | 3e-7 | `medical` |
| **thai** | [`scb10x/llama3.2-typhoon2-1b-instruct`](https://huggingface.co/scb10x/llama3.2-typhoon2-1b-instruct) | [`unsloth/Llama-3.1-8B-Instruct`](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) | 0.01 | **0.01** | 1e-7 | `fineweb_thai` ([`ChavyvAkvar/fineweb-2-1M-Sample-Thai`](https://huggingface.co/datasets/ChavyvAkvar/fineweb-2-1M-Sample-Thai)) |
| **finance** | [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | [`unsloth/Llama-3.1-8B-Instruct`](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) | 0.03 | **0.03** | 1e-6 | `finance` |
| **cantonese** | [`unsloth/Llama-3.2-1B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) | [`FlagAlpha/Llama3-Chinese-8B-Instruct`](https://huggingface.co/FlagAlpha/Llama3-Chinese-8B-Instruct) | 0.01 | **0.1** | 1e-8 | `cantonese` |
| **indonesian** | [`digo-prayudha/Llama-3.2-1B-Indonesian-lora`](https://huggingface.co/digo-prayudha/Llama-3.2-1B-Indonesian-lora) | [`unsloth/Llama-3.1-8B-Instruct`](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) | 0.01 | **0.1** | 1e-6 | `indonesian` |
| **malay** | [`mesolitica/Malaysian-Llama-3.2-1B-Instruct`](https://huggingface.co/mesolitica/Malaysian-Llama-3.2-1B-Instruct) | [`unsloth/Llama-3.1-8B-Instruct`](https://huggingface.co/unsloth/Llama-3.1-8B-Instruct) | 0.1 | **0.1** | 1e-6 | `malay` |

Common training settings (all six tasks): `MAX_SAMPLES=2000` (thai uses 8000), `BATCH_SIZE=1`, `GRAD_ACC=8`, `NUM_EPOCHS=1`, `BLOCK_SIZE=2048`, `lora_target_modules=q_proj,k_proj,v_proj,o_proj`, `fp16`. Fuse parameters (all): `LM_ONLY=true`, `ATTN_AUTOCAST=true`, `HOT_CHUNK_COLS=1024`, `HOT_DTYPE=float32`.

> "Fuse α" is the α used by `generate_hot_residual.py` to bake the transport-residual into Model A (Step 2). "Train α" is the α the residual is scaled by during the SFT pass (Step 3). They are *not* the same: the paper's main table picks the best of `ALPHA_SEARCH_LIST=(0.005…0.2)` after training.

## Secondary experiments (not the 6-task table)

| Experiment | Model A | Model B | Note |
|---|---|---|---|
| **math** (`trans_train_reasoning.sh`) | [`masani/SFT_math_Llama-3.2-1B_epoch_1_global_step_29`](https://huggingface.co/masani/SFT_math_Llama-3.2-1B_epoch_1_global_step_29) | A SFT'd 8B math checkpoint (e.g. `unsloth/Llama-3.1-8B-Instruct` further SFT'd; not released) | α-search `(0.01, 0.03, 0.05)`, lr `(5e-7, 1e-6)`, dataset `openr1` |
| **code** (`trans_train_code.sh`) | A 1B code SFT (not released) | A 8B code SFT (not released) | dataset `apps`, α=0.1; replace these two with public code LLMs (e.g. `meta-llama/CodeLlama-7b-Python-hf`) if reproducing |

## Datasets

- `medical`, `finance`, `cantonese`, `indonesian`, `malay`: built by `dataset_hot_texts.py` from publicly available HF sources (see file). Subset names map to dataset loaders inside that file.
- `fineweb_thai`: `ChavyvAkvar/fineweb-2-1M-Sample-Thai` — 18 parquet files pre-downloaded by `huggingface-cli download`; the run script sets `LOCAL_DATASET_PATH=$FINEWEB_THAI_CACHE_DIR` so the run is fully offline.
- `apps` (code task): the [APPS](https://huggingface.co/datasets/codeparrot/apps) coding benchmark.
- `openr1` (math task): the [OpenR1-Math](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) reasoning dataset.

## Evaluation harnesses

Three external benchmarks are used and live outside this repo:

| Benchmark | Repo | Used for |
|---|---|---|
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | EleutherAI | medical, thai, finance, indonesian; ablation suite |
| [MalayMMLU](https://github.com/UMxYTL-AI-Labs/MalayMMLU) | UMxYTL-AI-Labs | malay |
| [Yue-Benchmark](https://github.com/jiangjyjy/Yue-Benchmark) | jiangjyjy | cantonese (CMMLU-Yue) |

Set `LM_EVAL_REPO`, `MALAY_REPO`, `YUE_BENCHMARK_ROOT` env vars or clone them at `$WORKSPACE_ROOT/{lm-evaluation-harness,MalayMMLU,Yue-Benchmark}`.
