# Reproducing the paper

This walks through the exact commands the paper used. The full configurations of all 6 tasks (model paths, α, lr, dataset, eval tasks) live in [`MODELS.md`](MODELS.md) and are baked into `scripts/run_train_final.sh`.

## 0. Environment

```bash
# Recommended: Python 3.10 + CUDA 12.x, A100/H100 GPUs (≥1×80GB or 4×24GB).
conda create -n trans_opt python=3.10 -y && conda activate trans_opt
pip install -r requirements.txt
# Optional China mirror (the original setup):
export HF_ENDPOINT=https://hf-mirror.com
```

External evaluation harnesses (clone next to this repo or set env vars):

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
git clone https://github.com/UMxYTL-AI-Labs/MalayMMLU
git clone https://github.com/jiangjyjy/Yue-Benchmark
# Install lm-eval inside a separate env named "lm-eval":
conda create -n lm-eval python=3.10 -y && conda activate lm-eval
pip install -e ./lm-evaluation-harness
```

## 1. Download every model from Hugging Face

```bash
# Token only needed if you use gated models; the paper uses public mirrors of every model.
export HF_TOKEN=hf_xxx              # optional
python scripts/download_models.py   # ~80 GB total → ./models/
```

This populates `./models/` with the 7 directories `run_train_final.sh` expects: `llama3-1b-med, llama3.2-typhoon2-1b-instruct, Llama-3.2-1B-Instruct, Llama3-Chinese-8B-Instruct, Llama-3.2-1B-Indonesian-QLora, Malaysian-Llama-3.2-1B-Instruct-v0.1, Llama-3.1-8B-Instruct`.

(For the Thai task also pre-download FineWeb-Thai:
`huggingface-cli download ChavyvAkvar/fineweb-2-1M-Sample-Thai --repo-type dataset --local-dir ./hf_cache_eval/fineweb_thai_cli`.)

## 2. Run the full paper pipeline (6 tasks)

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash scripts/run_train_final.sh

# 4-GPU distributed
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash scripts/run_train_final.sh

# Run a single task only (e.g. malay):
TASK_CONFIGS='("config_malay")' bash scripts/run_train_final.sh
```

The script performs Step 1 (activations + transport) → Step 2 (fuse) → Step 3 (train HOT + nohot + ablation, then auto-eval and clean up) for each of the six tasks. Each task uses its own α / lr / dataset / eval set (see `config_<task>()` definitions at lines 215-365).

## 3. Reproduce a single task by hand (transparency)

The high-level pipeline is `Activations → Transport → Fuse → Train`. For the **malay** task (α_fuse=0.1, α_train=0.1, lr=1e-6):

```bash
export MODELS_DIR=./models                     # produced by scripts/download_models.py
export WORKSPACE_ROOT=$PWD

# 1. Activations + transport plans (P, Q)
python run_activs_and_hot.py \
  --model-a-path "$MODELS_DIR/Malaysian-Llama-3.2-1B-Instruct-v0.1" \
  --model-b-path "$MODELS_DIR/Llama-3.1-8B-Instruct" \
  --data-subset malay --max-samples 2000 \
  --batch-size-a 2 --batch-size-b 2 \
  --device-a cuda:0 --device-b cuda:0 \
  --hot-chunk-cols 1024 --hot-dtype float32 \
  --top-neuron-dir-a "$WORKSPACE_ROOT/top_neurons_llama3_mala_new" \
  --out-dir "$WORKSPACE_ROOT/transport_results/hot_results_llamamala_llama_norm"

# 2. Fuse Model A with the transport residual
python generate_hot_residual.py \
  --modelA_id "$MODELS_DIR/Malaysian-Llama-3.2-1B-Instruct-v0.1" \
  --modelB_id "$MODELS_DIR/Llama-3.1-8B-Instruct" \
  --hot_dir   "$WORKSPACE_ROOT/transport_results/hot_results_llamamala_llama_norm" \
  --hot_neuron_dir "$WORKSPACE_ROOT/top_neurons_llama3_mala_new" \
  --alpha 0.1 --lm_only --verbose \
  --attn_device cuda:0 --attn_max_mem_mb 1200 \
  --output_dir "$WORKSPACE_ROOT/maly_llama_fused_alpha01_fortrain_1b_select"

# 3. HOT SFT (and the no-HOT ablation)
python train_hot_residual_sft.py \
  --training_scenario hot --save_untrained_folded \
  --model_type llama --dataset_type malaysian_sft \
  --model_dir "$WORKSPACE_ROOT/maly_llama_fused_alpha01_fortrain_1b_select" \
  --tokenizer_dir "$MODELS_DIR/Malaysian-Llama-3.2-1B-Instruct-v0.1" \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 --num_train_epochs 1 --block_size 2048 \
  --alpha 0.1 \
  --hot_neuron_dir "$WORKSPACE_ROOT/top_neurons_llama3_mala_new" \
  --fp16 \
  --output_dir "$WORKSPACE_ROOT/hot_sft_runs_llamamala_1b/llama_malaysian_sft_train_alpha_lr_search/alpha0.1_lr1e-6/hot"
```

Repeat with `--training_scenario no_hot` (and without `--save_untrained_folded`) for the ablation pass.

## 4. Evaluation per task

`run_train_final.sh` runs evaluation inline at the end of Step 3; the exact eval tasks per task (from `config_<task>()`):

| Task | Eval task list |
|---|---|
| medical | `medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine` (lm-eval-harness) |
| thai | `belebele_tha_Thai,mgsm_direct_th,mgsm_native_cot_th,xcopa_th,mmlu_prox_lite_th_other,xquad_th,xnli_th` (lm-eval-harness) |
| finance | `financial_tweets,belebele_fin_Latn,multiblimp_fin,global_piqa_completions_fin_latn,global_piqa_prompted_fin_latn` (lm-eval-harness) |
| cantonese | CMMLU-Yue via `evaluation/generate_cmmlu_predictions.py` + `evaluation/evaluate_cmmlu_yue.py` (data at `$YUE_BENCHMARK_ROOT/data/latest_data/Yue-MMLU`) |
| indonesian | `belebele_ind_Latn,xcopa_id,arc_id,xstorycloze_id,truthfulqa_id_mc1,truthfulqa_id_mc2,hellaswag_id` (lm-eval-harness) |
| malay | MalayMMLU (zero-shot) via `$MALAY_REPO/...` (set `MALAY_HF_TOKEN` if needed) |

Common ablation eval (any task): `winogrande,social_iqa,arc_easy,piqa,commonsense_qa` (`lm-eval-harness`).

## Tips & gotchas

- **GPU memory**: fusion (Step 2) requires the 8B donor to be resident; with `--lm_only` (default for the six paper tasks) it fits in 24-40 GB.
- **`top_neurons_<task>`**: optional channel-select directories. `run_train_final.sh` looks for them at `$WORKSPACE_ROOT/top_neurons_<key>` (e.g. `top_neurons_medllama`). If absent, the fusion runs over all neurons — empirically the paper's reported numbers are with the select dirs.
- **CMMLU-Yue (cantonese)** uses two custom scripts shipped in `evaluation/` — they import the Yue-Benchmark data.
- **`hf_transfer` errors**: the scripts explicitly `unset HF_HUB_ENABLE_HF_TRANSFER` before lm-eval; keep that.
- **Reseed**: random seed is hard-coded inside each Python entry point; nothing to set in shell.
