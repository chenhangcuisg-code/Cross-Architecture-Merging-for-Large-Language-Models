#!/usr/bin/env bash
# 检查 lm-eval 环境能否正常启动、不报 import / AutoModelForVision2Seq 等错
# 用法: bash scripts/test_lm_eval_env.sh
# 不跑完整评测，只做最小检查（可选做 1 个 batch 的 boolq）

set -e
echo ">>> [1/3] 激活 lm-eval 环境并检查导入..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate lm-eval
python -c "
import lm_eval
import lm_eval.models
print('lm_eval 导入 OK')
"
echo ">>> [2/3] 检查 transformers 是否可加载 Qwen3 config（trust_remote_code）..."
python -c "
import transformers
path = '/data/chenhang/optimal_trans_new/hot_sft_runs_qwen3_1b/qwen_alpaca_train_alpha_lr_search_final_nolora_gen_qwen/alpha0.01_lr5e-7/hot/ablation_untrained_hot_fused'
cfg = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
print('config model_type:', getattr(cfg, 'model_type', '?'))
print('Qwen3 config 加载 OK')
"
echo ">>> [3/3] 最小 lm_eval 调用（boolq limit=0.001，约 1 个 batch 即停）..."
lm_eval --model hf \
  --model_args "pretrained=/data/chenhang/optimal_trans_new/hot_sft_runs_qwen3_1b/qwen_alpaca_train_alpha_lr_search_final_nolora_gen_qwen/alpha0.01_lr5e-7/hot/ablation_untrained_hot_fused,dtype=float,trust_remote_code=True" \
  --tasks boolq \
  --limit 0.001 \
  --batch_size 1 \
  --output_path /tmp/lm_eval_test_out
echo ">>> 全部检查通过。"
