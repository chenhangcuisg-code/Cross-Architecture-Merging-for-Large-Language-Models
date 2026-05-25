 #/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k copal_id /data/chenhang/models/llama3-1b-med ,mmlu_prox_lite_th
#xcopa_th,xquad_th,xnli_th  mgsm_direct_th,mmlu_prox_lite_th mmlu_prox
export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval

lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/Llama-3.1-8B-Instruct,dtype="float" \
    --tasks xcopa_th,xquad_th,xnli_th,mmlu_prox_lite_th \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama3.2-1bth-instruct/8b

lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans/hot_sft_runs_llamathai_1b/llama_fineweb_thai_train_hot/ablation_untrained_hot_fused,dtype="float"  \
    --tasks mgsm_direct_th,mmlu_prox_lite_th \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama3.2-1bth-instruct/ablation_untrained_hot_fused


lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans/hot_sft_runs_llamathai_1b/llama_fineweb_thai_train_hot,dtype="float"  \
    --tasks mgsm_direct_th,mmlu_prox_lite_th \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama3.2-1bth-instruct/train_hot


lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans/hot_sft_runs_llamathai_1b/llama_fineweb_thai_train_nohot,dtype="float"  \
    --tasks mgsm_direct_th,mmlu_prox_lite_th \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama3.2-1bth-instruct/train_nohot
