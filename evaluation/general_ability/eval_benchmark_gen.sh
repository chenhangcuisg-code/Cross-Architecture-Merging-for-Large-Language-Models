 #/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k copal_id /data/chenhang/models/llama3-1b-med

export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/Llama-3.2-1B-Instruct,dtype="float" \
    --tasks ifeval,mmlu \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/

