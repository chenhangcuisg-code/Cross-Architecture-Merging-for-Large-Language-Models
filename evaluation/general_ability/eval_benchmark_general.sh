 #/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k copal_id /data/chenhang/models/llama3-1b-med

export HF_DATASETS_CACHE=/data/chenhang/hf_cache_eval
lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans_new/hot_sft_runs_llama32_1b/llama_openr1_math_train_hot/ablation_untrained_hot_fused,dtype="float" \
    --tasks medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama32_1b/train_hot

lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans_new/hot_sft_runs_llama32_1b/llama_openr1_math_train_hot,dtype="float" \
    --tasks medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama32_1b/train_hot



lm_eval --model hf \
    --model_args pretrained=/data/chenhang/optimal_trans_new/hot_sft_runs_llama32_1b/llama_openr1_math_train_nohot,dtype="float" \
    --tasks medqa_4options,mmlu_anatomy,medmcqa,mmlu_clinical_knowledge,mmlu_college_biology,mmlu_college_medicine,mmlu_medical_genetics,mmlu_professional_medicine \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/llama32_1b/train_nohot
