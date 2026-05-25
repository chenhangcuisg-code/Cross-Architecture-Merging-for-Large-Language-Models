 #/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k  /data/chenhang/models/Llama-3.2-1B-Instruct-Indonesian-merged



 

lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/Llama-3.2-1B-Indonesian-QLora,dtype=float16,load_in_4bit=False,load_in_8bit=False \
    --tasks belebele_ind_Latn,xcopa_id,arc_id,xstorycloze_id,truthfulqa_id_mc1,truthfulqa_id_mc2 \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/Llama-3.2-1B-Indonesian-QLora-ori


lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/llamaindo_llama_fused_alpha01_select_sft_r_norm/llama_indonesian_conversation_train_hot,dtype=float16,load_in_4bit=False,load_in_8bit=False \
    --tasks copal_id \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/Llama-3.2-1B-Indonesian-QLora-mergetrain-train_hot_sftr

lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/llamaindo_llama_fused_alpha01_select_sft_r_norm/llama_indonesian_conversation_train_nohot,dtype=float16,load_in_4bit=False,load_in_8bit=False \
    --tasks copal_id \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/Llama-3.2-1B-Indonesian-QLora-mergetrain-train_nohot_sftr




