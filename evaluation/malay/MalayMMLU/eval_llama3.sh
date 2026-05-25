TOKEN="${HF_TOKEN:?Set HF_TOKEN to your Hugging Face access token}"
SHOT=0
#/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k /data/chenhang/models/Malaysian-Llama-3.2-1B-Instruct-v0.1 /data/chenhang/optimal_trans/llama_mala_hotres_ft_mergehot_full
## first token accuracy 


python src/evaluate.py  --by_letter --shot $SHOT --task=MalayMMLU \
                    --base_model=/data/chenhang/models/llamamala_llama_fused_alpha01_select_norm/llama_malaysian_sft_train_hot \
                    --output_folder=output/ --token $TOKEN




python src/evaluate.py  --by_letter --shot $SHOT --task=MalayMMLU \
                    --base_model=/data/chenhang/models/llamamala_llama_fused_alpha01_select_norm/llama_malaysian_sft_train_nohot \
                    --output_folder=output/ --token $TOKEN





python src/evaluate.py  --by_letter --shot $SHOT --task=MalayMMLU \
                    --base_model=/data/chenhang/models/llamamala_llama_fused_alpha01_select_norm/llama_malaysian_sft_train_hot/ablation_untrained_hot_fused \
                    --output_folder=output/ --token $TOKEN



