 #/data/chenhang/models/Llama-3.1-8B-Instruct /data/chenhang/models/malaysian-llama-3-8b-instruct-16k  /data/chenhang/models/Llama-3.2-1B-Instruct-Indonesian-merged /data/chenhang/models/Llama-3.2-1B-Indonesian-QLora



 
lm_eval --model hf \
    --model_args pretrained=/data/chenhang/models/Llama-3.2-1B-Instruct,load_in_4bit=False,load_in_8bit=False \
    --tasks winogrande,social_iqa,arc_easy,piqa,commonsense_qa \
    --device cuda:0 \
    --batch_size 1 \
    --output_path output/indonesian_general


