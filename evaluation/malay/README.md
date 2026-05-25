# Malay (MalayMMLU) Evaluation

Self-contained copy of the [MalayMMLU] code (only the runnable scripts —
no model checkpoints, no result dumps). Use this to score the fused model
on MalayMMLU.

## Layout

```
malay/
└── MalayMMLU/
    ├── eval_batch.py         # cross-shot result aggregator
    ├── eval_batch.sh         # batched re-run helper
    ├── eval_llama3.sh        # llama-3 specific wrapper
    ├── README.md             # upstream README
    ├── README_ms.md
    ├── requirements.txt
    ├── scripts/              # slurm + bare-metal templates
    │   ├── no_slurm_example.sh
    │   └── slurm_example.sh
    └── src/
        ├── evaluate.py       # main driver (used by paper)
        ├── eva_batch.py
        ├── calculate_accuracies.py
        ├── evaluate_glm.py / evaluate_gpt.py / evaluate_intern_vl.py /
        ├── evaluate_pixtral.py / evaluate_qwen_vl.py
        └── utils.py / utils_intern_vl.py / utils_vl.py
```

## Quick run (paper config)

```bash
cd MalayMMLU
pip install -r requirements.txt

python src/evaluate.py \
  --by_letter \
  --shot 0 \
  --task MalayMMLU \
  --base_model "${MODEL_PATH}" \
  --output_folder output/${MODEL_NAME} \
  --token "${HF_TOKEN}"
```

After all runs:

```bash
python eval_batch.py --all \
  --pred_dir output \
  --shot 0 \
  --output_dir output_batch
```

[MalayMMLU]: https://github.com/UMxYTL-AI-Labs/MalayMMLU
