# Malay reproduction — troubleshooting guide

If your reproduction of the malay task lands a **lower MalayMMLU score for
the trained checkpoint (`$HOT_OUT`) than for the merge-only baseline
(`$HOT_OUT/ablation_untrained_hot_fused`)**, almost always one of the
items below is the cause. Walk the list top to bottom.

> Quick context: `ablation_untrained_hot_fused/` is **the merge-only
> baseline**. It is produced by `train_hot_residual_sft.py` *before*
> any optimizer step — `enable_hot_residual_for_model` patches every
> `nn.Linear` of the loaded base, then `fold_hot_residual_into_weights`
> bakes `W_eff = (1-α)·W_base + α·W_hot` into the weight tensor and
> writes a plain HF checkpoint. The trained `$HOT_OUT/` does the same
> fold *after* SFT runs over the base parameters. So the two checkpoints
> are identical in architecture; only the base weights differ.

## 1. Pin the same paper config — don't change defaults

Paper-row config (see `MODELS.md`):

| Field | Value |
|---|---|
| Model A (1B donor) | `mesolitica/Malaysian-Llama-3.2-1B-Instruct` |
| Model B (8B base)  | `unsloth/Llama-3.1-8B-Instruct` |
| Fuse α (Step 2) | **0.1** |
| Train α (Step 3) | **0.1** |
| Train LR | **1e-6** |
| Dataset | `mesolitica/Malaysian-SFT`, split `google_translate_camel_ai`, **first 2000** rows |
| `per_device_train_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| `num_train_epochs` | 1 |
| `block_size` | 2048 |
| LoRA | **disabled** (paper used `final_nolora_same_param`; do **not** pass `--use_lora`) |
| Precision | `--fp16` |
| Freeze | `--freeze_strategy frozen_hot` (HOT residual is a buffer; only base trains) |
| Save mode | `--save_untrained_folded` (creates `ablation_untrained_hot_fused/`) |

If your run differs on any of these, fix it first.

## 2. The `mesolitica/Malaysian-Llama-3.2-1B-Instruct` repo

The paper's table cites the **non-versioned** name. The lab actually
trained on `Malaysian-Llama-3.2-1B-Instruct-v0.1`. The two are functionally
identical for our purposes (same base, same HF org), but if mesolitica
re-tags `main` in the future you may see a different donor. Pin the
revision:

```bash
huggingface-cli download mesolitica/Malaysian-Llama-3.2-1B-Instruct \
    --revision <commit-sha-from-paper-time> \
    --local-dir models/Malaysian-Llama-3.2-1B-Instruct
```

## 3. `top_neurons_*` directory must come from your own Step-1

The fold path used by `train_hot_residual_sft.py` is:

```python
fold_hot_residual_into_weights(model, hot_neuron_dir=args.hot_neuron_dir)
```

If `hot_neuron_dir` is set, fold runs in **neuron-select** mode
(only top-k rows of `Q/K/V/O` get the donor contribution). If you
re-use a `top_neurons_*` directory from a *different* Step-1 (different
data, different seed, different sample order), the trained ckpt and
the `ablation_untrained_hot_fused` ckpt are folded with different masks
than the ones the transport plan was actually computed for. The merge
loses meaning, and the trained version drifts from the baseline in an
unpredictable direction.

**Rule:** The `top_neurons_*` directory you pass to *both* `--hot_neuron_dir`
arguments (Step-2 fuse and Step-3 train) must be the one produced by
*your* Step-1 (`run_activs_and_hot.py`) on the *same* `data-subset`
(`malay`) and `max-samples` (2000) as Step-3 will use.

If you are reproducing without re-running Step-1, pass
`--hot_neuron_dir ""` everywhere (dense fold). The score will not match
the paper's neuron-select setting, but it is internally consistent and
won't trigger this bug.

## 4. `--model_dir` must point at the FUSED model, not at the base 8B

Step 2 (`generate_hot_residual.py`) writes a fused checkpoint at
`FUSED_MODEL_DIR`. Step 3's `--model_dir` must point at that
directory, *not* at the original `unsloth/Llama-3.1-8B-Instruct`. If you
point it at the base model, both the trained ckpt and the
"untrained_hot_fused" ckpt get re-built from the unfused base, and the
"merge-only" baseline becomes "untrained base + α·HOT residual at
forward time" — far weaker than the actually-fused model the paper
reports.

Sanity check after Step 2:

```bash
python -c "
from transformers import AutoModelForCausalLM
import torch, sys
m = AutoModelForCausalLM.from_pretrained(sys.argv[1], torch_dtype=torch.float32)
# Some layer of the fused model should NOT match the base model byte-wise.
import hashlib
w = m.model.layers[0].self_attn.q_proj.weight.detach().cpu().numpy().tobytes()
print('q_proj layer-0 hash:', hashlib.sha1(w).hexdigest()[:16])
" models/Llama-3.1-8B-Instruct
# repeat for the fused dir; the hash MUST differ.
```

## 5. Tokenizer / chat template at eval time

`mesolitica/Malaysian-Llama-3.2-1B-Instruct` and
`unsloth/Llama-3.1-8B-Instruct` ship slightly different tokenizers
(both LLaMA-3 family but different `chat_template` strings). After
Step-3 the saved tokenizer in `$HOT_OUT/` is whichever was loaded by
the *last* call inside the training script (paper code copies the
`model_dir` tokenizer). MalayMMLU's `evaluate.py` uses 0-shot **letter
mode** (`--by_letter --shot 0`) which depends on the chat template
tokenizing `"A"`/`"B"`/`"C"`/`"D"` to a single first token. If the
tokenizer sees a different chat-template prefix at eval time than at
train time, the first-token logits A/B/C/D may be split across two
sub-tokens, dropping accuracy ~10pts and easily letting "merge-only"
beat "merge+train".

Verify:

```bash
python -c "
from transformers import AutoTokenizer
import sys
tok = AutoTokenizer.from_pretrained(sys.argv[1])
for letter in ['A','B','C','D']:
    ids = tok.encode(letter, add_special_tokens=False)
    print(letter, ids, tok.decode(ids))
" /path/to/HOT_OUT
```

All four must encode to **single** ids. If not, copy the tokenizer
from the donor (`Malaysian-Llama-3.2-1B-Instruct`) into `$HOT_OUT/` and
re-evaluate.

## 6. The eval pointed at the wrong directory

The driver writes results to:

- `$HOT_OUT/...`  — trained (=merge+train)
- `$HOT_OUT/ablation_untrained_hot_fused/...`  — merge-only

Many alpha/lr sweep folders look identical at a glance. Make sure your
"merge-only" number really came from the `ablation_untrained_hot_fused/`
sub-directory of the **same** alpha/lr run. Comparing across alpha
values will show the wrong winner (merge-only at α=0.1 vs trained at
α=0.05, for example).

## 7. lr=1e-6 is at the upper edge for malay

Paper's lr-search for malay was `(1e-6, 1e-7)` and `1e-6` won. But
`1e-6` is on the edge — with full-param SFT on 2000 short instruction
samples, even a single bad batch (e.g. CamelAI translation noise) can
nudge the base far enough that the fused alignment between donor and
base is lost. If you cannot match the paper's number, drop to `5e-7`
or `1e-7` and re-run; if either of those beats merge-only, the issue
is *transient SFT instability*, not a code bug.

## 8. Eval batch size and dtype at eval time

MalayMMLU's `evaluate.py` defaults to `dtype=float16` and a batch
size that depends on free VRAM. The score for the same checkpoint
can drift by 1-3pt across `float16 / bfloat16 / float32` because of
the by-letter argmax tie-breaking. Always evaluate **both**
`ablation_untrained_hot_fused` and the trained ckpt with **the same
dtype** (the paper used `--token ... --shot 0` with the script's
default float16).

## 9. If all 8 checks pass: re-run with the seed pinned

```bash
python train_hot_residual_sft.py --seed 42 ...
```

The training script defaults to `seed=42`; some forks accidentally
remove this. Without a fixed seed, full-param SFT on 2000 examples
has ±2pt run-to-run noise.

## When to give up and accept "merge-only > merge+train"

After all of the above, if the trained checkpoint still loses to the
merge-only baseline, that is **a real result, not a bug**. The paper
itself documents alpha/lr sweeps where the trained variant is *worse*
on some tasks; on malay the paper reports the trained variant winning
*by a margin smaller than its run-to-run noise* (\~0.5pt). For a
mesolitica donor that is itself heavily SFT'd on Malay, it is plausible
that "merge alone" already absorbs most of the donor's domain
knowledge, and additional SFT on `Malaysian-SFT/google_translate_camel_ai`
(a noisier, machine-translated Camel AI subset) dilutes rather than
sharpens it.

In that case the right move is to **report both numbers** (merge-only
+ merge+train) and either:

1. Use merge-only as the malay row of your reproduction table, or
2. Try a cleaner Malay SFT split (e.g. `mesolitica/Malaysian-SFT` split
   `dolly` or `wiki_qa_ms`) instead of `google_translate_camel_ai`.

The pipeline is correct; the SFT data choice is the bottleneck.
