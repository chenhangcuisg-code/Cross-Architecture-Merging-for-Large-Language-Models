# dataset_gsm8k.py
from typing import Iterable, Iterator, List, Optional
from datasets import load_dataset

def load_gsm8k_texts(
    subset: str = "main",          # "main" or "socratic"
    split: str = "train",          # "train" or "test"
    streaming: bool = False,       # True for streaming mode
    shuffle: bool = False,
    seed: int = 42,
    max_samples: Optional[int] = None,   # Limit number of samples; None for all
    template: str = "{question}",  # Template for constructing input text
) -> List[str] | Iterator[str]:
    """
    Load GSM8K problem texts.
    
    Args:
        subset: "main" or "socratic"
        split: "train" or "test"
        streaming: Whether to use streaming mode
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        max_samples: Maximum number of samples (None for all)
        template: Template string for formatting questions
        
    Returns:
        List of text strings or iterator
    """
    ds = load_dataset("openai/gsm8k", subset, split=split, streaming=streaming)

    if streaming:
        # Streaming mode: yield on demand
        def _iter() -> Iterator[str]:
            itr = ds.shuffle(seed=seed) if shuffle else ds
            count = 0
            for row in itr:
                text = template.format(question=row["question"])
                yield text
                count += 1
                if max_samples is not None and count >= max_samples:
                    break
        return _iter()
    else:
        if shuffle:
            ds = ds.shuffle(seed=seed)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        texts = [template.format(question=row["question"]) for row in ds]
        return texts
