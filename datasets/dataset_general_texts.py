from datasets import load_dataset

def load_general_english_texts(subset="indonesian", split="train", max_samples=2000):
    """
    Load general text datasets.
    
    Args:
        subset: One of ["common", "wiki", "imdb", "c4", "indonesian", "malay", "en"]
        split: "train" or "test"
        max_samples: Maximum number of samples to load
        
    Returns:
        List of text strings
    """
    streaming = False

    if subset == "common":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    elif subset == "wiki":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    elif subset == "imdb":
        ds = load_dataset("imdb", split=split)

    elif subset == "c4":
        ds = load_dataset("allenai/c4", "en", split=split)

    elif subset == "eng":
        ds = load_dataset(
            "allenai/c4",
            "en",
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        streaming = True

    elif subset == "indonesian":
        ds = load_dataset(
            "allenai/c4",
            "id",
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        streaming = True

    elif subset == "malay":
        ds = load_dataset(
            "allenai/c4",
            "ms",
            split=split,
            streaming=True,
            trust_remote_code=True
        )
        streaming = True

    else:
        raise ValueError(f"Unknown subset: {subset}")

    # Handle streaming datasets
    if streaming:
        texts = []
        for i, sample in enumerate(ds):
            if i >= max_samples:
                break
            if "text" in sample:
                texts.append(sample["text"])
        return texts

    # Non-streaming datasets
    num = min(max_samples, len(ds))
    texts = [x["text"] for x in ds.select(range(num))]
    return texts
