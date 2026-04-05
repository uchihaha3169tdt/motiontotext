import numpy as np
import pandas as pd
import torch
from utils.metrics import normalize_gloss_sequence


class GaussianNoise:
    """Add Gaussian noise to a tensor — data augmentation for pose sequences."""
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std  = std

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.array(tensor))
        return tensor + torch.randn_like(tensor) * self.std + self.mean

    def __repr__(self):
        return f"GaussianNoise(mean={self.mean}, std={self.std})"


def invert_to_chars(sents, inv_ctc_map):
    """
    Convert padded label tensor (B, L) or (1, L) back to list of gloss strings.
    Stops at blank token (id=0).
    """
    if isinstance(sents, torch.Tensor):
        sents = sents.detach().cpu().numpy()
    outs = []
    for row in sents:
        for x in row:
            if int(x) == 0:
                break
            outs.append(inv_ctc_map[int(x)])
    return outs


def convert_text_for_ctc(dataset_name, train_csv, dev_csv):
    """
    Read annotation CSVs, build vocab, and encode glosses for CTC training.
    Supports isharah/csl (id|gloss) and generic (id|annotation) formats.
    """
    train_data = pd.read_csv(train_csv, delimiter="|")
    dev_data   = pd.read_csv(dev_csv,   delimiter="|")
    all_data   = pd.concat([train_data, dev_data], ignore_index=True)

    is_isharah = "isharah" in dataset_name.lower() or "csl" in dataset_name.lower()
    gloss_col  = "gloss" if is_isharah else "annotation"

    all_data = all_data[all_data["id"].notna() & all_data[gloss_col].notna()]

    all_glosses = set()
    for ann in all_data[gloss_col]:
        all_glosses.update(normalize_gloss_sequence(str(ann)).split())

    vocab_list    = ["_"] + sorted(all_glosses)
    vocab_map     = {g: i for i, g in enumerate(vocab_list)}
    inv_vocab_map = {i: g for g, i in vocab_map.items()}
    print(f"Vocabulary size: {len(vocab_map)}")

    def encode(df):
        df = df[df["id"].notna() & df[gloss_col].notna()].copy()
        df[gloss_col] = df[gloss_col].apply(lambda x: normalize_gloss_sequence(str(x)))
        df["enc"] = df[gloss_col].apply(
            lambda x: [vocab_map[g] for g in x.split() if g in vocab_map]
        )
        return df[["id", "enc"]]

    return encode(train_data), encode(dev_data), vocab_map, inv_vocab_map, vocab_list