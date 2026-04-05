import os
import re
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Keypoint layout: 231 dims = 77 joints x 3
# [  0: 24] Body 8j  [ 24:105] Face 27j
# [105:168] LHand 21j  [168:231] RHand 21j
IDX_LH_START = 105
IDX_RH_START = 168
HAND_DIMS    = 63


def _read_split_ids(split_path):
    split_file = Path(split_path)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def normalize_vietnamese_text(text):
    if not text:
        return ""
    confusion_map = {":": ".", ";": ".", "!": ".", "\u00a0": " ", "\t": " "}
    text = unicodedata.normalize("NFC", str(text)).lower()
    for wrong, right in confusion_map.items():
        text = text.replace(wrong, right)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,:;!?]+$", "", text).strip()
    return text


def _read_text_label(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    first = raw.split("#", 1)[0]
    return normalize_vietnamese_text(first)


def build_segment_text_for_ctc(dataset_root, train_split, dev_split,
                                min_freq=1, unk_token="<unk>"):
    """
    Build vocabulary từ TRAIN ONLY.
    OOV trong val/test → <unk>.
    """
    dataset_root = Path(dataset_root)
    text_dir     = dataset_root / "texts"

    train_ids = _read_split_ids(train_split)
    dev_ids   = _read_split_ids(dev_split)

    def build_df(sample_ids):
        rows = []
        for sid in sample_ids:
            tp = text_dir / f"{sid}.txt"
            if not tp.exists():
                continue
            gloss = _read_text_label(tp)
            if gloss:
                rows.append({"id": sid, "gloss": gloss})
        return pd.DataFrame(rows)

    train_data = build_df(train_ids)
    dev_data   = build_df(dev_ids)

    # Đếm tần suất từ trong train
    train_word_freq = Counter()
    for ann in train_data["gloss"]:
        train_word_freq.update(normalize_vietnamese_text(ann).split())

    train_vocab = {w for w, c in train_word_freq.items() if c >= min_freq}

    vocab_list    = ["_", unk_token] + sorted(train_vocab)
    vocab_map     = {g: i for i, g in enumerate(vocab_list)}
    inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}
    unk_id        = vocab_map[unk_token]

    # OOV analysis
    dev_word_freq = Counter()
    for ann in dev_data["gloss"]:
        dev_word_freq.update(normalize_vietnamese_text(ann).split())
    oov_words     = {w for w in dev_word_freq if w not in vocab_map}
    oov_cnt       = sum(dev_word_freq[w] for w in oov_words)
    total_dev_tok = sum(dev_word_freq.values())

    print(f"\n{'='*55}")
    print(f"Vocabulary built from TRAIN only")
    print(f"  Train vocab size  : {len(vocab_list)} (blank + unk + {len(train_vocab)} words)")
    print(f"  Dev unique words  : {len(dev_word_freq)}")
    print(f"  OOV in dev        : {len(oov_words)} words | "
          f"{oov_cnt}/{total_dev_tok} tokens = {100*oov_cnt/max(total_dev_tok,1):.1f}%")
    if oov_words:
        top = sorted(oov_words, key=lambda w: -dev_word_freq[w])[:10]
        print(f"  Top OOV           : {top}")
    print(f"{'='*55}\n")

    def encode_annotations(df):
        df = df.copy()
        df["gloss"] = df["gloss"].apply(normalize_vietnamese_text)
        df["enc"]   = df["gloss"].apply(
            lambda x: [vocab_map.get(g, unk_id) for g in x.split()]
        )
        # Bỏ sample mà toàn bộ token đều là unk
        df = df[df["enc"].apply(lambda e: any(t != unk_id for t in e))]
        return df[["id", "enc"]]

    return (encode_annotations(train_data), encode_annotations(dev_data),
            vocab_map, inv_vocab_map, vocab_list)


# ── Missing-hand interpolation ────────────────────────────────────────────────

def _interpolate_missing_hand(seq, start, length):
    hand    = seq[:, start:start + length]
    missing = (np.abs(hand).sum(axis=1) == 0)
    if missing.all():
        return seq
    last_valid = None
    for t in range(len(missing)):
        if not missing[t]:
            last_valid = hand[t].copy()
        elif last_valid is not None:
            hand[t] = last_valid
    first_valid = None
    for t in range(len(missing) - 1, -1, -1):
        if not missing[t]:
            first_valid = hand[t].copy()
        elif first_valid is not None:
            hand[t] = first_valid
    seq[:, start:start + length] = hand
    return seq


def fix_missing_hands(sequence):
    sequence = _interpolate_missing_hand(sequence, IDX_LH_START, HAND_DIMS)
    sequence = _interpolate_missing_hand(sequence, IDX_RH_START, HAND_DIMS)
    return sequence


# ── Dataset ───────────────────────────────────────────────────────────────────

class SegmentNPYDataset(Dataset):
    def __init__(self, dataset_root, split_file, target_enc_df,
                 transform=None, min_len=32, max_len=1000):
        self.dataset_root = Path(dataset_root)
        self.motion_dir   = self.dataset_root / "new_joints"
        self.text_dir     = self.dataset_root / "texts"
        self.transform    = transform
        self.min_len      = min_len
        self.max_len      = max_len

        if not self.motion_dir.exists():
            raise FileNotFoundError(f"Motion dir not found: {self.motion_dir}")
        if not self.text_dir.exists():
            raise FileNotFoundError(f"Text dir not found: {self.text_dir}")

        self.ids  = _read_split_ids(split_file)
        enc_map   = {str(row["id"]): row["enc"] for _, row in target_enc_df.iterrows()}

        self.files  = []
        self.labels = []
        for sid in self.ids:
            if ((self.motion_dir / f"{sid}.npy").exists()
                    and (self.text_dir / f"{sid}.txt").exists()
                    and sid in enc_map):
                self.files.append(sid)
                self.labels.append(enc_map[sid])

        print(f"Loaded {len(self.files)} segment samples from {split_file}")

    def __len__(self):
        return len(self.files)

    def _pad_or_crop(self, seq):
        T, D = seq.shape
        if T < self.min_len:
            pad = np.zeros((self.min_len - T, D), dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=0)
        if seq.shape[0] > self.max_len:
            seq = seq[:self.max_len]
        return seq

    def _read_motion(self, sample_id):
        pose = np.load(self.motion_dir / f"{sample_id}.npy", allow_pickle=False)
        if pose.ndim == 1:
            pose = pose[None, :]
        elif pose.ndim == 3:
            pose = pose.reshape(pose.shape[0], -1)
        if pose.ndim != 2:
            raise ValueError(f"Unexpected pose shape {pose.shape} for {sample_id}")
        return pose.astype(np.float32)

    def __getitem__(self, idx):
        sid  = self.files[idx]
        pose = self._read_motion(sid)
        pose = fix_missing_hands(pose)
        pose = self._pad_or_crop(pose)
        pose = torch.from_numpy(pose).float()
        if self.transform:
            pose = self.transform(pose)
        return sid, pose, torch.as_tensor(self.labels[idx])