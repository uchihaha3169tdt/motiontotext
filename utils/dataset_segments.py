import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ── Keypoint layout produced by YouTubeToSignSAMProcessor ─────────────────────
# Total: 231 dims = 77 joints × 3 coords (x, y, z)
#   [  0: 24] Body       8 joints × 3
#   [ 24:105] Face      27 joints × 3
#   [105:168] Left Hand 21 joints × 3
#   [168:231] Right Hand21 joints × 3
IDX_LH_START = 105   # flat index in the 231-dim vector
IDX_RH_START = 168
HAND_DIMS    = 63    # 21 joints × 3


def _read_split_ids(split_path):
    split_file = Path(split_path)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _extract_text_label(text_payload):
    # Files saved as: text#text#0.0#0.0
    return normalize_vietnamese_text(text_payload.split("#", 1)[0])


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
        return _extract_text_label(f.read().strip())


def build_segment_text_for_ctc(dataset_root, train_split, dev_split):
    dataset_root = Path(dataset_root)
    text_dir = dataset_root / "texts"

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

    all_data = pd.concat([train_data, dev_data], ignore_index=True)
    all_data = all_data[all_data["id"].notna() & all_data["gloss"].notna()]

    all_glosses = set()
    for ann in all_data["gloss"]:
        all_glosses.update(normalize_vietnamese_text(ann).split())

    vocab_list    = ["_"] + sorted(all_glosses)
    vocab_map     = {g: i for i, g in enumerate(vocab_list)}
    inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}
    print(f"Vocab size: {len(vocab_map)}")

    def encode_annotations(df):
        df = df.copy()
        df["gloss"] = df["gloss"].apply(normalize_vietnamese_text)
        df["enc"]   = df["gloss"].apply(lambda x: [vocab_map[g] for g in x.split()])
        return df[["id", "enc"]]

    return (encode_annotations(train_data), encode_annotations(dev_data),
            vocab_map, inv_vocab_map, vocab_list)


# ── Missing-hand interpolation ─────────────────────────────────────────────────

def _interpolate_missing_hand(seq, start, length):
    """
    Forward-fill then backward-fill a hand segment in a (T, D) sequence.
    A hand is considered 'missing' when all its dims are zero.
    Args:
        seq   : (T, D) float32 array
        start : start index of hand in the last dim
        length: number of dims for this hand (HAND_DIMS = 63)
    Returns:
        seq with missing hand frames filled in-place.
    """
    T = seq.shape[0]
    hand = seq[:, start:start + length]   # view

    missing = (np.abs(hand).sum(axis=1) == 0)  # (T,) bool

    if missing.all():
        return seq   # entire segment missing — leave zeros

    # Forward fill
    last_valid = None
    for t in range(T):
        if not missing[t]:
            last_valid = hand[t].copy()
        elif last_valid is not None:
            hand[t] = last_valid

    # Backward fill (for leading missing frames)
    first_valid = None
    for t in range(T - 1, -1, -1):
        if not missing[t]:
            first_valid = hand[t].copy()
        elif first_valid is not None:
            hand[t] = first_valid

    seq[:, start:start + length] = hand
    return seq


def fix_missing_hands(sequence):
    """
    Apply forward+backward interpolation to both hands.
    sequence: (T, D) float32  — D >= 231
    """
    sequence = _interpolate_missing_hand(sequence, IDX_LH_START, HAND_DIMS)
    sequence = _interpolate_missing_hand(sequence, IDX_RH_START, HAND_DIMS)
    return sequence


# ── Dataset ───────────────────────────────────────────────────────────────────

class SegmentNPYDataset(Dataset):
    def __init__(
        self,
        dataset_root,
        split_file,
        target_enc_df,
        transform=None,
        min_len=32,
        max_len=1000,
    ):
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
            if (self.motion_dir / f"{sid}.npy").exists() \
               and (self.text_dir / f"{sid}.txt").exists() \
               and sid in enc_map:
                self.files.append(sid)
                self.labels.append(enc_map[sid])

        print(f"Loaded {len(self.files)} segment samples from {split_file}")

    def __len__(self):
        return len(self.files)

    def _pad_or_crop(self, seq):
        """seq: (T, D)"""
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

        # ── Fix missing hands via interpolation ───────────────────────────
        pose = fix_missing_hands(pose)

        pose = self._pad_or_crop(pose)
        pose = torch.from_numpy(pose).float()

        if self.transform:
            pose = self.transform(pose)

        return sid, pose, torch.as_tensor(self.labels[idx])