import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_split_ids(split_path):
    split_file = Path(split_path)
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f.readlines() if line.strip()]

    return sample_ids


def _extract_text_label(text_payload):
    # Files are saved as: text#text#0.0#0.0
    first_part = text_payload.split("#", 1)[0]
    return normalize_vietnamese_text(first_part)


def normalize_vietnamese_text(text):
    if not text:
        return ""

    confusion_map = {
        ":": ".",
        ";": ".",
        "!": ".",
        "\u00a0": " ",
        "\t": " ",
    }

    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()

    for wrong, right in confusion_map.items():
        text = text.replace(wrong, right)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[.,:;!?]+$", "", text).strip()
    return text


def _read_text_label(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
    return _extract_text_label(content)


def build_segment_text_for_ctc(dataset_root, train_split, dev_split):
    dataset_root = Path(dataset_root)
    text_dir = dataset_root / "texts"

    train_ids = _read_split_ids(train_split)
    dev_ids = _read_split_ids(dev_split)

    def build_df(sample_ids):
        rows = []
        for sample_id in sample_ids:
            text_path = text_dir / f"{sample_id}.txt"
            if not text_path.exists():
                continue
            gloss = _read_text_label(text_path)
            if not gloss:
                continue
            rows.append({"id": sample_id, "gloss": gloss})
        return pd.DataFrame(rows)

    train_data = build_df(train_ids)
    dev_data = build_df(dev_ids)

    all_data = pd.concat([train_data, dev_data], ignore_index=True)
    all_data = all_data[all_data["id"].notna()]
    all_data = all_data[all_data["gloss"].notna()]

    all_glosses = set()
    for annotation in all_data["gloss"]:
        annotation = normalize_vietnamese_text(annotation)
        glosses = annotation.split()
        all_glosses.update(glosses)

    vocab_list = ["_"] + sorted(all_glosses)
    vocab_map = {g: i for i, g in enumerate(vocab_list)}
    inv_vocab_map = {i: g for i, g in enumerate(vocab_list)}

    def encode_annotations(df):
        df = df.copy()
        df["gloss"] = df["gloss"].apply(normalize_vietnamese_text)
        df["enc"] = df["gloss"].apply(lambda x: [vocab_map[g] for g in x.split()])
        return df[["id", "enc"]]

    train_processed = encode_annotations(train_data)
    dev_processed = encode_annotations(dev_data)

    return train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list


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
        self.motion_dir = self.dataset_root / "new_joints"
        self.text_dir = self.dataset_root / "texts"
        self.transform = transform
        self.min_len = min_len
        self.max_len = max_len

        if not self.motion_dir.exists():
            raise FileNotFoundError(f"Motion directory not found: {self.motion_dir}")
        if not self.text_dir.exists():
            raise FileNotFoundError(f"Text directory not found: {self.text_dir}")

        self.ids = _read_split_ids(split_file)
        enc_map = {
            str(row["id"]): row["enc"]
            for _, row in target_enc_df.iterrows()
        }

        self.files = []
        self.labels = []

        for sample_id in self.ids:
            npy_path = self.motion_dir / f"{sample_id}.npy"
            text_path = self.text_dir / f"{sample_id}.txt"
            if not npy_path.exists() or not text_path.exists():
                continue
            if sample_id not in enc_map:
                continue
            self.files.append(sample_id)
            self.labels.append(enc_map[sample_id])

        print(f"Loaded {len(self.files)} segment samples from {split_file}")

    def __len__(self):
        return len(self.files)

    def _pad_or_crop_sequence(self, sequence):
        t, f = sequence.shape

        if t < self.min_len:
            pad_len = self.min_len - t
            pad = np.zeros((pad_len, f), dtype=sequence.dtype)
            sequence = np.concatenate((sequence, pad), axis=0)

        if sequence.shape[0] > self.max_len:
            sequence = sequence[: self.max_len]

        return sequence

    def _read_motion(self, sample_id):
        npy_path = self.motion_dir / f"{sample_id}.npy"
        pose = np.load(npy_path, allow_pickle=False)

        if pose.ndim == 1:
            pose = pose[None, :]
        elif pose.ndim == 3:
            pose = pose.reshape(pose.shape[0], -1)

        if pose.ndim != 2:
            raise ValueError(f"Unexpected pose shape for {sample_id}: {pose.shape}")

        return pose.astype(np.float32)

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        pose = self._read_motion(sample_id)
        pose = self._pad_or_crop_sequence(pose)
        pose = torch.from_numpy(pose).float()

        if self.transform:
            pose = self.transform(pose)

        label = torch.as_tensor(self.labels[idx])
        return sample_id, pose, label