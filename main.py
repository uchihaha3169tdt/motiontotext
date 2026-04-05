import os
import random
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.text_ctc_utils import GaussianNoise, invert_to_chars, convert_text_for_ctc
from utils.decode import Decode
from utils.metrics import wer_list
from utils.datasetv2 import PoseDatasetV2
from utils.dataset_segments import SegmentNPYDataset, build_segment_text_for_ctc
from models.transformer import CSLRTransformer


def set_rng_state(seed=42):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_workdir(work_dir):
    if os.path.exists(work_dir):
        answer = input(f"'{work_dir}' exists. Remove and refresh? [y/n]: ")
        if answer.strip().lower() in ("y", "yes", "ok", "1"):
            shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(os.path.join(work_dir, "pred_outputs"), exist_ok=True)


def resolve_legacy_annotation_files(mode):
    ann_dir = os.path.join("./annotations_v2/isharah2000", mode)
    for ext in ("csv", "txt"):
        train = os.path.join(ann_dir, f"train.{ext}")
        dev = os.path.join(ann_dir, f"dev.{ext}")
        if os.path.exists(train) and os.path.exists(dev):
            return train, dev
    raise FileNotFoundError(f"Annotation files not found for mode={mode} in {ann_dir}")


def infer_input_dim(dataset):
    _, pose, _ = dataset[0]
    if pose.ndim == 3:
        return int(pose.shape[-2] * pose.shape[-1])
    return int(pose.shape[-1]) if pose.ndim == 2 else int(np.prod(pose.shape[1:]))


def get_target_lengths(labels, blank_id=0):
    return (labels != blank_id).sum(dim=1).to(dtype=torch.long)


def train_epoch(model, dataloader, optimizer, ctc_loss, device):
    model.train()
    total_loss, valid_steps, skipped = 0.0, 0, 0
    current_lr = optimizer.param_groups[0]["lr"]

    for _, poses, labels in tqdm(dataloader, desc="train", ncols=100):
        optimizer.zero_grad()
        poses = poses.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits = model(poses)
        # (B, T, C) -> (T, B, C) for CTC
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)

        T = log_probs.size(0)
        B = log_probs.size(1)
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        target_lengths = get_target_lengths(labels, blank_id=0)

        valid_mask = target_lengths <= input_lengths
        if not valid_mask.all():
            skipped += int((~valid_mask).sum().item())
            if valid_mask.sum() == 0:
                continue
            labels = labels[valid_mask]
            target_lengths = target_lengths[valid_mask]
            input_lengths = input_lengths[valid_mask]
            log_probs = log_probs[:, valid_mask, :]

        loss = ctc_loss(log_probs, labels, input_lengths, target_lengths).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_steps += 1

    return total_loss / max(valid_steps, 1), current_lr, skipped


def evaluate_model(model, dataloader, decoder, device, inv_vocab_map, work_dir, epoch):
    model.eval()
    preds, gt_labels = [], []
    empty_preds = 0
    pred_file_path = os.path.join(work_dir, "pred_outputs", f"predictions_epoch_{epoch+1}.txt")

    with open(pred_file_path, "w") as f:
        f.write(f"Epoch {epoch+1} Predictions\n{'='*50}\n")
        with torch.no_grad():
            for _, poses, labels in tqdm(dataloader, desc="valid", ncols=100):
                poses = poses.to(device)
                logits = model(poses)
                vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
                decoded = decoder.decode(logits, vid_lgt=vid_lgt, batch_first=True, probs=False)

                pred_str = " ".join(gloss for pred in decoded for gloss, _ in pred)
                if not pred_str.strip():
                    empty_preds += 1

                gt_str = " ".join(invert_to_chars(labels, inv_vocab_map))
                preds.append(pred_str)
                gt_labels.append(gt_str)
                f.write(f"GT: {gt_str}\nPred: {pred_str}\n\n")

    results = wer_list(gt_labels, preds)
    results["empty_preds"] = empty_preds
    results["total_samples"] = len(preds)
    return results


def build_datasets(args):
    if args.data_format == "legacy":
        train_csv, dev_csv = resolve_legacy_annotation_files(args.mode)
        train_proc, dev_proc, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc(
            "isharah", train_csv, dev_csv
        )
        ds_train = PoseDatasetV2(
            "isharah", train_csv, "train", train_proc,
            augmentations=True, transform=transforms.Compose([GaussianNoise()]), mode=args.mode,
        )
        ds_dev = PoseDatasetV2(
            "isharah", dev_csv, "dev", dev_proc, augmentations=False, mode=args.mode,
        )
    else:
        train_split = os.path.join(args.segments_root, args.train_split)
        dev_split = os.path.join(args.segments_root, args.dev_split)
        train_proc, dev_proc, vocab_map, inv_vocab_map, vocab_list = build_segment_text_for_ctc(
            args.segments_root, train_split, dev_split
        )
        ds_train = SegmentNPYDataset(
            args.segments_root, train_split, train_proc,
            transform=transforms.Compose([GaussianNoise()]),
            min_len=args.segment_min_len, max_len=args.segment_max_len,
        )
        ds_dev = SegmentNPYDataset(
            args.segments_root, dev_split, dev_proc,
            min_len=args.segment_min_len, max_len=args.segment_max_len,
        )
    return ds_train, ds_dev, vocab_map, inv_vocab_map, vocab_list


def main(args):
    set_rng_state(42)
    make_workdir(args.work_dir)

    cuda_ok = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if cuda_ok else "cpu")
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Device: {device}")

    ds_train, ds_dev, vocab_map, inv_vocab_map, vocab_list = build_datasets(args)

    if len(ds_train) == 0 or len(ds_dev) == 0:
        raise ValueError("Empty dataset split detected.")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    dev_loader = DataLoader(ds_dev, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    input_dim = infer_input_dim(ds_train)
    print(f"input_dim={input_dim}, vocab_size={len(vocab_map)}")

    model = CSLRTransformer(input_dim=input_dim, num_classes=len(vocab_map)).to(device)
    decoder = Decode(vocab_map, len(vocab_list), "beam")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

    log_file = os.path.join(args.work_dir, "training_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    best_wer, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        train_loss, lr, skipped = train_epoch(model, train_loader, optimizer, ctc_loss, device)
        wer_results = evaluate_model(model, dev_loader, decoder, device, inv_vocab_map, args.work_dir, epoch)
        scheduler.step(wer_results["wer"])

        if wer_results["wer"] < best_wer:
            best_wer = wer_results["wer"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.work_dir, "best_model.pt"))
        else:
            patience_counter += 1

        msg = (
            f"Loss={train_loss:.4f} | WER={wer_results['wer']:.2f} | BestWER={best_wer:.2f} "
            f"(ep{best_epoch+1}) | Empty={wer_results['empty_preds']}/{wer_results['total_samples']} "
            f"| Skipped={skipped} | LR={lr:.2e}"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if patience_counter >= args.patience:
            msg = f"Early stopping at epoch {epoch+1}. Best WER={best_wer:.2f} at epoch {best_epoch+1}."
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", default="./work_dir/test")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--mode", default="SI")
    parser.add_argument("--device", default="0")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--data_format", default="legacy", choices=["legacy", "segments"])
    parser.add_argument("--segments_root", default="./data/YOUTUBE_SIGN")
    parser.add_argument("--train_split", default="train.txt")
    parser.add_argument("--dev_split", default="val.txt")
    parser.add_argument("--segment_min_len", type=int, default=96)
    parser.add_argument("--segment_max_len", type=int, default=1000)

    args = parser.parse_args()
    main(args)