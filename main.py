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
        dev   = os.path.join(ann_dir, f"dev.{ext}")
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


def debug_first_batch(model, dataloader, device, inv_vocab_map):
    """Sanity-check the first batch before training starts."""
    model.eval()
    with torch.no_grad():
        for sid, poses, labels in dataloader:
            poses = poses.to(device)
            logits = model(poses)
            log_probs = F.log_softmax(logits, dim=-1)
            T_out    = logits.shape[1]
            tgt_len  = (labels != 0).sum().item()
            gt       = " ".join(invert_to_chars(labels, inv_vocab_map))
            blank_p  = log_probs[0, :, 0].exp().mean().item()
            status   = "OK" if T_out >= tgt_len else "BAD: T_out < target_len!"
            print(f"  sample      : {sid[0]}")
            print(f"  pose shape  : {poses.shape}")
            print(f"  logits shape: {logits.shape}  (B, T_out, C)")
            print(f"  T_out={T_out}  target_len={tgt_len}  [{status}]")
            print(f"  ground truth: {gt}")
            print(f"  mean blank prob (random init ~1/C): {blank_p:.4f}")
            break
    model.train()


def train_epoch(model, dataloader, optimizer, ctc_loss, device, grad_clip=5.0):
    model.train()
    total_loss, valid_steps, skipped = 0.0, 0, 0
    current_lr = optimizer.param_groups[0]["lr"]

    for _, poses, labels in tqdm(dataloader, desc="train", ncols=100):
        optimizer.zero_grad()
        poses  = poses.to(device)
        labels = labels.to(device, dtype=torch.long)

        logits    = model(poses)
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # (T, B, C)

        T_out          = log_probs.size(0)
        B              = log_probs.size(1)
        input_lengths  = torch.full((B,), T_out, dtype=torch.long, device=device)
        target_lengths = get_target_lengths(labels, blank_id=0)

        valid_mask = target_lengths <= input_lengths
        if not valid_mask.all():
            skipped += int((~valid_mask).sum().item())
            if valid_mask.sum() == 0:
                continue
            labels         = labels[valid_mask]
            target_lengths = target_lengths[valid_mask]
            input_lengths  = input_lengths[valid_mask]
            log_probs      = log_probs[:, valid_mask, :]

        loss = ctc_loss(log_probs, labels, input_lengths, target_lengths).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss  += loss.item()
        valid_steps += 1

    return total_loss / max(valid_steps, 1), current_lr, skipped


def evaluate_model(model, dataloader, decoder, device, inv_vocab_map,
                   work_dir, epoch, unk_token="<unk>"):
    model.eval()
    preds, gt_labels = [], []
    empty_preds = 0
    pred_path = os.path.join(work_dir, "pred_outputs", f"predictions_epoch_{epoch+1}.txt")

    with open(pred_path, "w", encoding="utf-8") as f:
        f.write(f"Epoch {epoch+1} Predictions\n{'='*50}\n")
        with torch.no_grad():
            for _, poses, labels in tqdm(dataloader, desc="valid", ncols=100):
                poses = poses.to(device)
                logits = model(poses)
                vid_lgt = torch.full(
                    (logits.size(0),), logits.size(1),
                    dtype=torch.long, device=device,
                )
                decoded = decoder.decode(logits, vid_lgt=vid_lgt,
                                         batch_first=True, probs=False)

                pred_str = " ".join(g for pred in decoded for g, _ in pred)
                if not pred_str.strip():
                    empty_preds += 1

                gt_str = " ".join(invert_to_chars(labels, inv_vocab_map))
                preds.append(pred_str)
                gt_labels.append(gt_str)
                f.write(f"GT:   {gt_str}\nPred: {pred_str}\n\n")

    results = wer_list(gt_labels, preds)
    results["empty_preds"]   = empty_preds
    results["total_samples"] = len(preds)

    # OOV rate in predictions (how often model outputs <unk>)
    all_pred_tokens = " ".join(preds).split()
    unk_pred_cnt    = all_pred_tokens.count(unk_token)
    results["unk_pred_rate"] = (
        unk_pred_cnt / max(len(all_pred_tokens), 1) * 100
    )
    return results


def build_datasets(args):
    if args.data_format == "legacy":
        train_csv, dev_csv = resolve_legacy_annotation_files(args.mode)
        train_proc, dev_proc, vocab_map, inv_vocab_map, vocab_list = (
            convert_text_for_ctc("isharah", train_csv, dev_csv)
        )
        ds_train = PoseDatasetV2(
            "isharah", train_csv, "train", train_proc,
            augmentations=True,
            transform=transforms.Compose([GaussianNoise(std=0.05)]),
            mode=args.mode,
        )
        ds_dev = PoseDatasetV2(
            "isharah", dev_csv, "dev", dev_proc,
            augmentations=False, mode=args.mode,
        )
    else:
        train_split = os.path.join(args.segments_root, args.train_split)
        dev_split   = os.path.join(args.segments_root, args.dev_split)
        train_proc, dev_proc, vocab_map, inv_vocab_map, vocab_list = (
            build_segment_text_for_ctc(
                args.segments_root, train_split, dev_split,
                min_freq=args.min_freq,
            )
        )
        ds_train = SegmentNPYDataset(
            args.segments_root, train_split, train_proc,
            transform=transforms.Compose([GaussianNoise(std=0.05)]),
            min_len=args.segment_min_len,
            max_len=args.segment_max_len,
        )
        ds_dev = SegmentNPYDataset(
            args.segments_root, dev_split, dev_proc,
            min_len=args.segment_min_len,
            max_len=args.segment_max_len,
        )
    return ds_train, ds_dev, vocab_map, inv_vocab_map, vocab_list


def main(args):
    set_rng_state(42)
    make_workdir(args.work_dir)

    cuda_ok = torch.cuda.is_available()
    device  = torch.device(f"cuda:{args.device}" if cuda_ok else "cpu")
    print(f"PyTorch {torch.__version__} | CUDA {torch.version.cuda} | Device: {device}")

    ds_train, ds_dev, vocab_map, inv_vocab_map, vocab_list = build_datasets(args)

    if len(ds_train) == 0 or len(ds_dev) == 0:
        raise ValueError("Empty dataset split detected.")

    train_label_lens = [len(lbl) for lbl in ds_train.labels]
    print(f"Train: {len(ds_train)} samples | "
          f"label len mean={np.mean(train_label_lens):.1f} "
          f"max={np.max(train_label_lens)} min={np.min(train_label_lens)}")
    print(f"Dev  : {len(ds_dev)} samples")
    print(f"Vocab: {len(vocab_map)} tokens (blank + <unk> + words)")

    pin          = torch.cuda.is_available()
    train_loader = DataLoader(ds_train, batch_size=1, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    dev_loader   = DataLoader(ds_dev, batch_size=1, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    input_dim = infer_input_dim(ds_train)
    print(f"input_dim={input_dim}")

    model = CSLRTransformer(
        input_dim=input_dim,
        num_classes=len(vocab_map),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    decoder   = Decode(vocab_map, len(vocab_list), "beam")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    ctc_loss  = nn.CTCLoss(blank=0, zero_infinity=True, reduction="none")

    print("\n--- Sanity check: first batch ---")
    debug_first_batch(model, train_loader, device, inv_vocab_map)
    print("--- End sanity check ---\n")

    log_file = os.path.join(args.work_dir, "training_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    best_wer, best_epoch, patience_counter = float("inf"), 0, 0

    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")

        train_loss, lr, skipped = train_epoch(
            model, train_loader, optimizer, ctc_loss, device,
            grad_clip=args.grad_clip,
        )
        wer_results = evaluate_model(
            model, dev_loader, decoder, device, inv_vocab_map,
            args.work_dir, epoch,
        )
        scheduler.step(wer_results["wer"])

        if wer_results["wer"] < best_wer:
            best_wer, best_epoch, patience_counter = wer_results["wer"], epoch, 0
            torch.save(model.state_dict(),
                       os.path.join(args.work_dir, "best_model.pt"))
        else:
            patience_counter += 1

        msg = (
            f"Loss={train_loss:.4f} | "
            f"WER={wer_results['wer']:.2f} "
            f"DEL={wer_results['del']:.2f} "
            f"INS={wer_results['ins']:.2f} "
            f"SUB={wer_results['sub']:.2f} | "
            f"BestWER={best_wer:.2f}(ep{best_epoch+1}) | "
            f"Empty={wer_results['empty_preds']}/{wer_results['total_samples']} | "
            f"UNK_pred={wer_results['unk_pred_rate']:.1f}% | "
            f"Skipped={skipped} | LR={lr:.2e}"
        )
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

        if patience_counter >= args.patience:
            msg = (f"Early stopping at epoch {epoch+1}. "
                   f"Best WER={best_wer:.2f} at epoch {best_epoch+1}.")
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument("--work_dir",        default="./work_dir/test")
    parser.add_argument("--mode",            default="SI")
    parser.add_argument("--device",          default="0")
    parser.add_argument("--data_format",     default="legacy",
                        choices=["legacy", "segments"])
    parser.add_argument("--segments_root",   default="./data/YOUTUBE_SIGN")
    parser.add_argument("--train_split",     default="train.txt")
    parser.add_argument("--dev_split",       default="val.txt")
    parser.add_argument("--segment_min_len", type=int,   default=96)
    parser.add_argument("--segment_max_len", type=int,   default=1000)
    # Vocab
    parser.add_argument("--min_freq",        type=int,   default=1,
                        help="Min word freq in train to include in vocab (OOV control)")
    # Training
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--num_epochs",      type=int,   default=300)
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--patience",        type=int,   default=15)
    parser.add_argument("--grad_clip",       type=float, default=5.0)
    # Model
    parser.add_argument("--d_model",         type=int,   default=256)
    parser.add_argument("--nhead",           type=int,   default=4)
    parser.add_argument("--num_layers",      type=int,   default=2)
    parser.add_argument("--dropout",         type=float, default=0.1)

    args = parser.parse_args()
    main(args)