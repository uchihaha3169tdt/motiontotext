import os
import random
from tqdm import tqdm
import numpy as np
import argparse
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.text_ctc_utils import * 
from utils.decode import Decode
from utils.metrics import wer_list
from torchvision import transforms
from utils.datasetv2 import PoseDatasetV2
from utils.dataset_segments import SegmentNPYDataset, build_segment_text_for_ctc

from models.transformer import CSLRTransformer


MODELS = {
    "base": CSLRTransformer,
}

def set_rng_state(seed):
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def make_workdir(work_dir):
    if os.path.exists(work_dir):
        answer = input('Current dir exists, do you want to remove and refresh it?\n')
        if answer in ['yes', 'y', 'ok', '1']:
            shutil.rmtree(work_dir)
            os.makedirs(work_dir)
    else:
        os.makedirs(work_dir)

    if not os.path.exists(os.path.join(work_dir, "pred_outputs")):
        os.mkdir(os.path.join(work_dir, "pred_outputs"))


def resolve_legacy_annotation_files(mode):
    ann_dir = os.path.join("./annotations_v2/isharah2000", mode)

    train_csv = os.path.join(ann_dir, "train.csv")
    dev_csv = os.path.join(ann_dir, "dev.csv")

    if not os.path.exists(train_csv):
        train_csv = os.path.join(ann_dir, "train.txt")
    if not os.path.exists(dev_csv):
        dev_csv = os.path.join(ann_dir, "dev.txt")

    if not os.path.exists(train_csv) or not os.path.exists(dev_csv):
        raise FileNotFoundError(
            f"Could not find annotation files for mode={mode} in {ann_dir}."
        )

    return train_csv, dev_csv


def infer_input_dim(dataset):
    _, pose, _ = dataset[0]
    if pose.ndim == 3:
        return int(pose.shape[-2] * pose.shape[-1])
    if pose.ndim == 2:
        return int(pose.shape[-1])
    return int(np.prod(pose.shape[1:]))


def get_target_lengths(labels, blank_id=0):
    # CTC target length = number of non-blank tokens in each sample.
    return (labels != blank_id).sum(dim=1).to(dtype=torch.long)

def train_epoch(model, dataloader, optimizer, loss_encoder, device):
    total_loss = 0
    current_lr = optimizer.param_groups[0]['lr']
    skipped_invalid_ctc = 0
    valid_steps = 0

    model.train()
    for i, (_, poses, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="train", ncols=100):
        optimizer.zero_grad()

        poses = poses.to(device)
        labels = labels.to(device, dtype=torch.long)
        logits = model(poses)
        log_probs_enc = F.log_softmax(logits, dim=-1).permute(1, 0, 2)  # Required for CTC Loss
        log_probs_enc = log_probs_enc - (torch.tensor([1.0], device=log_probs_enc.device) * 
                                        (torch.arange(log_probs_enc.shape[-1], device=log_probs_enc.device) == 0).float())

        input_lengths = torch.full(
            (log_probs_enc.size(1),),
            log_probs_enc.size(0),
            dtype=torch.long,
            device=labels.device,
        )
        target_lengths = get_target_lengths(labels, blank_id=0)

        valid_mask = target_lengths <= input_lengths
        if not torch.all(valid_mask):
            skipped_invalid_ctc += int((~valid_mask).sum().item())
            if valid_mask.sum() == 0:
                continue

            # Keep valid samples only (supports future batch_size > 1).
            labels = labels[valid_mask]
            target_lengths = target_lengths[valid_mask]
            input_lengths = input_lengths[valid_mask]
            log_probs_enc = log_probs_enc[:, valid_mask, :]

        loss_enc = loss_encoder(log_probs_enc, labels, input_lengths=input_lengths, target_lengths=target_lengths)
        loss = loss_enc.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        valid_steps += 1

    avg_loss = total_loss / max(valid_steps, 1)
    return avg_loss, current_lr, skipped_invalid_ctc


def evaluate_model(model, dataloader, decoder_dec, device, inv_vocab_map, work_dir, epoch):
    preds = []
    gt_labels = []
    empty_preds = 0

    model.eval()
    predictions_file = f"{work_dir}/pred_outputs/predictions_epoch_{epoch+1}.txt"
    with open(predictions_file, "w") as pred_file:
        pred_file.write(f"Epoch {epoch+1} Predictions\n")
        pred_file.write("=" * 50 + "\n")

        with torch.no_grad():
            for i, (file, poses, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="valid", ncols=100):
                poses = poses.to(device)

                logits = model(poses)

                vid_lgt = torch.full((logits.size(0),), logits.size(1), dtype=torch.long).to(device)
                decoded_list = decoder_dec.decode(logits, vid_lgt=vid_lgt, batch_first=True, probs=False)
                flat_preds = [gloss for pred in decoded_list for gloss, _ in pred]  # Flatten list
                current_preds = ' '.join(flat_preds)  # Convert list to string
                if not current_preds.strip():
                    empty_preds += 1

                preds.append(current_preds)
                ground_truth = ' '.join(invert_to_chars(labels, inv_vocab_map))
                gt_labels.append(ground_truth)

                pred_file.write(f"GT: {ground_truth}\nPred: {current_preds}\n\n")

    # wer_list expects (references, hypotheses) = (ground_truth, predictions)
    wer_results = wer_list(gt_labels, preds)
    wer_results["empty_preds"] = empty_preds
    wer_results["total_samples"] = len(preds)
    
    return wer_results

def main(args):
    set_rng_state(42)
    make_workdir(args.work_dir)
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.device}" if cuda_available else "cpu")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA build in torch: {torch.version.cuda}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        gpu_idx = int(args.device)
        if gpu_idx < torch.cuda.device_count():
            print(f"Using device: cuda:{gpu_idx} ({torch.cuda.get_device_name(gpu_idx)})")
        else:
            print(
                f"Requested cuda:{gpu_idx} but only {torch.cuda.device_count()} GPU(s) detected. "
                "Falling back to cuda:0."
            )
            device = torch.device("cuda:0")
            print(f"Using device: cuda:0 ({torch.cuda.get_device_name(0)})")
    else:
        print("Using device: cpu")
        print("Tip: On Kaggle, enable GPU Accelerator and avoid CPU-only torch wheels.")

    if args.data_format == "legacy":
        train_csv, dev_csv = resolve_legacy_annotation_files(args.mode)
        train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = convert_text_for_ctc(
            "isharah", train_csv, dev_csv
        )

        dataset_train = PoseDatasetV2(
            "isharah",
            train_csv,
            "train",
            train_processed,
            augmentations=True,
            transform=transforms.Compose([GaussianNoise()]),
            mode=args.mode,
        )
        dataset_dev = PoseDatasetV2(
            "isharah",
            dev_csv,
            "dev",
            dev_processed,
            augmentations=False,
            mode=args.mode,
        )
    else:
        train_split = os.path.join(args.segments_root, args.train_split)
        dev_split = os.path.join(args.segments_root, args.dev_split)

        train_processed, dev_processed, vocab_map, inv_vocab_map, vocab_list = build_segment_text_for_ctc(
            args.segments_root,
            train_split,
            dev_split,
        )

        dataset_train = SegmentNPYDataset(
            dataset_root=args.segments_root,
            split_file=train_split,
            target_enc_df=train_processed,
            transform=transforms.Compose([GaussianNoise()]),
            min_len=args.segment_min_len,
            max_len=args.segment_max_len,
        )
        dataset_dev = SegmentNPYDataset(
            dataset_root=args.segments_root,
            split_file=dev_split,
            target_enc_df=dev_processed,
            min_len=args.segment_min_len,
            max_len=args.segment_max_len,
        )

    if len(dataset_train) == 0 or len(dataset_dev) == 0:
        raise ValueError(
            "Empty dataset split detected. Please verify split files and input data paths."
        )

    traindataloader = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    devdataloader = DataLoader(
        dataset_dev,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_input_dim = infer_input_dim(dataset_train)
    print(f"Detected model input_dim={model_input_dim}")
    model = MODELS[args.model](input_dim=model_input_dim, num_classes=len(vocab_map)).to(device)

    if args.data_format == "segments":
        train_label_lens = [len(x) for x in dataset_train.labels]
        print(
            "Train target length (tokens) "
            f"mean={np.mean(train_label_lens):.2f}, max={np.max(train_label_lens)}"
        )

    decoder_dec = Decode(vocab_map, len(vocab_list), 'beam')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_encoder = nn.CTCLoss(blank=0, zero_infinity=True, reduction='none')

    log_file = f"{args.work_dir}/training_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    best_wer = float("inf") 
    best_epoch = 0
    patience = 10
    patience_counter = 0

    for epoch in range(args.num_epochs):
        print(f"\n\nEpoch [{epoch+1}/{args.num_epochs}]")
        train_loss, current_lr, skipped_invalid_ctc = train_epoch(
            model,
            traindataloader,
            optimizer,
            loss_encoder,
            device,
        )
        dev_wer_results = evaluate_model(model, devdataloader, decoder_dec, device, inv_vocab_map, args.work_dir, epoch)
        scheduler.step(dev_wer_results['wer'])

        if dev_wer_results['wer'] < best_wer:
            best_wer = dev_wer_results['wer']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), f"{args.work_dir}/best_model.pt")
        else:
            patience_counter += 1
        
         log_msg = (f"Train Loss: {train_loss:.4f} "
               f"- Dev WER: {dev_wer_results['wer']:.4f} - Best Dev WER: {best_wer:.4f} - Best epoch: {best_epoch+1} "
             f"- Empty Preds: {dev_wer_results['empty_preds']}/{dev_wer_results['total_samples']} "
             f"- Skipped Invalid CTC: {skipped_invalid_ctc} "
             f"- Learning Rate: {current_lr:.8f}")
        
        print(log_msg)
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

        if patience_counter >= patience:
            print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
            log_msg = (f"Early stopping triggered! No improvement for {patience} consecutive epochs. Best WER: {best_wer:.4f} - Best epoch: {best_epoch+1}" )
            
            with open(log_file, "a") as f:
                f.write(log_msg + "\n")

            break

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--work_dir', dest='work_dir', default="./work_dir/test")
    parser.add_argument('--data_dir', dest='data_dir', default="./data")
    parser.add_argument('--mode', dest='mode', default="SI")
    parser.add_argument('--model', dest='model', default="base")
    parser.add_argument('--device', dest='device', default="0")
    parser.add_argument('--lr', dest='lr', default="0.0001")
    parser.add_argument('--num_epochs', dest='num_epochs', default="300")
    parser.add_argument('--num_workers', dest='num_workers', default="0")
    parser.add_argument('--data_format', dest='data_format', default="legacy", choices=["legacy", "segments"])
    parser.add_argument('--segments_root', dest='segments_root', default="./data/YOUTUBE_SIGN")
    parser.add_argument('--train_split', dest='train_split', default="train.txt")
    parser.add_argument('--dev_split', dest='dev_split', default="val.txt")
    parser.add_argument('--segment_min_len', dest='segment_min_len', default="96")
    parser.add_argument('--segment_max_len', dest='segment_max_len', default="1000")

    args=parser.parse_args()
    args.lr = float(args.lr)
    args.num_epochs = int(args.num_epochs)
    args.num_workers = int(args.num_workers)
    args.segment_min_len = int(args.segment_min_len)
    args.segment_max_len = int(args.segment_max_len)
    
    main(args)