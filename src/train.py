# src/train.py
import os
import math
import time
import argparse
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm
from timm.data import Mixup
from tqdm import tqdm


# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def soft_target_cross_entropy(logits, targets):
    """targets: (B, C) soft labels"""
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_acc, classes, args):
    ckpt = {
        "epoch": epoch,
        "best_acc": best_acc,
        "classes": classes,
        "model_name": args.model,
        "img_size": args.img_size,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    if optimizer is not None and "optimizer" in ckpt and ckpt["optimizer"] is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            # scheduler 설정이 바뀌면 로드가 실패할 수 있음
            pass

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_acc = float(ckpt.get("best_acc", 0.0))
    classes = ckpt.get("classes", None)
    return start_epoch, best_acc, classes


# ---------------------------
# Data / Model
# ---------------------------
def build_loaders(data_dir, img_size, batch, workers):
    # Strong aug for cropped faces
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise FileNotFoundError(
            f"Expected folders:\n- {train_path}\n- {val_path}\n"
            f"(ImageFolder 구조: data/train/class_x/*.jpg, data/val/class_x/*.jpg)"
        )

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    classes = train_ds.classes
    n_train, n_val = len(train_ds), len(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_loader, val_loader, classes, n_train, n_val


def build_model(model_name: str, num_classes: int):
    # ✅ pretrained=True is critical for 7k-scale cropped dataset
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes
    )
    return model


# ---------------------------
# Main
# ---------------------------
@dataclass
class WarmupConfig:
    warmup_epochs: int = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--out", type=str, default="weights")
    parser.add_argument("--img_size", type=int, default=224)

    # tuning options
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--smoothing", type=float, default=0.1)     # label smoothing
    parser.add_argument("--mixup", type=float, default=0.2)         # 0 disables
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=8)          # early stopping
    parser.add_argument("--resume", type=str, default="")           # path to .pth checkpoint
    parser.add_argument("--save_last", action="store_true")         # also save last checkpoint
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: NO GPU")

    # Data
    train_loader, val_loader, classes, n_train, n_val = build_loaders(
        args.data, args.img_size, args.batch, args.workers
    )
    num_classes = len(classes)
    print(f"Classes: {classes}")
    print(f"Train images: {n_train}")
    print(f"Val images: {n_val}")

    # Model
    model = build_model(args.model, num_classes=num_classes).to(device)

    # Loss
    criterion = nn.CrossEntropyLoss(label_smoothing=args.smoothing)

    # Mixup/Cutmix
    mixup_fn = None
    if args.mixup and args.mixup > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.mixup,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.smoothing,
            num_classes=num_classes,
        )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler: Warmup + Cosine (LambdaLR)
    def lr_lambda(epoch_idx: int):
        # epoch_idx: 0-based
        if epoch_idx < args.warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, args.warmup_epochs))
        progress = (epoch_idx - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # Output paths
    ensure_dir(args.out)
    best_path = os.path.join(args.out, "best.pth")
    last_path = os.path.join(args.out, "last.pth")

    start_epoch = 1
    best_acc = 0.0
    no_improve = 0

    # Resume (if provided)
    if args.resume and os.path.isfile(args.resume):
        print(f"🔁 Resuming from: {args.resume}")
        start_epoch, best_acc, ckpt_classes = load_checkpoint(
            args.resume, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        if ckpt_classes is not None and ckpt_classes != classes:
            print("⚠️ Warning: classes in checkpoint differ from current dataset classes.")
            print(f"  ckpt: {ckpt_classes}")
            print(f"  now : {classes}")
        print(f"✅ Resume OK. start_epoch={start_epoch}, best_acc={best_acc*100:.2f}%")

    # Training
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, total=len(train_loader), ncols=110)
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                outputs = model(images)
                if labels.ndim == 2:  # soft labels from mixup/cutmix
                    loss = soft_target_cross_entropy(outputs, labels)
                else:
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = images.size(0)
            epoch_loss += loss.item() * bsz
            seen += bsz
            pbar.set_postfix(train_loss=f"{(epoch_loss / max(1, seen)):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # step scheduler per-epoch (LambdaLR designed for per-epoch stepping)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        count = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                bsz = images.size(0)
                val_loss += loss.item() * bsz
                val_acc += (outputs.argmax(1) == labels).float().sum().item()
                count += bsz

        val_loss /= max(1, count)
        val_acc = val_acc / max(1, count)
        print(f"[VAL] loss={val_loss:.4f} acc={val_acc*100:.2f}%")

        # Save last (optional)
        if args.save_last:
            save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_acc, classes, args)

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            no_improve = 0
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_acc, classes, args)
            print(f"✅ BEST saved (acc={best_acc*100:.2f}%)")
        else:
            no_improve += 1

        # Early stopping
        if args.patience > 0 and no_improve >= args.patience:
            print(f"⏹ Early stopping: no improvement for {args.patience} epochs.")
            break

    print(f"Done. Best acc: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"Best saved to: {best_path}")
    if args.save_last:
        print(f"Last saved to: {last_path}")


if __name__ == "__main__":
    main()
