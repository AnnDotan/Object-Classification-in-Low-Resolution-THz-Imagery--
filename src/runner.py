# src/runner.py
from __future__ import annotations

import os
import warnings
import csv
import argparse
import time
from pathlib import Path

# Disable HuggingFace symlink warning on Windows (must be set early)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Suppress the specific CIFAR/NumPy warning noise
warnings.filterwarnings("ignore", message=r"dtype\(\): align")

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import timm

from src.data.datasets import DataConfig, THzLikeCIFAR10
from src.models.transnext_wrapper import create_transnext_model

def train_one_epoch(model, loader, optimizer, criterion, device: str):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device: str):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def format_lr_for_name(lr: float) -> str:
    if lr < 1e-2:
        return f"{lr:.0e}"
    return str(lr).replace(".", "p")


def make_unique_run_dir(base_dir: Path, base_name: str) -> Path:
    candidate = base_dir / base_name
    if not candidate.exists():
        return candidate

    version = 2
    while True:
        candidate = base_dir / f"{base_name}__v{version}"
        if not candidate.exists():
            return candidate
        version += 1


def run_experiment(
    model_name: str,
    pretrained: bool,
    out_size: int,
    low_res: int,
    epochs: int,
    batch_size: int,
    train_subset: int,
    val_subset: int,
    lr: float = 1e-3,
    tag: str = "",
    group: str = "pilot",
    freeze_backbone: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # run folder
    tag_prefix = f"{tag}_" if tag else ""
    weights_tag = "pt" if pretrained else "scratch"
    lr_str = format_lr_for_name(lr)

    run_name = (
        f"{tag_prefix}{model_name}_{weights_tag}_out{out_size}_"
        f"lowres{low_res}_lr{lr_str}"
    )

    base_group_dir = Path("runs") / group
    base_group_dir.mkdir(parents=True, exist_ok=True)

    run_dir = make_unique_run_dir(base_group_dir, run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    # חשוב: אם נוצר __v2 / __v3 נשמור את השם האמיתי
    run_name = run_dir.name

    log_path = run_dir / "log.txt"

    def log(msg: str):
        print(msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # save a small run config snapshot
    run_config_path = run_dir / "run_config.txt"
    with open(run_config_path, "w", encoding="utf-8") as f:
        f.write(f"device={device}\n")
        f.write(f"model_name={model_name}\n")
        f.write(f"pretrained={pretrained}\n")
        f.write(f"out_size={out_size}\n")
        f.write(f"low_res={low_res}\n")
        f.write(f"epochs={epochs}\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"train_subset={train_subset}\n")
        f.write(f"val_subset={val_subset}\n")
        f.write(f"lr={lr}\n")
        f.write(f"tag={tag}\n")
        f.write(f"group={group}\n")
        f.write(f"run_name={run_name}\n")
        f.write(f"freeze_backbone={freeze_backbone}\n")

    log(f"Device: {device}")
    log(f"Model: {model_name}, pretrained={pretrained}")
    log(f"Data: CIFAR10 degraded | out_size={out_size}, low_res={low_res}")
    log(f"Train subset={train_subset}, Val subset={val_subset}")
    log(f"Epochs={epochs}, batch_size={batch_size}, lr={lr}")
    log(f"Group: {group}")
    log(f"Freeze backbone: {freeze_backbone}")
    log(f"Saved run config: {run_config_path}")

    if model_name.startswith("transnext_") and out_size != 224:
        print(f"[INFO] Overriding out_size from {out_size} to 224 for TransNeXt")
        out_size = 224

    # data
    cfg_train = DataConfig(train=True, out_size=out_size, low_res=low_res, root="./data")
    cfg_val = DataConfig(train=False, out_size=out_size, low_res=low_res, root="./data")

    train_ds = THzLikeCIFAR10(cfg_train)
    val_ds = THzLikeCIFAR10(cfg_val)

    # subsets for CPU speed
    if train_subset > 0:
        train_ds = Subset(train_ds, range(min(train_subset, len(train_ds))))
    if val_subset > 0:
        val_ds = Subset(val_ds, range(min(val_subset, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=max(batch_size, 64), shuffle=False, num_workers=0)

    # model
    # model
    TRANSNEXT_NAMES = {
        "transnext_micro",
        "transnext_tiny",
        "transnext_small",
        "transnext_base",
    }

    if model_name in TRANSNEXT_NAMES:
        transnext_ckpt = None

        if pretrained and model_name == "transnext_micro":
            transnext_ckpt = r"C:\Users\ib94\Documents\NewFinalProject\artifacts\weights\transnext_micro_224_1k.pth"

        model = create_transnext_model(
            model_name=model_name,
            num_classes=10,
            pretrained=pretrained,
            checkpoint_path=transnext_ckpt,
        )
    else:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=10,
        )

    model.to(device)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

        if hasattr(model, "head"):
            for p in model.head.parameters():
                p.requires_grad = True
        else:
            raise RuntimeError("freeze_backbone=True but model has no attribute 'head'")

    criterion = nn.CrossEntropyLoss()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    # metrics + best tracking
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = -1.0
    best_path = run_dir / "best.pt"

    t0 = time.time()
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)

        log(
            f"Epoch {ep}/{epochs} | "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"Val: loss={va_loss:.4f}, acc={va_acc:.4f}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "epoch": ep,
                    "model_name": model_name,
                    "pretrained": pretrained,
                    "out_size": out_size,
                    "low_res": low_res,
                    "lr": lr,
                    "group": group,
                    "run_name": run_name,
                },
                best_path,
            )
            log(f"New best: val_acc={best_val_acc:.4f} (epoch {ep}) -> {best_path}")

    dt = time.time() - t0
    log(f"Total time: {dt:.1f}s")
    log(f"Saved log: {log_path}")
    log(f"Saved metrics: {metrics_path}")
    log(f"Saved best checkpoint: {best_path} (best_val_acc={best_val_acc:.4f})")

    # save last checkpoint
    ckpt_path = run_dir / "model_last.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": model_name,
            "pretrained": pretrained,
            "out_size": out_size,
            "low_res": low_res,
            "lr": lr,
            "group": group,
            "run_name": run_name,
        },
        ckpt_path,
    )
    log(f"Saved checkpoint: {ckpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--out_size", type=int, default=32)
    p.add_argument("--low_res", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--train_subset", type=int, default=2000)
    p.add_argument("--val_subset", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--group", type=str, default="pilot", choices=["pilot", "official"])
    p.add_argument("--freeze_backbone", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    run_experiment(
        model_name=args.model,
        pretrained=args.pretrained,
        out_size=args.out_size,
        low_res=args.low_res,
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        val_subset=args.val_subset,
        lr=args.lr,
        tag=args.tag,
        group=args.group,
        freeze_backbone=args.freeze_backbone,
    )