# src/runner.py
from __future__ import annotations

import os
import sys
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
from timm.data import Mixup
from timm.scheduler import CosineLRScheduler

from src.data.datasets import DataConfig, THzLikeCIFAR10, THzLikeMNIST
from src.models.transnext_wrapper import create_transnext_model
from src.tools.visualize_run import RunVisualizer

def train_one_epoch(model, loader, optimizer, criterion, device: str,
                    mixup_fn=None, max_grad_norm: float = 0.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        # When mixup is active, y is soft labels — skip accuracy count
        if mixup_fn is None:
            correct += (logits.argmax(dim=1) == y).sum().item()
        else:
            correct += (logits.argmax(dim=1) == y.argmax(dim=1)).sum().item()
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
    backbone_lr: float | None = None,
    degradation_type: str = "all",
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.0,
    scheduler_type: str = "none",
    warmup_epochs: int = 0,
    max_grad_norm: float = 0.0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    drop_path_rate: float = 0.0,
    # Custom degradation parameters (override defaults when provided)
    blur_kernel: int | None = None,
    blur_sigma: float | None = None,
    gaussian_noise_std: float | None = None,
    salt_pepper_amount: float | None = None,
    p_grayscale: float | None = None,
    early_stopping_patience: int = 0,
    dataset: str = "cifar10",
):
    import random as _random
    import numpy as _np
    _random.seed(42)
    _np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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
        f.write(f"backbone_lr={backbone_lr}\n")
        f.write(f"degradation_type={degradation_type}\n")
        f.write(f"tag={tag}\n")
        f.write(f"group={group}\n")
        f.write(f"run_name={run_name}\n")
        f.write(f"freeze_backbone={freeze_backbone}\n")
        f.write(f"weight_decay={weight_decay}\n")
        f.write(f"label_smoothing={label_smoothing}\n")
        f.write(f"scheduler_type={scheduler_type}\n")
        f.write(f"warmup_epochs={warmup_epochs}\n")
        f.write(f"max_grad_norm={max_grad_norm}\n")
        f.write(f"mixup_alpha={mixup_alpha}\n")
        f.write(f"cutmix_alpha={cutmix_alpha}\n")
        f.write(f"drop_path_rate={drop_path_rate}\n")
        f.write(f"blur_kernel={blur_kernel}\n")
        f.write(f"blur_sigma={blur_sigma}\n")
        f.write(f"gaussian_noise_std={gaussian_noise_std}\n")
        f.write(f"salt_pepper_amount={salt_pepper_amount}\n")
        f.write(f"p_grayscale={p_grayscale}\n")
        f.write(f"early_stopping_patience={early_stopping_patience}\n")
        f.write(f"dataset={dataset}\n")

    ds_label = "CIFAR10" if dataset == "cifar10" else "MNIST"
    log(f"Device: {device}")
    log(f"Model: {model_name}, pretrained={pretrained}")
    log(f"Data: {ds_label} degraded | out_size={out_size}, low_res={low_res}")
    log(f"Train subset={train_subset}, Val subset={val_subset}")
    log(f"Epochs={epochs}, batch_size={batch_size}, lr={lr}, backbone_lr={backbone_lr}")
    log(f"Degradation type: {degradation_type}")
    log(f"Group: {group}")
    log(f"Freeze backbone: {freeze_backbone}")
    log(f"Weight decay: {weight_decay}, Label smoothing: {label_smoothing}")
    log(f"Scheduler: {scheduler_type}, Warmup: {warmup_epochs} epochs")
    log(f"Mixup: {mixup_alpha}, CutMix: {cutmix_alpha}, DropPath: {drop_path_rate}")
    log(f"Max grad norm: {max_grad_norm}")
    log(f"Saved run config: {run_config_path}")

    if model_name.startswith("transnext_") and out_size != 224:
        print(f"[INFO] Overriding out_size from {out_size} to 224 for TransNeXt")
        out_size = 224

    # data – pass custom degradation overrides if provided
    deg_overrides = {}
    if blur_kernel is not None:
        deg_overrides["blur_kernel"] = blur_kernel
    if blur_sigma is not None:
        deg_overrides["blur_sigma"] = blur_sigma
    if gaussian_noise_std is not None:
        deg_overrides["gaussian_noise_std"] = gaussian_noise_std
    if salt_pepper_amount is not None:
        deg_overrides["salt_pepper_amount"] = salt_pepper_amount
    if p_grayscale is not None:
        deg_overrides["p_grayscale"] = p_grayscale

    cfg_train = DataConfig(dataset=dataset, train=True, out_size=out_size, low_res=low_res, root="./data", degradation_type=degradation_type, **deg_overrides)
    cfg_val = DataConfig(dataset=dataset, train=False, out_size=out_size, low_res=low_res, root="./data", degradation_type=degradation_type, **deg_overrides)

    DatasetClass = THzLikeMNIST if dataset == "mnist" else THzLikeCIFAR10
    train_ds = DatasetClass(cfg_train)
    val_ds = DatasetClass(cfg_val)

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
            _project_root = Path(__file__).resolve().parents[1]
            _candidate = _project_root / "artifacts" / "weights" / "transnext_micro_224_1k.pth"
            if _candidate.exists():
                transnext_ckpt = str(_candidate)

        model = create_transnext_model(
            model_name=model_name,
            num_classes=10,
            pretrained=pretrained,
            checkpoint_path=transnext_ckpt,
            drop_path_rate=drop_path_rate,
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

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Mixup / CutMix setup
    mixup_fn = None
    if mixup_alpha > 0 or cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=label_smoothing,
            num_classes=10,
        )
        # When using timm Mixup, it handles label smoothing internally
        # Use soft cross-entropy instead
        criterion = nn.CrossEntropyLoss()  # mixup provides soft targets

    if backbone_lr is not None:
        backbone_params = [p for n, p in model.named_parameters() if 'head' not in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if 'head' in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
            {'params': head_params, 'lr': lr, 'weight_decay': weight_decay}
        ])
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # LR Scheduler
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs,
            lr_min=1e-5,
            warmup_t=warmup_epochs,
            warmup_lr_init=1e-6,
            warmup_prefix=True,
        )

    # metrics + best tracking
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = -1.0
    best_path = run_dir / "best.pt"
    epochs_without_improvement = 0

    t0 = time.time()
    for ep in range(1, epochs + 1):
        if scheduler is not None:
            scheduler.step(ep - 1)

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            mixup_fn=mixup_fn, max_grad_norm=max_grad_norm,
        )
        va_loss, va_acc = eval_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[-1]['lr']
        log(
            f"Epoch {ep}/{epochs} | "
            f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f} | "
            f"Val: loss={va_loss:.4f}, acc={va_acc:.4f} | "
            f"LR={current_lr:.2e}"
        )

        with open(metrics_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ep, f"{tr_loss:.6f}", f"{tr_acc:.6f}", f"{va_loss:.6f}", f"{va_acc:.6f}"])

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            epochs_without_improvement = 0
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
        else:
            epochs_without_improvement += 1

        # Early stopping
        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            log(f"[EARLY STOP] No improvement for {early_stopping_patience} epochs. "
                f"Best val_acc={best_val_acc:.4f}. Stopping at epoch {ep}.")
            break

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

    # Auto-generate learning curve visualizations
    try:
        log("\n[VISUALIZE] Generating learning curve visualizations...")
        visualizer = RunVisualizer(run_dir)
        visualizer.generate_all()
        log("[OK] Visualizations complete!")
    except Exception as e:
        log(f"[WARN] Visualization failed: {e}")

    # Auto-update experiment plan dashboard after each experiment
    try:
        log("[DASHBOARD] Updating experiment plan dashboard...")
        import importlib
        # Ensure project root is on sys.path for the import
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        dashboard_mod = importlib.import_module("src.tools.generate_experiment_plan_dashboard")
        dashboard_mod.main()
        log("[OK] Dashboard updated: artifacts/dashboard_experiment_plan.html")
    except Exception as e:
        log(f"[WARN] Dashboard update failed: {e}")


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
    p.add_argument("--backbone_lr", type=float, default=None)
    p.add_argument("--degradation_type", type=str, default="all",
                  choices=["all", "downsampling", "blur", "noise", "salt_pepper"],
                  help="Type of degradation: all, downsampling, blur, noise, or salt_pepper")
    p.add_argument("--tag", type=str, default="")
    p.add_argument("--group", type=str, default="pilot", choices=["pilot", "official"])
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--scheduler_type", type=str, default="none", choices=["none", "cosine"])
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--mixup_alpha", type=float, default=0.0)
    p.add_argument("--cutmix_alpha", type=float, default=0.0)
    p.add_argument("--drop_path_rate", type=float, default=0.0)
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
        backbone_lr=args.backbone_lr,
        degradation_type=args.degradation_type,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        scheduler_type=args.scheduler_type,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        drop_path_rate=args.drop_path_rate,
    )