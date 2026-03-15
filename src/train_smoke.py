# src/train_smoke.py
from __future__ import annotations

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import timm

from src.data.datasets import DataConfig, THzLikeCIFAR10

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Small "smoke test" settings
##############################################################################Change to 224 after training
    #cfg_train = DataConfig(train=True, out_size=224, low_res=16, root="./data")
    #cfg_val = DataConfig(train=False, out_size=224, low_res=16, root="./data")
    cfg_train = DataConfig(train=True, out_size=32, low_res=16, root="./data")
    cfg_val = DataConfig(train=False, out_size=32, low_res=16, root="./data")

    train_ds = THzLikeCIFAR10(cfg_train)
    val_ds = THzLikeCIFAR10(cfg_val)

############################################################running on small subset
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, range(2000))
    val_ds = Subset(val_ds, range(1000))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # Start with a simple model (ResNet18) to verify pipeline works
    model = timm.create_model("resnet50", pretrained=True, num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def run_epoch(loader, train: bool):
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = criterion(logits, y)

                if train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        return total_loss / total, correct / total

    t0 = time.time()
    train_loss, train_acc = run_epoch(train_loader, train=True)
    val_loss, val_acc = run_epoch(val_loader, train=False)
    dt = time.time() - t0

    print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
    print(f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")
    print(f"Time:  {dt:.1f}s")


if __name__ == "__main__":
    main()