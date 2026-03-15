# src/verify_labels.py
from collections import Counter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.data.datasets import DataConfig, THzLikeCIFAR10

def main():
    # dataset שלנו (degraded)
    ds = THzLikeCIFAR10(DataConfig(train=True, out_size=32, low_res=16, root="./data"))
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    x, y = next(iter(loader))
    counts = Counter(y.tolist())

    # CIFAR original (רק בשביל שמות הקלאסים)
    base = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    class_names = base.classes

    print("x shape:", tuple(x.shape))  # צפוי: (256, 3, 32, 32)
    print("label counts in one batch:", counts)
    print("example labels -> class names:")
    for lab in sorted(list(counts.keys()))[:5]:
        print(lab, "->", class_names[lab])

if __name__ == "__main__":
    main()