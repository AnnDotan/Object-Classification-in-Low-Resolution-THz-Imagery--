# src/check_labels.py
from collections import Counter
from torch.utils.data import DataLoader
from src.data.datasets import DataConfig, THzLikeCIFAR10

def main():
    ds = THzLikeCIFAR10(DataConfig(train=True, out_size=32, low_res=16, root="./data"))
    loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)

    counts = Counter()
    x, y = next(iter(loader))
    for label in y.tolist():
        counts[label] += 1

    print("Batch label distribution:", counts)
    print("x shape:", tuple(x.shape))

if __name__ == "__main__":
    main()