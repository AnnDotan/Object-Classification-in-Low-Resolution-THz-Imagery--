# src/inspect_data.py
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from src.data.degrade import DegradeConfig, degrade_image

def main():
    base = datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())

    cfg = DegradeConfig(low_res=16, out_size=32)  # אותו out_size שאת מריצה כרגע
    n = 8

    plt.figure(figsize=(12, 4))
    for i in range(n):
        x, y = base[i]  # clean tensor [3,32,32]
        x_deg = degrade_image(x, cfg)

        # clean
        plt.subplot(2, n, i + 1)
        plt.imshow(x.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"clean {y}", fontsize=8)

        # degraded
        plt.subplot(2, n, n + i + 1)
        plt.imshow(x_deg.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"deg {y}", fontsize=8)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()