import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
import os

# טען CIFAR10
dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=None
)

# קח תמונה אחת לדוגמה
img, label = dataset[0]
img_np = np.array(img)


def degrade(image, size):
    small = Image.fromarray(image).resize((size, size), Image.BILINEAR)
    upsample = small.resize((32, 32), Image.NEAREST)
    return np.array(upsample)


img16 = degrade(img_np, 16)
img8 = degrade(img_np, 8)

fig, ax = plt.subplots(1, 3, figsize=(9, 3))

ax[0].imshow(img_np)
ax[0].set_title("Original")

ax[1].imshow(img16)
ax[1].set_title("LowRes 16")

ax[2].imshow(img8)
ax[2].set_title("LowRes 8")

for a in ax:
    a.axis("off")

plt.tight_layout()

os.makedirs("artifacts/figures", exist_ok=True)
plt.savefig("artifacts/figures/degradation_example.png", dpi=300)

plt.show()