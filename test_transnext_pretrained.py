from src.models.transnext_wrapper import create_transnext_model
import torch

checkpoint = r"C:\Users\ib94\Documents\NewFinalProject\artifacts\weights\transnext_micro_224_1k.pth"

print("Loading model...")

model = create_transnext_model(
    "transnext_micro",
    num_classes=10,
    pretrained=True,
    checkpoint_path=checkpoint,
)

print("Model loaded:", type(model))

print("Running forward pass...")

x = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    y = model(x)

print("Output shape:", y.shape)