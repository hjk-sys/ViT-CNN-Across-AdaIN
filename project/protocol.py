# Control-corrected protocol, run from the repo root: .venv/bin/python project/protocol.py resnet (or vit, vit_small)

import os, sys, csv, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Style_Transfer"))
from sample import generate
from Evaluate import evaluate_all, load, rho, attractors, DATASETS, RESULTS_DIR, CLASSES
from Resnet import ResNet50FineTune
from ViTModel import load_model
from transformers import ViTForImageClassification

# (style image, alpha, resize only)
conditions = {
    "resize_only": ("cubism.jpg", 1.0, True),
    "alpha0": ("cubism.jpg", 0.0, False),
    "oil_painting": ("oil_painting.jpeg", 1.0, False),
    "abstract": ("abstract.jpeg", 1.0, False),
    "post_impression": ("starry_night.jpg", 1.0, False),
    "watercolor": ("watercolor.jpg", 1.0, False),
    "sketch": ("sketch.jpeg", 1.0, False),
    "cubism": ("cubism.jpg", 1.0, False),
}
styles = ["oil_painting", "abstract", "post_impression", "watercolor", "sketch", "cubism"]


""" models """

def load_resnet(device):
    model = ResNet50FineTune(num_classes=10)
    model.load_state_dict(torch.load("project/models/finetune_resnet_cifar_model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model


def load_vit(device):
    return load_model("project/models/vit_base_16_cifar10_original.pth", num_classes=10, device=device)


# Hugging Face models return an object, this returns just the class scores
class HuggingFaceModel(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(name)

    def forward(self, images):
        return self.model(pixel_values=images).logits


def load_vit_small(device):
    # Downloads the first time (about 90 MB)
    model = HuggingFaceModel("MF21377197/vit-small-patch16-224-finetuned-Cifar10")
    model.to(device)
    model.eval()
    return model


# Transforming to fit pretrained Imagenet architecture (same as ResnetInput.py and ViTInput.py)
imagenet_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# ViT-Small preprocessing (from its preprocessor_config.json)
vit_small_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    )
])

models = {
    "resnet": (load_resnet, imagenet_transform),
    "vit": (load_vit, imagenet_transform),
    "vit_small": (load_vit_small, vit_small_transform)
}

name = sys.argv[1] if len(sys.argv) > 1 else "resnet"
if name not in models:
    sys.exit(f"Unknown model '{name}'. Choose from: {', '.join(models)}")


""" Generate datasets """
for condition, (style, alpha, resize_only) in conditions.items():
    folder = DATASETS[condition]
    if len(glob.glob(f"{folder}/*/*.png")) == 10000:
        continue
    print(f"Generating {condition} (about 30 minutes)")
    generate(f"project/Style_Transfer/styles/{style}", folder, alpha, resize_only)


""" Evaluate """
# Apple silicon device
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

loader, transform = models[name]
model = loader(device)
evaluate_all(model, device, name, transform)


""" Stage table """
def accuracy(condition):
    confusion = load(name, condition)
    return np.trace(confusion) / confusion.sum() * 100

baseline = accuracy("baseline")
resized = accuracy("resize_only")
control = accuracy("alpha0")

rows = [["Condition", "Accuracy (%)", "Points lost", "Effect"]]
rows.append(["baseline", f"{baseline:.2f}", "", ""])
rows.append(["resize_only", f"{resized:.2f}", f"{baseline - resized:.2f}", "resizing (baseline - resize only)"])
rows.append(["alpha0", f"{control:.2f}", f"{resized - control:.2f}", "reconstruction (resize only - alpha 0)"])

style_effects = []
for style in styles:
    style_accuracy = accuracy(style)
    effect = control - style_accuracy
    style_effects.append(effect)
    rows.append([style, f"{style_accuracy:.2f}", f"{effect:.2f}", "style (alpha 0 - style)"])
rows.append(["mean style", "", f"{np.mean(style_effects):.2f}", "style"])

path = f"{RESULTS_DIR}/protocol_{name}_stages.csv"
with open(path, "w", newline="") as f:
    csv.writer(f).writerows(rows)

for row in rows:
    print(f"{row[0]:16s} {row[1]:>12s} {row[2]:>12s}   {row[3]}")
print(f"Saved {path}")


""" Rho heatmap """
rows = ["baseline", "resize_only", "alpha0"] + styles
grid = np.array([rho(name, row) for row in rows])
control = grid[2]

fig, ax = plt.subplots(figsize=(6.5, 3.6))
colors = LinearSegmentedColormap.from_list("rho", ["#1c5cab", "#ffffff", "#b8322f"])   # white at 1x
image = ax.imshow(np.clip(grid, 0.25, 4), cmap=colors, norm=LogNorm(vmin=0.25, vmax=4), aspect="auto")

for i in range(len(rows)):
    found = attractors(name, rows[i])
    for j in range(10):
        value = grid[i, j]
        attractor = rows[i] in styles and found[j]
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                fontweight="bold" if attractor else "normal")
        if attractor:
            ax.add_patch(Rectangle((j - 0.45, i - 0.42), 0.9, 0.84, fill=False, linewidth=1.1))
            print(f"Attractor: {rows[i]}, {CLASSES[j]} ({value:.2f}x, control {control[j]:.2f}x)")

ax.set_xticks(range(10), [c.capitalize() for c in CLASSES], fontsize=7, rotation=25, ha="right")
ax.set_yticks(range(len(rows)), rows, fontsize=7)
ax.set_title(f"{name}: predicted / expected, attractors outlined", fontsize=9)
colorbar = fig.colorbar(image, ax=ax)
colorbar.ax.minorticks_off()
colorbar.set_ticks([0.25, 0.5, 1, 2, 4], labels=["≤0.25×", "0.5×", "1×", "2×", "≥4×"])

path = f"{RESULTS_DIR}/protocol_{name}_rho.png"
fig.savefig(path, dpi=300, bbox_inches="tight")
print(f"Saved {path}")
