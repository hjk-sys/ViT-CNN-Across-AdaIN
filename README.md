# ViT-CNN-Classifiers-Across-AdaIN

From the paper: "From photographs to paintings: Evaluating computer vision architecture across artistic domains."

Compares how a CNN (fine-tuned **ResNet-50**) and a Vision Transformer (**ViT-Base-16**) hold up when the same images are shown in different artistic styles.

The test set is the 10,000 CIFAR-10 test images. Each image is also rendered in six painting styles using AdaIN style transfer, giving 7 versions of the test set (original + 6 styles) and 70,000 images in total. Both models are evaluated on every version.

For each model and version, `Evaluate.py` reports overall accuracy, per-class accuracy, a confusion matrix, per-class precision / recall / F1, parameter count, and inference speed (images/sec).

## Setup

Developed with Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Build the dataset

The images are not stored in this repo. `protocol.py` generates any stylized version that's missing (see [PROTOCOL.md](project/PROTOCOL.md)). To make a single one, edit the `generate(...)` call at the bottom of `project/Style_Transfer/sample.py` and run:

```bash
python project/Style_Transfer/sample.py
```

This takes a long time per style. It skips images that already exist, so you can stop it and resume later.

The AdaIN checkpoint and the six reference paintings are included in `project/Style_Transfer/`.

| Output folder | Reference painting |
|---|---|
| `Base Copy` | original CIFAR-10 images (no style) |
| `abstract-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/3d8c32c5-4737-4841-83a9-991aecdba53e" />|
| `cubism-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/2d2eef68-cacf-4e93-bc2c-0c7ff3cfba57" />|
| `oil_painting-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/632b6470-9a20-4198-b159-4be2d8b396c7" />|
| `Post_Impression-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/e369d8bc-5d4d-4e6f-a1bf-181d0df340f2" />|
| `sketch-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/cb8e79d8-9b28-4831-9c35-b2e29c32ea8e" />|
| `watercolor-Batch` | <img width="250" height="316" alt="image" src="https://github.com/user-attachments/assets/e6cb4658-a115-41d1-801b-d84d262607c2" />|

Each folder contains one subfolder per CIFAR-10 class, under `project/Images/`.

## 2. Trained models

This project evaluates two existing CIFAR-10 checkpoints from Hugging Face. They are about 420 MB combined and are not included in this repo. Download these two files and place them in `project/models/`:

| File | Model | Source |
|---|---|---|
| `finetune_resnet_cifar_model.pt` | ResNet-50 fine-tuned on CIFAR-10 | [AnjanSB/finetune-resnet-cifar10](https://huggingface.co/AnjanSB/finetune-resnet-cifar10/tree/main) |
| `vit_base_16_cifar10_original.pth` | ViT-Base-16 fine-tuned on CIFAR-10 | [Yurim0507/vit-base-16-cifar10-unlearning](https://huggingface.co/Yurim0507/vit-base-16-cifar10-unlearning) |

The ViT repository contains 11 checkpoints. Only the `original` one (trained on all 10 classes) is used here.

## 3. Run the experiments

The scripts use paths relative to the repo root, so run them from there:

```bash
python project/ResnetInput.py   # evaluate ResNet-50
python project/ViTInput.py      # evaluate ViT-Base-16
python project/results.py       # tables and figures from the saved results
```

Each input script evaluates every dataset listed in `DATASETS` in `Evaluate.py` that is fully generated and not saved yet. Results go to `project/results/`; delete a file there to redo that dataset.

For the full control-corrected protocol, see [PROTOCOL.md](project/PROTOCOL.md).

The scripts use an Apple Silicon GPU (MPS) when one is available and fall back to the CPU otherwise.

## Repository layout

```
project/
├── protocol.py           # control-corrected protocol for one model
├── results.py            # tables and figures for the paper
├── Evaluate.py           # accuracy, confusion matrix, precision/recall/F1, speed
├── Resnet.py             # ResNet-50 fine-tuning model
├── ResnetInput.py        # runs ResNet-50 on the datasets
├── ViTModel.py           # ViT-Base-16 model loading
├── ViTInput.py           # runs ViT-Base-16 on the datasets
├── baseline.py           # dataset loading / sanity checks
├── Style_Transfer/       # AdaIN code, checkpoint, style paintings, sample.py
├── figures/              # figures
├── results/              # saved results, tables and figures
├── models/               # trained weights (not in the repo)
└── Images/               # generated dataset (not in the repo)
```

## Credits

- ResNet-50 CIFAR-10 checkpoint: [AnjanSB/finetune-resnet-cifar10](https://huggingface.co/AnjanSB/finetune-resnet-cifar10) (MIT license)
- ViT-Base-16 CIFAR-10 checkpoint: [Yurim0507/vit-base-16-cifar10-unlearning](https://huggingface.co/Yurim0507/vit-base-16-cifar10-unlearning) (MIT license)
- Style transfer: AdaIN (Huang and Belongie, 2017)
- Dataset: CIFAR-10 (Krizhevsky, 2009)
