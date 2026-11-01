# Control-corrected evaluation protocol

Style-transfer robustness tests usually compare stylized images to the original photos. But AdaIN changes images even when no style is applied: it resizes them and rebuilds them through an encoder and decoder. If you compare against the original photos, that damage gets counted as an effect of style.

This protocol runs a model through four conditions on the CIFAR-10 test set, so each step's effect can be measured on its own:

| Condition | What changes |
|---|---|
| Baseline | Original CIFAR-10 test images |
| Resize only | 32 → 256 → 32 px resize, no AdaIN |
| AdaIN, α = 0 | Full AdaIN pipeline with no style (the control) |
| Styles | AdaIN, α = 1, with one reference painting per style |

From these it reports:

- **Resizing effect** = baseline − resize only
- **Reconstruction effect** = resize only − α = 0
- **Style effect** = α = 0 − style, for each style
- **Attractor classes**: classes predicted at least 1.5× as often as expected (ρ_c ≥ 1.5) *and* at least 0.25 above the same class under the α = 0 control, so bias the pipeline already causes isn't credited to style

It has been tested on CIFAR-10 image classification with three models (ResNet-50, ViT-B/16 and ViT-Small).

## Running it

From the repo root:

```
.venv/bin/python project/protocol.py resnet
.venv/bin/python project/protocol.py vit
.venv/bin/python project/protocol.py vit_small
```

The first run generates any missing condition datasets (about 30 minutes each on an M1 CPU; resize-only is faster) and evaluates the model on each. Later runs skip anything already done.

Outputs, in `project/results/`:

- `protocol_<model>_stages.csv`: accuracy at each condition and the points lost at each stage
- `protocol_<model>_rho.png`: ρ_c heatmap with attractors outlined
- `<model>__<condition>_predictions.csv`: true and predicted class for every test image

## Setup

- Install the packages in `requirements.txt`.
- CIFAR-10 downloads automatically into `project/Images/`.
- Model weights aren't in the repo (too large for GitHub). Put them in `project/models/`:
  - `finetune_resnet_cifar_model.pt` from https://huggingface.co/AnjanSB/finetune-resnet-cifar10
  - `vit_base_16_cifar10_original.pth` from https://huggingface.co/Yurim0507/vit-base-16-cifar10-unlearning
- The AdaIN decoder weights are included in `project/Style_Transfer/weights/`.

## Adding a model

Write a function that loads your model and returns it in eval mode, then add it to `MODELS` in `protocol.py` together with the preprocessing its checkpoint was trained with:

```python
def load_my_model(device):
    model = ...
    model.to(device)
    model.eval()
    return model

MODELS = {
    "resnet": (load_resnet, imagenet_transform),
    "vit": (load_vit, imagenet_transform),
    "vit_small": (load_vit_small, vit_small_transform),
    "my_model": (load_my_model, my_transform)
}
```

The model must output 10 class scores in CIFAR-10 order (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). Hugging Face models can be wrapped with `HuggingFaceModel`, as ViT-Small is. Check the label order in the checkpoint's `config.json` and its preprocessing in `preprocessor_config.json`, and confirm the clean baseline matches the reported accuracy before running everything. The datasets are shared, so a new model only needs the evaluation step.

ViT-Small (`vit_small`, 21.7M parameters, close to ResNet-50's 24.6M) was added this way as a size-matched comparison. Its clean baseline is 98.48% with its own preprocessing (resize to 224, mean = std = 0.5).

## Example output (from the paper)
| Condition | ResNet-50 acc. (%) | Points lost | ViT-B/16 acc. (%) | Points lost |
|---|---|---|---|---|
| Baseline | 92.57 | | 98.47 | |
| Resize only | 88.95 | 3.62 (resizing) | 96.52 | 1.95 (resizing) |
| AdaIN, α = 0 | 79.82 | 9.13 (reconstruction) | 90.17 | 6.35 (reconstruction) |
| Oil painting | 73.29 | 6.53 | 88.80 | 1.37 |
| Abstract | 67.02 | 12.80 | 86.99 | 3.18 |
| Post-Impressionism | 63.47 | 16.35 | 88.10 | 2.07 |
| Watercolor | 54.84 | 24.98 | 82.60 | 7.57 |
| Sketch | 53.03 | 26.79 | 73.14 | 17.03 |
| Cubism | 49.07 | 30.75 | 78.81 | 11.36 |
| **Mean style effect** | | **19.70** | | **7.10** |

Attractors found: ResNet-50 — airplane (Post-Impressionism, watercolor, cubism), cat (watercolor, sketch), frog (cubism). ViT-B/16 — cat (sketch), frog (cubism).

## Limitations

Tested only on 32 × 32 CIFAR-10 images and two models, with one reference painting per style in the main conditions. The 1.5× and 0.25 attractor thresholds are choices, not derived values.
