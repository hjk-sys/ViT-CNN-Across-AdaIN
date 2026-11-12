# VLM_Test: CLIP and LLaVA on stylized CIFAR-10

A pilot test, kept apart from the paper experiments in `project/`. It asks whether vision-language models degrade, and hallucinate objects, on the same AdaIN-stylized CIFAR-10 images.

- **CLIP** (`openai/clip-vit-base-patch32`): zero-shot classification accuracy across the 10 classes.
- **LLaVA-1.5-7B** (4-bit): POPE-style yes/no probing. For each image it asks "Is there a [true class]?" (a yes counts toward recall) and "Is there a [random wrong class]?" (a yes counts as a hallucination). Each image gets the same wrong class in every style.

The same 100 images (10 per class, seed 0) are used in every style, so the comparisons are paired. `resize_only` is the control: the plain upscaled images, with no style applied.

## Results (n = 100 per style)

| Style | CLIP acc | LLaVA recall | LLaVA hallucination |
|---|---|---|---|
| resize_only (control) | 0.79 | 0.87 | 0.04 |
| abstract | 0.56 | 0.68 | 0.09 |
| cubism | 0.46 | 0.60 | 0.06 |
| oil_painting | 0.57 | 0.67 | 0.08 |
| Post_Impression | 0.56 | 0.77 | 0.06 |
| sketch | 0.55 | 0.69 | 0.11 |
| watercolor | 0.49 | 0.64 | 0.08 |

Raw numbers: [`summary.csv`](summary.csv).

- Every style lowers both CLIP accuracy (by 22–33 points) and LLaVA recall (by 10–27 points). This drop is clear even at n = 100.
- The hallucination rate rises from 4% to 8% on average across the styles (49 of 600 answers). With only 100 images per style, that gap could be chance, so it is **not statistically significant yet**. A run with `--n 500` should settle it.
- CIFAR images contain one object each, so this setup cannot test co-occurrence hallucination. That needs multi-object scenes, such as COCO.

## Running it

The images are not in the repo (see the main README). The script reads `<data_root>/<style>/<class>/<file>.png`.

```bash
# 1. Locally: copy the sampled images into a small folder to upload
python VLM_Test/vlm_style_probe.py --data_root project/Images --n 100 --make_subset probe_subset

# 2. Colab (T4 GPU), after uploading probe_subset and the script to Drive
pip install -q transformers accelerate bitsandbytes pillow pandas
python vlm_style_probe.py --data_root /content/drive/MyDrive/probe_subset --n 100 --skip_llava   # CLIP only
python vlm_style_probe.py --data_root /content/drive/MyDrive/probe_subset --n 100                # + LLaVA
```

The full run downloads about 14 GB of LLaVA weights and asks 1,400 questions. Results are written to `vlm_style_results/summary.csv` and `vlm_style_results/llava_per_image.csv`.
