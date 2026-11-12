"""
vlm_style_probe.py — quick check: do CLIP / LLaVA degrade (and hallucinate) on stylized CIFAR-10?

Two tests, run on the SAME sampled images in every style folder (paired comparison):
  1. CLIP zero-shot classification accuracy per style.
  2. LLaVA POPE-style yes/no probing per style:
       - positive question: "Is there a <true class> in the image?"   -> want "yes"
       - negative question: "Is there a <random wrong class> ...?"     -> "yes" = hallucination

Layout (labels come from the class subfolder, no labels.csv needed):
  data_root/
    resize_only-Batch/airplane/10.png ...   <- reference condition (first in STYLE_DIRS)
    cubism-Batch/airplane/10.png ...
  The same <class>/<filename> must exist in every style folder.

Step 1 (locally, on your Mac) — copy a small subset to upload to Drive:
  python vlm_style_probe.py --data_root Images --n 100 --make_subset probe_subset

Step 2 (Colab, T4 GPU):
  !pip install -q transformers accelerate bitsandbytes pillow pandas
  !python vlm_style_probe.py --data_root /content/drive/MyDrive/probe_subset --n 100 --skip_llava
  !python vlm_style_probe.py --data_root /content/drive/MyDrive/probe_subset --n 100
"""

import argparse
import os
import random
import shutil

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
# Friendlier words for prompts (VLMs know "car" better than "automobile")
PROMPT_NAMES = {"automobile": "car"}

# First entry is the reference condition.
STYLE_DIRS = ["resize_only-Batch", "abstract-Batch", "cubism-Batch", "oil_painting-Batch",
              "Post_Impression-Batch", "sketch-Batch", "watercolor-Batch"]


def pname(c):
    return PROMPT_NAMES.get(c, c)


def img_path(data_root, style, fn, label):
    return os.path.join(data_root, style, label, fn)


def load_labels(data_root, styles):
    """Images (filename, label) present in every style folder, labels from class subfolders."""
    common = None
    for style in styles:
        found = set()
        for c in CLASSES:
            d = os.path.join(data_root, style, c)
            if os.path.isdir(d):
                found |= {(f, c) for f in os.listdir(d) if f.lower().endswith(".png")}
        common = found if common is None else common & found
    rows = sorted(common or [])
    if not rows:
        raise SystemExit("No images found in all style folders — check --data_root and STYLE_DIRS.")
    return rows


def load_img(path):
    from PIL import Image
    # Upscale 32x32 -> 224 with bicubic so every condition gets the same resize.
    return Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC)


# ---------------------------------------------------------------- CLIP
def run_clip(samples, data_root, styles, device):
    import pandas as pd
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(name).to(device).eval()
    proc = CLIPProcessor.from_pretrained(name)
    texts = [f"a photo of a {pname(c)}" for c in CLASSES]

    rows = []
    for style in styles:
        correct = 0
        for fn, label in samples:
            img = load_img(img_path(data_root, style, fn, label))
            inp = proc(text=texts, images=img, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = model(**inp).logits_per_image[0]
            correct += CLASSES[int(logits.argmax())] == label
        acc = correct / len(samples)
        rows.append({"style": style, "clip_acc": acc})
        print(f"[CLIP] {style:24s} acc = {acc:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- LLaVA
def run_llava(samples, data_root, styles, device, seed):
    import pandas as pd
    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              LlavaForConditionalGeneration)
    name = "llava-hf/llava-1.5-7b-hf"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(
        name, quantization_config=quant, device_map="auto")
    proc = AutoProcessor.from_pretrained(name)

    def ask_yes(img, obj):
        prompt = (f"USER: <image>\nIs there a {pname(obj)} in the image? "
                  f"Answer with only yes or no. ASSISTANT:")
        inp = proc(images=img, text=prompt, return_tensors="pt").to(model.device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=3, do_sample=False)
        ans = proc.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        return ans.strip().lower().startswith("yes")

    # Fix the wrong class per image so every style gets the identical question.
    rng = random.Random(seed)
    negs = {fn: rng.choice([c for c in CLASSES if c != lab]) for fn, lab in samples}

    rows, detail = [], []
    for style in styles:
        tp = fp = 0
        for fn, label in samples:
            img = load_img(img_path(data_root, style, fn, label))
            pos = ask_yes(img, label)
            neg = ask_yes(img, negs[fn])
            tp += pos
            fp += neg
            detail.append({"style": style, "file": fn, "label": label,
                           "neg_class": negs[fn], "said_yes_true": pos,
                           "said_yes_wrong": neg})
        n = len(samples)
        rows.append({"style": style,
                     "recall_true_object": tp / n,        # higher = better
                     "hallucination_rate": fp / n})       # higher = worse
        print(f"[LLaVA] {style:24s} recall = {tp/n:.3f}   hallucination = {fp/n:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(detail)


def make_subset(samples, data_root, styles, out_dir):
    for style in styles:
        for fn, label in samples:
            dst = img_path(out_dir, style, fn, label)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(img_path(data_root, style, fn, label), dst)
    print(f"Copied {len(samples)} images x {len(styles)} styles to {out_dir} — upload this folder to Drive.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--n", type=int, default=100, help="images sampled (same set for every style)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_llava", action="store_true")
    ap.add_argument("--make_subset", metavar="OUT_DIR",
                    help="only copy the sampled images to OUT_DIR (for uploading) and exit")
    ap.add_argument("--out", default="vlm_style_results")
    args = ap.parse_args()

    styles = [s for s in STYLE_DIRS if os.path.isdir(os.path.join(args.data_root, s))]
    missing = set(STYLE_DIRS) - set(styles)
    if missing:
        print("Skipping missing folders:", sorted(missing))

    pairs = load_labels(args.data_root, styles)
    # Stratified sample: equal images per class so no class dominates.
    per_class = max(1, args.n // len(CLASSES))
    rng = random.Random(args.seed)
    samples = []
    for c in CLASSES:
        group = [p for p in pairs if p[1] == c]
        samples += rng.sample(group, min(per_class, len(group)))

    if args.make_subset:
        make_subset(samples, args.data_root, styles, args.make_subset)
        return

    import pandas as pd
    import torch
    device ="cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(samples)} images x {len(styles)} styles on {device}")

    os.makedirs(args.out, exist_ok=True)
    summary = run_clip(samples, args.data_root, styles, device)

    if not args.skip_llava:
        llava_sum, llava_detail = run_llava(samples, args.data_root, styles, device, args.seed)
        summary = summary.merge(llava_sum, on="style")
        llava_detail.to_csv(os.path.join(args.out, "llava_per_image.csv"), index=False)

    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
