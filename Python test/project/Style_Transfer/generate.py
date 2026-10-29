"""
Builds a stylized copy of the CIFAR-10 test set with the same steps as sample.py,
but loads the network and encodes the style image once instead of per image.

Run from the repo root:
    # Second painting for a style
    .venv/bin/python project/Style_Transfer/generate.py --style project/Style_Transfer/styles/cubism_b.jpg --out project/Images/cubism_b-Batch

    # No-style control: identical AdaIN pipeline, alpha = 0 (decoder rebuilds the content features)
    .venv/bin/python project/Style_Transfer/generate.py --alpha 0 --out project/Images/alpha0-Batch

    # Resize-only control: 32 -> 256 -> 32 round trip and uint8 save, no encoder/decoder
    .venv/bin/python project/Style_Transfer/generate.py --resize-only --out project/Images/resize_only-Batch
"""
import argparse, os, sys, time
import torch
from torchvision.datasets import CIFAR10

# project/ on the path too, so infer.py works with either "from utils" or "from Style_Transfer.utils"
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer import load_image, postprocess
from model import StyleNet

CKPT = "project/Style_Transfer/weights/cktp-style-net-30000.tar"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output folder, e.g. project/Images/alpha0-Batch")
    parser.add_argument("--style", default="project/Style_Transfer/styles/cubism.jpg",
                        help="style image (ignored by --resize-only; has no effect at --alpha 0)")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--resize-only", action="store_true")
    parser.add_argument("--device", default="cpu",
                        help="cpu matches the original datasets exactly; mps is faster but not bit-identical")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (for testing)")
    args = parser.parse_args()

    dataset = CIFAR10(root="project/Images", train=False, download=False)
    for class_name in dataset.classes:
        os.makedirs(f"{args.out}/{class_name}", exist_ok=True)

    device = torch.device(args.device)
    if not args.resize_only:
        model = StyleNet(CKPT).to(device).eval()
        style_image, _ = load_image(args.style, 256)
        with torch.no_grad():
            style_feats = model.encoder(style_image.float().to(device), return_last=True)

    total = len(dataset) if args.limit is None else args.limit
    start = time.perf_counter()
    for i in range(total):
        image, label = dataset[i]
        save_path = f"{args.out}/{dataset.classes[label]}/{i}.png"
        if os.path.exists(save_path):
            continue

        content_image, c_size = load_image(image, 256)
        content_image = content_image.float().to(device)

        if args.resize_only:
            out = content_image
        else:
            with torch.no_grad():
                content_feats = model.encoder(content_image, return_last=True)
                out = model.generate(content_feats, style_feats, args.alpha, infer=True)

        postprocess(out, c_size).save(save_path)

        if i % 500 == 0:
            print(f"{i}/{total}  ({time.perf_counter() - start:.0f} s)")

    print(f"done: {total} images in {time.perf_counter() - start:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
