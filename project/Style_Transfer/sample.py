import os
import torch
from torchvision.datasets import CIFAR10
from infer import load_image, postprocess
from model import StyleNet

checkpoint = "project/Style_Transfer/weights/cktp-style-net-30000.tar"


def generate(style, out_folder, alpha, resize_only):
    dataset = CIFAR10(
        root="project/Images",
        train=False,
        download=True
    )

    for class_name in dataset.classes:
        os.makedirs(f"{out_folder}/{class_name}", exist_ok=True)

    # Load model + style once
    model = StyleNet(checkpoint).eval()
    style_image, _ = load_image(style, 256)
    with torch.no_grad():
        style_feats = model.encoder(style_image.float(), return_last=True)

    for i in range(len(dataset)):

        image, label = dataset[i]
        class_name = dataset.classes[label]

        save_path = f"{out_folder}/{class_name}/{i}.png"

        if i % 10 == 0:
            print(f"Last file loaded: {save_path}")
            print(f"------------------------------------------------------     {round((i/len(dataset))*100,2)}%     -------------------------------------------------")

        if os.path.exists(save_path):
            continue

        content_image, size = load_image(image, 256)
        content_image = content_image.float()

        if resize_only:
            styled_image = content_image
        else:
            with torch.no_grad():
                content_feats = model.encoder(content_image, return_last=True)
                styled_image = model.generate(content_feats, style_feats, alpha, infer=True)

        postprocess(styled_image, size).save(save_path)


if __name__ == "__main__":
    generate("project/Style_Transfer/styles/cubism.jpg", "project/Images/cubism-Batch", 1.0, False)
