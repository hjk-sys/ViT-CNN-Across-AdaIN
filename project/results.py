# Makes every table and figure in the paper. Run from the repo root after ResnetInput.py and ViTInput.py

import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.datasets import CIFAR10

from Evaluate import load, rho, attractors, CLASSES, DATASETS, RESULTS_DIR

MODELS = {"resnet": "ResNet-50", "vit": "ViT-B/16"}

# Order of the conditions in every figure
ORDER = ["baseline", "alpha0", "oil_painting", "abstract", "post_impression", "watercolor", "sketch", "cubism"]

# The first painting of each style is the original
PAINTINGS = {
    "cubism": ["cubism", "cubism_b", "cubism_c"],
    "sketch": ["sketch", "sketch_b", "sketch_c"]
}
PAINTING_NAMES = {
    "cubism": "Cubism A",
    "cubism_b": "Cubism B",
    "cubism_c": "Cubism C",
    "sketch": "Sketch A",
    "sketch_b": "Sketch B",
    "sketch_c": "Sketch C"
}
PAINTING_TITLES = {
    "cubism": "The Weeping Woman",
    "cubism_b": "Table by a Window",
    "cubism_c": "Guitar and Mandolin",
    "sketch": "Curls",
    "sketch_b": "Weary Woman Sketch",
    "sketch_c": "Peaceful Slumber Portrait"
}

FIGURES_DIR = f"{RESULTS_DIR}/figures"
TABLES_DIR = f"{RESULTS_DIR}/tables"


def fmt(value, digits):
    # Rounds like the paper (12.75 -> 12.8), Python's rounding can go down on these
    return f"{value + 1e-9:.{digits}f}"


""" Part 1: numbers from the saved results """

def accuracy(confusion):
    return np.trace(confusion) / confusion.sum()


def per_class(confusion):
    correct = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0)
    actual = confusion.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = correct / predicted
        recall = correct / actual
        f1 = 2 * precision * recall / (precision + recall)
    rho = predicted / (confusion.sum() / 10)    # Eq. 4
    return precision, recall, f1, predicted, rho


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {path}")


def make_csvs():
    summary = [["model", "condition", "accuracy_%", "degradation_pts", "drop_from_alpha0_pts"]]
    classes = [["model", "condition", "class", "predicted", "rho", "precision_%", "recall_%", "f1_%"]]

    for model in MODELS:
        baseline = accuracy(load(model, "baseline"))
        control = accuracy(load(model, "alpha0"))

        for condition in DATASETS:
            if not os.path.exists(f"{RESULTS_DIR}/{model}__{condition}.npz"):
                continue
            confusion = load(model, condition)
            acc = accuracy(confusion)
            summary.append([model, condition, round(acc * 100, 2), round((baseline - acc) * 100, 2),
                            round((control - acc) * 100, 2)])

            precision, recall, f1, predicted, rho = per_class(confusion)
            for k in range(10):
                classes.append([model, condition, CLASSES[k], int(predicted[k]), round(rho[k], 3),
                                round(precision[k] * 100, 1), round(recall[k] * 100, 1), round(f1[k] * 100, 1)])

    write_csv(f"{RESULTS_DIR}/summary.csv", summary)
    write_csv(f"{RESULTS_DIR}/per_class.csv", classes)


def make_tables():
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Table 1: baseline, controls, size and speed
    resnet = np.load(f"{RESULTS_DIR}/resnet__baseline.npz")
    vit = np.load(f"{RESULTS_DIR}/vit__baseline.npz")
    table1 = [
        ["", "ResNet-50", "ViT-B/16"],
        ["Baseline accuracy (%)", fmt(accuracy(resnet["confusion"]) * 100, 2), fmt(accuracy(vit["confusion"]) * 100, 2)],
        ["Resize-only accuracy (%)", fmt(accuracy(load("resnet", "resize_only")) * 100, 2), fmt(accuracy(load("vit", "resize_only")) * 100, 2)],
        ["Control accuracy, α = 0 (%)", fmt(accuracy(load("resnet", "alpha0")) * 100, 2), fmt(accuracy(load("vit", "alpha0")) * 100, 2)],
        ["Parameters", f"{int(resnet['parameters']):,}", f"{int(vit['parameters']):,}"],
        ["Inference time, 10,000 images (s)", fmt(float(resnet["seconds"]), 2), fmt(float(vit["seconds"]), 2)],
        ["Images per second", fmt(10000 / float(resnet["seconds"]), 1), fmt(10000 / float(vit["seconds"]), 1)],
    ]
    write_csv(f"{TABLES_DIR}/table1_baseline_control.csv", table1)

    # Table 2: accuracy and attractors for every cubism and sketch painting
    table2 = [["Style", "Painting", "Title", "ResNet-50 accuracy (%)", "ResNet-50 attractors (ρ_c)",
               "ViT-B/16 accuracy (%)", "ViT-B/16 attractors (ρ_c)"]]
    for style in PAINTINGS:
        accuracies = {"resnet": [], "vit": []}

        for painting in PAINTINGS[style]:
            row = [style.capitalize(), PAINTING_NAMES[painting], PAINTING_TITLES[painting]]
            for model in MODELS:
                confusion = load(model, painting)
                precision, recall, f1, predicted, rho_values = per_class(confusion)
                accuracies[model].append(accuracy(confusion) * 100)

                # Attractors, biggest first
                is_attractor = attractors(model, painting)
                found = [k for k in np.argsort(-rho_values) if is_attractor[k]]
                listed = ", ".join(f"{CLASSES[k].capitalize()} {fmt(rho_values[k], 2)}" for k in found)
                row += [fmt(accuracies[model][-1], 2), listed or "None"]
            table2.append(row)

        row = [style.capitalize(), "Mean (range)", ""]
        for model in MODELS:
            a = accuracies[model]
            row += [f"{fmt(np.mean(a), 2)} ({fmt(min(a), 2)}–{fmt(max(a), 2)})", ""]
        table2.append(row)
    write_csv(f"{TABLES_DIR}/table2_robustness.csv", table2)


def print_discussion_numbers():
    styles = ORDER[2:]

    print("\nWhere the accuracy goes (points lost)")
    for model in MODELS:
        baseline = accuracy(load(model, "baseline")) * 100
        resized = accuracy(load(model, "resize_only")) * 100
        control = accuracy(load(model, "alpha0")) * 100
        style_mean = np.mean([accuracy(load(model, s)) * 100 for s in styles])
        print(f"  {MODELS[model]:10s} resizing {baseline - resized:.2f}, reconstruction {resized - control:.2f}, "
              f"style (mean) {control - style_mean:.2f}")

    print("\nShared attractors, and how alike the two models' rho values are (r of log rho)")
    for condition in styles + ["cubism_b", "cubism_c", "sketch_b", "sketch_c"]:
        resnet = attractors("resnet", condition)
        vit = attractors("vit", condition)
        shared = [CLASSES[k] for k in range(10) if resnet[k] and vit[k]]
        r = np.corrcoef(np.log(rho("resnet", condition)), np.log(rho("vit", condition)))[0, 1]
        print(f"  {condition:16s} r = {r:+.2f}   shared: {', '.join(shared) or '-'}")

    print("\nShip -> airplane and dog -> frog (% of the true class)")
    for condition in ["baseline", "resize_only"] + ORDER[1:] + ["cubism_b", "cubism_c", "sketch_b", "sketch_c"]:
        line = f"  {condition:16s}"
        for model in MODELS:
            confusion = load(model, condition)
            ship = confusion[8, 0] / confusion[8].sum() * 100   # ship -> airplane
            dog = confusion[5, 6] / confusion[5].sum() * 100    # dog -> frog
            line += f"   {MODELS[model]} {ship:5.1f} {dog:5.1f}"
        print(line)


""" Part 2: figures """

# Numbers from the Part 1 output, columns in ORDER unless it says otherwise
ACCURACY = {   # overall accuracy (%)
    "resnet": [92.57, 79.82, 73.29, 67.02, 63.47, 54.84, 53.03, 49.07],
    "vit": [98.47, 90.17, 88.80, 86.99, 88.10, 82.60, 73.14, 78.81],
}
EXTRA_PAINTING_ACCURACY = {   # paintings B and C (%), the dots on the sketch and cubism bars
    "resnet": {"sketch": [60.25, 59.50], "cubism": [58.51, 51.23]},
    "vit": {"sketch": [74.61, 74.35], "cubism": [84.89, 83.84]},
}
PER_CLASS_ACCURACY = {   # recall (%), one row per class
    "resnet": [
        [ 92.2,  87.1,  69.3,  78.1,  82.9,  90.0,  28.8,  77.6],   # airplane
        [ 96.8,  87.6,  84.1,  81.6,  77.4,  78.5,  63.3,  76.6],   # automobile
        [ 88.3,  68.6,  63.8,  64.4,  68.0,  46.6,  36.2,  58.8],   # bird
        [ 83.4,  79.3,  71.2,  69.2,  49.1,  70.7,  86.3,  39.4],   # cat
        [ 94.9,  78.7,  63.8,  46.2,  53.3,  18.7,  52.1,  13.7],   # deer
        [ 86.2,  76.6,  76.0,  49.9,  33.0,  36.8,  63.2,   9.6],   # dog
        [ 98.4,  72.9,  70.6,  79.8,  58.0,  35.0,  43.7,  73.2],   # frog
        [ 93.1,  81.7,  87.3,  79.2,  58.5,  62.0,  51.4,  46.5],   # horse
        [ 97.0,  76.6,  57.4,  32.8,  68.7,  39.6,  43.8,  19.4],   # ship
        [ 95.4,  89.1,  89.4,  89.0,  85.8,  70.5,  61.5,  75.9],   # truck
    ],
    "vit": [
        [ 99.2,  94.8,  89.5,  88.2,  94.4,  93.8,  62.0,  80.9],   # airplane
        [ 98.9,  92.7,  91.4,  91.7,  91.3,  89.6,  76.9,  88.0],   # automobile
        [ 97.9,  91.4,  88.4,  89.6,  90.7,  90.2,  64.0,  84.5],   # bird
        [ 97.1,  91.6,  91.9,  88.6,  82.0,  77.3,  88.1,  69.2],   # cat
        [ 99.1,  86.3,  83.8,  85.0,  85.8,  75.7,  68.3,  77.2],   # deer
        [ 96.9,  82.1,  85.8,  76.5,  77.3,  64.3,  84.9,  66.6],   # dog
        [ 99.6,  91.5,  86.0,  92.3,  92.1,  88.5,  67.2,  95.6],   # frog
        [ 98.5,  89.1,  90.4,  85.3,  83.7,  76.9,  67.9,  76.8],   # horse
        [ 99.5,  88.2,  86.9,  80.2,  89.6,  86.3,  72.6,  60.2],   # ship
        [ 98.0,  94.0,  93.9,  92.5,  94.1,  83.4,  79.5,  89.1],   # truck
    ],
}
RHO = {   # rho per class (columns = CLASSES), rows in ORDER
    "resnet": [
        [0.98, 1.02, 0.94, 0.96, 1.07, 0.92, 1.12, 0.96, 1.03, 1.00],   # baseline
        [1.18, 0.95, 0.80, 1.48, 1.01, 1.05, 0.80, 0.92, 0.83, 0.99],   # alpha0
        [1.01, 0.92, 0.86, 1.44, 0.84, 1.17, 0.96, 1.16, 0.61, 1.04],   # oil_painting
        [1.46, 0.98, 0.92, 1.51, 0.58, 0.65, 1.43, 1.04, 0.35, 1.09],   # abstract
        [1.92, 0.94, 1.30, 0.96, 0.74, 0.39, 0.91, 0.65, 0.92, 1.28],   # post_impression
        [3.00, 0.94, 0.75, 2.01, 0.22, 0.49, 0.47, 0.86, 0.44, 0.83],   # watercolor
        [0.33, 0.70, 0.54, 3.74, 1.04, 1.14, 0.66, 0.62, 0.55, 0.68],   # sketch
        [2.25, 1.13, 1.37, 1.08, 0.16, 0.11, 2.16, 0.53, 0.24, 0.97],   # cubism
    ],
    "vit": [
        [1.00, 1.01, 0.99, 1.01, 1.00, 1.00, 1.01, 0.99, 1.00, 0.99],   # baseline
        [1.07, 0.95, 1.09, 1.25, 0.93, 0.94, 0.98, 0.92, 0.92, 0.98],   # alpha0
        [0.97, 0.94, 1.03, 1.38, 0.90, 1.00, 0.97, 0.93, 0.91, 0.99],   # oil_painting
        [0.98, 0.94, 1.09, 1.34, 0.94, 0.86, 1.18, 0.88, 0.82, 0.97],   # abstract
        [1.13, 0.94, 1.12, 1.09, 0.97, 0.86, 1.07, 0.85, 0.94, 1.02],   # post_impression
        [1.19, 0.95, 1.30, 1.20, 0.92, 0.73, 1.13, 0.81, 0.91, 0.87],   # watercolor
        [0.67, 0.78, 0.72, 2.34, 0.92, 1.46, 0.75, 0.70, 0.81, 0.85],   # sketch
        [1.01, 0.93, 1.23, 0.97, 0.92, 0.74, 1.83, 0.78, 0.63, 0.97],   # cubism
    ],
}
EXTRA_PAINTING_RHO = {   # rho per class for paintings B and C
    "resnet": {
        "cubism_b": [1.59, 0.95, 0.83, 1.13, 0.28, 0.55, 1.57, 1.13, 0.28, 1.70],
        "cubism_c": [2.64, 0.91, 0.39, 1.85, 0.10, 0.51, 0.51, 1.28, 0.24, 1.58],
        "sketch_b": [0.66, 0.86, 0.49, 3.07, 0.87, 1.39, 0.39, 0.96, 0.52, 0.77],
        "sketch_c": [0.79, 0.92, 0.51, 3.30, 0.61, 1.25, 0.21, 0.90, 0.77, 0.75],
    },
    "vit": {
        "cubism_b": [1.07, 0.93, 1.08, 1.21, 0.88, 0.95, 1.25, 0.88, 0.73, 1.04],
        "cubism_c": [1.21, 0.92, 1.00, 1.28, 0.79, 1.03, 1.06, 0.92, 0.75, 1.04],
        "sketch_b": [0.83, 0.82, 0.77, 2.35, 0.70, 1.53, 0.65, 0.83, 0.72, 0.80],
        "sketch_c": [0.89, 0.88, 0.80, 2.37, 0.58, 1.49, 0.49, 0.80, 0.86, 0.84],
    },
}
CUBISM_CONFUSION = {   # % of each true class (rows) predicted as each class (columns), cubism A
    "resnet": [
        [77.6,  4.4,  2.9,  1.7,  0.2,  0.0, 10.1,  0.1,  1.9,  1.1],   # airplane
        [ 9.6, 76.6,  1.2,  1.7,  0.0,  0.1,  4.3,  0.0,  0.3,  6.2],   # automobile
        [16.1,  1.1, 58.8,  5.7,  0.4,  0.2, 16.1,  0.4,  0.3,  0.9],   # bird
        [10.2,  2.1, 16.7, 39.4,  0.1,  0.4, 27.8,  0.4,  0.9,  2.0],   # cat
        [19.4,  2.7, 14.4, 15.9, 13.7,  0.3, 28.0,  3.3,  0.3,  2.0],   # deer
        [ 7.0,  1.3, 24.8, 21.9,  0.2,  9.6, 31.5,  1.5,  0.4,  1.8],   # dog
        [ 8.3,  1.0,  6.2,  9.7,  0.0,  0.0, 73.2,  0.2,  0.3,  1.1],   # frog
        [17.3,  2.0,  9.3,  7.1,  1.4,  0.2, 13.8, 46.5,  0.3,  2.1],   # horse
        [50.5, 13.4,  1.6,  3.2,  0.0,  0.0,  7.8,  0.0, 19.4,  4.1],   # ship
        [ 9.4,  8.7,  1.0,  1.5,  0.0,  0.0,  3.1,  0.2,  0.2, 75.9],   # truck
    ],
    "vit": [
        [80.9,  0.0,  3.8,  2.0,  1.9,  0.3,  9.2,  0.0,  1.1,  0.8],   # airplane
        [ 0.6, 88.0,  0.7,  0.9,  0.1,  0.1,  4.6,  0.0,  0.8,  4.2],   # automobile
        [ 2.2,  0.0, 84.5,  2.9,  2.2,  0.7,  7.3,  0.1,  0.1,  0.0],   # bird
        [ 1.3,  0.2,  6.9, 69.2,  1.5,  3.5, 16.8,  0.1,  0.3,  0.2],   # cat
        [ 1.1,  0.0,  7.7,  3.3, 77.2,  0.8,  9.0,  0.5,  0.1,  0.3],   # deer
        [ 0.8,  0.1,  8.0, 10.5,  2.4, 66.6, 10.9,  0.7,  0.0,  0.0],   # dog
        [ 0.4,  0.0,  2.1,  1.5,  0.2,  0.2, 95.6,  0.0,  0.0,  0.0],   # frog
        [ 0.9,  0.0,  5.8,  3.5,  5.3,  1.3,  6.2, 76.8,  0.0,  0.2],   # horse
        [10.2,  1.4,  2.9,  1.9,  0.9,  0.0, 20.2,  0.0, 60.2,  2.3],   # ship
        [ 2.1,  3.0,  0.9,  1.0,  0.1,  0.1,  3.1,  0.2,  0.4, 89.1],   # truck
    ],
}

# Points lost: pipeline (baseline - control), then style effect (control - style) for ORDER[2:]
EFFECTS = {}
EXTRA_PAINTING_EFFECTS = {}
for model in MODELS:
    a = ACCURACY[model]
    EFFECTS[model] = [round(a[0] - a[1], 2)]
    for i in range(2, len(ORDER)):
        EFFECTS[model].append(round(a[1] - a[i], 2))
    EXTRA_PAINTING_EFFECTS[model] = {}
    for style in EXTRA_PAINTING_ACCURACY[model]:
        EXTRA_PAINTING_EFFECTS[model][style] = [round(a[1] - b, 2) for b in EXTRA_PAINTING_ACCURACY[model][style]]

# Rows of Fig. 7
PAINTING_ROWS = ["alpha0", "cubism", "cubism_b", "cubism_c", "sketch", "sketch_b", "sketch_c"]
PAINTING_RHO = {}
for model in MODELS:
    PAINTING_RHO[model] = []
    for condition in PAINTING_ROWS:
        if condition in ORDER:
            PAINTING_RHO[model].append(RHO[model][ORDER.index(condition)])
        else:
            PAINTING_RHO[model].append(EXTRA_PAINTING_RHO[model][condition])

HIGHLIGHTED_ERRORS = [("ship", "airplane"), ("dog", "frog")]   # (true, predicted), outlined in Fig. 6

# Labels
NAMES = {
    "baseline": "Baseline",
    "alpha0": "Control (α = 0)",
    "oil_painting": "Oil Painting",
    "abstract": "Abstract",
    "post_impression": "Post-Impressionism",
    "watercolor": "Watercolor",
    "sketch": "Sketch",
    "cubism": "Cubism"
}
BAR_NAMES = {**NAMES, "alpha0": "Control\n(α = 0)", "oil_painting": "Oil\nPainting", "post_impression": "Post-\nImpressionism"}
PAINTING_FILES = {
    "cubism": "cubism.jpg",
    "cubism_b": "cubism_b.jpg",
    "cubism_c": "cubism_c.jpg",
    "sketch": "sketch.jpeg",
    "sketch_b": "sketch_b.jpg",
    "sketch_c": "sketch_c.jpg"
}

# Colors
COLORS = {"resnet": "#2a78d6", "vit": "#eb6834"}
INK = "#1a1a19"
MUTED = "#6b6a65"
GRID = "#e4e3df"
ACCURACY_CMAP = matplotlib.colormaps["viridis_r"]
CONFUSION_CMAP = LinearSegmentedColormap.from_list("conf", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
RHO_CMAP = LinearSegmentedColormap.from_list("rho", ["#1c5cab", "#ffffff", "#b8322f"])   # white at 1x
RHO_NORM = LogNorm(vmin=0.25, vmax=4)

PAGE_WIDTH = 6.5   # inches

plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 8.5,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = f"{FIGURES_DIR}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def hide_frame(ax):
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def stylized_image(condition, index, dataset):
    if condition == "baseline":
        return dataset[index][0]
    path = f"{DATASETS[condition]}/{dataset.classes[dataset[index][1]]}/{index}.png"
    if os.path.exists(path):
        return Image.open(path)
    return None


def fig1_examples():
    examples = [0, 1]   # CIFAR-10 test images 0 (cat) and 1 (ship)
    dataset = CIFAR10(
        root="project/Images",
        train=False,
        download=False
    )

    top_height = 0.8 * len(examples) + 0.35
    fig = plt.figure(figsize=(PAGE_WIDTH, top_height + 2.9))
    top, bottom = fig.subfigures(2, 1, height_ratios=[top_height, 2.9])

    # Top: examples under every condition
    axes = top.subplots(len(examples), len(ORDER), squeeze=False)
    top.subplots_adjust(left=0.04, right=1, top=0.82, bottom=0.02, wspace=0.08, hspace=0.08)
    for row, index in enumerate(examples):
        for col, condition in enumerate(ORDER):
            ax = axes[row, col]
            image = stylized_image(condition, index, dataset)
            if image is not None:
                ax.imshow(image, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(BAR_NAMES[condition], fontsize=6.5)
        axes[row, 0].set_ylabel(dataset.classes[dataset[index][1]].capitalize(), fontsize=7)

    # Bottom: each painting and the first example stylized with it
    axes = bottom.subplots(2, 6)
    bottom.subplots_adjust(left=0.04, right=1, top=0.9, bottom=0.02, wspace=0.12, hspace=0.35)
    for row, style in enumerate(PAINTINGS):
        for slot, painting in enumerate(PAINTINGS[style]):
            painting_ax = axes[row, 2 * slot]
            output_ax = axes[row, 2 * slot + 1]

            painting_ax.imshow(Image.open(f"project/Style_Transfer/styles/{PAINTING_FILES[painting]}").convert("RGB"))
            painting_ax.set_title(PAINTING_NAMES[painting], fontsize=6.5)
            output_ax.imshow(stylized_image(painting, examples[0], dataset), interpolation="nearest")
            output_ax.set_title("output", fontsize=6.5, color=MUTED)

            for ax in (painting_ax, output_ax):
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    save(fig, "fig1_examples.png")


def fig2_accuracy():
    x = np.arange(len(ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.0))

    for offset, model in zip([-width / 2, width / 2], MODELS):
        bars = ax.bar(x + offset, ACCURACY[model], width - 0.03, color=COLORS[model], label=MODELS[model], zorder=2)
        bars[1].set_hatch("////")
        bars[1].set_edgecolor("white")

        for i, bar in enumerate(bars):
            value = ACCURACY[model][i]
            center = bar.get_x() + bar.get_width() / 2
            if i == 1:   # label above the hatched bar
                ax.text(center, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=5.8, color=INK,
                        fontweight="bold")
            else:
                ax.text(center, value - 1.5, f"{value:.1f}", ha="center", va="top", fontsize=5.8, color="white",
                        fontweight="bold")

        # Other paintings
        for style in EXTRA_PAINTING_ACCURACY[model]:
            ax.scatter(x[ORDER.index(style)] + offset + np.array([-0.06, 0.06]), EXTRA_PAINTING_ACCURACY[model][style],
                       s=16, color="white", edgecolor=INK, linewidth=0.9, zorder=3)

    ax.set_xticks(x, [BAR_NAMES[c] for c in ORDER], fontsize=7)
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    dot = Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white", markeredgecolor=INK)
    ax.legend([Rectangle((0, 0), 1, 1, color=COLORS["resnet"]), Rectangle((0, 0), 1, 1, color=COLORS["vit"]), dot],
              ["ResNet-50", "ViT-B/16", "Other reference paintings"],
              frameon=False, loc="upper center", ncols=3, bbox_to_anchor=(0.5, 1.12), fontsize=7)
    save(fig, "fig2_accuracy.png")


def fig3_effects():
    groups = ["pipeline"] + ORDER[2:]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.9))

    for offset, model in zip([-width / 2, width / 2], MODELS):
        bars = ax.bar(x + offset, EFFECTS[model], width - 0.03, color=COLORS[model], zorder=2)
        bars[0].set_hatch("////")
        bars[0].set_edgecolor("white")
        for bar, value in zip(bars, EFFECTS[model]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=6)

        # Other paintings
        for style in EXTRA_PAINTING_EFFECTS[model]:
            ax.scatter(x[groups.index(style)] + offset + np.array([-0.06, 0.06]), EXTRA_PAINTING_EFFECTS[model][style],
                       s=16, color="white", edgecolor=INK, linewidth=0.9, zorder=3)

    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x, ["Pipeline\n(baseline →\ncontrol)"] + [BAR_NAMES[c] for c in ORDER[2:]], fontsize=7)
    ax.text(3.5, 1.02, "Style effect (control → style)", transform=ax.get_xaxis_transform(), ha="center",
            fontsize=7, color=MUTED)
    ax.set_ylabel("Accuracy lost (points)")
    ax.set_ylim(0, 35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)

    dot = Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white", markeredgecolor=INK)
    ax.legend([Rectangle((0, 0), 1, 1, color=COLORS["resnet"]), Rectangle((0, 0), 1, 1, color=COLORS["vit"]), dot],
              ["ResNet-50", "ViT-B/16", "Other reference paintings"],
              frameon=False, loc="upper left", bbox_to_anchor=(0.17, 1), fontsize=7)
    save(fig, "fig3_effects.png")


def fig4_per_class():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), sharey=True)

    for ax, model in zip(axes, MODELS):
        grid = np.array(PER_CLASS_ACCURACY[model])
        image = ax.imshow(grid, cmap=ACCURACY_CMAP, vmin=0, vmax=100, aspect="auto")
        for row in range(10):
            for col in range(len(ORDER)):
                value = grid[row, col]
                ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=5.2,
                        color="white" if value >= 55 else INK)
        ax.set_title(MODELS[model])
        ax.set_xticks(range(len(ORDER)), [NAMES[c] for c in ORDER], rotation=40, ha="right",
                      rotation_mode="anchor", fontsize=6.5)
        ax.set_yticks(range(10), [c.capitalize() for c in CLASSES], fontsize=6.5)
        hide_frame(ax)

    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.85, pad=0.02, aspect=25)
    colorbar.set_label("Per-class accuracy (recall, %)")
    colorbar.outline.set_visible(False)
    save(fig, "fig4_per_class.png")


# ResNet on top, ViT below
def rho_heatmap(values, rows, labels, name):
    panel_height = 0.24 * len(rows) + 0.35
    fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 2 * panel_height + 0.6), sharex=True,
                             gridspec_kw={"hspace": 0.28})

    for ax, model in zip(axes, MODELS):
        grid = np.array(values[model])
        image = ax.imshow(np.clip(grid, 0.25, 4), cmap=RHO_CMAP, norm=RHO_NORM, aspect="auto")
        for row, condition in enumerate(rows):
            found = attractors(model, condition)
            for col in range(10):
                value = grid[row, col]
                boxed = found[col]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.3,
                        fontweight="bold" if boxed else "normal",
                        color="white" if value >= 2.6 or value <= 0.38 else INK)
                if boxed:
                    ax.add_patch(Rectangle((col - 0.45, row - 0.42), 0.9, 0.84, fill=False, edgecolor=INK, linewidth=1.1))
        ax.set_title(MODELS[model], loc="left")
        ax.set_xticks(range(10), [c.capitalize() for c in CLASSES], fontsize=6.8, rotation=25, ha="right",
                      rotation_mode="anchor")
        ax.set_yticks(range(len(rows)), labels, fontsize=6.8)
        hide_frame(ax)

    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.7, pad=0.02, aspect=25)
    colorbar.ax.minorticks_off()
    colorbar.set_ticks([0.25, 0.5, 1, 2, 4], labels=["≤0.25×", "0.5×", "1×", "2×", "≥4×"])
    colorbar.set_label(r"$\rho_c$ (predicted ÷ expected)")
    colorbar.outline.set_visible(False)
    save(fig, name)


def fig6_confusion_cubism():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2), sharey=True)

    for ax, model in zip(axes, MODELS):
        grid = np.array(CUBISM_CONFUSION[model])
        image = ax.imshow(grid, cmap=CONFUSION_CMAP, vmin=0, vmax=100, aspect="equal")
        for row in range(10):
            for col in range(10):
                ax.text(col, row, f"{grid[row, col]:.1f}", ha="center", va="center", fontsize=5.2,
                        fontweight="bold" if row == col else "normal",
                        color="white" if grid[row, col] >= 55 else INK)
            ax.add_patch(Rectangle((row - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=0.8))

        # Red boxes on the confusions in the text
        for true, predicted in HIGHLIGHTED_ERRORS:
            ax.add_patch(Rectangle((CLASSES.index(predicted) - 0.46, CLASSES.index(true) - 0.46), 0.92, 0.92,
                                   fill=False, edgecolor="#b8322f", linewidth=1.5))

        ax.set_title(f"{MODELS[model]}, cubism ({ACCURACY[model][ORDER.index('cubism')]:.2f}%)")
        ax.set_xticks(range(10), [c.capitalize() for c in CLASSES], rotation=40, ha="right",
                      rotation_mode="anchor", fontsize=6.2)
        ax.set_yticks(range(10), [c.capitalize() for c in CLASSES], fontsize=6.2)
        ax.set_xlabel("Predicted class", fontsize=7)
        hide_frame(ax)
    axes[0].set_ylabel("True class", fontsize=7)

    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.8, pad=0.02, aspect=25)
    colorbar.set_label("Share of true class (%)")
    colorbar.outline.set_visible(False)
    save(fig, "fig6_confusion_cubism.png")


# Part 1
make_csvs()
make_tables()
print_discussion_numbers()

# Part 2
os.makedirs(FIGURES_DIR, exist_ok=True)
fig1_examples()
fig2_accuracy()
fig3_effects()
fig4_per_class()
rho_heatmap(RHO, ORDER, [NAMES[c] for c in ORDER], "fig5_attractors.png")
fig6_confusion_cubism()
labels = [NAMES["alpha0"]] + [PAINTING_NAMES[p] for p in PAINTING_ROWS[1:]]
rho_heatmap(PAINTING_RHO, PAINTING_ROWS, labels, "fig7_paintings_rho.png")
