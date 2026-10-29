"""
Turns the saved evaluation results into every number, table and figure in the paper.

Run from the repo root after ResnetInput.py and ViTInput.py:
    .venv/bin/python project/results.py
    .venv/bin/python project/results.py --pairs ship:airplane dog:frog cat:dog --examples 0 1

Part 1 computes from project/results/*.npz and writes to project/results/:
    summary.csv            accuracy and degradation per model and condition (Eq. 1, Eq. 3), also from the alpha = 0 control
    per_class.csv          P / R / F1 and attractor ratio rho_c per class (Eq. 2, Eq. 4)
    confusion_pairs.csv    the largest off-diagonal confusions per model and condition
    robustness.csv         accuracy and top attractor for every cubism and sketch painting
    tables/table1_baseline_control.csv, tables/table2_robustness.csv
and prints the shared-attractor, control and painting comparisons.

Part 2 draws the figures from numbers typed into this file (copied from the Part 1 output, final results)
and writes to project/results/figures/ (all 6.5 in wide, for a single-column page):
    fig1_examples.png  fig2_accuracy.png  fig3_effects.png  fig4_per_class.png
    fig5_attractors.png  fig6_confusion_cubism.png  fig7_paintings_rho.png
If a result ever changes, update the matching block in Part 2 by hand.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision.datasets import CIFAR10

from Evaluate import CLASSES, DATASETS as CONDITIONS, RESULTS_DIR


# =====================================================================================================
# Part 1: numbers computed from the saved results
# =====================================================================================================

FIG_DIR = f"{RESULTS_DIR}/figures"
TABLE_DIR = f"{RESULTS_DIR}/tables"
PAINTING_NAMES = {"cubism": "Cubism A", "cubism_b": "Cubism B", "cubism_c": "Cubism C",
                  "sketch": "Sketch A", "sketch_b": "Sketch B", "sketch_c": "Sketch C"}


def fmt(value, digits=1):
    # 12.75 is stored as 12.7499999..., so nudge before rounding to match the paper's 2-decimal values
    return f"{value + 1e-9:.{digits}f}"

MODELS = {"resnet": "ResNet-50", "vit": "ViT-B/16"}
STYLES = ["oil_painting", "abstract", "post_impression", "watercolor", "sketch", "cubism"]   # paper order
CONTROLS = ["resize_only", "alpha0"]
# Every reference painting per style; the first is the one in the original paper
PAINTINGS = {"cubism": ["cubism", "cubism_b", "cubism_c"], "sketch": ["sketch", "sketch_b", "sketch_c"]}

# A class is an attractor when rho_c >= ATTRACTOR and it sits at least CONTROL_MARGIN above the same
# class under the alpha = 0 control, so over-prediction the pipeline already causes isn't credited to style.
# Both are choices: state them in Methods if you change them.
ATTRACTOR = 1.5
CONTROL_MARGIN = 0.25


def load_confusions():
    confusions = {}
    for model in MODELS:
        for condition in CONDITIONS:
            path = f"{RESULTS_DIR}/{model}__{condition}.npz"
            if os.path.exists(path):
                confusions[model, condition] = np.load(path)["confusion"]
    return confusions


def accuracy(confusion):
    return np.trace(confusion) / confusion.sum()


def per_class(confusion):
    tp = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0)          # column sums
    actual = confusion.sum(axis=1)             # row sums
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        recall = tp / actual
        f1 = 2 * precision * recall / (precision + recall)
    rho = predicted / (confusion.sum() / len(CLASSES))   # Eq. 4
    return precision, recall, f1, predicted, rho


def attractors(confusions, model, condition):
    """Boolean mask over CLASSES: rho_c >= ATTRACTOR and clearly above the control's rho_c for that class."""
    rho = per_class(confusions[model, condition])[4]
    mask = rho >= ATTRACTOR
    if (model, "alpha0") in confusions:
        mask &= rho - per_class(confusions[model, "alpha0"])[4] >= CONTROL_MARGIN
    return mask


def write_csv(name, header, rows):
    with open(f"{RESULTS_DIR}/{name}", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def analysis(args):
    confusions = load_confusions()
    if not confusions:
        raise SystemExit("No results yet. Run project/ResnetInput.py and project/ViTInput.py first.")
    present = [c for c in CONDITIONS if any((m, c) in confusions for m in MODELS)]

    # Accuracy, degradation (Eq. 3) and per-class metrics
    summary, class_rows, pair_rows = [], [], []
    for model in MODELS:
        if (model, "baseline") not in confusions:
            continue
        base = accuracy(confusions[model, "baseline"])
        control = accuracy(confusions[model, "alpha0"]) if (model, "alpha0") in confusions else None
        style_accs = []
        for condition in present:
            if (model, condition) not in confusions:
                continue
            confusion = confusions[model, condition]
            acc = accuracy(confusion)
            if condition in STYLES:
                style_accs.append(acc)
            summary.append([model, condition, round(acc * 100, 2), round((base - acc) * 100, 2),
                            "" if control is None else round((control - acc) * 100, 2)])

            precision, recall, f1, predicted, rho = per_class(confusion)
            for k, name in enumerate(CLASSES):
                class_rows.append([model, condition, name, int(predicted[k]), round(rho[k], 3),
                                   round(precision[k] * 100, 1), round(recall[k] * 100, 1), round(f1[k] * 100, 1)])

            off = confusion.copy()
            np.fill_diagonal(off, 0)
            for flat in np.argsort(off, axis=None)[::-1][:5]:
                true, pred = np.unravel_index(flat, off.shape)
                pair_rows.append([model, condition, CLASSES[true], CLASSES[pred], int(off[true, pred]),
                                  round(off[true, pred] / confusion[true].sum() * 100, 1)])

        if len(style_accs) == len(STYLES):
            print(f"{MODELS[model]}: baseline {base * 100:.2f}%, style mean {np.mean(style_accs) * 100:.2f}%, "
                  f"population σ {np.std(style_accs) * 100:.2f}")

    write_csv("summary.csv", ["model", "condition", "accuracy_%", "degradation_pts", "drop_from_alpha0_pts"], summary)
    write_csv("per_class.csv", ["model", "condition", "class", "predicted", "rho", "precision_%", "recall_%", "f1_%"], class_rows)
    write_csv("confusion_pairs.csv", ["model", "condition", "true", "predicted", "count", "share_of_true_%"], pair_rows)

    # Do the two architectures collapse into the same classes?
    print(f"\nShared attractors (ρ_c ≥ {ATTRACTOR} and ≥ control + {CONTROL_MARGIN}, in both models) "
          "and agreement of the two ρ profiles:")
    for condition in present:
        if ("resnet", condition) not in confusions or ("vit", condition) not in confusions:
            continue
        rho_r = per_class(confusions["resnet", condition])[4]
        rho_v = per_class(confusions["vit", condition])[4]
        r = np.corrcoef(np.log(np.maximum(rho_r, 1e-3)), np.log(np.maximum(rho_v, 1e-3)))[0, 1]
        att_r = attractors(confusions, "resnet", condition)
        att_v = attractors(confusions, "vit", condition)
        shared = [CLASSES[k] for k in range(len(CLASSES)) if att_r[k] and att_v[k]]
        only_r = [CLASSES[k] for k in range(len(CLASSES)) if att_r[k] and not att_v[k]]
        only_v = [CLASSES[k] for k in range(len(CLASSES)) if att_v[k] and not att_r[k]]
        print(f"  {condition:16s} r(log ρ) = {r:+.2f}   shared: {shared or '-'}   ResNet only: {only_r or '-'}   ViT only: {only_v or '-'}")

    # How much of the drop is the pipeline rather than the style?
    if any(c in present for c in CONTROLS):
        print("\nControl ladder (accuracy %): baseline -> resize only -> AdaIN α=0 -> style mean")
        for model in MODELS:
            steps = [f"{accuracy(confusions[model, c]) * 100:.2f}" if (model, c) in confusions else "n/a"
                     for c in ["baseline"] + CONTROLS]
            styles = [accuracy(confusions[model, c]) for c in STYLES if (model, c) in confusions]
            print(f"  {MODELS[model]:10s} " + " -> ".join(steps) + f" -> {np.mean(styles) * 100:.2f}")

    # Are the results a property of the style or of one painting?
    robustness = []
    for style, paintings in PAINTINGS.items():
        rows = [p for p in paintings if all((m, p) in confusions for m in MODELS)]
        if len(rows) < 2:
            continue
        print(f"\nRobustness across {style} paintings (accuracy %, top attractor):")
        for painting in rows:
            row = [style, painting]
            text = f"  {painting:10s}"
            for model in MODELS:
                rho = per_class(confusions[model, painting])[4]
                top = int(np.argmax(rho))
                acc = accuracy(confusions[model, painting]) * 100
                counts = bool(attractors(confusions, model, painting)[top])
                row += [round(acc, 2), CLASSES[top], round(rho[top], 2), "yes" if counts else "no"]
                text += f"   {MODELS[model]} {acc:5.2f}  {CLASSES[top]} {rho[top]:.2f}×{'' if counts else ' (not an attractor)'}"
            robustness.append(row)
            print(text)
        for model in MODELS:
            accs = [accuracy(confusions[model, p]) * 100 for p in rows]
            print(f"  {MODELS[model]:10s} mean {np.mean(accs):.2f}, range {min(accs):.2f}–{max(accs):.2f}")
    if robustness:
        write_csv("robustness.csv", ["style", "painting"] + [f"{m}_{col}" for m in MODELS
                  for col in ["accuracy_%", "top_class", "rho", "is_attractor"]], robustness)

        # Does the flip (donor under sketch, attractor under cubism) hold for every painting?
        k = CLASSES.index(args.flip_class)
        print(f"\n{args.flip_class.capitalize()} ρ_c for every painting:")
        for model in MODELS:
            cells = [f"{p} {per_class(confusions[model, p])[4][k]:.2f}×"
                     for style in PAINTINGS for p in PAINTINGS[style] if (model, p) in confusions]
            print(f"  {MODELS[model]:10s} " + "   ".join(cells))

    # Track specific confusions across every condition (for the "why" claims)
    if args.pairs:
        print("\nTracked confusions (% of the true class sent to the predicted class):")
        print(f"  {'condition':16s}" + "".join(f"{m + ' ' + p:>26s}" for m in MODELS for p in args.pairs))
        for condition in present:
            cells = []
            for model in MODELS:
                for pair in args.pairs:
                    true, pred = (CLASSES.index(x) for x in pair.split(":"))
                    if (model, condition) in confusions:
                        confusion = confusions[model, condition]
                        cells.append(f"{confusion[true, pred] / confusion[true].sum() * 100:.1f}")
                    else:
                        cells.append("n/a")
            print(f"  {condition:16s}" + "".join(f"{c:>26s}" for c in cells))


def write_table(name, rows):
    path = f"{TABLE_DIR}/{name}"
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote {path}")


def tables(confusions):
    results = {m: np.load(f"{RESULTS_DIR}/{m}__baseline.npz") for m in MODELS}
    write_table("table1_baseline_control.csv", [
        ["", *MODELS.values()],
        ["Baseline accuracy (%)", *[fmt(accuracy(confusions[m, "baseline"]) * 100, 2) for m in MODELS]],
        ["Control accuracy, α = 0 (%)", *[fmt(accuracy(confusions[m, "alpha0"]) * 100, 2) for m in MODELS]],
        ["Parameters", *[f"{int(results[m]['parameters']):,}" for m in MODELS]],
        ["Inference time, 10,000 images (s)", *[fmt(float(results[m]["seconds"]), 2) for m in MODELS]],
        ["Images per second", *[fmt(10000 / float(results[m]["seconds"])) for m in MODELS]],
    ])

    rows = [["Style", "Painting"] + [f"{MODELS[m]} {col}" for m in MODELS
                                     for col in ["accuracy (%)", "attractors (ρ_c)"]]]
    for style, paintings in PAINTINGS.items():
        for painting in paintings:
            if not all((m, painting) in confusions for m in MODELS):
                continue
            row = [style.capitalize(), PAINTING_NAMES[painting]]
            for model in MODELS:
                rho = per_class(confusions[model, painting])[4]
                # Every class that counts as an attractor (the outlined cells in Fig. 7), largest first
                boxed = sorted(np.where(attractors(confusions, model, painting))[0], key=lambda k: -rho[k])
                row += [fmt(accuracy(confusions[model, painting]) * 100, 2),
                        ", ".join(f"{CLASSES[k].capitalize()} {fmt(rho[k], 2)}" for k in boxed) or "None"]
            rows.append(row)
        accs = {m: [accuracy(confusions[m, p]) * 100 for p in paintings if (m, p) in confusions] for m in MODELS}
        if all(len(a) > 1 for a in accs.values()):
            row = [style.capitalize(), "Mean (range)"]
            for model in MODELS:
                a = accs[model]
                row += [f"{fmt(np.mean(a), 2)} ({fmt(min(a), 2)}–{fmt(max(a), 2)})", ""]
            rows.append(row)
    write_table("table2_robustness.csv", rows)


# =====================================================================================================
# Part 2: figures, drawn from the numbers below
# =====================================================================================================

ORDER = ["baseline", "alpha0", "oil_painting", "abstract", "post_impression", "watercolor", "sketch", "cubism"]
PAINTING_ROWS = ["alpha0", "cubism", "cubism_b", "cubism_c", "sketch", "sketch_b", "sketch_c"]

# ----- Final numbers, copied from the Part 1 output (summary.csv, per_class.csv) -----

ACCURACY = {   # overall accuracy (%), columns in ORDER
    "resnet": [92.57, 79.82, 73.29, 67.02, 63.47, 54.84, 53.03, 49.07],
    "vit": [98.47, 90.17, 88.80, 86.99, 88.10, 82.60, 73.14, 78.81],
}
EXTRA_PAINTING_ACCURACY = {   # paintings B and C (%), drawn as dots on the sketch and cubism bars
    "resnet": {"sketch": [60.25, 59.50], "cubism": [58.51, 51.23]},
    "vit": {"sketch": [74.61, 74.35], "cubism": [84.89, 83.84]},
}
EFFECTS = {   # points lost: pipeline (baseline - control), then style effect (control - style) for ORDER[2:]
    "resnet": [12.75, 6.53, 12.80, 16.35, 24.98, 26.79, 30.75],
    "vit": [8.30, 1.37, 3.18, 2.07, 7.57, 17.03, 11.36],
}
EXTRA_PAINTING_EFFECTS = {   # style effect (control - painting) for paintings B and C, drawn as dots in Fig. 3
    "resnet": {"sketch": [19.57, 20.32], "cubism": [21.31, 28.59]},
    "vit": {"sketch": [15.56, 15.82], "cubism": [5.28, 6.33]},
}
HIGHLIGHTED_ERRORS = [("ship", "airplane"), ("dog", "frog")]   # (true, predicted) cells outlined in Fig. 6
PER_CLASS_ACCURACY = {   # recall (%), one row per class, columns in ORDER
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
RHO = {   # rho_c per class (columns = CLASSES), rows in ORDER
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
RHO_OUTLINED = {   # (row, class) cells that count as attractors by the rule in Part 1
    "resnet": [('post_impression', 'airplane'), ('watercolor', 'airplane'), ('watercolor', 'cat'), ('sketch', 'cat'), ('cubism', 'airplane'), ('cubism', 'frog')],
    "vit": [('sketch', 'cat'), ('cubism', 'frog')],
}
PAINTING_RHO = {   # rho_c per class for the control and every cubism/sketch painting, rows in PAINTING_ROWS
    "resnet": [
        [1.18, 0.95, 0.80, 1.48, 1.01, 1.05, 0.80, 0.92, 0.83, 0.99],   # alpha0
        [2.25, 1.13, 1.37, 1.08, 0.16, 0.11, 2.16, 0.53, 0.24, 0.97],   # cubism
        [1.59, 0.95, 0.83, 1.13, 0.28, 0.55, 1.57, 1.13, 0.28, 1.70],   # cubism_b
        [2.64, 0.91, 0.39, 1.85, 0.10, 0.51, 0.51, 1.28, 0.24, 1.58],   # cubism_c
        [0.33, 0.70, 0.54, 3.74, 1.04, 1.14, 0.66, 0.62, 0.55, 0.68],   # sketch
        [0.66, 0.86, 0.49, 3.07, 0.87, 1.39, 0.39, 0.96, 0.52, 0.77],   # sketch_b
        [0.79, 0.92, 0.51, 3.30, 0.61, 1.25, 0.21, 0.90, 0.77, 0.75],   # sketch_c
    ],
    "vit": [
        [1.07, 0.95, 1.09, 1.25, 0.93, 0.94, 0.98, 0.92, 0.92, 0.98],   # alpha0
        [1.01, 0.93, 1.23, 0.97, 0.92, 0.74, 1.83, 0.78, 0.63, 0.97],   # cubism
        [1.07, 0.93, 1.08, 1.21, 0.88, 0.95, 1.25, 0.88, 0.73, 1.04],   # cubism_b
        [1.21, 0.92, 1.00, 1.28, 0.79, 1.03, 1.06, 0.92, 0.75, 1.04],   # cubism_c
        [0.67, 0.78, 0.72, 2.34, 0.92, 1.46, 0.75, 0.70, 0.81, 0.85],   # sketch
        [0.83, 0.82, 0.77, 2.35, 0.70, 1.53, 0.65, 0.83, 0.72, 0.80],   # sketch_b
        [0.89, 0.88, 0.80, 2.37, 0.58, 1.49, 0.49, 0.80, 0.86, 0.84],   # sketch_c
    ],
}
PAINTING_RHO_OUTLINED = {   # (row, class) cells that count as attractors by the rule in Part 1
    "resnet": [('cubism', 'airplane'), ('cubism', 'frog'), ('cubism_b', 'airplane'), ('cubism_b', 'frog'), ('cubism_b', 'truck'), ('cubism_c', 'airplane'), ('cubism_c', 'cat'), ('cubism_c', 'truck'), ('sketch', 'cat'), ('sketch_b', 'cat'), ('sketch_c', 'cat')],
    "vit": [('cubism', 'frog'), ('sketch', 'cat'), ('sketch_b', 'cat'), ('sketch_b', 'dog'), ('sketch_c', 'cat')],
}
CUBISM_CONFUSION = {   # % of each true class (rows) predicted as each class (columns), cubism painting A
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

STYLES_DIR = "project/Style_Transfer/styles"
PAGE_WIDTH = 6.5   # inches

NAMES = {"baseline": "Baseline", "alpha0": "Control (α = 0)", "oil_painting": "Oil Painting", "abstract": "Abstract",
         "post_impression": "Post-Impressionism", "watercolor": "Watercolor", "sketch": "Sketch", "cubism": "Cubism"}
BAR_NAMES = {**NAMES, "alpha0": "Control\n(α = 0)", "oil_painting": "Oil\nPainting", "post_impression": "Post-\nImpressionism"}
PAINTING_FILES = {"cubism": "cubism.jpg", "cubism_b": "cubism_b.jpg", "cubism_c": "cubism_c.jpg",
                  "sketch": "sketch.jpeg", "sketch_b": "sketch_b.jpg", "sketch_c": "sketch_c.jpg"}

COLORS = {"resnet": "#2a78d6", "vit": "#eb6834"}   # ResNet blue, ViT orange, in every figure
INK, MUTED, GRID = "#1a1a19", "#6b6a65", "#e4e3df"
# Per-class accuracy: viridis, light (low) -> dark (high); one direction, no arbitrary midpoint, colorblind-safe
ACCURACY_CMAP = matplotlib.colormaps["viridis_r"]
# Confusion matrices: white -> dark blue
CONFUSION_CMAP = LinearSegmentedColormap.from_list("conf", ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
# rho_c: blue (under-predicted) -> white at exactly 1x -> red (over-predicted); the log scale centers 0.25-4 on 1
RHO_CMAP = LinearSegmentedColormap.from_list("rho", ["#1c5cab", "#ffffff", "#b8322f"])
RHO_NORM = LogNorm(vmin=0.25, vmax=4)
ERROR_PATH_COLOR = "#b8322f"   # outlines the confusions the text discusses in Fig. 6
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 8.5,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def accuracy_text_color(value):
    return "white" if value >= 55 else INK   # viridis_r turns dark teal/purple above ~55%


def confusion_text_color(value):
    return "white" if value >= 55 else INK


def save(fig, name):
    path = f"{FIG_DIR}/{name}"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def clean_heatmap_axes(ax):
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def stylized(condition, index, dataset):
    if condition == "baseline":
        return dataset[index][0]
    path = f"{CONDITIONS[condition]}/{dataset.classes[dataset[index][1]]}/{index}.png"
    return Image.open(path) if os.path.exists(path) else None


def fig1_examples(examples):
    dataset = CIFAR10(root="project/Images", train=False, download=False)
    top_height = 0.8 * len(examples) + 0.35
    fig = plt.figure(figsize=(PAGE_WIDTH, top_height + 2.9))
    top, bottom = fig.subfigures(2, 1, height_ratios=[top_height, 2.9])

    axes = top.subplots(len(examples), len(ORDER), squeeze=False)
    top.subplots_adjust(left=0.04, right=1, top=0.82, bottom=0.02, wspace=0.08, hspace=0.08)
    for row, index in enumerate(examples):
        for col, condition in enumerate(ORDER):
            ax = axes[row, col]
            image = stylized(condition, index, dataset)
            if image is not None:
                ax.imshow(image, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(BAR_NAMES[condition], fontsize=6.5)
        axes[row, 0].set_ylabel(dataset.classes[dataset[index][1]].capitalize(), fontsize=7)

    # Every reference painting next to the first example stylized with it
    axes = bottom.subplots(2, 6)
    bottom.subplots_adjust(left=0.04, right=1, top=0.9, bottom=0.02, wspace=0.12, hspace=0.35)
    for row, style in enumerate(PAINTINGS):
        for slot, painting in enumerate(PAINTINGS[style]):
            art_ax, out_ax = axes[row, 2 * slot], axes[row, 2 * slot + 1]
            art_path = f"{STYLES_DIR}/{PAINTING_FILES[painting]}"
            output = stylized(painting, examples[0], dataset) if os.path.isdir(CONDITIONS[painting]) else None
            if os.path.exists(art_path):
                art_ax.imshow(Image.open(art_path).convert("RGB"))
            else:
                art_ax.text(0.5, 0.5, "painting\nnot added", ha="center", va="center", fontsize=6, color=MUTED,
                            transform=art_ax.transAxes)
            if output is not None:
                out_ax.imshow(output, interpolation="nearest")
            else:
                out_ax.text(0.5, 0.5, "not yet\ngenerated", ha="center", va="center", fontsize=6, color=MUTED,
                            transform=out_ax.transAxes)
            art_ax.set_title(PAINTING_NAMES[painting], fontsize=6.5)
            out_ax.set_title("output", fontsize=6.5, color=MUTED)
            for ax in (art_ax, out_ax):
                ax.set_xticks([]), ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    save(fig, "fig1_examples.png")


def fig2_accuracy():
    x = np.arange(len(ORDER))
    width = 0.38
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 3.0))
    for offset, model in zip([-width / 2, width / 2], MODELS):
        values = ACCURACY[model]
        bars = ax.bar(x + offset, values, width - 0.03, color=COLORS[model], label=MODELS[model], zorder=2)
        bars[1].set_hatch("////")
        bars[1].set_edgecolor("white")
        # Values sit just inside the bar tops, leaving the space above free for the extra-painting dots
        for i, (bar, value) in enumerate(zip(bars, values)):
            center = bar.get_x() + bar.get_width() / 2
            if i == 1:   # hatched control bars: label above the bar, where it stays readable
                ax.text(center, value + 1, fmt(value), ha="center", va="bottom", fontsize=5.8, color=INK,
                        fontweight="bold")
            else:
                ax.text(center, value - 1.5, fmt(value), ha="center", va="top", fontsize=5.8, color="white",
                        fontweight="bold")
        for style in EXTRA_PAINTING_ACCURACY[model]:
            extra = EXTRA_PAINTING_ACCURACY[model][style]
            jitter = np.linspace(-0.06, 0.06, len(extra)) if len(extra) > 1 else np.zeros(len(extra))
            ax.scatter(x[ORDER.index(style)] + offset + jitter, extra, s=16, color="white", edgecolor=INK,
                       linewidth=0.9, zorder=3, label="Other reference paintings" if model == "resnet" and
                       style == "cubism" else None)
    ax.set_xticks(x, [BAR_NAMES[c] for c in ORDER], fontsize=7)
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(MODELS["resnet"]), labels.index(MODELS["vit"])] + \
            [i for i, l in enumerate(labels) if l not in MODELS.values()]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, loc="upper center",
              ncols=3, bbox_to_anchor=(0.5, 1.12), fontsize=7)
    save(fig, "fig2_accuracy.png")


def fig3_effects():
    groups = ["pipeline"] + ORDER[2:]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(PAGE_WIDTH, 2.9))
    for offset, model in zip([-width / 2, width / 2], MODELS):
        values = EFFECTS[model]
        bars = ax.bar(x + offset, values, width - 0.03, color=COLORS[model], label=MODELS[model], zorder=2)
        bars[0].set_hatch("////")
        bars[0].set_edgecolor("white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, fmt(value), ha="center", va="bottom", fontsize=6)
        # Same dots as Fig. 2: style effect for the other sketch and cubism paintings
        for style, extra in EXTRA_PAINTING_EFFECTS[model].items():
            ax.scatter(x[groups.index(style)] + offset + np.linspace(-0.06, 0.06, len(extra)), extra, s=16,
                       color="white", edgecolor=INK, linewidth=0.9, zorder=3)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(x, ["Pipeline\n(baseline →\ncontrol)"] + [BAR_NAMES[c] for c in ORDER[2:]], fontsize=7)
    ax.text(3.5, 1.02, "Style effect (control → style)", transform=ax.get_xaxis_transform(), ha="center",
            fontsize=7, color=MUTED)
    ax.set_ylabel("Accuracy lost (points)")
    ax.set_ylim(0, 35)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    dot = Line2D([], [], marker="o", linestyle="", markersize=4, markerfacecolor="white", markeredgecolor=INK)
    ax.legend([Rectangle((0, 0), 1, 1, color=COLORS[m]) for m in MODELS] + [dot],
              list(MODELS.values()) + ["Other reference paintings"],
              frameon=False, loc="upper left", bbox_to_anchor=(0.17, 1), fontsize=7)
    save(fig, "fig3_effects.png")


def fig4_per_class():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.3), sharey=True)
    for ax, model in zip(axes, MODELS):
        grid = np.array(PER_CLASS_ACCURACY[model])
        image = ax.imshow(grid, cmap=ACCURACY_CMAP, vmin=0, vmax=100, aspect="auto")
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                ax.text(col, row, fmt(grid[row, col]), ha="center", va="center", fontsize=5.2,
                        color=accuracy_text_color(grid[row, col]))
        ax.set_title(MODELS[model])
        ax.set_xticks(range(len(ORDER)), [NAMES[c] for c in ORDER], rotation=40, ha="right",
                      rotation_mode="anchor", fontsize=6.5)
        ax.set_yticks(range(len(CLASSES)), [c.capitalize() for c in CLASSES], fontsize=6.5)
        clean_heatmap_axes(ax)
    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.85, pad=0.02, aspect=25)
    colorbar.set_label("Per-class accuracy (recall, %)")
    colorbar.outline.set_visible(False)
    save(fig, "fig4_per_class.png")


def rho_heatmap(values, outlined_cells, rows, labels, name):
    # Panels stacked so each gets the full page width; side by side, two-decimal values run together
    panel_height = 0.24 * len(rows) + 0.35
    fig, axes = plt.subplots(2, 1, figsize=(PAGE_WIDTH, 2 * panel_height + 0.6), sharex=True,
                             gridspec_kw={"hspace": 0.28})
    for ax, model in zip(axes, MODELS):
        grid = np.array(values[model])
        image = ax.imshow(np.clip(grid, 0.25, 4), cmap=RHO_CMAP, norm=RHO_NORM, aspect="auto")
        for row, condition in enumerate(rows):
            outlined = [(condition, name) in outlined_cells[model] for name in CLASSES]
            for col in range(len(CLASSES)):
                value = grid[row, col]
                ax.text(col, row, fmt(value, 2), ha="center", va="center", fontsize=6.3,
                        fontweight="bold" if outlined[col] else "normal",
                        color="white" if value >= 2.6 or value <= 0.38 else INK)
                if outlined[col]:
                    ax.add_patch(Rectangle((col - 0.45, row - 0.42), 0.9, 0.84, fill=False, edgecolor=INK,
                                           linewidth=1.1))
        ax.set_title(MODELS[model], loc="left")
        ax.set_xticks(range(len(CLASSES)), [c.capitalize() for c in CLASSES], fontsize=6.8, rotation=25,
                      ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(rows)), labels, fontsize=6.8)
        clean_heatmap_axes(ax)
    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.7, pad=0.02, aspect=25)
    colorbar.ax.minorticks_off()
    colorbar.set_ticks([0.25, 0.5, 1, 2, 4], labels=["≤0.25×", "0.5×", "1×", "2×", "≥4×"])
    colorbar.set_label(r"$\rho_c$ (predicted ÷ expected)")
    colorbar.outline.set_visible(False)
    save(fig, name)


def fig5_attractors():
    rho_heatmap(RHO, RHO_OUTLINED, ORDER, [NAMES[c] for c in ORDER], "fig5_attractors.png")


def fig7_paintings_rho():
    rho_heatmap(PAINTING_RHO, PAINTING_RHO_OUTLINED, PAINTING_ROWS,
                [NAMES["alpha0"]] + [PAINTING_NAMES[p] for p in PAINTING_ROWS[1:]],
                "fig7_paintings_rho.png")


def fig6_confusion_cubism():
    fig, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, 3.2), sharey=True)
    for ax, model in zip(axes, MODELS):
        grid = np.array(CUBISM_CONFUSION[model])
        image = ax.imshow(grid, cmap=CONFUSION_CMAP, vmin=0, vmax=100, aspect="equal")
        for row in range(10):
            for col in range(10):
                ax.text(col, row, fmt(grid[row, col]), ha="center", va="center", fontsize=5.2,
                        fontweight="bold" if row == col else "normal",
                        color=confusion_text_color(grid[row, col]))
            ax.add_patch(Rectangle((row - 0.5, row - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=0.8))
        for true, predicted in HIGHLIGHTED_ERRORS:
            ax.add_patch(Rectangle((CLASSES.index(predicted) - 0.46, CLASSES.index(true) - 0.46), 0.92, 0.92,
                                   fill=False, edgecolor=ERROR_PATH_COLOR, linewidth=1.5))
        ax.set_title(f"{MODELS[model]}, cubism ({fmt(ACCURACY[model][ORDER.index('cubism')], 2)}%)")
        ax.set_xticks(range(10), [c.capitalize() for c in CLASSES], rotation=40, ha="right",
                      rotation_mode="anchor", fontsize=6.2)
        ax.set_yticks(range(10), [c.capitalize() for c in CLASSES], fontsize=6.2)
        ax.set_xlabel("Predicted class", fontsize=7)
        clean_heatmap_axes(ax)
    axes[0].set_ylabel("True class", fontsize=7)
    colorbar = fig.colorbar(image, ax=axes.tolist(), shrink=0.8, pad=0.02, aspect=25)
    colorbar.set_label("Share of true class (%)")
    colorbar.outline.set_visible(False)
    save(fig, "fig6_confusion_cubism.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="*", default=["ship:airplane", "dog:frog"],
                        help="true:predicted class pairs to track across every condition")
    parser.add_argument("--flip-class", default="airplane",
                        help="class whose rho_c is compared across every cubism and sketch painting")
    parser.add_argument("--examples", nargs="+", type=int, default=[0, 1],
                        help="CIFAR-10 test indices to show in Fig. 1 (default: 0 = cat, 1 = ship)")
    args = parser.parse_args()

    # Part 1
    analysis(args)
    os.makedirs(TABLE_DIR, exist_ok=True)
    tables(load_confusions())

    # Part 2
    os.makedirs(FIG_DIR, exist_ok=True)
    fig1_examples(args.examples)
    fig2_accuracy()
    fig3_effects()
    fig4_per_class()
    fig5_attractors()
    fig6_confusion_cubism()
    fig7_paintings_rho()


if __name__ == "__main__":
    main()
