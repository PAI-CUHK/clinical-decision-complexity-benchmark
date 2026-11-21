#!/usr/bin/env python3
"""Plot per-model accuracy at each of the 10 levels for all nine problems.

Usage:
    python -m analysis.plot_problem_level_accuracy
    python -m analysis.plot_problem_level_accuracy --out plot/my_accuracy.png

Each cell is correct / total for one model, problem, and level.
Missing results remain blank.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Support both ``python -m analysis.plot_problem_level_accuracy`` and direct
# execution as ``python analysis/plot_problem_level_accuracy.py``.
if __package__:
    from .plot_data import DATA_DIR
    from .figure_style import apply_style, save_figure
    from .plot_model_performance import MODEL_NAMES
    from .task_labels import TASK_SHORT_NAMES
else:
    from plot_data import DATA_DIR
    from figure_style import apply_style, save_figure
    from plot_model_performance import MODEL_NAMES
    from task_labels import TASK_SHORT_NAMES


# Row 1: P; row 2: NP-complete; row 3: NP-hard.
PROBLEMS = (
    ("P", "ECG_Pattern_Detection", TASK_SHORT_NAMES["ECG_Pattern_Detection"]),
    ("P", "ED_Scheduling", TASK_SHORT_NAMES["ED_Scheduling"]),
    ("P", "Medical_Risk", TASK_SHORT_NAMES["Medical_Risk"]),
    ("NP-complete", "Disease_Diagnosis_NP-complete", TASK_SHORT_NAMES["Disease_Diagnosis_NP-complete"]),
    ("NP-complete", "Nurse_Scheduling", TASK_SHORT_NAMES["Nurse_Scheduling"]),
    ("NP-complete", "OR_Scheduling_NP-complete", TASK_SHORT_NAMES["OR_Scheduling_NP-complete"]),
    ("NP-hard", "Disease_Diagnosis", TASK_SHORT_NAMES["Disease_Diagnosis"]),
    ("NP-hard", "OR_Scheduling", TASK_SHORT_NAMES["OR_Scheduling"]),
    ("NP-hard", "PASP", TASK_SHORT_NAMES["PASP"]),
)
PROBLEM_LABELS = {problem: label for _, problem, label in PROBLEMS}
LEVELS = range(1, 11)






def read_csv(path: Path):
    """Restore integer counts; do not use the rounded accuracy column."""
    by_label = {label: problem for _, problem, label in PROBLEMS}
    models = []
    counts = defaultdict(lambda: [0, 0])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if model not in models:
                models.append(model)
            key = (model, by_label[row["problem"]], int(row["level"]))
            counts[key] = [int(row["correct"]), int(row["total"])]
    return models, counts


def plot(path: Path, models, counts):
    apply_style()
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("white")
    layouts = [(PROBLEMS, (3, 3), (21, 17), path)]
    for problems, shape, size, output in layouts:
        fig, axes = plt.subplots(*shape, figsize=size)
        for ax, (_, problem, label) in zip(np.ravel(axes), problems):
            values = np.full((len(models), len(LEVELS)), np.nan)
            for row, model in enumerate(models):
                for col, level in enumerate(LEVELS):
                    correct, total = counts[(model, problem, level)]
                    if total:
                        values[row, col] = correct / total
            image = ax.imshow(values, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_title(label, pad=9, loc="left", fontsize=18, fontweight="bold")
            ax.set_xticks(range(10), [f"L{level}" for level in LEVELS])
            ax.set_yticks(range(len(models)), [MODEL_NAMES.get(m, m) for m in models])
            ax.tick_params(length=0)
            ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.8)
            ax.tick_params(which="minor", bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            for row in range(len(models)):
                for col in range(10):
                    value = values[row, col]
                    if np.isfinite(value):
                        ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                                fontsize=8, color="white" if value >= 0.48 else "#202020")
        if shape == (3, 3):
            fig.subplots_adjust(left=0.11, right=0.925, top=0.97, bottom=0.06,
                                wspace=0.34, hspace=0.46)
            colorbar_ax = fig.add_axes([0.945, 0.25, 0.014, 0.50])
        else:
            fig.subplots_adjust(left=0.22, right=0.89, top=0.96, bottom=0.07,
                                hspace=0.50)
            colorbar_ax = fig.add_axes([0.92, 0.25, 0.018, 0.50])
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.outline.set_visible(False)
        colorbar.set_label("Accuracy")
        colorbar.set_ticks(np.linspace(0, 1, 6))
        save_figure(fig, output)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=DATA_DIR / "problem_level_accuracy.csv")
    parser.add_argument("--out", type=Path,
                        default=Path("plot/problem_level_accuracy.png"))
    args = parser.parse_args()

    models, counts = read_csv(args.data)
    plot(args.out, models, counts)
    print(f"绘图完成: {args.out}")
    print(f"问题数: {len(PROBLEMS)}，模型数: {len(models)}，结果数: {sum(v[1] for v in counts.values())}")


if __name__ == "__main__":
    main()
