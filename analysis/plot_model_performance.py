#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical NP-problem benchmark: multi-model performance figure.

Reads saved intermediate data and outputs only
weighted accuracy (WA) in the figure and table, grouped by NP-Complexity
(P / NP-Complete / NP-Hard).

Metric definitions (per the task specification)
-----------------------------------------------
Each result JSON carries a ``complexity`` field in {P, NP-complete, NP-hard}
and a difficulty ``Level`` (1..10) parsed from the file name.

Weighted Accuracy (WA) for a complexity class c:

        WA_c = sum_i ( w_i * Acc_i ) / sum_i w_i

    where i indexes the difficulty level (w_i = i, level 1 -> weight 1, ...),
    Acc_i = correct_i / total_i is the accuracy at level i, and only levels with
    data are summed. A problem counts as "correct" when the validator marks it
    ``valid`` and the raw output is not an error stub.

The black "All models" curve is the unweighted mean over models present in a
given complexity class.

Usage:
    python -m analysis.plot_model_performance [--data FILE] [--out FILE]
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if __package__:
    from .figure_style import apply_style, save_figure
    from .plot_data import load_data
else:
    from figure_style import apply_style, save_figure
    from plot_data import load_data

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_OUT = "plot/model_performance.png"

# order of the x axis, and its display labels
COMPLEXITY_ORDER = ["P", "NP-complete", "NP-hard"]
COMPLEXITY_LABELS = ["P", "NP-Complete", "NP-Hard"]


# Pretty display names for the model folders.
MODEL_NAMES = {
    "gpt_claude-sonnet-5": "Claude Sonnet 5",
    "gpt_gpt-5.6-sol-ca": "GPT 5.6 Sol",
    "gpt_qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "vllm_Mistral_4": "Mistral Small 4 119B 2603",
    "vllm_Qwen3_6_35B_A3B": "Qwen3.6 35B A3B",
    "vllm_Qwen3_8_27B": "Qwen3.8 27B",
    "vllm_deepseek_v4_flash": "DeepSeek V4 Flash 0731",
    "vllm_gpt_oss_120b": "GPT OSS 120B",
}
MODEL_ORDER = (
    "gpt_gpt-5.6-sol-ca",
    "gpt_claude-sonnet-5",
    "vllm_deepseek_v4_flash",
    "gpt_qwen3.5-397b-a17b",
    "vllm_Qwen3_6_35B_A3B",
    "vllm_Qwen3_8_27B",
    "vllm_Mistral_4",
    "vllm_gpt_oss_120b",
)
MODEL_MARKERS = {
    "gpt_gpt-5.6-sol-ca": "^",
    "gpt_claude-sonnet-5": "^",
}
MODEL_FOLDERS_BY_NAME = {name: folder for folder, name in MODEL_NAMES.items()}

# Stable, muted colors inspired by the supplied reference. Every model uses an
# open square marker so color, rather than marker shape, identifies the series.
MODEL_STYLES = [
    "#3182BD", "#9DBCE5", "#8B6BCB", "#C5A6D8",
    "#39A845", "#91D982", "#C49A8F", "#9B6D5D",
]


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #






# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _series(models, key):
    """2-D array [n_models, 3] of a metric, keeping NaN for missing classes."""
    data = np.full((len(models), len(COMPLEXITY_ORDER)), np.nan)
    for i, (_, metrics) in enumerate(models):
        for j, c in enumerate(COMPLEXITY_ORDER):
            data[i, j] = metrics[c][key]
    return data


def plot(models, out_path: str):
    apply_style()
    x = np.arange(len(COMPLEXITY_ORDER))
    wa = _series(models, "WA")

    # One WA panel, with a right-side model legend and a black aggregate curve.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    handles = []
    # Individual models: thin dash-dot lines with open square markers.
    for i, (name, _) in enumerate(models):
        color = MODEL_STYLES[i % len(MODEL_STYLES)]
        marker = MODEL_MARKERS.get(MODEL_FOLDERS_BY_NAME.get(name), "s")
        (line,) = ax.plot(
            x, wa[i], color=color, marker=marker, markersize=4.6,
            markerfacecolor="none", markeredgewidth=1.1,
            linewidth=1.0, linestyle="-.", alpha=0.95,
            label=name, zorder=2,
        )
        handles.append((line, name))

    # Aggregate "All models" = unweighted mean across models (ignoring NaN).
    with np.errstate(invalid="ignore"):
        agg = np.nanmean(wa, axis=0)
    agg_line, = ax.plot(
        x, agg, color="#000000", marker="o", markersize=5,
        markerfacecolor="#000000", markeredgecolor="#000000",
        linewidth=2.0, linestyle="-", label="All models", zorder=5,
    )
    for xi, val in zip(x, agg):
        if not np.isnan(val):
            ax.annotate(
                f"{val:.2f}", (xi, val),
                textcoords="offset points", xytext=(7, 5),
                ha="left", fontsize=8.5, color="#111111", zorder=6,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(["P", "NP-Complete", "NP-Hard"])
    ax.set_xlabel("Complexity class")
    ax.set_ylabel("Weighted accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.12, len(COMPLEXITY_ORDER) - 0.88)
    ax.tick_params(labelsize=8, direction="out", length=3, color="#333333")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#666666")
        spine.set_linewidth(0.8)

    legend_handles = [h for h, _ in handles] + [agg_line]
    legend_labels = [n for _, n in handles] + ["All models"]
    fig.legend(
        legend_handles, legend_labels,
        loc="center left", bbox_to_anchor=(0.72, 0.5), ncol=1,
        title="Model", fontsize=8, title_fontsize=9,
        frameon=True, edgecolor="#BBBBBB", facecolor="white",
        framealpha=1, labelspacing=0.42, handlelength=2.0,
    )
    fig.subplots_adjust(left=0.11, right=0.67, top=0.92, bottom=0.22)

    save_figure(fig, out_path)
    plt.close(fig)


def print_table(models):
    print(f"{'Model':<20}" + "".join(f"{lbl:>18}" for lbl in COMPLEXITY_LABELS))
    print(f"{'':<20}" + "".join(f"{'WA':>18}" for _ in COMPLEXITY_LABELS))
    print("-" * (20 + 18 * len(COMPLEXITY_LABELS)))
    for name, metrics in models:
        row = f"{name:<20}"
        for c in COMPLEXITY_ORDER:
            m = metrics[c]
            wa = f"{m['WA']:.3f}" if not np.isnan(m["WA"]) else "  -  "
            row += f"{wa:>18}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Plot multi-model performance figure")
    parser.add_argument("--data", help="intermediate model_performance.json")
    parser.add_argument("--out", default=DEFAULT_OUT, help="output image path")
    args = parser.parse_args()

    models = load_data("model_performance", args.data)["models"]
    if not models:
        raise SystemExit("No models in intermediate data")

    print_table(models)
    plot(models, args.out)
    print(f"\nSaved figure -> {args.out}")


if __name__ == "__main__":
    main()
