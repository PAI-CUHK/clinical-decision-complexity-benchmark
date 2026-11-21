#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-composition figure for the medical NP benchmark.

Reads intermediate error counts. The plot has:

    three side-by-side panels = complexity classes P / NP-Complete / NP-Hard;
    one horizontal 100%-stacked bar per model (plus an aggregated "All models"
    row), with the model names as row labels and the failure counts in an
    ``n`` column on the right;
    each bar's segments = error types TE / OF / HE / HV / OE / Unclassified,
    using the Okabe-Ito colour-blind-safe palette.

Because the classifier assigns exactly one label per invalid instance, the
stacked segments partition that model's failures at that complexity. The grey
``Unclassified`` band is the remainder (valid=False instances for which no
pattern matched -- mainly plain "wrong answers" on P / NP-complete problems).

Output: plot/error_composition.{png,svg,pdf}
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .figure_style import apply_style
from .plot_model_performance import MODEL_NAMES, MODEL_ORDER
from .plot_data import load_data
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_OUT = "plot/error_composition.png"

COMPLEXITY_ORDER = ["P", "NP-complete", "NP-hard"]
COMPLEXITY_LABELS = ["P", "NP-C", "NP-H"]
PANEL_TAGS = ["P", "NP-Complete", "NP-Hard"]
ERROR_ORDER = ["TE", "OF", "HE", "HV", "OE", "Unclassified"]
ERROR_LABELS = {
    "TE": "TE: Token-limit / no output",
    "OF": "OF: Output-format error",
    "HV": "HV: Hard-constraint violation",
    "OE": "OE: Optimization error (suboptimal)",
    "HE": "HE: Hallucination Error",
    "Unclassified": "Other invalid answers",
}
# Column-major order so the 3x2 legend reads:
#   TE | OF | HE
#   HV | OE | Unclassified
LEGEND_ORDER = ["TE", "HV", "OF", "OE", "HE", "Unclassified"]
# Okabe-Ito colour-blind-safe palette
ERROR_COLORS = {
    "TE": "#0072B2",
    "OF": "#E69F00",
    "HV": "#D55E00",
    "OE": "#009E73",
    "HE": "#CC79A7",
    "Unclassified": "#D0D3D8",
}



# --------------------------------------------------------------------------- #
# Data collection: {model: {complexity: {error_type: count}}}
# --------------------------------------------------------------------------- #


def _round_pcts(fracs):
    """Integer percentages summing to 100 (largest-remainder method)."""
    scaled = [f * 100.0 for f in fracs]
    base = [int(x) for x in scaled]
    need = 100 - sum(base)
    order = sorted(range(len(fracs)), key=lambda i: scaled[i] - base[i], reverse=True)
    for k in range(max(0, need)):
        base[order[k % len(order)]] += 1
    return base


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot(models, out_path: str, formats):
    folders = [folder for folder in MODEL_ORDER if folder in models]
    folders.extend(folder for folder in models if folder not in MODEL_ORDER)
    names = [MODEL_NAMES.get(f, f) for f in folders] + ["All models"]

    # aggregated "All models" row: sum raw counts, then re-normalise per bar
    agg = {c: {"errors": defaultdict(int), "total": 0} for c in COMPLEXITY_ORDER}
    for folder in folders:
        for c in COMPLEXITY_ORDER:
            agg[c]["total"] += models[folder][c].get("total", 0)
            for e, v in models[folder][c].get("errors", {}).items():
                agg[c]["errors"][e] += v

    rows = [models[f] for f in folders]
    rows.append({c: {"errors": dict(agg[c]["errors"]), "total": agg[c]["total"]}
                 for c in COMPLEXITY_ORDER})

    nrows = len(rows)
    ypos = np.arange(nrows)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 6.4), sharey=True,
                             gridspec_kw={"wspace": 0.40})

    for ax, complexity, tag in zip(axes, COMPLEXITY_ORDER, PANEL_TAGS):
        for i, per_comp in enumerate(rows):
            cell = per_comp[complexity]
            total = sum(max(0, v) for v in cell.get("errors", {}).values())
            if total == 0:
                ax.text(50, ypos[i], "not run", ha="center", va="center",
                        fontsize=8, color="#999999", zorder=3)
                ax.text(103, ypos[i], "\u2013", ha="left", va="center",
                        fontsize=8.5, color="#666666")
                continue

            fracs = [max(0, cell["errors"].get(e, 0)) / total for e in ERROR_ORDER]
            pcts = _round_pcts(fracs)
            left = 0.0
            for k, e in enumerate(ERROR_ORDER):
                w = fracs[k] * 100.0
                if w <= 0:
                    continue
                ax.barh(ypos[i], w, left=left, height=0.62, color=ERROR_COLORS[e],
                        edgecolor="white", linewidth=0.6, zorder=2)
                if w >= 8.0:  # label only when the segment is wide enough
                    ax.text(left + w / 2, ypos[i], f"{pcts[k]}%", ha="center",
                            va="center", fontsize=7.6, fontweight="bold",
                            color="#3A3A3A" if e == "Unclassified" else "white",
                            zorder=3)
                left += w

            ax.text(103, ypos[i], f"{total}", ha="left", va="center",
                    fontsize=8.5, color="#333333")

        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_ylim(nrows - 0.5, -0.5)
        ax.set_title(tag, fontsize=12, fontweight="bold", pad=14)
        ax.set_xlabel("Share of failures (%)", fontsize=10)
        ax.grid(axis="x", color="#EDEDED", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#B5B5B5")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=9, color="#B5B5B5")
        ax.text(1.031, 1.02, "n", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=9.5, color="#333333")

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(names, fontsize=9)
    for lbl in axes[0].get_yticklabels():
        if lbl.get_text() == "All models":
            lbl.set_fontweight("bold")

    handles = [Patch(facecolor=ERROR_COLORS[e], label=ERROR_LABELS[e]) for e in LEGEND_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.065),
               handlelength=1.3, columnspacing=2.6, labelspacing=0.6)

    # Explicit margins: bottom leaves a dedicated band for the 2-row legend so
    # the x-axis labels can never collide with it.
    fig.subplots_adjust(left=0.12, right=0.96, top=0.89, bottom=0.24,
                        wspace=0.40)

    base, _ = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Error-composition figure (all models)")
    ap.add_argument("--data", help="intermediate error_composition.json")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--formats", nargs="+", default=["png", "svg", "pdf"])
    args = ap.parse_args()

    models = load_data("error_composition", args.data)["models"]
    if not models:
        raise SystemExit("No models found.")

    # summary table
    print(f"\n{'Model':<20}" + "".join(f"{l:>34}" for l in COMPLEXITY_LABELS))
    for folder, per_comp in models.items():
        row = f"{MODEL_NAMES.get(folder, folder):<20}"
        for c in COMPLEXITY_ORDER:
            cell = per_comp[c]
            if cell["total"] == 0:
                row += f"{'not run':>34}"
                continue
            errs = cell["errors"]
            fails = sum(max(0, v) for v in errs.values())
            parts = " ".join(f"{e}={errs[e]}" for e in ERROR_ORDER if errs.get(e, 0))
            row += f"{(parts + (' (0 fail)' if fails == 0 else '')):>34}"
        print(row)

    plot(models, args.out, args.formats)
    print(f"\nSaved figure -> {os.path.splitext(args.out)[0]}.{{{','.join(args.formats)}}}")


if __name__ == "__main__":
    main()
