"""Compare weighted accuracy with and without medical semantics.

Run from the repository root::

    python -m analysis.plot_medical_semantics_wa

WA weights level 1..10 accuracies
weighted by their level number. An answer counts as correct when ``valid``
is true and the raw model output is not an API error.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from .figure_style import apply_style, isolate_axes
from .task_labels import TASK_SHORT_NAMES
from .plot_data import load_data
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ("Medical semantics", "No medical semantics")
OUT = ROOT / "plot/medical_semantics_wa"
PANEL_OUT = OUT / "medical_semantics_wa_comparison_panels"
PROBLEMS = [
    "Disease_Diagnosis", "Disease_Diagnosis_NP-complete",
    "ECG_Pattern_Detection", "ED_Scheduling", "Medical_Risk",
    "Nurse_Scheduling", "OR_Scheduling", "OR_Scheduling_NP-complete", "PASP",
]
COMPLEXITY = {
    "Disease_Diagnosis": "NP-hard",
    "Disease_Diagnosis_NP-complete": "NP-complete",
    "ECG_Pattern_Detection": "P",
    "ED_Scheduling": "P",
    "Medical_Risk": "P",
    "Nurse_Scheduling": "NP-complete",
    "OR_Scheduling": "NP-hard",
    "OR_Scheduling_NP-complete": "NP-complete",
    "PASP": "NP-hard",
}
MODEL_LABELS = {
    "gpt_gpt-5.6-sol-ca": "GPT 5.6 Sol",
    "gpt_claude-sonnet-5": "Claude Sonnet 5",
    "vllm_deepseek_v4_flash": "DeepSeek V4 Flash 0731",
    "gpt_qwen3.5-397b-a17b": "Qwen3.5 397B A17B",
    "vllm_Qwen3_6_35B_A3B": "Qwen3.6 35B A3B",
    "vllm_Qwen3_8_27B": "Qwen3.8 27B",
    "vllm_Mistral_4": "Mistral Small 4 119B 2603",
    "vllm_gpt_oss_120b": "GPT OSS 120B",
}
COLORS = ["#277DA1", "#E76F51"]










def value(summary, scope, condition, **filters):
    matches = [r for r in summary if r["scope"] == scope and r["condition"] == condition
               and all(r[k] == v for k, v in filters.items())]
    if len(matches) != 1:
        raise ValueError((scope, condition, filters, matches))
    return matches[0]["wa_percent"]


def save_panel_crops(fig, panels):
    """Save each panel without its title as a separate, tightly cropped figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    for filename, axes in panels:
        titles = [ax.get_title(loc="left") for ax in axes]
        for ax in axes:
            ax.set_title("", loc="left")
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = Bbox.union([ax.get_tightbbox(renderer) for ax in axes])
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted()).expanded(1.06, 1.08)
        base = PANEL_OUT / filename
        with isolate_axes(fig, axes):
            for extension in ("png", "pdf", "svg"):
                fig.savefig(base.with_suffix(f".{extension}"), bbox_inches=bbox,
                            dpi=600 if extension == "png" else None)
        for ax, title in zip(axes, titles):
            ax.set_title(title, loc="left")


def draw_model_change(ax, models, model_delta):
    """Use the standalone panel's styling and plot aspect in every export."""
    y = np.arange(len(models))
    ax.barh(y, model_delta, color="#D4654B", height=.65)
    ax.set_yticks(y, [MODEL_LABELS[model] for model in models])
    ax.invert_yaxis()
    ax.set_xlim(-1, 6)
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_xlabel("Change in WA (pp)", fontsize=8, labelpad=4)
    for i, val in enumerate(model_delta):
        ax.text(val + .10, i, f"{val:+.1f}", va="center", fontsize=8.5)
    ax.axvline(0, color="#65717D", linewidth=.9)
    ax.grid(axis="x", color="#E8EDF2", linewidth=.7)
    ax.set_axisbelow(True)
    ax.spines[:].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5, length=0)
    ax.set_box_aspect((3.7 * .81) / (5.6 * .53))


def save_standalone_model_change(models, model_delta):
    """Render the model-change panel in a narrower standalone layout."""
    fig, ax = plt.subplots(figsize=(5.6, 3.7))
    draw_model_change(ax, models, model_delta)
    fig.subplots_adjust(left=.43, right=.96, top=.97, bottom=.16)
    base = PANEL_OUT / "b_model_change"
    for extension in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{extension}"),
                    dpi=600 if extension == "png" else None)
    plt.close(fig)


def plot(summary):
    """Make a change-first dashboard, with absolute scores for context."""
    apply_style()
    fig = plt.figure(figsize=(19, 15))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.35, 2.7, 2.0],
                            left=.12, right=.95, top=.97, bottom=.13, hspace=.36)
    top = grid[0].subgridspec(1, 3, width_ratios=[1.0, 1.9, 1.25], wspace=.48)
    ax_overall, ax_model, ax_class = [fig.add_subplot(top[0, i]) for i in range(3)]
    ax_matrix = fig.add_subplot(grid[1])
    ax_problem = fig.add_subplot(grid[2])
    conds = list(RESULTS)
    overall = [value(summary, "overall", c) for c in conds]
    bars = ax_overall.bar([0, 1], overall, color=COLORS, width=.6)
    ax_overall.set_xticks([0, 1], ["Medical semantics", "No medical semantics"])
    ax_overall.set_ylim(0, 55)
    ax_overall.set_ylabel("WA (%)")
    ax_overall.bar_label(bars, labels=[f"{x:.1f}%" for x in overall], padding=3, fontsize=10)
    ax_overall.spines[["top", "right"]].set_visible(False)
    ax_overall.grid(axis="y", color="#E8EDF2")
    ax_overall.set_axisbelow(True)

    models = list(MODEL_LABELS)
    model_delta = [value(summary, "model", conds[1], model=m) -
                   value(summary, "model", conds[0], model=m) for m in models]
    draw_model_change(ax_model, models, model_delta)

    classes = ["P", "NP-complete", "NP-hard"]
    class_delta = [value(summary, "complexity", conds[1], complexity=c) -
                   value(summary, "complexity", conds[0], complexity=c) for c in classes]
    y = np.arange(len(classes))
    ax_class.barh(y, class_delta, color="#D4654B", height=.6)
    ax_class.set_yticks(y, classes)
    ax_class.invert_yaxis()
    ax_class.set_xlim(-2, 10)
    ax_class.set_xlabel("Change in WA (pp)")
    for i, val in enumerate(class_delta):
        ax_class.text(val + .15, i, f"{val:+.1f}", va="center", fontsize=10)
    ax_class.axvline(0, color="#65717D", linewidth=.9)
    ax_class.grid(axis="x", color="#E8EDF2")
    ax_class.set_axisbelow(True)
    ax_class.spines[:].set_visible(False)
    ax_class.tick_params(length=0)

    # Group columns by complexity so subject-level effects are easier to scan.
    ordered = ["ECG_Pattern_Detection", "ED_Scheduling", "Medical_Risk",
               "Disease_Diagnosis_NP-complete", "Nurse_Scheduling",
               "OR_Scheduling_NP-complete", "Disease_Diagnosis", "OR_Scheduling", "PASP"]
    short = [TASK_SHORT_NAMES[p] for p in ordered]
    delta = np.array([[value(summary, "model_problem", conds[1], model=m, problem=p) -
                       value(summary, "model_problem", conds[0], model=m, problem=p)
                       for p in ordered] for m in models])
    cmap = LinearSegmentedColormap.from_list("change", ["#2366A1", "#F7F8FA", "#C94631"])
    heat = ax_matrix.imshow(delta, aspect="auto", cmap=cmap,
                            norm=TwoSlopeNorm(vmin=-10, vcenter=0, vmax=35))
    ax_matrix.set_xticks(np.arange(9), short, fontsize=8)
    ax_matrix.set_yticks(np.arange(8), [MODEL_LABELS[m] for m in models], fontsize=10)
    ax_matrix.tick_params(length=0, pad=8)
    for boundary in (2.5, 5.5):
        ax_matrix.axvline(boundary, color="white", linewidth=5)
    for i in range(8):
        for j in range(9):
            val = delta[i, j]
            ax_matrix.text(j, i, f"{val:+.1f}", ha="center", va="center", fontsize=9.5,
                           color="white" if val > 18 or val < -6 else "#25303A")
    for center, label in [(1, "P"), (4, "NP-complete"), (7, "NP-hard")]:
        ax_matrix.text(center, -0.68, label, ha="center", va="center", fontweight="bold",
                       fontsize=10, clip_on=False)
    ax_matrix.spines[:].set_visible(False)
    cbar = fig.colorbar(heat, ax=ax_matrix, fraction=.02, pad=.015)
    cbar.set_label("Change (pp)")

    problem_pairs = np.array([[value(summary, "problem", c, problem=p) for c in conds]
                              for p in ordered])
    y = np.arange(len(ordered))
    for j in range(2):
        positions = y + (j - .5) * .38
        ax_problem.barh(positions, problem_pairs[:, j], color=COLORS[j], height=.34,
                        label=conds[j])
        for pos, val in zip(positions, problem_pairs[:, j]):
            ax_problem.text(val + .7, pos, f"{val:.1f}", va="center", fontsize=8.5)
    ax_problem.set_yticks(y, [s.replace("\n", " ") for s in short])
    ax_problem.invert_yaxis()
    ax_problem.set_xlim(0, 93)
    ax_problem.set_xlabel("Weighted accuracy (%)")
    ax_problem.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24),
                      frameon=False, ncol=2)
    ax_problem.grid(axis="x", color="#E8EDF2")
    ax_problem.set_axisbelow(True)
    ax_problem.spines[:].set_visible(False)
    ax_problem.tick_params(length=0)
    for boundary in (2.5, 5.5):
        ax_problem.axhline(boundary, color="#AAB4BD", linewidth=.8)
    # Export all panels on one opaque canvas for direct insertion into slides.
    # SVG is exported from the artists themselves, preserving tick/curve strokes.
    for extension in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"medical_semantics_wa_comparison.{extension}",
                    dpi=600 if extension == "png" else None,
                    facecolor="white", edgecolor="none", transparent=False)
    fig.canvas.draw()
    save_panel_crops(fig, [
        ("a_overall", [ax_overall]),
        ("b_model_change", [ax_model]),
        ("c_complexity_change", [ax_class]),
        ("d_model_task_change", [ax_matrix, cbar.ax]),
        ("e_task_wa", [ax_problem]),
    ])
    save_standalone_model_change(models, model_delta)
    plt.close(fig)


def main():
    global OUT, PANEL_OUT
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="intermediate medical_semantics_wa.json")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_dir
    PANEL_OUT = OUT / "medical_semantics_wa_comparison_panels"
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = load_data("medical_semantics_wa", args.data)
    summary = prepared["summary"]
    plot(summary)
    print(f"Rendered figures under {OUT} from intermediate data")
    for condition in RESULTS:
        print(condition, f"WA={value(summary, 'overall', condition):.2f}%")


if __name__ == "__main__":
    main()
