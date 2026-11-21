"""Draw the medical-metrics panels from saved intermediate rows."""
import math
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .figure_style import TWO_COLUMN, apply_style
from .task_labels import TASK_SHORT_NAMES
from tasks.pra_cc.penalty import CP_WEIGHT_SCALE, CP_WEIGHTS

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


C_NOANS  = "#A9B0B7"   # No valid output or unassessable (TE/OF, no schedule or triage, etc.)


C_SEVERE = "#67001F"   # Most severe error


C_MAJOR  = "#B2182B"


C_MOD    = "#D6604D"


C_MINOR  = "#F4A582"


C_MILD   = "#FDDBC7"


C_NONE   = "#C9E2F2"   # No identifiable error category


DD_PANEL = [
    ("dd_cand_unassessable", "Unassessable", C_NOANS),
    ("dd_cand_reference_missed", "Reference omission", C_SEVERE),
    ("dd_cand_group_mismatch", "Phenotype misclassification", C_MAJOR),
    ("dd_cand_reference_grouped", "Reference concordance", C_NONE),
]


RISK_PANEL = [
    ("risk_category_unassessable", "Unassessable",         C_NOANS),
    ("risk_category_under",        "Risk underestimation", C_SEVERE),
    ("risk_category_over",         "Risk overestimation",  C_MOD),
    ("risk_category_score_error",  "Category match; score error",   C_MILD),
    ("risk_category_score_match",  "Category and score match",      C_NONE),
]


ECG_PANEL = [
    ("ecg_pos_unassessable", "Unassessable", C_NOANS),
    ("ecg_pos_missed", "Position omission", C_SEVERE),
    ("ecg_pos_wrong_set", "Label discordance", C_MAJOR),
    ("ecg_pos_reference_match", "Reference concordance", C_NONE),
]


ED_PANEL = [
    ("ed_no_sched",   "Unassessable",                 C_NOANS),
    ("ed_urgency",    "Acuity-priority discordance",  C_SEVERE),
    ("ed_vulnerable", "Vulnerability discordance",    C_MAJOR),
    ("ed_nrs",        "Pain-score discordance",       C_MOD),
    ("ed_fifo",       "Arrival-order discordance",    C_MINOR),
    ("ed_correct",    "Rule concordance",             C_NONE),
]


OR_DISPOSITION_PANEL = [
    ("or_case_unassessable", "Unassessable", C_NOANS),
    ("or_case_emergency_unscheduled", "Emergency unscheduled", C_SEVERE),
    ("or_case_urgent_unscheduled", "Urgent unscheduled", C_MAJOR),
    ("or_case_elective_unscheduled", "Elective unscheduled", C_MOD),
    ("or_case_specialty_mismatch", "Specialty discordance", C_MINOR),
    ("or_case_matched", "Specialty concordance", C_NONE),
]


PASP_PLACEMENT_PANEL = [
    ("pasp_day_unassessable", "Unassessable", C_NOANS),
    ("pasp_day_uncovered", "Coverage failure", C_SEVERE),
    ("pasp_day_capacity", "Capacity violation", C_MAJOR),
    ("pasp_day_both_mismatch", "Specialty and demographic mismatch", C_MOD),
    ("pasp_day_specialty", "Specialty mismatch", C_MINOR),
    ("pasp_day_demographic", "Demographic mismatch", C_MILD),
    ("pasp_day_fit", "Requirement concordance", C_NONE),
]


STACK_PANELS = [
    (ECG_PANEL,  f"{TASK_SHORT_NAMES['ECG_Pattern_Detection']} — reference-positive RR positions", "ecg_interval_total"),
    (ED_PANEL,   f"{TASK_SHORT_NAMES['ED_Scheduling']} — selection-rule concordance", "ed_sel_total"),
    (RISK_PANEL, f"{TASK_SHORT_NAMES['Medical_Risk']} — risk-classification outcomes", "risk_n"),
    (DD_PANEL,   f"{TASK_SHORT_NAMES['Disease_Diagnosis']} — reference-positive diseases", "dd_reference_disease_total"),
    (OR_DISPOSITION_PANEL, f"{TASK_SHORT_NAMES['OR_Scheduling']} — surgery disposition", "or_case_total"),
    (PASP_PLACEMENT_PANEL, f"{TASK_SHORT_NAMES['PASP']} — required patient-days", "pasp_day_total"),
]


def _text_on(hex_color: str) -> str:
    """Choose a legible text color based on the background brightness."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#FFFFFF" if 0.299 * r + 0.587 * g + 0.114 * b < 150 else "#333333"


def plot_metrics(rows, out_path: str, formats, ncol: int = 1, panels=None,
                 overview=False):
    """Plot task outcomes at their natural units; PABP dimensions may overlap."""
    if panels is None:
        base, ext = os.path.splitext(out_path)
        plot_metrics(rows, out_path, formats, ncol=3,
                     panels=STACK_PANELS, overview=True)
        for group in range(0, len(STACK_PANELS), 2):
            group_path = f"{base}_detail_{group // 2 + 1}{ext}"
            plot_metrics(rows, group_path, formats, ncol=1,
                         panels=STACK_PANELS[group:group + 2])
        return
    model_rank = {model: idx for idx, model in enumerate(MODEL_ORDER)}
    order = sorted(rows, key=lambda r: (model_rank.get(r["model"], len(model_rank)),
                                        r["model_name"]))
    names = [r["model_name"] for r in order]
    nrow = (len(panels) + ncol - 1) // ncol
    panel_h = 0.42 * len(names) + 2.1

    apply_style()
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=((18.0, 14.5) if ncol == 2 else
                                      (20.0, 12.0) if ncol == 3 else
                                      (18.0, panel_h * nrow + 1.0)) if overview
                             else (TWO_COLUMN, 8.4))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (segs, panel_title, total_key) in enumerate(panels):
        ax = axes[idx]
        ypos = list(range(len(order)))



        if total_key == "or_case_total":
            for y, row in enumerate(order):
                values = row["values"]
                total = values.get(total_key, 0) or 0
                if sum((values.get(key, 0) or 0) for key, _, _ in segs) != total:
                    raise AssertionError(f"OR categories do not cover all surgeries: {row['model']}")
                left = 0.0
                for key, _label, color in segs:
                    count = values.get(key, 0) or 0
                    if not count:
                        continue
                    width = count
                    midpoint = left + width / 2
                    ax.barh(y, width, left=left, height=.62, color=color,
                            edgecolor="white", linewidth=.5, zorder=2)
                    if width >= total * (0.03 if overview else 0.027):
                        ax.text(midpoint, y, str(count), ha="center", va="center",
                                fontsize=5.8 if overview else 7.0,
                                color=_text_on(color), zorder=3)
                    left += width
            max_surgeries = max((row["values"].get(total_key, 0) or 0
                                 for row in order), default=0)
            ax.set_xlim(0, max(max_surgeries * 1.05, 1))
            ax.set_ylim(len(order) - .5, -.5)
            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=7.0 if overview else 7.3)
            ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Surgeries (count)",
                          fontsize=6.2 if overview else 7.0)
            ax.grid(axis="x", color="#EDEDED", linewidth=.7, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#B5B5B5")
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=6.5 if overview else 7.0,
                           color="#B5B5B5")
            handles = [Patch(facecolor=color, label=label, edgecolor="white")
                       for _key, label, color in segs]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.16),
                      ncol=3 if overview else 2, fontsize=5.3 if overview else 6.4, frameon=False,
                      handletextpad=.25, columnspacing=.55)
            continue

        if total_key in ("or_new_issue_total", "or_case_model_issue_total"):
            rates = []
            counts = []
            for y, row in enumerate(order):
                values = row["values"]
                all_surgeries = values.get("or_case_total", 0) or 0
                left = 0.0
                total_count = 0
                narrow_segments = []
                for segment_index, (key, _label, color) in enumerate(segs):
                    count = values.get(key, 0) or 0
                    total_count += count
                    width = 100 * count / all_surgeries if all_surgeries else 0.0
                    if width:
                        ax.barh(y, width, left=left, height=.62,
                                color=color, edgecolor="white", linewidth=.5,
                                zorder=2)
                        midpoint = left + width / 2
                        if width >= (0.46 if overview else 0.26):
                            ax.text(midpoint, y, str(count), ha="center", va="center",
                                    fontsize=5.8 if overview else 7.0,
                                    color="#333333" if segment_index == 3 else "white",
                                    fontweight="bold", zorder=3)
                        else:
                            narrow_segments.append((midpoint, count))
                    left += width
                for midpoint, count in narrow_segments:
                    ax.annotate(str(count), xy=(midpoint, y - .31),
                                xytext=(midpoint, y - .44),
                                ha="center", va="center",
                                fontsize=5.7 if overview else 6.8,
                                color="#333333",
                                arrowprops={"arrowstyle": "-", "color": "#555555",
                                            "lw": .5}, zorder=4)
                rates.append(left)
                counts.append(total_count)
            maximum = max(max(rates, default=0), 1.0)
            ax.set_xlim(0, maximum * 1.16)
            for y, (rate, count) in enumerate(zip(rates, counts)):
                ax.text(rate + maximum * .012, y, str(count),
                        ha="left", va="center", fontsize=6.2 if overview else 7.0,
                        color="#333333")
            ax.set_ylim(len(order) - .5, -.56)
            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=7.0 if overview else 7.3)
            ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Affected surgeries per 100 total surgeries (labels: counts)",
                          fontsize=6.2 if overview else 7.0)
            ax.grid(axis="x", color="#EDEDED", linewidth=.7, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#B5B5B5")
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=6.5 if overview else 7.0,
                           color="#B5B5B5")
            handles = [Patch(facecolor=color, label=label, edgecolor="white")
                       for _key, label, color in segs]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.16),
                      ncol=2 if overview else 4,
                      fontsize=5.5 if overview else 6.0, frameon=False,
                      handletextpad=.25, columnspacing=.6)
            continue

        if total_key == "pasp_day_total":
            # Raw violation types can overlap and have different counting
            # units. Optional hard counts are assessed independently; soft
            # penalties remain restricted to hard-feasible plans.
            issue_specs = (
                ("specialism", "Specialty mismatch", "#0072B2"),
                ("gender_policy", "Gender mismatch", "#D55E00"),
                ("age_policy", "Age mismatch", "#E69F00"),
                ("room_preference", "Room preference", "#009E73"),
                ("room_transfer", "Room transfer", "#CC79A7"),
            )

            hard_specs = [(key.removeprefix("pasp_hard_"), label, color)
                          for key, label, color in segs
                          if key.startswith("pasp_hard_")]
            count_specs = hard_specs + list(issue_specs)
            # Stack the scorer's raw violation units, which can overlap
            # across types and use the prompt's type-specific count units.
            count_rows = []
            for row in order:
                values = row["values"]
                issue_counts = [int(values[f"pasp_hard_{key}"])
                                for key, _label, _color in hard_specs]
                for key, _label, _color in issue_specs:
                    penalty = values.get(f"pasp_penalty_{key}", 0) or 0
                    units = penalty * CP_WEIGHT_SCALE / CP_WEIGHTS[key]
                    count = round(units)
                    if abs(units - count) > 1e-6:
                        raise ValueError(f"Non-integral PABP violation count: {key}={units}")
                    issue_counts.append(count)
                count_rows.append(issue_counts)
            max_total = max((sum(counts) for counts in count_rows), default=0)
            max_total = max(max_total, 1)
            for y, issue_counts in enumerate(count_rows):
                left = 0
                for (_key, _label, color), count in zip(count_specs, issue_counts):
                    if not count:
                        continue
                    ax.barh(y, count, left=left, height=.62,
                            color=color, edgecolor="white", linewidth=.5, zorder=2)
                    if count >= max_total * (.06 if overview else .05):
                        ax.text(left + count / 2, y, str(count),
                                ha="center", va="center",
                                fontsize=5.7 if overview else 7.0,
                                color=_text_on(color), zorder=3)
                    left += count
            ax.set_xlim(0, max_total * 1.04)
            ax.set_ylim(len(order) - .5, -.5)
            ax.set_yticks(ypos)
            ax.set_yticklabels(names, fontsize=6.5 if overview else 7.3)
            ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Violation units (count)" if hard_specs else
                          "Raw scorer violation units (count)",
                          fontsize=6.2 if overview else 7.0)
            ax.grid(axis="x", color="#EDEDED", linewidth=.7, zorder=0)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#B5B5B5")
            ax.tick_params(axis="y", length=0, pad=2)
            ax.tick_params(axis="x", labelsize=6.5 if overview else 7.0,
                           color="#B5B5B5")
            handles = [Patch(facecolor=color, label=label, edgecolor="white")
                       for _key, label, color in count_specs]
            if overview:
                # Matplotlib fills legend columns first; arrange the input
                # so the displayed rows follow the stack's left-to-right order.
                handles = [handles[i] for col in range(3)
                           for i in range(col, len(handles), 3)]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.15),
                      ncol=3 if overview else 5,
                      fontsize=5.3 if overview else 6.2, frameon=False,
                      handletextpad=.25, columnspacing=.6)
            continue

        # Main segments sum to the reference-positive denominator. Hatched extras
        # are appended visually, but remain outside that denominator.
        panel_max_total = max((r["values"].get(total_key) or 0.0 for r in order),
                              default=0.0)
        for y in ypos:
            left = 0.0
            for seg_key, _seg_label, seg_col in segs:
                w = order[y]["values"].get(seg_key) or 0.0
                if w > 0:
                    ax.barh(y, w, left=left, height=0.62, color=seg_col,
                            edgecolor="white", linewidth=0.6, zorder=2)
                total_for_row = order[y]["values"].get(total_key) or 0.0
                label_min = max(6.0, total_for_row * 0.06)
                if total_key == "or_penalty_model_total":
                    label_min = max(label_min, panel_max_total * .028)
                if w >= label_min:
                    value_label = (f"{w:.1f}" if total_key == "or_penalty_model_total"
                                   and abs(w - round(w)) > 1e-6 else f"{w:.0f}")
                    ax.text(left + w / 2, y, value_label, ha="center", va="center",
                            fontsize=6.8, color=_text_on(seg_col), zorder=3)
                left += w
        tot = [r["values"].get(total_key) or 0.0 for r in order]
        over_key = ("dd_false_positive_candidates" if total_key == "dd_reference_disease_total" else
                    "ecg_false_positive_patterns" if total_key == "ecg_interval_total" else None)
        extras = []
        if over_key:
            for y, row in enumerate(order):
                extra = row["values"].get(over_key, 0) or 0
                if total_key == "dd_reference_disease_total":
                    extra += row["values"].get("dd_selected_outside_candidates", 0) or 0
                extras.append(extra)
                if extra:
                    ax.barh(y, extra, left=tot[y], height=0.62, color="#FFFFFF",
                            edgecolor="#333333", linewidth=0.8, hatch="////", zorder=2)
                    ax.text(tot[y] + extra + max(tot) * 0.008, y, f"+{extra}",
                            ha="left", va="center", fontsize=6.5, color="#222222", zorder=3)
        max_total = max(max(tot, default=1.0), 1.0)
        max_end = max((tot[i] + (extras[i] if extras else 0) for i in range(len(tot))), default=max_total)
        limit_lo, limit_hi = 0.0, max(max_total, max_end) + max_total * (0.14 if over_key else 0.07)
        if total_key == "or_case_total" and not any(
                "or_assessable_instances" in row["values"] for row in order):
            limit_hi = max(limit_hi, max_total * 1.24)
            for y, row in enumerate(order):
                values = row["values"]
                assessed = values.get("or_idle_assessable_instances", 0) or 0
                idle_sum = values.get("or_idle_time_sum", 0) or 0
                idle_mean = idle_sum / assessed if assessed else 0
                ax.text(tot[y] + max_total * 0.015, y,
                        f"Idle avg. {idle_mean:.0f} min / instance",
                        ha="left", va="center", fontsize=5.8, color="#444444")

        ax.set_title(panel_title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlim(limit_lo, limit_hi)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=8)
        unit_txt = ("reference-positive diseases + extra predictions" if total_key == "dd_reference_disease_total" else
                    "reference-positive RR positions + extra patterns" if total_key == "ecg_interval_total" else
                    "patients" if total_key == "risk_n" else
                    "selections" if total_key == "ed_sel_total" else
                    "surgeries" if total_key == "or_case_total" else
                    "penalty points" if total_key == "or_penalty_model_total" else
                    "patient-days" if total_key == "pasp_day_total" else "instances")
        ax.set_xlabel(unit_txt if total_key == "or_penalty_model_total"
                      else f"{unit_txt} (count)", fontsize=7.5)
        ax.grid(axis="x", color="#EDEDED", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#B5B5B5")
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.5, color="#B5B5B5")

        seg_handles = [Patch(facecolor=c, label=lb, edgecolor="#BBBBBB", linewidth=0.5)
                       for _k, lb, c in segs]
        if over_key:
            over_label = ("Extra disease predictions (outside denominator)"
                          if total_key == "dd_reference_disease_total"
                          else "False-positive patterns (outside denominator)")
            seg_handles.append(Patch(facecolor="#FFFFFF", label=over_label,
                                     edgecolor="#333333", hatch="////", linewidth=0.8))
        ax.legend(handles=seg_handles, loc="upper left", bbox_to_anchor=(0.0, -0.13),
                  ncol=3 if overview else 2, fontsize=5.3 if overview else 6.6, frameon=False, handlelength=1.1,
                  handletextpad=0.4, columnspacing=1.0, labelspacing=0.3)

    if overview:
        short_labels = {
            "Reference omission": "Missed reference",
            "Phenotype misclassification": "Wrong phenotype",
            "Reference concordance": "Reference match",
            "Risk underestimation": "Risk underestimated",
            "Risk overestimation": "Risk overestimated",
            "Category match; score error": "Score mismatch",
            "Category and score match": "Category + score match",
            "Position omission": "Missed position",
            "Label discordance": "Wrong label",
            "Acuity-priority discordance": "Triage priority error",
            "Vulnerability discordance": "Vulnerability priority error",
            "Pain-score discordance": "Pain priority error",
            "Arrival-order discordance": "Arrival order error",
            "Rule concordance": "Rule match",
            "Extra disease predictions (outside denominator)": "Extra diseases",
            "False-positive patterns (outside denominator)": "Extra patterns",
            "Scheduled: specialty mismatch": "Specialty mismatch",
            "Scheduled: specialty matched": "Specialty match",
            "Non-contiguous segments": "Non-contiguous",
        }
        for ax in axes[:len(panels)]:
            legend = ax.get_legend()
            if legend is not None:
                for label in legend.get_texts():
                    label.set_text(short_labels.get(label.get_text(), label.get_text()))
                legend.set_loc("upper center")
                legend.set_bbox_to_anchor((0.5, -0.16), transform=ax.transAxes)

    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")

    if overview:
        if ncol == 2:
            fig.subplots_adjust(left=.13, right=.99, top=.96, bottom=.08,
                                wspace=.42, hspace=.56)
        elif ncol == 3:
            fig.subplots_adjust(left=.12, right=.99, top=.95, bottom=.13,
                                wspace=.42, hspace=.62)
        else:
            fig.subplots_adjust(left=0.14, right=0.99, top=0.96, bottom=0.12,
                                wspace=0.54, hspace=0.58)
    elif len(panels) == 1:
        bottom = (0.24 if panels[0][2] == "pasp_day_total" else
                  0.25 if panels[0][2] in ("or_case_total", "or_penalty_model_total")
                  else 0.12)
        fig.subplots_adjust(left=0.25, right=0.96,
                            top=0.96, bottom=bottom)
    else:
        fig.subplots_adjust(left=0.25, right=0.96, top=0.96, bottom=0.12,
                            hspace=0.58)

    base, _ = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    for fmt in formats:
        fig.savefig(f"{base}.{fmt}", dpi=600)
    plt.close(fig)

