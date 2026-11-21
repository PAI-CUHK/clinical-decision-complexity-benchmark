"""Draw the final medical metrics figure from saved intermediate data."""
import argparse
import json
from pathlib import Path
from . import count_medical_metrics as metrics
from .plot_data import DATA_DIR, ROOT

INTERMEDIATE = DATA_DIR / "medical_metrics.json"
FIGURE = ROOT / "plot/medical_metrics.png"

OR_MODEL_PANEL = [
    ("or_case_unassessable", "Unassessable", metrics.C_NOANS),
    ("or_case_model_cancel_emergency", "Emergency cancelled", metrics.C_SEVERE),
    ("or_case_model_cancel_urgent", "Urgent cancelled", metrics.C_MAJOR),
    ("or_case_model_cancel_elective", "Elective cancelled", metrics.C_MOD),
    ("or_case_model_specialty_mismatch", "Scheduled: specialty mismatch", metrics.C_MINOR),
    ("or_case_model_compliant", "Scheduled: specialty matched", metrics.C_NONE),
]
OR_MODEL_ISSUE_PANEL = OR_MODEL_PANEL[1:5]
OR_MODEL_PENALTY_PANEL = [
    ("or_penalty_model_cancel_emergency", "Emergency cancellation", metrics.C_SEVERE),
    ("or_penalty_model_cancel_urgent", "Urgent cancellation", metrics.C_MAJOR),
    ("or_penalty_model_cancel_elective", "Elective cancellation", metrics.C_MOD),
    ("or_penalty_model_specialty", "Specialty mismatch", metrics.C_MINOR),
    ("or_penalty_model_idle", "Idle time", "#56B4E9"),
]
PASP_ISSUE_TYPES = (
    ("specialism", "Specialty mismatch", "#0072B2"),
    ("gender_policy", "Gender-policy mismatch", "#D55E00"),
    ("age_policy", "Age-policy mismatch", "#E69F00"),
    ("room_preference", "Room-preference violation", "#009E73"),
    ("room_transfer", "Room transfer", "#CC79A7"),
)
PASP_HARD_ISSUE_TYPES = (
    ("capacity", "Capacity exceeded", "#7B3294"),
    ("coverage_days", "Coverage days", "#A50026"),
    ("non_contiguous_segment", "Non-contiguous segments", "#8C510A"),
)
PASP_PANEL = [
    (f"pasp_day_answer_{key}", label, color)
    for key, label, color in PASP_ISSUE_TYPES
] + [(f"pasp_hard_{key}", label, color)
     for key, label, color in PASP_HARD_ISSUE_TYPES]
PANELS = metrics.STACK_PANELS[:4] + [
    (OR_MODEL_PANEL, "SCS (O) — scheduling outcomes", "or_case_total"),
    (PASP_PANEL, "PRA-CC — violation counts by type", "pasp_day_total"),
]



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=INTERMEDIATE)
    parser.add_argument("--out", type=Path, default=FIGURE)
    args = parser.parse_args()
    with args.data.open(encoding="utf-8") as stream:
        prepared = json.load(stream)
    if prepared.get("schema_version") != 13:
        raise ValueError("Medical metrics schema 13 is required")
    metrics.plot_metrics(prepared["rows"], str(args.out), ("png", "svg", "pdf"),
                         ncol=3, panels=PANELS, overview=True)


if __name__ == "__main__":
    main()
