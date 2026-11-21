#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWS2 Risk Assessment Instance Generator

This is a P-class problem:
- NEWS2 (National Early Warning Score 2) calculation
- Linear time complexity O(n·K) = O(n) where n is the number of patients and
  K (snapshots per patient) is a small constant (K=3).
- Each patient has K timestamped snapshots; each snapshot has 7 physiological
  variables contributing to the score.

Difficulty model
----------------
Only ``num_patients`` varies with the level (``10 + level*3``). Other sampling
parameters are fixed across levels (see ``data_generation_protocol.py``).

Three combined mechanisms (all keep the problem in P):
1. **Time dimension (multi-snapshot)**: each patient has K=3 snapshots. The
   NEWS2 total is the sum over components of the per-component MAX (worst)
   score across snapshots — i.e. ``total = Σ_k max_t(score_k(t))``. This is
   NOT the same as ``max_t Σ_k score_k(t)`` (picking the worst snapshot).
2. **Self-selected SpO2 scale**: ``spo2_scale`` is no longer given; the LLM
   must derive it from the patient's ``conditions`` (COPD /
   hypercapnic_resp_failure -> Scale 2, else Scale 1).
3. **Variable cohort size**: n = 10 + level*3.

Scoring covers:
- Respiratory Rate
- Oxygen Saturation (with SpO2 Scale 1 or Scale 2, derived from conditions)
- Supplemental Oxygen
- Temperature
- Systolic Blood Pressure
- Heart Rate
- Consciousness (ACVPU)

Data generation follows the NEWS2 Data Generation Protocol defined in
``data_generation_protocol.py``.
"""

import random
from typing import Dict, List

from tasks.ecd_rs.protocol import (
    sample_respiratory_rate,
    sample_systolic_bp,
    sample_temperature,
    sample_spo2,
    sample_heart_rate,
    sample_supplemental_oxygen,
    sample_consciousness,
    sample_conditions,
    conditions_to_scale,
    get_protocol_config,
    NUM_SNAPSHOTS,
)


class MedicalRiskInstance:
    """NEWS2 Risk Assessment Instance"""

    def __init__(self, patients: Dict[str, Dict]):
        self.patients = patients

    def metadata(self):
        return {
            "num_patients": len(self.patients),
            "field_names": [
                "age", "conditions", "snapshots",
            ],
        }


# ---------------------------------------------------------------------------
# Scoring functions — mirror the NEWS2 rules exactly (unchanged)
# ---------------------------------------------------------------------------

def score_respiratory_rate(rr: int) -> int:
    """Respiratory Rate scoring."""
    if rr <= 8:
        return 3
    elif rr <= 11:
        return 1
    elif rr <= 20:
        return 0
    elif rr <= 24:
        return 2
    else:  # rr >= 25
        return 3


def score_spo2_scale1(spo2: int) -> int:
    """SpO2 Scale 1 scoring."""
    if spo2 >= 96:
        return 0
    elif spo2 >= 94:
        return 1
    elif spo2 >= 92:
        return 2
    else:  # spo2 <= 91
        return 3


def score_spo2_scale2(spo2: int) -> int:
    """SpO2 Scale 2 scoring."""
    if spo2 >= 97:
        return 3
    elif spo2 >= 95:
        return 2
    elif spo2 >= 93:
        return 1
    elif spo2 >= 88:
        return 0
    elif spo2 >= 86:
        return 1
    elif spo2 >= 84:
        return 2
    else:  # spo2 <= 83
        return 3


def score_supplemental_oxygen(oxygen: str) -> int:
    """Supplemental Oxygen scoring."""
    return 2 if oxygen == "Supplemental Oxygen" else 0


def score_temperature(temp: float) -> int:
    """Temperature scoring."""
    if temp <= 35.0:
        return 3
    elif temp <= 36.0:
        return 1
    elif temp <= 38.0:
        return 0
    elif temp <= 39.0:
        return 1
    else:  # temp >= 39.1
        return 2


def score_systolic_bp(sbp: int) -> int:
    """Systolic Blood Pressure scoring."""
    if sbp <= 90:
        return 3
    elif sbp <= 100:
        return 2
    elif sbp <= 110:
        return 1
    elif sbp <= 219:
        return 0
    else:  # sbp >= 220
        return 3


def score_heart_rate(hr: int) -> int:
    """Heart Rate scoring."""
    if hr <= 40:
        return 3
    elif hr <= 50:
        return 1
    elif hr <= 90:
        return 0
    elif hr <= 110:
        return 1
    elif hr <= 130:
        return 2
    else:  # hr >= 131
        return 3


def score_consciousness(level: str) -> int:
    """Consciousness (ACVPU) scoring."""
    if level == "Alert":
        return 0
    # New Confusion, Voice, Pain, Unresponsive all score +3
    return 3


# ---------------------------------------------------------------------------
# Full assessment pipeline (multi-snapshot, per-component max)
# ---------------------------------------------------------------------------

# The seven NEWS2 components, in canonical order.
COMPONENTS = ("rr", "spo2", "oxygen", "temp", "sbp", "hr", "cons")


def _snapshot_component_scores(snapshot: Dict, scale: int) -> Dict[str, int]:
    """Score all seven components for a single snapshot using the given scale."""
    return {
        "rr": score_respiratory_rate(snapshot["respiratory_rate"]),
        "spo2": (
            score_spo2_scale1(snapshot["spo2"])
            if scale == 1
            else score_spo2_scale2(snapshot["spo2"])
        ),
        "oxygen": score_supplemental_oxygen(snapshot["supplemental_oxygen"]),
        "temp": score_temperature(snapshot["temperature"]),
        "sbp": score_systolic_bp(snapshot["systolic_bp"]),
        "hr": score_heart_rate(snapshot["heart_rate"]),
        "cons": score_consciousness(snapshot["consciousness"]),
    }


def _legacy_snapshot(patient: Dict) -> Dict:
    """
    Convert a legacy scalar patient (single observation, explicit spo2_scale)
    into a single-snapshot dict. Used for backward compatibility with
    instances generated before the multi-snapshot refactor.
    """
    return {
        "timestamp": "T1",
        "respiratory_rate": patient["respiratory_rate"],
        "spo2": patient["spo2"],
        "supplemental_oxygen": patient["supplemental_oxygen"],
        "systolic_bp": patient["systolic_bp"],
        "heart_rate": patient["heart_rate"],
        "temperature": patient["temperature"],
        "consciousness": patient["consciousness"],
    }


def _resolve_snapshots_and_scale(patient: Dict):
    """
    Return (snapshots, scale) for either the new conditions+snapshots
    structure or the legacy scalar structure.
    """
    snapshots = patient.get("snapshots")
    if snapshots:
        conditions = patient.get("conditions", [])
        scale = conditions_to_scale(conditions)
        return snapshots, scale
    # Legacy scalar patient
    return [_legacy_snapshot(patient)], patient.get("spo2_scale", 1)


def compute_news2_score(patient: Dict) -> Dict:
    """
    Compute the complete NEWS2 assessment for a single patient.

    Multi-snapshot aggregation: for each of the seven components, take the
    MAXIMUM (worst) score across all snapshots, then sum the seven component
    maxima to obtain the total NEWS2 score:

        total = Σ_k max_t(score_k(t))

    This is intentionally NOT ``max_t Σ_k score_k(t)`` (the worst snapshot's
    sum): deterioration may be spread across different snapshots and
    different components, and each component's worst value must be
    accumulated independently.

    Returns a dict with component scores, total score, risk category, and
    urgent review flag.
    """
    snapshots, scale = _resolve_snapshots_and_scale(patient)

    # Per-component scores across snapshots
    per_snapshot = [_snapshot_component_scores(snap, scale) for snap in snapshots]

    # Final component score = max (worst) across snapshots
    final = {c: max(ps[c] for ps in per_snapshot) for c in COMPONENTS}

    rr_score = final["rr"]
    spo2_score = final["spo2"]
    oxygen_score = final["oxygen"]
    temp_score = final["temp"]
    sbp_score = final["sbp"]
    hr_score = final["hr"]
    cons_score = final["cons"]

    total_score = (
        rr_score + spo2_score + oxygen_score + temp_score
        + sbp_score + hr_score + cons_score
    )

    # --- Risk category ---
    # Apply the "highest applicable" rule order.
    # NEWS2 = 0 → Low Risk
    # NEWS2 = 1-4 → Low Risk
    # NEWS2 = 3 in any single physiological parameter → Low-Medium Risk
    # NEWS2 = 5-6 → Medium Risk
    # NEWS2 >= 7 → High Risk
    # If multiple rules apply, assign the highest applicable risk category.

    if total_score >= 7:
        risk_category = "High Risk"
    elif total_score >= 5:
        risk_category = "Medium Risk"
    elif total_score >= 3 and _has_critical_physiological(rr_score, spo2_score, temp_score, sbp_score, hr_score):
        # Single physiological variable scoring +3
        risk_category = "Low-Medium Risk"
    elif total_score >= 1:
        risk_category = "Low Risk"
    else:
        risk_category = "Low Risk"  # total_score == 0

    # --- Urgent review ---
    urgent = _requires_urgent_review(total_score, rr_score, spo2_score, temp_score, sbp_score, hr_score)

    return {
        "rr_score": rr_score,
        "spo2_score": spo2_score,
        "oxygen_score": oxygen_score,
        "temp_score": temp_score,
        "sbp_score": sbp_score,
        "hr_score": hr_score,
        "cons_score": cons_score,
        "total_score": total_score,
        "risk_category": risk_category,
        "urgent_review": urgent,
    }


def _has_critical_physiological(rr_score, spo2_score, temp_score, sbp_score, hr_score) -> bool:
    """Check if any single physiological variable scored +3."""
    return any(s == 3 for s in [rr_score, spo2_score, temp_score, sbp_score, hr_score])


def _requires_urgent_review(total_score, rr_score, spo2_score, temp_score, sbp_score, hr_score) -> bool:
    """
    Determine if urgent clinical review is required.

    Urgent review is required if either condition is satisfied:
    - NEWS2 score >= 5
    - Any single physiological variable receives a score of +3
      (Respiratory Rate, SpO2, Temperature, Systolic Blood Pressure, Heart Rate)

    Supplemental Oxygen and Consciousness are NOT considered physiological
    variables for this criterion.

    Note: with multi-snapshot per-component-max aggregation, "any single
    physiological variable +3" is equivalent to "any snapshot had a +3 in a
    physiological variable", which is the clinically correct semantics.
    """
    if total_score >= 5:
        return True
    if _has_critical_physiological(rr_score, spo2_score, temp_score, sbp_score, hr_score):
        return True
    return False


def compute_full_assessment(patients: Dict[str, Dict], rules=None) -> Dict:
    """
    Compute the full NEWS2 assessment for all patients.

    Args:
        patients: Dict of patient_id -> patient_data (new conditions+snapshots
            structure, or legacy scalar structure).

    Returns:
        Dict with 'scores', 'risk_categories', and 'urgent_review'.
    """
    scores = {}
    risk_categories = {}
    urgent_review = []

    for pid, patient in patients.items():
        assessment = compute_news2_score(patient)
        scores[pid] = assessment["total_score"]
        risk_categories[pid] = assessment["risk_category"]
        if assessment["urgent_review"]:
            urgent_review.append(pid)

    urgent_review.sort()

    return {
        "scores": scores,
        "risk_categories": risk_categories,
        "urgent_review": urgent_review,
        "details": {
            pid: compute_news2_score(patient)
            for pid, patient in patients.items()
        },
    }


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def generate_patient(rng: random.Random, level: int) -> Dict:
    """
    Generate a synthetic patient with K timestamped snapshots using the Data
    Generation Protocol (ordinary random sampling, no guaranteed trap).
    """
    K = NUM_SNAPSHOTS
    conditions = sample_conditions(rng, level)
    scale = conditions_to_scale(conditions)

    snapshots = []
    for t in range(K):
        snap = {
            "timestamp": f"T{t + 1}",
            "respiratory_rate": sample_respiratory_rate(rng, level),
            "spo2": sample_spo2(rng, level, scale),
            "supplemental_oxygen": sample_supplemental_oxygen(rng, level),
            "systolic_bp": sample_systolic_bp(rng, level),
            "heart_rate": sample_heart_rate(rng, level),
            "temperature": sample_temperature(rng, level),
            "consciousness": sample_consciousness(rng, level),
        }
        snapshots.append(snap)

    return {
        "age": rng.randint(18, 100),
        "conditions": conditions,
        "snapshots": snapshots,
    }


def generate_patients(num_patients: int, level: int, rng: random.Random) -> Dict[str, Dict]:
    """
    Generate multiple synthetic patients.

    Patients are sampled independently without guaranteed risk categories.
    """
    return {f"P{i + 1}": generate_patient(rng, level)
            for i in range(num_patients)}


def generate_medical_risk_instance(config: Dict) -> MedicalRiskInstance:
    """
    Generate a NEWS2 risk assessment instance.

    Args:
        config:
            - num_patients: number of patients (default 10 + level*3)
            - seed: random seed
            - level: difficulty level (1-10)

    Returns:
        MedicalRiskInstance with patient data.
    """
    seed = config.get("seed", random.randint(0, 10000))
    level = config.get("level", 5)
    num_patients = config.get("num_patients", 10 + level * 3)

    rng = random.Random(seed)
    patients = generate_patients(num_patients, level, rng)

    return MedicalRiskInstance(patients)
