#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWS2 Data Generation Protocol

This module defines the data generation protocol for the NEWS2 (National Early
Warning Score 2) Risk Assessment benchmark.

Difficulty model
----------------
Only the number of patients ``n`` varies with the difficulty level (driven by
the adapter). The sampling parameters are otherwise fixed across levels:

- Number of snapshots per patient K = 3            (fixed)
- Boundary-value sampling ratio     = 0.00         (fixed, all variables)
- Scale-2 (COPD / hypercapnic) patient ratio = 0.40 (fixed)
- Dispersed-deterioration trap ratio = 0.00         (fixed)
- Supplemental Oxygen ratio          = 0.35         (fixed)
- Non-Alert consciousness ratio      = 0.25         (fixed)

These ratios are DATA-SAMPLING parameters (the composition of the synthetic
cohort), NOT the NEWS2 scoring rules. The NEWS2 scoring rules themselves live
in ``tasks/ecd_rs/generator.py`` (the ``score_*`` functions) and are never
altered here.

The SpO2 scale is no longer stored on the patient; it is derived from the
patient's ``conditions`` field (COPD / hypercapnic_resp_failure -> Scale 2,
otherwise Scale 1). This forces the LLM to select the scale itself.
"""

import random
from typing import Dict, List

# ---------------------------------------------------------------------------
# Constants: the exact NEWS2 scoring boundaries
# ---------------------------------------------------------------------------

# Respiratory Rate thresholds
RR_BOUNDARY_LOW  = 8    # ≤8 -> +3
RR_BOUNDARY_HIGH = 25   # ≥25 -> +3
RR_LOW_MAX       = 11   # 9-11 -> +1
RR_NORMAL_HIGH   = 20   # 12-20 -> +0
RR_HIGH_LOW      = 21   # 21-24 -> +2

# Systolic Blood Pressure thresholds
SBP_CRITICAL     = 90   # ≤90 -> +3
SBP_LOW_MAX      = 100  # 91-100 -> +2
SBP_NORMAL_MIN   = 111  # 111-219 -> +0
SBP_VERY_HIGH    = 220  # ≥220 -> +3

# Temperature thresholds
TEMP_HYPO        = 35.0  # ≤35.0 -> +3
TEMP_COOL_MAX    = 36.0  # 35.1-36.0 -> +1
TEMP_NORMAL_MIN  = 36.1  # 36.1-38.0 -> +0
TEMP_WARN_MAX    = 39.0  # 38.1-39.0 -> +1
TEMP_FEVER       = 39.1  # ≥39.1 -> +2

# SpO2 Scale 1 thresholds
SPO2_S1_NORMAL   = 96   # ≥96 -> +0
SPO2_S1_MILD     = 95   # 94-95 -> +1
SPO2_S1_MODERATE = 93   # 92-93 -> +2
SPO2_S1_SEVERE   = 91   # ≤91 -> +3

# SpO2 Scale 2 thresholds (more granular, reserved for specific patients)
SPO2_S2_NORMAL_HIGH = 97  # ≥97 -> +3
SPO2_S2_MODERATE_HIGH = 96  # 95-96 -> +2
SPO2_S2_MILD_HIGH = 94   # 93-94 -> +1
SPO2_S2_NORMAL_LOW = 92  # 88-92 -> +0
SPO2_S2_MILD_LOW = 87    # 86-87 -> +1
SPO2_S2_MODERATE_LOW = 85  # 84-85 -> +2
SPO2_S2_SEVERE = 83      # ≤83 -> +3

# ---------------------------------------------------------------------------
# Fixed protocol parameters (level-independent)
# ---------------------------------------------------------------------------

NUM_SNAPSHOTS = 3                       # K: timestamped snapshots per patient
BOUNDARY_RATIO = 0.0                    # sample from the ordinary value range
SCALE2_CONDITION_RATIO = 0.40           # fraction of patients with COPD/hypercapnia -> Scale 2
DISPERSED_DETERIORATION_RATIO = 0.0
SUPPLEMENTAL_OXYGEN_RATIO = 0.35        # fraction of readings on Supplemental Oxygen
NON_ALERT_CONSCIOUSNESS_RATIO = 0.25    # fraction of non-Alert consciousness readings

# Conditions that mandate SpO2 Scale 2 (hypercapnic respiratory failure risk)
CONDITIONS_FOR_SCALE2 = ("COPD", "hypercapnic_resp_failure")


# ---------------------------------------------------------------------------
# Protocol configuration
# ---------------------------------------------------------------------------

def get_protocol_config(level: int) -> Dict:
    """
    Return the data generation protocol configuration.

    All factors are identical across levels;
    ``level`` is accepted only for interface compatibility and is ignored.
    Only ``num_patients`` varies with level (handled by the adapter).
    """
    non_alert = NON_ALERT_CONSCIOUSNESS_RATIO
    consciousness_dist = {
        "Alert": 1.0 - non_alert,
        "Voice": non_alert / 3,
        "Pain": non_alert / 3,
        "Unresponsive": non_alert / 3,
    }

    return {
        "num_snapshots": NUM_SNAPSHOTS,
        "dispersed_deterioration_ratio": DISPERSED_DETERIORATION_RATIO,
        "boundary_ratios": {
            "RR_boundary": BOUNDARY_RATIO,
            "SBP_boundary": BOUNDARY_RATIO,
            "Temp_boundary": BOUNDARY_RATIO,
            "SpO2_S1_boundary": BOUNDARY_RATIO,
            "SpO2_S2_boundary": BOUNDARY_RATIO,
        },
        "categorical": {
            "scale2_condition_ratio": SCALE2_CONDITION_RATIO,
            "supplemental_oxygen_ratio": SUPPLEMENTAL_OXYGEN_RATIO,
            "consciousness_distribution": consciousness_dist,
        },
    }


# ---------------------------------------------------------------------------
# Conditions -> SpO2 scale derivation
# ---------------------------------------------------------------------------

def sample_conditions(rng: random.Random, level: int) -> List[str]:
    """
    Sample a patient's clinical conditions list.

    A non-empty result containing COPD / hypercapnic_resp_failure mandates
    SpO2 Scale 2; an empty list means Scale 1. The fraction of Scale-2
    patients is fixed at ``SCALE2_CONDITION_RATIO`` (0.40) across all levels.
    """
    cfg = get_protocol_config(level)
    if rng.random() < cfg["categorical"]["scale2_condition_ratio"]:
        return [rng.choice(CONDITIONS_FOR_SCALE2)]
    return []


def conditions_to_scale(conditions) -> int:
    """
    Derive the SpO2 scale from a patient's conditions.

    Returns 2 if any condition mandates hypercapnic scale (COPD /
    hypercapnic_resp_failure), otherwise 1.
    """
    if conditions:
        for c in conditions:
            if c in CONDITIONS_FOR_SCALE2:
                return 2
    return 1


# ---------------------------------------------------------------------------
# Sampling helpers (used by the ECD-RS generator)
# ---------------------------------------------------------------------------

def sample_respiratory_rate(rng: random.Random, level: int) -> int:
    """
    Sample a respiratory rate value.
    With probability ``boundary`` the value is placed within ±1 of a scoring
    boundary. Otherwise it is sampled uniformly from the physiological range.
    """
    cfg = get_protocol_config(level)
    boundary_p = cfg["boundary_ratios"]["RR_boundary"]

    if rng.random() < boundary_p:
        # Choose a boundary to cluster around
        boundary_centers = [8, 11, 20, 21, 25]
        center = rng.choice(boundary_centers)
        offset = rng.randint(-1, 1)
        val = max(4, min(40, center + offset))
    else:
        val = rng.randint(4, 40)
    return val


def sample_systolic_bp(rng: random.Random, level: int) -> int:
    """Sample a systolic blood pressure value."""
    cfg = get_protocol_config(level)
    boundary_p = cfg["boundary_ratios"]["SBP_boundary"]

    if rng.random() < boundary_p:
        boundary_centers = [90, 91, 100, 110, 111, 219, 220]
        center = rng.choice(boundary_centers)
        offset = rng.randint(-2, 2)
        val = max(60, min(250, center + offset))
    else:
        val = rng.randint(60, 250)
    return val


def sample_temperature(rng: random.Random, level: int) -> float:
    """Sample a body temperature value (rounded to 0.1)."""
    cfg = get_protocol_config(level)
    boundary_p = cfg["boundary_ratios"]["Temp_boundary"]

    if rng.random() < boundary_p:
        boundary_centers = [35.0, 35.1, 36.0, 36.1, 38.0, 38.1, 39.0, 39.1]
        center = rng.choice(boundary_centers)
        offset = round(rng.uniform(-0.2, 0.2), 1)
        val = round(max(33.0, min(41.0, center + offset)), 1)
    else:
        val = round(rng.uniform(33.0, 41.0), 1)
    return val


def sample_spo2(rng: random.Random, level: int, spo2_scale: int) -> int:
    """
    Sample an SpO2 value appropriate for the given scale.
    Boundary values are near scale-specific thresholds.
    """
    cfg = get_protocol_config(level)
    boundary_p = (
        cfg["boundary_ratios"]["SpO2_S1_boundary"]
        if spo2_scale == 1
        else cfg["boundary_ratios"]["SpO2_S2_boundary"]
    )

    if rng.random() < boundary_p:
        if spo2_scale == 1:
            centers = [91, 92, 93, 94, 95, 96]
        else:
            centers = [83, 84, 85, 86, 87, 92, 93, 94, 95, 96, 97]
        center = rng.choice(centers)
        offset = rng.randint(-1, 1)
        val = max(80, min(100, center + offset))
    else:
        val = rng.randint(80, 100)
    return val


def sample_heart_rate(rng: random.Random, level: int) -> int:
    """Sample a heart rate value."""
    cfg = get_protocol_config(level)
    # Heart rate uses the same fixed boundary ratio as the other variables.
    boundary_p = cfg["boundary_ratios"]["SBP_boundary"]

    if rng.random() < boundary_p:
        centers = [40, 50, 90, 91, 110, 111, 131]
        center = rng.choice(centers)
        offset = rng.randint(-2, 2)
        val = max(30, min(160, center + offset))
    else:
        val = rng.randint(30, 160)
    return val


def sample_supplemental_oxygen(rng: random.Random, level: int) -> str:
    """
    Decide supplemental oxygen status.
    Returns 'Room Air' or 'Supplemental Oxygen'.
    """
    cfg = get_protocol_config(level)
    if rng.random() < cfg["categorical"]["supplemental_oxygen_ratio"]:
        return "Supplemental Oxygen"
    return "Room Air"


def sample_consciousness(rng: random.Random, level: int) -> str:
    """
    Sample a consciousness (ACVPU) level.
    Returns one of: Alert, Voice, Pain, Unresponsive.
    """
    cfg = get_protocol_config(level)
    dist = cfg["categorical"]["consciousness_distribution"]
    levels = list(dist.keys())
    weights = [dist[l] for l in levels]
    return rng.choices(levels, weights=weights, k=1)[0]
