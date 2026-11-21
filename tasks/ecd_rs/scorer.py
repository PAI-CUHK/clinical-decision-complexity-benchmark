#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWS2 Risk Assessment Scorer

Validates and scores LLM solutions for the NEWS2 benchmark.
"""

import json
import re
from collections import defaultdict
from typing import Dict, Tuple

from tasks.ecd_rs.generator import compute_full_assessment


def extract_risk_assessment(raw_output: str) -> Dict:
    """
    Extract the NEWS2 assessment from an LLM output string.

    Looks for:
        <root>...<final_answer>{"scores": {...}, "risk_category": {...}, "urgent_review": [...]}</final_answer>...</root>

    Returns:
        Dict with 'scores', 'risk_category', and 'urgent_review'.
    """
    if not isinstance(raw_output, str):
        raise ValueError("raw_output must be a string")

    # Strip <root> tags if present
    root_match = re.search(
        r"<root>\s*([\s\S]*?)\s*</root>", raw_output, flags=re.IGNORECASE
    )
    if root_match:
        raw_output = root_match.group(1)

    # Find <final_answer>
    match = re.search(
        r"<final_answer>\s*([\s\S]*?)\s*</final_answer>",
        raw_output,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Missing <final_answer> block")

    answer_text = match.group(1).strip()

    # Strip markdown code fences
    answer_text = re.sub(r"^```(?:json)?\s*", "", answer_text)
    answer_text = re.sub(r"\s*```$", "", answer_text)

    try:
        data = json.loads(answer_text)
    except json.JSONDecodeError:
        # Try to find a JSON object inside
        obj_match = re.search(r"\{[\s\S]*\}", answer_text)
        if obj_match:
            try:
                data = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                data = {}
        else:
            data = {}

    scores = data.get("scores", {})
    risk_category = data.get("risk_category", {})
    urgent_review = data.get("urgent_review", [])

    # Normalize: ensure all patient keys are present
    if not isinstance(scores, dict):
        scores = {}
    if not isinstance(risk_category, dict):
        risk_category = {}
    if not isinstance(urgent_review, list):
        urgent_review = []

    return {
        "scores": scores,
        "risk_category": risk_category,
        "urgent_review": urgent_review,
    }


def validate_medical_risk_assessment(benchmark: Dict, raw_output) -> Tuple[bool, Dict, float]:
    """
    Validate a NEWS2 risk assessment solution.

    Args:
        benchmark: Dict with 'instance' key (containing patient data) or
            'patients' key.
        raw_output: LLM output string or pre-parsed dict.

    Returns:
        (is_valid, errors_dict, accuracy_score)
    """
    errors = defaultdict(int)

    # Extract patient data
    if "instance" in benchmark:
        patients = benchmark["instance"].get("patients", {})
    elif "patients" in benchmark:
        patients = benchmark["patients"]
    else:
        errors["missing_data"] = 1
        return False, dict(errors), 0.0

    if not patients:
        errors["no_patients"] = 1
        return False, dict(errors), 0.0

    # Compute ground truth
    ground_truth = compute_full_assessment(patients)

    # Parse LLM answer
    try:
        if isinstance(raw_output, dict):
            llm_answer = {
                "scores": raw_output.get("scores", {}),
                "risk_category": raw_output.get("risk_category", {}),
                "urgent_review": raw_output.get("urgent_review", []),
            }
        elif isinstance(raw_output, str):
            llm_answer = extract_risk_assessment(raw_output)
        else:
            raise ValueError(f"Invalid input type: {type(raw_output).__name__}")
    except ValueError as e:
        errors["parse_error"] = str(e)
        return False, dict(errors), 0.0

    total_correct = 0
    total_patients = len(patients)

    for patient_id in patients:
        # Score comparison
        if patient_id not in llm_answer["scores"]:
            errors[f"missing_{patient_id}_score"] += 1
        else:
            expected_score = ground_truth["scores"][patient_id]
            actual_score = llm_answer["scores"][patient_id]
            if actual_score == expected_score:
                total_correct += 1
            else:
                errors[f"score_mismatch_{patient_id}"] += 1

        # Risk category comparison
        if patient_id not in llm_answer["risk_category"]:
            errors[f"missing_{patient_id}_risk_category"] += 1
        else:
            expected_cat = ground_truth["risk_categories"][patient_id]
            actual_cat = llm_answer["risk_category"][patient_id]
            # Normalize to handle en-dash (–) vs hyphen (-) mismatches
            norm_expected = expected_cat.replace("\u2013", "-").replace("\u2014", "-").strip() if isinstance(expected_cat, str) else expected_cat
            norm_actual = actual_cat.replace("\u2013", "-").replace("\u2014", "-").strip() if isinstance(actual_cat, str) else actual_cat
            if norm_actual != norm_expected:
                errors[f"risk_category_mismatch_{patient_id}"] += 1

    # Urgent review comparison
    expected_urgent = set(ground_truth["urgent_review"])
    actual_urgent = set(llm_answer["urgent_review"])

    for pid in actual_urgent - expected_urgent:
        errors[f"false_positive_urgent_{pid}"] += 1
    for pid in expected_urgent - actual_urgent:
        errors[f"false_negative_urgent_{pid}"] += 1

    # Accuracy: percentage of patients with ALL three (score + category + urgent) correct
    fully_correct = total_patients - sum(
        1 for pid in patients
        if f"score_mismatch_{pid}" in errors or
           f"risk_category_mismatch_{pid}" in errors or
           (pid in actual_urgent) != (pid in expected_urgent)
    )
    accuracy = fully_correct / total_patients if total_patients > 0 else 0.0
    is_valid = len(errors) == 0

    return is_valid, dict(errors), accuracy


def extract_risk_scores(raw_output: str) -> Dict:
    """Backward-compatible: extract just the scores dict."""
    result = extract_risk_assessment(raw_output)
    return result["scores"]
