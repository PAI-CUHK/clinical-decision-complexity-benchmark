#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG RR-Interval Pattern Recognition Scorer

Validates and scores ECG RR-interval pattern recognition solutions against the
deterministic baseline for the four predefined screening patterns
(Pattern 1 - Pattern 4).
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Union

from tasks.rr_iapd.generator import compute_ground_truth

PATTERN_NAMES = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4"]


def parse_ecg_prompt(prompt: str) -> List[float]:
    """
    Parse an ECG prompt to extract the RR interval sequence.

    Supports two instance-data formats:
    1. Bracketed list: ``RR = [0.9000, 0.9000, ...]`` (legacy)
    2. Numbered list: ``1. RR_1 = 0.9000`` / ``2. RR_2 = 0.9000`` (current)

    Returns:
        List of RR interval values (seconds).
    """
    if not isinstance(prompt, str):
        return []

    rr_intervals: List[float] = []

    # 1. Prefer a bracketed list that appears on a line starting with "RR =".
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("RR") and "=" in stripped:
            list_match = re.search(r"\[([^\]]*)\]", stripped)
            if list_match:
                rr_intervals = _parse_number_list(list_match.group(1))
                if rr_intervals:
                    return rr_intervals

    # 2. Numbered list format: "N. RR_N = value" (one per line).
    numbered = re.findall(
        r"^\s*\d+\.\s*RR_\d+\s*=\s*([0-9]+\.?[0-9]*)",
        prompt,
        flags=re.MULTILINE,
    )
    if numbered:
        values: List[float] = []
        for token in numbered:
            try:
                values.append(float(token))
            except ValueError:
                continue
        if values:
            return values

    # 3. Fallback: the first bracketed list of numbers anywhere in the prompt.
    for list_match in re.finditer(r"\[([^\]]*)\]", prompt):
        values = _parse_number_list(list_match.group(1))
        if len(values) >= 2:
            return values

    return rr_intervals


def _parse_number_list(text: str) -> List[float]:
    """Parse a comma/space separated list of numbers from a string."""
    values: List[float] = []
    for token in re.split(r"[,\s]+", text.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def extract_patterns_from_output(raw_output: str) -> Dict:
    """
    Extract pattern detection results from an LLM output string.

    Expects a ``<final_answer> ... </final_answer>`` block containing a JSON
    object mapping each pattern name to a list of 1-based positions, e.g.::

        {"Pattern 1": [6, 9], "Pattern 2": [], "Pattern 3": [14], "Pattern 4": [15]}

    Returns:
        Dict mapping each pattern name to a dict with 'count' and 'positions'.
    """
    if not isinstance(raw_output, str):
        raise ValueError("raw_output must be a string")

    match = re.search(
        r"<final_answer>\s*([\s\S]*?)\s*</final_answer>",
        raw_output,
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("Missing <final_answer> block")

    answer_text = match.group(1).strip()

    # Strip optional markdown code fences.
    answer_text = re.sub(r"^```(?:json)?\s*", "", answer_text)
    answer_text = re.sub(r"\s*```$", "", answer_text)

    result = {name: {"count": 0, "positions": []} for name in PATTERN_NAMES}

    parsed = None
    try:
        parsed = json.loads(answer_text)
    except json.JSONDecodeError:
        # Try to locate the first JSON object within the text.
        obj_match = re.search(r"\{[\s\S]*\}", answer_text)
        if obj_match:
            try:
                parsed = json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, dict):
        for name in PATTERN_NAMES:
            positions = parsed.get(name, [])
            positions = _coerce_positions(positions)
            result[name] = {"count": len(positions), "positions": positions}
        return result

    # Last-resort fallback: scan for "Pattern N": [...] entries.
    for name in PATTERN_NAMES:
        pattern_match = re.search(
            rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*\[([^\]]*)\]',
            answer_text,
            re.IGNORECASE,
        )
        if pattern_match:
            positions = _parse_number_list(pattern_match.group(1))
            positions = [int(round(v)) for v in positions]
            result[name] = {"count": len(positions), "positions": positions}

    return result


def _coerce_positions(value) -> List[int]:
    """Coerce a value into a list of integer 1-based positions."""
    if value is None:
        return []
    if isinstance(value, (int, float)):
        return [int(round(value))] if value else []
    if isinstance(value, list):
        positions: List[int] = []
        for v in value:
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                positions.append(int(round(v)))
            elif isinstance(v, str):
                try:
                    positions.append(int(round(float(v))))
                except ValueError:
                    continue
        return positions
    return []


def normalize_solution(solution: Union[str, Dict]) -> Dict:
    """
    Normalize a solution into the {pattern: {"count", "positions"}} form.

    Accepts either:
    - a raw LLM output string (parsed via ``extract_patterns_from_output``), or
    - a dict of the form {pattern: [positions]} or
      {pattern: {"count", "positions"}}.
    """
    if isinstance(solution, str):
        return extract_patterns_from_output(solution)

    if isinstance(solution, dict):
        result = {name: {"count": 0, "positions": []} for name in PATTERN_NAMES}
        for name in PATTERN_NAMES:
            value = solution.get(name)
            if value is None:
                continue
            if isinstance(value, dict):
                positions = _coerce_positions(value.get("positions", []))
                result[name] = {"count": len(positions), "positions": positions}
            else:
                positions = _coerce_positions(value)
                result[name] = {"count": len(positions), "positions": positions}
        return result

    raise ValueError("Unsupported solution type")


def validate_ecg_detection(benchmark: Dict, solution: Union[str, Dict]) -> Tuple[bool, Dict, float]:
    """
    Validate an ECG RR-interval pattern detection solution.

    Args:
        benchmark: Dict with an 'instance' key (containing 'rr_intervals') and/or
            a 'prompt' key.
        solution: Either the raw LLM output string or an already-extracted
            solution dict.

    Returns:
        (is_valid, errors_dict, accuracy_score)
    """
    errors: Dict = defaultdict(int)

    # Obtain the RR intervals.
    rr_intervals: List[float] = []
    if "instance" in benchmark and "rr_intervals" in benchmark["instance"]:
        rr_intervals = list(benchmark["instance"]["rr_intervals"])
    if not rr_intervals and "prompt" in benchmark:
        rr_intervals = parse_ecg_prompt(benchmark["prompt"])

    if not rr_intervals:
        errors["no_rr_intervals"] = 1
        return False, dict(errors), 0.0

    # Ground truth from the deterministic baseline.
    ground_truth = compute_ground_truth(rr_intervals)

    # Normalize the candidate solution.
    try:
        llm_answer = normalize_solution(solution)
    except ValueError as e:
        errors["parse_error"] = str(e)
        return False, dict(errors), 0.0

    total_correct = 0
    total_predictions = 0
    total_ground_truth = 0

    for name in PATTERN_NAMES:
        gt_positions = set(ground_truth[name])
        llm_positions = set(llm_answer[name]["positions"])

        if llm_answer[name]["count"] != len(llm_positions):
            errors[f"{name}_count_mismatch"] += 1

        correct = len(gt_positions & llm_positions)
        total_correct += correct
        total_predictions += len(llm_positions)
        total_ground_truth += len(gt_positions)

        false_positives = llm_positions - gt_positions
        if false_positives:
            errors[f"{name}_false_positives"] = len(false_positives)

        false_negatives = gt_positions - llm_positions
        if false_negatives:
            errors[f"{name}_false_negatives"] = len(false_negatives)

    accuracy = _f1_accuracy(total_correct, total_predictions, total_ground_truth)
    is_valid = len(errors) == 0
    return is_valid, dict(errors), accuracy


def _f1_accuracy(total_correct: int, total_predictions: int, total_ground_truth: int) -> float:
    """Compute an overall F1-style accuracy across all patterns."""
    if total_predictions == 0 and total_ground_truth == 0:
        return 1.0
    if total_predictions == 0 or total_ground_truth == 0:
        return 0.0
    precision = total_correct / total_predictions
    recall = total_correct / total_ground_truth
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def format_detection_result(pattern: str, count: int, positions: List[int]) -> str:
    """Format the detection result for a single pattern."""
    return f"{pattern}: count = {count}, positions = {positions}"
