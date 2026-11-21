#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG RR-Interval Pattern Recognition Baseline

Provides the deterministic baseline solution for the four predefined
RR-interval screening patterns.

This is a P-class problem: a linear scan that uses a rolling five-sample
median as the Baseline RR.

Pattern 1 - Short-Long RR Pattern
Pattern 2 - Periodic Short-Long RR Pattern
Pattern 3 - Prolonged RR Interval
Pattern 4 - Relative RR Difference Episode
"""

import json
from statistics import median
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants (mirror the rule definitions exactly)
# ---------------------------------------------------------------------------
SHORT_RATIO = 0.80            # Short  : RR <  80% of Baseline RR
LONG_RATIO = 1.20             # Long   : RR > 120% of Baseline RR
PROLONGED_PAUSE = 2.0         # Pattern 3 pause criterion (seconds)
REL_DIFF_THRESHOLD = 0.0325   # Pattern 4 relative RR difference (3.25%)
PATTERN4_WINDOW = 10          # Pattern 4 sliding-window size (RR intervals)
PATTERN4_MIN_EVENTS = 8       # min Relative RR Difference Events per window
PATTERN2_MIN_REPEATS = 3      # min consecutive repeats for Pattern 2
BASELINE_WINDOW = 5           # number of preceding RR intervals used for Baseline

PATTERN_NAMES = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4"]


def compute_baseline_rr(rr_intervals: List[float]) -> List[Optional[float]]:
    """
    Compute the Baseline RR for every position.

    Beginning with RR_6 (1-based), the Baseline RR for RR_i is the median of
    the five immediately preceding RR intervals (RR_{i-5} .. RR_{i-1}).
    The first five RR intervals are reference intervals only and therefore
    have no Baseline RR (represented as None).
    """
    n = len(rr_intervals)
    baseline: List[Optional[float]] = [None] * n
    for i in range(n):
        # 0-based index i corresponds to 1-based position i + 1.
        # Baseline is defined beginning with RR_6, i.e. i + 1 >= 6  =>  i >= 5.
        if i >= BASELINE_WINDOW:
            window = rr_intervals[i - BASELINE_WINDOW:i]
            baseline[i] = median(window)
    return baseline


def classify_rr(rr_intervals: List[float]) -> List[str]:
    """
    Classify each RR interval as 'Short', 'Long', 'Normal' or 'Reference'.

    'Reference' is used for the first five intervals, which have no Baseline RR
    and therefore cannot be classified as Short / Long / Normal.
    """
    baseline = compute_baseline_rr(rr_intervals)
    labels: List[str] = []
    for i, rr in enumerate(rr_intervals):
        b = baseline[i]
        if b is None:
            labels.append("Reference")
            continue
        if rr < SHORT_RATIO * b:
            labels.append("Short")
        elif rr > LONG_RATIO * b:
            labels.append("Long")
        else:
            labels.append("Normal")
    return labels


def detect_pattern1(labels: List[str]) -> List[int]:
    """
    Pattern 1 - Short-Long RR Pattern.

    Occurs when RR_i is Short and RR_{i+1} is Long.
    Reports the 1-based position of RR_i.
    """
    positions: List[int] = []
    n = len(labels)
    for i in range(n - 1):
        if labels[i] == "Short" and labels[i + 1] == "Long":
            positions.append(i + 1)
    return positions


def detect_pattern2(labels: List[str]) -> List[int]:
    """
    Pattern 2 - Periodic Short-Long RR Pattern.

    A repeated Short-Long timing pattern with constant periodicity.

    Each repeated unit is one Short RR immediately followed by one Long RR.
    The number of consecutive Normal RR intervals between neighboring Short RR
    intervals must remain constant, and the same pattern must repeat at least
    three consecutive times.

    Reports the 1-based position of the first Short RR interval of every
    qualifying periodic run. Overlapping runs (which start at the boundary
    Short of a previous run when the periodicity changes) are also reported.
    """
    n = len(labels)

    # A "unit start" is a position p (0-based) where RR_p is Short and
    # RR_{p+1} is Long.
    unit_starts = [
        p for p in range(n - 1)
        if labels[p] == "Short" and labels[p + 1] == "Long"
    ]

    positions: List[int] = []
    m = len(unit_starts)
    if m < PATTERN2_MIN_REPEATS:
        return positions

    i = 0
    while i < m:
        chain = [unit_starts[i]]
        gap: Optional[int] = None
        j = i
        while j + 1 < m:
            p_cur = unit_starts[j]
            p_next = unit_starts[j + 1]
            # Number of consecutive Normal RR intervals strictly between the
            # Long of the current unit (p_cur + 1) and the next Short (p_next).
            g = p_next - p_cur - 2
            if g < 0:
                break  # overlapping units - cannot form the pattern
            # All intervening positions must be Normal.
            if any(labels[k] != "Normal" for k in range(p_cur + 2, p_next)):
                break
            if gap is None:
                gap = g
            if g != gap:
                break
            chain.append(p_next)
            j += 1

        if len(chain) >= PATTERN2_MIN_REPEATS:
            positions.append(chain[0] + 1)  # 1-based position of first Short
            # Allow a new periodic pattern (with a different periodicity) to
            # start at the last unit of this run.
            i = j
        else:
            i += 1
    return positions


def detect_pattern3(rr_intervals: List[float]) -> List[int]:
    """
    Pattern 3 - Prolonged RR Interval.

    Occurs when RR_i >= 2.0 seconds.
    Reports the 1-based position of RR_i.
    """
    positions: List[int] = []
    for i, rr in enumerate(rr_intervals):
        if rr >= PROLONGED_PAUSE:
            positions.append(i + 1)
    return positions


def detect_pattern4(rr_intervals: List[float]) -> List[int]:
    """
    Pattern 4 - Relative RR Difference Episode.

    For every adjacent pair (RR_i, RR_{i+1}) compute the relative RR difference
    |RR_{i+1} - RR_i| / RR_i. If it is at least 3.25%, record one Relative RR
    Difference Event for that pair.

    Then examine every sliding window of 10 consecutive RR intervals. A window
    covering indices w .. w+9 contains the 9 adjacent pairs w .. w+8. If at
    least 8 of those pairs are events, Pattern 4 is detected at the window's
    1-based starting position.
    """
    n = len(rr_intervals)
    if n < 2:
        return []

    # event[k] is True when pair (k, k+1) is a Relative RR Difference Event.
    event = [False] * (n - 1)
    for k in range(n - 1):
        prev = rr_intervals[k]
        if prev <= 0:
            continue
        rel_diff = abs(rr_intervals[k + 1] - prev) / prev
        if rel_diff >= REL_DIFF_THRESHOLD:
            event[k] = True

    positions: List[int] = []
    if n < PATTERN4_WINDOW:
        return positions

    for w in range(n - PATTERN4_WINDOW + 1):
        # pairs belonging to window w .. w + 9 are w .. w + 8 (9 pairs)
        events_in_window = sum(1 for k in range(w, w + PATTERN4_WINDOW - 1) if event[k])
        if events_in_window >= PATTERN4_MIN_EVENTS:
            positions.append(w + 1)
    return positions


def detect_patterns(rr_intervals: List[float]) -> Dict:
    """
    Detect all four predefined RR-interval screening patterns.

    Returns:
        Dict with keys "Pattern 1" .. "Pattern 4", each mapping to a dict with
        'count' and 'positions' (1-based).
    """
    labels = classify_rr(rr_intervals)
    p1 = detect_pattern1(labels)
    p2 = detect_pattern2(labels)
    p3 = detect_pattern3(rr_intervals)
    p4 = detect_pattern4(rr_intervals)
    return {
        "Pattern 1": {"count": len(p1), "positions": p1},
        "Pattern 2": {"count": len(p2), "positions": p2},
        "Pattern 3": {"count": len(p3), "positions": p3},
        "Pattern 4": {"count": len(p4), "positions": p4},
    }


def format_baseline_output(detections: Dict) -> str:
    """
    Format baseline detection results as a compact JSON string mapping each
    pattern name to its list of 1-based starting positions.
    """
    result = {name: detections[name]["positions"] for name in PATTERN_NAMES}
    return json.dumps(result, ensure_ascii=False)


def generate_ecg_baseline(prompt: str, rr_intervals: List[float] = None) -> str:
    """
    Generate the baseline solution for an ECG RR-interval pattern task.

    Args:
        prompt: The problem prompt (used only to parse RR intervals when
            ``rr_intervals`` is not provided).
        rr_intervals: List of RR intervals (seconds). If None, parsed from the
            prompt.

    Returns:
        Formatted baseline output wrapped in <final_answer> tags.
    """
    if rr_intervals is None:
        from tasks.rr_iapd.scorer import parse_ecg_prompt
        rr_intervals = parse_ecg_prompt(prompt)

    if not rr_intervals:
        raise ValueError("No RR intervals provided or parsed")

    detections = detect_patterns(rr_intervals)
    output = format_baseline_output(detections)
    return f"<final_answer>\n{output}\n</final_answer>"


if __name__ == "__main__":
    # Quick self-test.
    test_rr = [
        0.82, 0.79, 0.80, 0.81, 0.78,   # reference intervals (positions 1-5)
        0.50, 1.10, 0.80,                # Short-Long unit (positions 6-8)
        0.50, 1.10, 0.80,                # Short-Long unit (positions 9-11)
        0.50, 1.10,                      # Short-Long unit (positions 12-13)
        2.30,                            # prolonged pause (position 14)
        0.55, 1.15, 0.55, 1.15, 0.55, 1.15, 0.55, 1.15, 0.55, 1.15,  # high-variance window
    ]

    print("Test RR intervals:", test_rr)
    print("Classification   :", classify_rr(test_rr))
    print("Baseline output  :", format_baseline_output(detect_patterns(test_rr)))
