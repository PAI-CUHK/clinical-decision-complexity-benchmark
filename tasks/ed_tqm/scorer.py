import re
import json
from typing import Dict, Optional, List, Tuple
from tasks.ed_tqm.protocol import (
    ESCALATION_THRESHOLDS,
    MAX_ESCALATIONS,
)


def extract_ed_output(output: str) -> Optional[Dict]:
    """Extract ED scheduling results from LLM output."""
    root_match = re.search(r'<root>([\s\S]*?)</root>', output, re.DOTALL)
    if not root_match:
        return None
    root_content = root_match.group(1)
    final_answer_match = re.search(
        r'<final_answer>([\s\S]*?)</final_answer>', root_content, re.DOTALL
    )
    if not final_answer_match:
        return None
    final_answer_content = final_answer_match.group(1).strip()
    try:
        return json.loads(final_answer_content)
    except json.JSONDecodeError:
        return None


def compute_effective_triage(patient: Dict, wait_time: float) -> int:
    """
    Compute effective triage level via level-specific safe waiting targets.

    Incremental cascade: after each escalation, the threshold consumed so far
    is subtracted from the remaining wait time. The new level's threshold is
    checked against this *remaining* wait, not the original total.

    Example: ESI-3, wait=35min
      35 > 30 → upgrade to ESI-2, remaining wait = 35 - 30 = 5min
      5 > 10?  No → stop at ESI-2

    Example: ESI-3, wait=45min
      45 > 30 → upgrade to ESI-2, remaining wait = 45 - 30 = 15min
      15 > 10 → upgrade to ESI-1, remaining wait = 15 - 10 = 5min
      MAX_ESCALATIONS reached → stop at ESI-1
    """
    effective = patient["triage_level"]
    escalations = 0
    remaining_wait = wait_time

    while escalations < MAX_ESCALATIONS:
        threshold = ESCALATION_THRESHOLDS.get(effective)
        if threshold is not None and remaining_wait > threshold:
            remaining_wait -= threshold
            effective -= 1
            escalations += 1
        else:
            break

    return max(1, effective)


def _is_vulnerable(patient: Dict) -> bool:
    """< 2 years or >= 65 years = vulnerable (tie-breaker priority)."""
    age = patient.get("age", 0)
    return age < 2 or age >= 65


def calculate_ed_baseline(patients: List[Dict],
                          num_doctors: int = 1) -> Dict:
    """
    Ground-truth baseline: event-driven single-doctor simulation.

    Priority sort key at each step:
      effective_triage (asc) -> vulnerable (True first) -> NRS (desc)
      -> arrival_time (asc)
    """
    patient_dict = {p["id"]: p for p in patients}
    total_patients = len(patients)

    doctor_free_time = 0.0
    current_time = 0.0

    waiting_times: Dict[str, float] = {}
    order: List[str] = []
    scheduled = set()

    while len(scheduled) < total_patients:
        candidates = []
        for p in patients:
            pid = p["id"]
            if pid in scheduled:
                continue
            if p["arrival_time"] > current_time:
                continue

            wait_time = max(0, current_time - p["arrival_time"])
            effective_triage = compute_effective_triage(p, wait_time)

            candidates.append({
                "id": pid,
                "effective_triage": effective_triage,
                "vulnerable": _is_vulnerable(p),
                "nrs": p.get("nrs", 0),
                "age": p["age"],
                "arrival_time": p["arrival_time"],
                "processing_time": p["processing_time"],
                "triage_level": p["triage_level"],
            })

        if not candidates:
            next_arrivals = [
                p["arrival_time"] for p in patients
                if p["arrival_time"] > current_time
            ]
            if not next_arrivals:
                break
            current_time = min(next_arrivals)
            continue

        candidates.sort(
            key=lambda c: (
                c["effective_triage"],
                0 if c["vulnerable"] else 1,
                -c["nrs"],
                c["arrival_time"],
            )
        )

        best = candidates[0]
        pid = best["id"]
        p = patient_dict[pid]

        start_time = max(current_time, doctor_free_time)
        waiting_times[pid] = round(start_time - p["arrival_time"], 2)

        finish_time = start_time + p["processing_time"]
        doctor_free_time = finish_time
        current_time = finish_time

        order.append(pid)
        scheduled.add(pid)

    if waiting_times:
        avg_wait = round(sum(waiting_times.values()) / len(waiting_times), 2)
    else:
        avg_wait = 0.0

    return {
        "order": order,
        "waiting_times": waiting_times,
        "avg_wait": avg_wait,
    }


def compute_order_metrics(patients: List[Dict],
                           order: List[str]) -> Optional[Dict]:
    """
    Compute implied waiting times for a GIVEN processing order
    (single doctor, non-preemptive, arrival-aware).
    """
    patient_dict = {p["id"]: p for p in patients}
    expected_ids = set(patient_dict.keys())

    if not isinstance(order, list):
        return None
    if len(order) != len(patient_dict):
        return None
    if len(set(order)) != len(order):
        return None
    if set(order) != expected_ids:
        return None

    doctor_free_time = 0.0
    waiting_times: Dict[str, float] = {}
    for pid in order:
        p = patient_dict[pid]
        start_time = max(doctor_free_time, p["arrival_time"])
        waiting_times[pid] = round(start_time - p["arrival_time"], 2)
        doctor_free_time = start_time + p["processing_time"]

    if waiting_times:
        avg_wait = round(sum(waiting_times.values()) / len(waiting_times), 2)
    else:
        avg_wait = 0.0

    return {"waiting_times": waiting_times, "avg_wait": avg_wait}


def validate_ed_scheduling(instance: Dict, llm_output) -> Tuple[bool, Dict, Optional[float]]:
    """Validate LLM's ED scheduling result."""
    if isinstance(llm_output, dict):
        llm_result = llm_output
    elif isinstance(llm_output, str):
        llm_result = extract_ed_output(llm_output)
        if llm_result is None:
            return False, {"error": "Failed to extract ED scheduling results"}, None
    else:
        return False, {"error": f"Invalid input type: {type(llm_output).__name__}"}, None

    patients = instance.get("patients", [])
    num_doctors = instance.get("num_doctors", 1)

    baseline = calculate_ed_baseline(patients, num_doctors)

    errors = {}

    if "order" not in llm_result:
        errors["order"] = "Missing order in LLM output"
    elif not isinstance(llm_result["order"], list):
        errors["order"] = "Order must be a list"
    elif len(llm_result["order"]) != len(baseline["order"]):
        errors["order"] = (
            f"Incorrect order length: expected {len(baseline['order'])}, "
            f"got {len(llm_result['order'])}"
        )
    elif llm_result["order"] != baseline["order"]:
        # Record first mismatch position for debugging
        for i, (a, b) in enumerate(zip(llm_result["order"], baseline["order"])):
            if a != b:
                errors["order"] = (
                    f"Order differs from baseline at index {i} "
                    f"(baseline={b}, llm={a})"
                )
                break

    feasible = len(errors) == 0
    return feasible, errors, baseline["avg_wait"]
