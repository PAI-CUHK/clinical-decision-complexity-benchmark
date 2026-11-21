"""PASP answer validation and comparison with the CP-SAT baseline."""

import json
import re

from core.validator import Validator
from .baseline import generate_baseline_with_validation
from .scorer import validate_instance_segments


MAX_PENALTY = 1000.0


def evaluate_pasp_answer(solution, patients, rooms, baseline=None):
    """Score one answer with the exact policy rules used by the baseline.

    The scorer and CP-SAT model share ``pasp_penalty``. Only room preference
    accrues per patient-day; gender, age and specialism accrue per segment,
    while transfers accrue per room change.
    """
    raw = solution if isinstance(solution, str) else json.dumps(solution)
    feasible, violations, penalty = validate_instance_segments(
        {"prompt": ""}, raw, patients=patients, rooms=rooms,
    )
    errors = dict(violations)
    score = max(0.0, 1.0 - penalty / MAX_PENALTY) if feasible else 0.0

    if baseline is None:
        matched = bool(feasible)
        baseline_comparison = {}
    else:
        baseline_penalty = float(baseline["total_penalty"])
        matched = bool(feasible and penalty <= baseline_penalty + 1e-9)
        if feasible and not matched:
            errors["penalty_higher_than_baseline"] = {
                "solution_penalty": penalty,
                "baseline_penalty": baseline_penalty,
            }
        baseline_score = max(0.0, 1.0 - baseline_penalty / MAX_PENALTY)
        baseline_comparison = {
            "baseline_score": baseline_score,
            "llm_score": score,
            "llm_better": score >= baseline_score,
            "baseline_matched": matched,
        }

    return {
        "valid": matched,
        "violations": list(errors),
        "score": score,
        "details": {
            "errors": errors,
            "penalty": penalty,
            "feasible": bool(feasible),
            "baseline_matched": matched,
        },
        "baseline_comparison": baseline_comparison,
    }


class PASPValidator(Validator):
    """Validate PASP room schedules and score them against a baseline."""

    def extract_answer(self, output: str):
        if not isinstance(output, str):
            return output
        for pattern in (
            r"<final_answer>\s*([\s\S]*?)\s*</final_answer>",
            r"<root>\s*([\s\S]*?)\s*</root>",
        ):
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except (TypeError, ValueError):
                    pass
        try:
            return json.loads(output)
        except (TypeError, ValueError):
            return output

    def _extract_patients_rooms(self, problem):
        patients = {}
        for item in problem.instance_data.get("patients", []):
            patient = item.copy()
            if "stay" in patient:
                patient["start_day"], patient["end_day"] = patient["stay"]
            patients[patient["id"]] = patient
        rooms = {room["id"]: room for room in problem.instance_data.get("rooms", [])}
        return patients, rooms

    def validate_solution(self, problem, solution):
        try:
            patients, rooms = self._extract_patients_rooms(problem)
            result = evaluate_pasp_answer(
                solution, patients, rooms, getattr(problem, "baseline", None),
            )
            return {
                "valid": result["valid"],
                "violations": result["violations"],
                "details": result["details"],
            }
        except Exception as exc:
            return {
                "valid": False,
                "violations": [f"Validation error: {exc}"],
                "details": {"error": str(exc), "baseline_matched": False},
            }

    def calculate_baseline(self, problem):
        patients, rooms = self._extract_patients_rooms(problem)
        return generate_baseline_with_validation(patients, rooms)

    def calculate_score(self, problem, solution):
        try:
            patients, rooms = self._extract_patients_rooms(problem)
            return evaluate_pasp_answer(solution, patients, rooms)["score"]
        except Exception:
            return 0.0

    def evaluate_solution(self, problem, solution, baseline=None):
        try:
            baseline = baseline if baseline is not None else getattr(problem, "baseline", None)
            if not baseline:
                self._ensure_baseline(problem)
            patients, rooms = self._extract_patients_rooms(problem)
            return evaluate_pasp_answer(solution, patients, rooms, baseline)
        except Exception as exc:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {exc}"],
                "score": 0.0,
                "details": {"error": str(exc), "baseline_matched": False},
                "baseline_comparison": {},
            }
