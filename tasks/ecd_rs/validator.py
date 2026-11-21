"""
NEWS2 Risk Assessment Validator

Implements validation and evaluation logic for the NEWS2 benchmark.
"""

from core.validator import Validator
from tasks.ecd_rs.generator import compute_full_assessment
from tasks.ecd_rs.scorer import validate_medical_risk_assessment, extract_risk_assessment


def _normalize_risk_category(value):
    """Normalize risk category strings for robust comparison.

    Handles Unicode variants like en-dash (–, U+2013) vs hyphen (-, U+002D)
    that cause spurious mismatches between LLM output and baseline.
    """
    if not isinstance(value, str):
        return value
    # Replace en-dash and em-dash with regular hyphen
    return value.replace("\u2013", "-").replace("\u2014", "-").strip()


class MedicalRiskValidator(Validator):
    """NEWS2 Risk Assessment Validator."""

    def extract_answer(self, output: str):
        """
        Extract NEWS2 assessment from LLM output.

        Returns:
            Dict with 'scores', 'risk_category', 'urgent_review'.
        """
        try:
            return extract_risk_assessment(output)
        except Exception:
            return output

    def validate_solution(self, problem, solution):
        """Validate a NEWS2 solution against the baseline."""
        try:
            errors = {}
            valid = True

            baseline = getattr(problem, 'baseline', None)

            if baseline:
                solution_data = solution if isinstance(solution, dict) else {}
                baseline_data = baseline

                # Compare scores, risk categories, and urgent_review
                for key in ['scores', 'risk_category']:
                    bl = baseline_data.get(key, {})
                    sl = solution_data.get(key, {})
                    for pid in set(sl.keys()) | set(bl.keys()):
                        if pid not in bl:
                            valid = False
                            errors[f"extra_{pid}_{key}"] = 1
                        elif pid not in sl:
                            valid = False
                            errors[f"missing_{pid}_{key}"] = 1
                        elif sl[pid] != bl[pid]:
                            valid = False
                            errors[f"mismatch_{pid}_{key}"] = 1

                bl_urgent = set(baseline_data.get('urgent_review', []))
                sl_urgent = set(solution_data.get('urgent_review', []))
                for pid in sl_urgent - bl_urgent:
                    valid = False
                    errors[f"false_positive_urgent_{pid}"] = 1
                for pid in bl_urgent - sl_urgent:
                    valid = False
                    errors[f"false_negative_urgent_{pid}"] = 1
            else:
                benchmark = {
                    "prompt": "",
                    "instance": problem.instance_data,
                }
                feasible, errors_dict, _ = validate_medical_risk_assessment(benchmark, solution)
                if not feasible:
                    valid = False
                    errors.update(errors_dict)

            return {
                "valid": valid,
                "violations": list(errors.keys()),
                "details": {
                    "errors": dict(errors),
                    "baseline_matched": baseline is not None and valid,
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Validation error: {str(e)}"],
                "details": {"error": str(e)},
            }

    def calculate_baseline(self, problem):
        """Calculate the NEWS2 baseline for a problem."""
        patients = problem.instance_data.get("patients", {})
        if not patients:
            return {"scores": {}, "risk_category": {}, "urgent_review": []}

        assessment = compute_full_assessment(patients)
        return {
            "scores": assessment["scores"],
            "risk_category": assessment["risk_categories"],
            "urgent_review": assessment["urgent_review"],
        }

    def calculate_score(self, problem, solution):
        """Calculate the NEWS2 solution score."""
        try:
            benchmark = {
                "prompt": "",
                "instance": problem.instance_data,
            }
            _, _, accuracy = validate_medical_risk_assessment(benchmark, solution)
            return accuracy
        except Exception:
            return 0.0

    def evaluate_solution(self, problem, solution, baseline=None):
        """Evaluate a NEWS2 solution with full baseline comparison."""
        try:
            if baseline is None:
                baseline = self.calculate_baseline(problem)

            benchmark = {
                "prompt": "",
                "instance": problem.instance_data,
            }

            # Normalize solution
            if isinstance(solution, str):
                norm_solution = extract_risk_assessment(solution)
            elif isinstance(solution, dict):
                norm_solution = solution
            else:
                norm_solution = {}

            errors = {}
            baseline_matched = True

            for key in ['scores', 'risk_category']:
                bl = baseline.get(key, {})
                sl = norm_solution.get(key, {})
                for pid in set(sl.keys()) | set(bl.keys()):
                    if pid not in bl:
                        baseline_matched = False
                        errors[f"extra_{pid}_{key}"] = 1
                    elif pid not in sl:
                        baseline_matched = False
                        errors[f"missing_{pid}_{key}"] = 1
                    else:
                        # Normalize risk_category strings to handle en-dash vs hyphen
                        sl_val = _normalize_risk_category(sl[pid]) if key == 'risk_category' else sl[pid]
                        bl_val = _normalize_risk_category(bl[pid]) if key == 'risk_category' else bl[pid]
                        if sl_val != bl_val:
                            baseline_matched = False
                            errors[f"mismatch_{pid}_{key}"] = 1

            bl_urgent = set(baseline.get('urgent_review', []))
            sl_urgent = set(norm_solution.get('urgent_review', []))
            for pid in sl_urgent - bl_urgent:
                baseline_matched = False
                errors[f"false_positive_urgent_{pid}"] = 1
            for pid in bl_urgent - sl_urgent:
                baseline_matched = False
                errors[f"false_negative_urgent_{pid}"] = 1

            # Compute accuracy
            _, _, accuracy = validate_medical_risk_assessment(benchmark, norm_solution)

            return {
                "valid": baseline_matched,
                "violations": list(errors.keys()) if errors else [],
                "score": accuracy,
                "details": {
                    "errors": dict(errors),
                    "accuracy": accuracy,
                    "baseline_matched": baseline_matched,
                },
                "baseline_comparison": {
                    "baseline_answer": baseline,
                    "llm_answer": norm_solution,
                    "llm_accuracy": accuracy,
                    "baseline_matched": baseline_matched,
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {str(e)}"],
                "score": 0.0,
                "details": {"error": str(e)},
            }
