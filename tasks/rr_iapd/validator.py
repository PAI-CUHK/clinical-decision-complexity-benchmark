"""
ECG RR-Interval Pattern Recognition Validator

Implements validation and evaluation logic for the four predefined
RR-interval screening patterns (Pattern 1 - Pattern 4).
"""

from core.validator import Validator
from tasks.rr_iapd.generator import compute_ground_truth
from tasks.rr_iapd.scorer import (
    extract_patterns_from_output,
    validate_ecg_detection,
    PATTERN_NAMES,
)


class ECGValidator(Validator):
    """ECG RR-Interval Pattern Recognition Validator."""

    def extract_answer(self, output: str):
        """
        Extract the ECG pattern detection solution from an LLM output.

        Returns:
            Unified positions-only dict:
            {"Pattern 1": [positions], ..., "Pattern 4": [positions]}
        """
        try:
            parsed = extract_patterns_from_output(output)
            return {name: parsed[name]["positions"] for name in PATTERN_NAMES}
        except Exception:
            return output

    def validate_solution(self, problem, solution):
        """
        Validate an ECG pattern detection solution.

        The primary criterion is exact agreement with the deterministic
        baseline. When no baseline is available, the scorer is used.

        Args:
            problem: MedicalNPProblem object
            solution: positions-only dict {pattern: [positions]} or raw output

        Returns:
            dict: validation result
        """
        try:
            errors = {}
            valid = True

            baseline = getattr(problem, "baseline", None)

            if baseline:
                # Normalize both sides to positions-only dicts before comparing.
                formatted = self._to_positions_only(solution)
                if formatted != baseline:
                    valid = False
                    errors["solution_mismatch"] = "Solution differs from baseline"
            else:
                benchmark = {
                    "prompt": getattr(problem, "prompt", ""),
                    "instance": problem.instance_data,
                }
                feasible, errors_dict, _ = validate_ecg_detection(benchmark, solution)
                if not feasible:
                    valid = False
                    errors.update(errors_dict)

            return {
                "valid": valid,
                "violations": list(errors.keys()),
                "details": {
                    "errors": errors,
                    "baseline_matched": baseline is not None and valid,
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Validation error: {str(e)}"],
                "details": {"error": str(e), "baseline_matched": False},
            }

    def calculate_baseline(self, problem):
        """
        Compute the baseline (ground-truth) solution.

        Returns:
            Positions-only dict: {"Pattern 1": [...], ..., "Pattern 4": [...]}
        """
        rr_intervals = problem.instance_data.get("rr_intervals", [])
        if not rr_intervals:
            return {name: [] for name in PATTERN_NAMES}
        return compute_ground_truth(rr_intervals)

    def calculate_score(self, problem, solution):
        """
        Compute the F1-style accuracy score of a solution.

        Args:
            problem: MedicalNPProblem object
            solution: positions-only dict or raw LLM output

        Returns:
            float: accuracy score in [0, 1]
        """
        try:
            benchmark = {
                "prompt": getattr(problem, "prompt", ""),
                "instance": problem.instance_data,
            }
            _, _, accuracy = validate_ecg_detection(benchmark, solution)
            return accuracy
        except Exception:
            return 0.0

    def evaluate_solution(self, problem, solution, baseline=None):
        """
        Evaluate an ECG pattern detection solution against the baseline.

        Args:
            problem: MedicalNPProblem object
            solution: positions-only dict {pattern: [positions]} or raw output
            baseline: baseline solution (optional)

        Returns:
            dict: evaluation result
        """
        try:
            errors = {}

            if baseline is None:
                baseline = self.calculate_baseline(problem)

            formatted_solution = self._to_positions_only(solution)

            baseline_matched = (formatted_solution == baseline) if baseline else False

            if not baseline_matched and baseline:
                for name in PATTERN_NAMES:
                    solution_positions = set(formatted_solution.get(name, []))
                    baseline_positions = set(baseline.get(name, []))

                    false_positives = solution_positions - baseline_positions
                    false_negatives = baseline_positions - solution_positions

                    if false_positives:
                        errors[f"{name}_false_positives"] = len(false_positives)
                    if false_negatives:
                        errors[f"{name}_false_negatives"] = len(false_negatives)

            # Overall F1-style accuracy across all patterns.
            total_correct = 0
            total_predictions = 0
            total_ground_truth = 0

            for name in PATTERN_NAMES:
                solution_positions = set(formatted_solution.get(name, []))
                baseline_positions = set(baseline.get(name, [])) if baseline else set()

                total_correct += len(solution_positions & baseline_positions)
                total_predictions += len(solution_positions)
                total_ground_truth += len(baseline_positions)

            accuracy = self._f1(total_correct, total_predictions, total_ground_truth)

            evaluation_result = {
                "valid": baseline_matched,
                "violations": list(errors.keys()) if errors else [],
                "score": accuracy,
                "details": {
                    "errors": errors,
                    "accuracy": accuracy,
                    "baseline_matched": baseline_matched,
                },
                "baseline_comparison": {
                    "baseline_answer": baseline,
                    "llm_answer": formatted_solution,
                    "llm_accuracy": accuracy,
                    "baseline_matched": baseline_matched,
                },
            }

            return evaluation_result
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {str(e)}"],
                "score": 0.0,
                "details": {"error": str(e)},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_positions_only(solution):
        """
        Normalize a solution into a positions-only dict keyed by pattern name.

        Accepts a raw output string, a {pattern: [positions]} dict, or a
        {pattern: {"count", "positions"}} dict.
        """
        if isinstance(solution, str):
            parsed = extract_patterns_from_output(solution)
            return {name: parsed[name]["positions"] for name in PATTERN_NAMES}

        if isinstance(solution, dict):
            result = {name: [] for name in PATTERN_NAMES}
            for name in PATTERN_NAMES:
                value = solution.get(name, [])
                if isinstance(value, dict):
                    result[name] = list(value.get("positions", []))
                elif isinstance(value, list):
                    result[name] = list(value)
            return result

        return {name: [] for name in PATTERN_NAMES}

    @staticmethod
    def _f1(total_correct, total_predictions, total_ground_truth):
        if total_predictions == 0 and total_ground_truth == 0:
            return 1.0
        if total_predictions == 0 or total_ground_truth == 0:
            return 0.0
        precision = total_correct / total_predictions
        recall = total_correct / total_ground_truth
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
