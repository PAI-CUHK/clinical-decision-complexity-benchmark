"""
ED Scheduling Problem Validator — Single Doctor with Dynamic Priority Escalation

P-class problem with a single deterministic solution: validation is done by
directly comparing the LLM answer's processing order against the baseline
order, mirroring the ECG validator pattern. (Waiting times / avg_wait are no
longer requested from the LLM; they are derived internally for reporting.)
"""

from core.validator import Validator
from tasks.ed_tqm.scorer import (
    extract_ed_output,
    calculate_ed_baseline,
    compute_order_metrics,
)


class EDValidator(Validator):
    """ED Scheduling problem validator (baseline-consistency version)."""

    def extract_answer(self, output: str):
        """
        Extract ED scheduling solution from LLM output.

        Args:
            output: LLM output text

        Returns:
            Parsed solution dict or raw output
        """
        try:
            return extract_ed_output(output)
        except Exception:
            return output

    def validate_solution(self, problem, solution):
        """
        Validate an ED scheduling solution by direct comparison against the
        deterministic baseline.

        For a P-class problem the schedule is uniquely determined by the
        rules, so the LLM answer is correct iff its processing order
        exactly matches the baseline order.

        Args:
            problem: MedicalNPProblem object
            solution: LLM solution dict

        Returns:
            dict: validation result with valid, violations, details
        """
        try:
            errors = {}

            if not isinstance(solution, dict):
                return self._fail("invalid_format",
                                  "Solution must be a dictionary")

            # Resolve baseline (do NOT call _ensure_baseline here to avoid
            # raising; fall back to computing on the fly).
            baseline = getattr(problem, "baseline", None)
            if baseline is None:
                baseline = self.calculate_baseline(problem)

            # --- order: exact list equality (only required field) ---
            if "order" not in solution:
                errors["missing_order"] = "Missing 'order' field"
            else:
                llm_order = solution["order"]
                base_order = baseline.get("order", [])
                if not isinstance(llm_order, list):
                    errors["order_type"] = "'order' must be a list"
                elif llm_order != base_order:
                    errors["order_mismatch"] = (
                        f"Order differs from baseline at "
                        f"{self._first_diff_index(base_order, llm_order)}"
                    )

            valid = len(errors) == 0
            return self._result(valid, errors,
                                baseline_matched=valid)

        except Exception as e:
            return self._fail("validation_error", f"Validation error: {e}")

    def calculate_baseline(self, problem):
        """
        Calculate the dynamic-priority ED baseline.

        Args:
            problem: MedicalNPProblem object

        Returns:
            Baseline solution dict {order, waiting_times, avg_wait}
        """
        patients = problem.instance_data.get("patients", [])
        num_doctors = problem.instance_data.get("num_doctors", 1)
        return calculate_ed_baseline(patients, num_doctors)

    def calculate_score(self, problem, solution):
        """
        Score a solution. For a P-class problem with a unique solution,
        the score is binary: 1.0 if it matches the baseline, else 0.0.

        Args:
            problem: MedicalNPProblem object
            solution: Solution dict

        Returns:
            float: 1.0 or 0.0
        """
        try:
            validation = self.validate_solution(problem, solution)
            return 1.0 if validation["valid"] else 0.0
        except Exception:
            return 0.0

    def evaluate_solution(self, problem, solution, baseline=None):
        """
        Evaluate ED scheduling solution against the baseline.

        Args:
            problem: MedicalNPProblem
            solution: LLM solution
            baseline: Baseline (optional, computed if None)

        Returns:
            dict with evaluation results
        """
        try:
            self._ensure_baseline(problem)

            if baseline is None:
                baseline = problem.baseline

            validation = self.validate_solution(problem, solution)
            score = self.calculate_score(problem, solution)

            baseline_avg_wait = baseline.get("avg_wait", 0.0)

            # The LLM no longer reports waiting times; derive the avg_wait
            # implied by its proposed order for reporting/comparison.
            llm_avg_wait = 0.0
            if isinstance(solution, dict) and isinstance(solution.get("order"), list):
                patients = problem.instance_data.get("patients", [])
                metrics = compute_order_metrics(patients, solution["order"])
                if metrics is not None:
                    llm_avg_wait = metrics["avg_wait"]

            return {
                "valid": validation["valid"],
                "violations": validation["violations"],
                "score": score,
                "details": {
                    "errors": validation["details"]["errors"],
                    "llm_avg_wait": llm_avg_wait,
                    "baseline_avg_wait": baseline_avg_wait,
                    "baseline_matched": validation["valid"],
                },
                "baseline_comparison": {
                    "baseline_avg_wait": baseline_avg_wait,
                    "llm_avg_wait": llm_avg_wait,
                    "baseline_matched": validation["valid"],
                },
            }
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
    def _result(valid, errors, baseline_matched):
        return {
            "valid": valid,
            "violations": list(errors.keys()),
            "details": {
                "errors": errors,
                "baseline_matched": baseline_matched,
            },
        }

    @staticmethod
    def _fail(error_key, message):
        return {
            "valid": False,
            "violations": [error_key],
            "details": {
                "errors": {error_key: message},
                "baseline_matched": False,
            },
        }

    @staticmethod
    def _first_diff_index(a, b):
        """Return a human-readable description of the first index where two
        lists differ (or differ in length)."""
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return f"index {i} (baseline={a[i]}, llm={b[i]})"
        if len(a) != len(b):
            return f"length (baseline={len(a)}, llm={len(b)})"
        return "unknown"
