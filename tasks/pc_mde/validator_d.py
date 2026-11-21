"""Validator for the PC-MDE decision task.

The decision and optimization tasks share the objective and weights defined in
tasks/pc_mde/objective.py. The instance generator establishes the threshold
and expected answer from a feasible reference and an objective upper bound.
The validator compares the model YES/NO answer with stored has_solution."""

from core.validator import Validator
from tasks.pc_mde.scorer import extract_yes_no_answer


class DiseaseDiagnosisNPCompleteValidator(Validator):
    """Validator for the PC-MDE decision task."""

    def extract_answer(self, output: str):
        """Extract a YES/NO answer from the model output."""
        try:
            return extract_yes_no_answer(output)
        except Exception:
            return output

    def _truth(self, problem):
        """Return Yes if a valid solution reaches the threshold; otherwise No."""
        try:
            has_solution = problem.instance_data.get("has_solution", False)
        except Exception:
            has_solution = False
        return "Yes" if has_solution else "No"

    def calculate_baseline(self, problem):
        """Return the expected Yes/No answer stored in the instance."""
        return self._truth(problem)

    def validate_solution(self, problem, solution):
        """Check whether the model YES/NO answer matches the expected answer."""
        try:
            truth = self._truth(problem)

            def normalize(answer):
                if isinstance(answer, bool):
                    return "yes" if answer else "no"
                if answer is None:
                    return None
                return str(answer).lower().strip()

            sol = normalize(solution)
            tru = normalize(truth)

            if sol is None:
                return {
                    "valid": False,
                    "violations": ["no_valid_answer"],
                    "details": {
                        "errors": {"no_valid_answer": "No valid <final_answer> tag found"},
                        "truth": truth,
                        "answer": solution,
                    },
                }

            if sol != tru:
                return {
                    "valid": False,
                    "violations": ["answer_mismatch"],
                    "details": {
                        "errors": {"answer_mismatch": f"Answer {solution} differs from truth {truth}"},
                        "truth": truth,
                        "answer": solution,
                    },
                }

            return {
                "valid": True,
                "violations": [],
                "details": {"truth": truth, "answer": solution, "correct": True},
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Validation error: {str(e)}"],
                "details": {"errors": {"error": str(e)}},
            }

    def calculate_score(self, problem, solution):
        """Return 1.0 for a correct answer and 0.0 otherwise."""
        return 1.0 if self.validate_solution(problem, solution)["valid"] else 0.0

    def evaluate_solution(self, problem, solution, baseline=None):
        """Evaluate the model answer against the stored or provided baseline."""
        try:
            if baseline is None:
                baseline = self._truth(problem)

            def normalize(answer):
                if isinstance(answer, bool):
                    return "yes" if answer else "no"
                if answer is None:
                    return None
                return str(answer).lower().strip()

            sol = normalize(solution)
            tru = normalize(baseline)
            correct = sol is not None and sol == tru

            return {
                "valid": correct,
                "violations": [] if correct else (["no_valid_answer"] if sol is None else ["answer_mismatch"]),
                "score": 1.0 if correct else 0.0,
                "details": {
                    "errors": {} if correct else {
                        "no_valid_answer" if sol is None else "answer_mismatch": (
                            "No valid <final_answer> tag found" if sol is None
                            else f"Answer {solution} differs from baseline {baseline}"
                        )
                    },
                    "correct": correct,
                },
                "baseline_comparison": {
                    "baseline_answer": baseline,
                    "llm_answer": solution,
                    "correct": correct,
                },
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {str(e)}"],
                "score": 0.0,
                "details": {"errors": {"error": str(e)}},
            }
