"""Validation and evaluation for the nurse rostering decision task."""

from core.validator import Validator
from tasks.hnr_d.scorer import extract_yes_no_answer, normalize_yes_no
from tasks.hnr_d.baseline import generate_nurse_baseline


class NurseValidator(Validator):
    """Validator for the nurse rostering decision task."""
    
    def extract_answer(self, output: str):
        """Extract a nurse rostering answer from model output.
        
        Args:
            output: Model output text.
        
        Returns:
            The extracted solution."""
        try:
            answer = extract_yes_no_answer(output)
            return answer
        except Exception as e:
            return output
    
    def validate_solution(self, problem, solution):
        """Validate a nurse rostering answer against the baseline.
        
        Args:
            problem: MedicalNPProblem instance.
            solution: Model answer.
        
        Returns:
            Validation result dictionary."""
        try:
            errors = {}
            
            self._ensure_baseline(problem)
            
            baseline = problem.baseline
            if isinstance(baseline, dict):
                if "baseline_answer" not in baseline:
                    errors["missing_baseline_answer"] = "Baseline dict must contain 'baseline_answer' field"
                    return {
                        "valid": False,
                        "violations": list(errors.keys()),
                        "details": {
                            "errors": errors,
                            "baseline_matched": False
                        }
                    }
                baseline_answer = baseline["baseline_answer"]
            elif isinstance(baseline, str):
                baseline_answer = baseline
            else:
                baseline_answer = str(baseline)

            llm_answer = str(solution).strip()
            valid = normalize_yes_no(solution) == normalize_yes_no(baseline_answer)

            if not valid:
                errors["answer_mismatch"] = (
                    f"LLM answer '{llm_answer}' differs from baseline '{baseline_answer}'"
                )

            return {
                "valid": valid,
                "violations": list(errors.keys()),
                "details": {
                    "errors": errors,
                    "baseline_matched": valid,
                    "llm_answer": llm_answer,
                    "baseline_answer": baseline_answer,
                }
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Validation error: {str(e)}"],
                "details": {
                    "error": str(e),
                    "baseline_matched": False
                }
            }
    
    def calculate_baseline(self, problem):
        """Compute the reference answer for a nurse rostering problem.
        
        Raises:
            Exception: If reference generation fails."""
        return generate_nurse_baseline(problem.instance_data)
    
    def calculate_score(self, problem, solution):
        """Return the score for a nurse rostering answer compared with the baseline."""
        try:
            self._ensure_baseline(problem)
            
            baseline = problem.baseline
            if isinstance(baseline, dict):
                baseline_answer = baseline.get("baseline_answer", "NO")
            elif isinstance(baseline, str):
                baseline_answer = baseline
            else:
                baseline_answer = str(baseline)
            
            score = 1.0 if normalize_yes_no(solution) == normalize_yes_no(baseline_answer) else 0.0
            
            return score
        except Exception as e:
            return 0.0
    
    def evaluate_solution(self, problem, solution, baseline=None):
        """Evaluate a Yes/No nurse rostering answer against an optional baseline."""
        try:
            if baseline is None:
                baseline = self.calculate_baseline(problem)

            if isinstance(baseline, dict):
                baseline_answer = baseline.get("baseline_answer", "No")
            else:
                baseline_answer = str(baseline)

            llm_answer = str(solution).strip()
            correct = normalize_yes_no(solution) == normalize_yes_no(baseline_answer)

            evaluation_result = {
                "valid": correct,
                "violations": [] if correct else ["answer_mismatch"],
                "score": 1.0 if correct else 0.0,
                "details": {
                    "errors": {} if correct else {
                        "answer_mismatch": (
                            f"Answer '{llm_answer}' differs from baseline '{baseline_answer}'"
                        )
                    },
                    "correct": correct,
                },
            }

            evaluation_result["baseline_comparison"] = {
                "baseline_answer": baseline_answer,
                "llm_answer": llm_answer,
                "correct": correct,
            }
            
            return evaluation_result
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {str(e)}"],
                "score": 0.0,
                "details": {
                    "error": str(e),
                    "correct": False
                }
            }
