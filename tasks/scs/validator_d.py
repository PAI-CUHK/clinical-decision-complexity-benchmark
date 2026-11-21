"""Validation and evaluation for the operating-room scheduling decision task."""

import re
from core.validator import Validator


class ORSchedulingNPCompleteValidator(Validator):
    """Validator for the operating-room scheduling decision task."""
    
    def extract_answer(self, output: str):
        """Extract Yes/No from the final_answer tag in model output.
        
        Return None when the tag is absent or its content is not Yes/No, including
        when the output is truncated or incorrectly formatted."""
        try:
            # Extract only from the final_answer tag
            match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', output, re.DOTALL | re.IGNORECASE)
            if match:
                answer = match.group(1).strip().lower()
                if answer in ['yes', 'no']:
                    return answer

            # Return None when no valid final_answer tag is found
            return None
        except Exception as e:
            return None
    
    def validate_solution(self, problem, solution):
        """Validate a Yes/No scheduling answer against the reference answer."""
        try:
            errors = {}
            valid = True
            
            # Convert the answer to a lowercase string
            solution_str = str(solution).lower().strip()
            
            # Compare with the baseline, the main validation criterion
            baseline = problem.baseline if hasattr(problem, 'baseline') and problem.baseline else None
            
            if baseline:
                # Get the reference answer
                baseline_answer = str(baseline).lower().strip()
                
                # Check whether the answer matches the reference
                if solution_str != baseline_answer:
                    valid = False
                    errors["answer_mismatch"] = f"Answer '{solution}' differs from baseline '{baseline}'"
            else:
                # Validate the answer format if no baseline is available
                if solution_str not in ['yes', 'no']:
                    valid = False
                    errors["invalid_answer_format"] = f"Answer '{solution}' is not 'Yes' or 'No'"
            
            return {
                "valid": valid,
                "violations": list(errors.keys()),
                "details": {
                    "errors": errors,
                    "baseline_matched": baseline is not None and valid
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
        """Compute the Yes/No reference using the OR-Tools optimal penalty OPT.
        
        Return Yes if OPT <= k and No otherwise. Penalty settings from instance_data
        include urgency-based cancellations, a specialty mismatch penalty of 3, and
        an idle-time penalty of 0.01 per minute. Capacity includes turnover time:
        10 minutes for the same specialty and 20 minutes for different specialties."""
        from tasks.scs.baseline import generate_baseline_with_ortools
        
        # Get the problem parameters
        instance_data = problem.instance_data
        or_days = instance_data.get('or_days', [])
        surgeries = instance_data.get('surgeries', [])
        k = instance_data.get('k', 5)
        turnover = instance_data.get('turnover', {'same_specialty': 10, 'different_specialty': 20})
        
        # Compute the optimal penalty with OR-Tools
        baseline_data = {
            'surgeries': surgeries,
            'or_days': or_days,
            'turnover': turnover
        }
        
        try:
            baseline_result = generate_baseline_with_ortools(baseline_data)
            if baseline_result.get('status') != 'optimal':
                raise RuntimeError("Solver did not prove an optimal baseline")
            opt_penalty = baseline_result['total_penalty']
            
            # Check whether a feasible solution exists
            if opt_penalty <= k:
                return "Yes"
            else:
                return "No"
        except Exception as e:
            # Raise an error if OR-Tools fails
            raise RuntimeError(f"OR-Tools baseline generation failed: {str(e)}")
    
    def calculate_score(self, problem, solution):
        """Return 1.0 for a correct Yes/No answer and 0.0 otherwise."""
        try:
            # Get the reference answer
            baseline = self.calculate_baseline(problem)
            
            # Compare answers
            solution_str = str(solution).lower().strip()
            baseline_str = str(baseline).lower().strip()
            
            return 1.0 if solution_str == baseline_str else 0.0
        except Exception as e:
            return 0.0
    
    def evaluate_solution(self, problem, solution, baseline=None):
        """Evaluate a Yes/No scheduling answer against an optional reference."""
        try:
            # Convert the answer to a lowercase string
            solution_str = str(solution).lower().strip()
            
            # Compute a baseline if none was supplied
            if baseline is None:
                baseline = self.calculate_baseline(problem)
            
            # Get the reference answer
            baseline_answer = str(baseline).lower().strip()
            
            # Check whether the answer is correct
            correct = solution_str == baseline_answer
            
            # Build the evaluation result
            evaluation_result = {
                "valid": correct,
                "violations": [] if correct else ["answer_mismatch"],
                "score": 1.0 if correct else 0.0,
                "details": {
                    "errors": {} if correct else {"answer_mismatch": f"Answer '{solution}' differs from baseline '{baseline}'"},
                    "correct": correct
                }
            }
            
            # Add baseline comparison details
            evaluation_result["baseline_comparison"] = {
                "baseline_answer": baseline,
                "llm_answer": solution,
                "correct": correct
            }
            
            return evaluation_result
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Evaluation error: {str(e)}"],
                "score": 0.0,
                "details": {
                    "error": str(e)
                }
            }
