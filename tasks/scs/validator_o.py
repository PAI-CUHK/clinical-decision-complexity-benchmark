"""Validation and evaluation for the operating-room scheduling optimization task."""

from core.validator import Validator
from tasks.scs.scorer import validate_or_schedule, extract_schedule_from_output
from tasks.scs.baseline import generate_baseline_with_ortools
from tasks.scs.entity_check import or_entity_errors


class ORValidator(Validator):
    """Validator for the operating-room scheduling optimization task."""
    
    def _validate_schedule_dict(self, benchmark, solution_dict):
        """Validate a parsed operating-room schedule.
        
        Args:
            benchmark: Dictionary containing instance data.
            solution_dict: Parsed solution.
        
        Returns:
            (is_feasible, errors_dict, unused_time, specialty_penalty)."""
        from collections import defaultdict
        
        errors = defaultdict(int)
        is_feasible = True
        specialty_penalty = 0
        
        # Get surgery and OR-day data
        instance_data = benchmark.get("instance", {})
        surgeries_list = instance_data.get("surgeries", [])
        or_days_list = instance_data.get("or_days", [])
        
        surgeries = {s["id"]: s for s in surgeries_list}
        or_days = {o["id"]: o for o in or_days_list}
        
        # Get the schedule
        schedule = solution_dict.get("schedule", {})
        
        # Track which surgeries are assigned
        assigned_surgeries = set()
        # Initialize usage for all OR-days so empty days count toward unused time
        or_day_usage = {
            did: {"used": 0, "capacity": info["capacity"], "unused": info["capacity"]}
            for did, info in or_days.items()
        }
        
        # Get turnover configuration from instance data
        instance_turnover = instance_data.get("turnover", {})
        same_specialty_turnover = instance_turnover.get("same_specialty", 10)
        different_specialty_turnover = instance_turnover.get("different_specialty", 20)
        
        # Read penalty settings from instance_data with generator-compatible defaults
        penalties = instance_data.get("penalties", {})
        penalty_per_mismatch = penalties.get("specialty_mismatch", 3)
        
        # Validate each OR-day assignment
        for or_day_id, surgery_list in schedule.items():
            # Check if OR-day exists
            if or_day_id not in or_days:
                errors["invalid_or_day"] += 1
                is_feasible = False
                continue
            
            or_day = or_days[or_day_id]
            total_duration = 0
            
            # Get specialties supported by each OR-day
            or_specialties = or_day.get("specialties", [])
            
            # Track specialties in this OR-day for turnover calculation
            specialties_in_or = set()
            
            for surgery_id in surgery_list:
                # Check if surgery exists
                if surgery_id not in surgeries:
                    errors["invalid_surgery"] += 1
                    is_feasible = False
                    continue
                
                # Check for duplicate assignment
                if surgery_id in assigned_surgeries:
                    errors["duplicate_assignment"] += 1
                    is_feasible = False
                else:
                    assigned_surgeries.add(surgery_id)
                
                # Add to total duration
                total_duration += surgeries[surgery_id]["duration"]
                
                # Track specialty for turnover calculation
                surgery_specialty = surgeries[surgery_id].get("specialty")
                if surgery_specialty:
                    specialties_in_or.add(surgery_specialty)
                
                # Check specialty matching (soft constraint)
                if or_specialties and surgery_specialty and surgery_specialty not in or_specialties:
                    specialty_penalty += penalty_per_mismatch
                    errors["specialty_mismatch"] = errors.get("specialty_mismatch", 0) + 1
            
            # Calculate turnover time
            # For n surgeries with s distinct specialties: turnover = 10*n + 10*s - 20
            num_surgeries_in_or = len(surgery_list)
            num_distinct_specialties = len(specialties_in_or)
            if num_surgeries_in_or >= 1:
                turnover_time = same_specialty_turnover * num_surgeries_in_or + same_specialty_turnover * num_distinct_specialties - 2 * same_specialty_turnover
            else:
                turnover_time = 0
            
            # Total occupied time = surgery duration + turnover
            total_occupied = total_duration + turnover_time
            
            # Check capacity constraint (considering turnover)
            if total_occupied > or_day["capacity"]:
                errors["capacity_exceeded"] += 1
                is_feasible = False
            
            or_day_usage[or_day_id] = {
                "used": total_duration,
                "turnover": turnover_time,
                "occupied": total_occupied,
                "capacity": or_day["capacity"],
                "unused": or_day["capacity"] - total_occupied
            }
        
        # Determine cancelled surgeries
        all_surgeries = set(surgeries.keys())
        inferred_cancelled = all_surgeries - assigned_surgeries
        cancelled = inferred_cancelled
        entity_errors = or_entity_errors(instance_data, solution_dict)
        if entity_errors:
            is_feasible = False
            errors["entity_hallucination"] = entity_errors
        
        # Calculate total unused time
        total_unused_time = sum(usage["unused"] for usage in or_day_usage.values())
        
        # Verify reported unused time (if provided)
        if "total_unused_time" in solution_dict:
            reported_unused = solution_dict["total_unused_time"]
            if abs(reported_unused - total_unused_time) > 0.1:
                errors["incorrect_unused_time_calculation"] += 1
        
        # Store objective components
        errors["cancellations"] = len(cancelled)
        errors["cancellation_time"] = sum(
            surgeries[sid]["duration"] for sid in cancelled if sid in surgeries
        )
        errors["idle_time"] = total_unused_time
        errors["specialty_penalty"] = specialty_penalty
        
        return is_feasible, dict(errors), total_unused_time, specialty_penalty
    
    def extract_answer(self, output: str):
        """Extract an operating-room schedule from model output."""
        try:
            return extract_schedule_from_output(output)
        except Exception as e:
            return output
    
    def validate_solution(self, problem, solution, baseline=None):
        """Validate a schedule and return valid, violations, and details including
        errors, unused_time, and specialty_penalty."""
        try:
            errors = {}
            valid = True
            unused_time = 0.0
            specialty_penalty = 0
            total_penalty = 0
            
            # Build the benchmark object
            instance_data = problem.instance_data if hasattr(problem, 'instance_data') else {}
            
            # Check whether instance_data has the required fields
            if not instance_data or "surgeries" not in instance_data or "or_days" not in instance_data:
                # Try to obtain data from problem entities
                if hasattr(problem, 'entities'):
                    entity_surgeries = problem.entities.get('surgeries', [])
                    entity_or_days = problem.entities.get('or_days', [])
                    if entity_surgeries and entity_or_days:
                        instance_data = {
                            "surgeries": entity_surgeries,
                            "or_days": entity_or_days
                        }
            
            benchmark = {
                "prompt": "",
                "instance": instance_data
            }
            
            # Validate the parsed solution dictionary
            if isinstance(solution, dict):
                feasible, errors_dict, unused_time, specialty_penalty = self._validate_schedule_dict(benchmark, solution)
            else:
                feasible, errors_dict, unused_time = validate_or_schedule(benchmark, str(solution))
                specialty_penalty = errors_dict.get("specialty_penalty", 0)
            
            if not feasible:
                valid = False

            # A missing or malformed final answer has no measurable objective.
            # The scorer reports unused_time=None in this case; never turn it
            # into a zero penalty (which would appear to beat the baseline).
            if unused_time is None:
                return {
                    "valid": False,
                    "violations": list(errors_dict.keys()) or ["parse_error"],
                    "details": {
                        "errors": errors_dict or {"parse_error": 1},
                        "unused_time": None,
                        "specialty_penalty": None,
                        "total_penalty": None,
                        "feasible": False,
                        "baseline_matched": False,
                    },
                }
            
            # Include all errors, including non-feasibility errors such as time calculation errors
            errors.update(errors_dict)
            
            # Read penalty settings from instance_data
            penalties = instance_data.get("penalties", {})
            cancellation_penalties = penalties.get("cancellation", {})
            emergency_penalty = cancellation_penalties.get("emergency", 20)
            urgent_penalty = cancellation_penalties.get("urgent", 8)
            elective_penalty = cancellation_penalties.get("elective", 1)
            idle_time_coefficient = penalties.get("idle_time", 0.01)
            
            # Calculate total cancellation and specialty mismatch penalties
            cancellation_penalty = 0
            surgeries_dict = {s["id"]: s for s in instance_data.get("surgeries", [])}
            
            # Count actual unassigned surgeries; a declaration cannot reduce cost.
            if isinstance(solution, dict):
                assigned_surgeries = {
                    sid for ids in solution.get("schedule", {}).values() for sid in ids}
                for sid in set(surgeries_dict) - assigned_surgeries:
                    urgency = surgeries_dict[sid].get("urgency", "elective")
                    cancellation_penalty += {
                        "emergency": emergency_penalty, "urgent": urgent_penalty,
                        "elective": elective_penalty,
                    }.get(urgency, elective_penalty)
            else:
                # Estimate from error counts if the data is not a dictionary
                cancellation_count = errors_dict.get("cancellations", 0)
                cancellation_penalty = cancellation_count * urgent_penalty  # Use an average estimate
            
            # Calculate the idle-time penalty
            idle_time_penalty = unused_time * idle_time_coefficient
            errors["idle_time_penalty"] = idle_time_penalty
            
            # Total penalty = cancellation + specialty mismatch + idle-time penalties
            total_penalty = cancellation_penalty + specialty_penalty + idle_time_penalty
            errors["total_penalty"] = total_penalty
            
            # Compare against the same cached baseline used by evaluate_solution.
            baseline_penalty = None
            try:
                if baseline is None:
                    baseline = getattr(problem, "baseline", None)
                if baseline is None:
                    baseline = self.calculate_baseline(problem)
                if baseline and isinstance(baseline, dict):
                    baseline_penalty = baseline.get("total_penalty")
            except Exception as e:
                pass

            if not isinstance(baseline_penalty, (int, float)):
                valid = False
                errors["missing_baseline_penalty"] = True
            else:
                errors["baseline_penalty"] = baseline_penalty
            
            # Mark the solution invalid if its penalty exceeds the baseline
            if baseline_penalty is not None and total_penalty > baseline_penalty:
                valid = False
                errors["penalty_exceeds_baseline"] = True
                errors["penalty_excess"] = total_penalty - baseline_penalty
            
            return {
                "valid": valid,
                "violations": [] if valid else list(errors.keys()),
                "details": {
                    "errors": errors,
                    "unused_time": unused_time,
                    "specialty_penalty": specialty_penalty,
                    "cancellation_penalty": cancellation_penalty,
                    "total_penalty": total_penalty,
                    "baseline_penalty": baseline_penalty,
                    "feasible": feasible,
                    "baseline_matched": valid
                }
            }
        except Exception as e:
            return {
                "valid": False,
                "violations": [f"Validation error: {str(e)}"],
                "details": {
                    "error": str(e),
                    "errors": {"validation_error": 1},
                    "unused_time": None,
                    "specialty_penalty": None,
                    "total_penalty": None,
                    "feasible": False,
                    "baseline_matched": False
                }
            }
    
    def calculate_baseline(self, problem):
        """Compute an operating-room scheduling reference using OR-Tools CP-SAT.
        
        Raises:
            Exception: If reference generation fails."""
        # Generate the reference with OR-Tools CP-SAT
        instance_data = problem.instance_data
        baseline = generate_baseline_with_ortools(instance_data)
        return baseline
    
    def calculate_score(self, problem, solution):
        """Calculate the score of an operating-room schedule."""
        try:
            # Build the benchmark object
            instance_data = problem.instance_data if hasattr(problem, 'instance_data') else {}
            
            # Check whether instance_data has the required fields
            if not instance_data or "surgeries" not in instance_data or "or_days" not in instance_data:
                # Try to obtain data from problem entities
                if hasattr(problem, 'entities'):
                    entity_surgeries = problem.entities.get('surgeries', [])
                    entity_or_days = problem.entities.get('or_days', [])
                    if entity_surgeries and entity_or_days:
                        instance_data = {
                            "surgeries": entity_surgeries,
                            "or_days": entity_or_days
                        }
            
            benchmark = {
                "prompt": "",  # The OR scheduling validator requires a prompt
                "instance": instance_data
            }
            
            # Validate the solution
            valid, errors, unused_time = validate_or_schedule(benchmark, solution)
            
            if not valid:
                return 0.0
            
            # Calculate the score: less unused time gives a higher score
            # Assume a total duration of 8 hours (480 minutes)
            total_time = 480.0
            score = max(0.0, 1.0 - (unused_time / total_time))
            
            return score
        except Exception as e:
            return 0.0
    
    def evaluate_solution(self, problem, solution, baseline=None):
        """Evaluate a parsed operating-room schedule against an optional baseline."""
        try:
            # Require a baseline and raise an error if missing
            self._ensure_baseline(problem)
            
            # Build the benchmark object
            instance_data = problem.instance_data if hasattr(problem, 'instance_data') else {}
            
            # Check whether instance_data has the required fields
            if not instance_data or "surgeries" not in instance_data or "or_days" not in instance_data:
                # Try to obtain data from problem entities
                if hasattr(problem, 'entities'):
                    entity_surgeries = problem.entities.get('surgeries', [])
                    entity_or_days = problem.entities.get('or_days', [])
                    if entity_surgeries and entity_or_days:
                        instance_data = {
                            "surgeries": entity_surgeries,
                            "or_days": entity_or_days
                        }
            
            benchmark = {
                "prompt": "",
                "instance": instance_data
            }
            
            # Resolve the baseline once and use it for both validation and scoring.
            if baseline is None:
                baseline = getattr(problem, "baseline", None)
            if baseline is None:
                baseline = self.calculate_baseline(problem)

            # Validate constraints and compare with the baseline using validate_solution
            validation_result = self.validate_solution(problem, solution, baseline)
            valid = validation_result["valid"]
            errors = validation_result.get("details", {}).get("errors", {})
            unused_time = validation_result.get("details", {}).get("unused_time")
            specialty_penalty = validation_result.get("details", {}).get("specialty_penalty")
            feasible = validation_result.get("details", {}).get("feasible", False)
            
            # Compute the score from total_penalty and baseline_total_penalty
            # Get the model total_penalty
            llm_total_penalty = errors.get("total_penalty")
            
            # Get the baseline total_penalty
            baseline_total_penalty = baseline.get("total_penalty") if baseline and isinstance(baseline, dict) else None
            
            # Lower model penalties yield higher scores
            # score = max(0, 1 - (llm_penalty - baseline_penalty) / baseline_penalty)
            # If llm_penalty <= baseline_penalty, the score is 1.0
            # If llm_penalty > baseline_penalty, the score decreases
            if not feasible or llm_total_penalty is None or baseline_total_penalty is None:
                score = 0.0
            elif baseline_total_penalty > 0:
                if llm_total_penalty <= baseline_total_penalty:
                    score = 1.0
                else:
                    # Fraction by which the penalty exceeds the baseline
                    excess_ratio = (llm_total_penalty - baseline_total_penalty) / baseline_total_penalty
                    score = max(0.0, 1.0 - excess_ratio)
            else:
                # When baseline_total_penalty is zero, score 1.0 only for zero model penalty
                score = 1.0 if llm_total_penalty == 0 else 0.0
            
            # Build the evaluation result
            evaluation_result = {
                "valid": valid,
                "violations": validation_result.get("violations", []),
                "score": score,
                "details": {
                    "errors": errors,
                    "unused_time": unused_time,
                    "specialty_penalty": specialty_penalty,
                    "llm_total_penalty": llm_total_penalty,
                    "baseline_total_penalty": baseline_total_penalty
                }
            }
            
            # Add baseline comparison details
            evaluation_result["baseline_comparison"] = {
                "baseline_total_penalty": baseline_total_penalty,
                "llm_total_penalty": llm_total_penalty,
                "llm_better": bool(
                    valid and llm_total_penalty is not None
                    and baseline_total_penalty is not None
                    and llm_total_penalty <= baseline_total_penalty
                )
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
