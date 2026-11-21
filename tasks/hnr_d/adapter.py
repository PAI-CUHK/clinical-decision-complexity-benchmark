import random
from typing import Tuple
from core import MedicalProblemAdapter, MedicalNPProblem, Constraint, Objective

class NurseSchedulingNPCompleteAdapter(MedicalProblemAdapter):
    """Adapter for the nurse rostering decision task.
    
    Shifts are day (D), evening (E), night (N), or off (O). Rosters include
    daily coverage requirements, contract workloads, shift transition limits,
    limits on consecutive work and nights, and weekly night and rest constraints.
    The task asks whether a valid roster has total penalty at most k."""
    
    # Penalty weights shared with the baseline
    PENALTY_WORKLOAD_LOW = 10    # Underwork penalty
    PENALTY_WORKLOAD_HIGH = 20   # Overwork penalty
    PENALTY_NIGHT_EXCESS = 50    # Excess night-shift penalty
    PENALTY_REST_LOW = 10        # Insufficient rest penalty
    PENALTY_REST_HIGH = 15       # Excess rest penalty
    PENALTY_CONSECUTIVE_WORK = 50  # Excess consecutive working-days penalty
    PENALTY_CONSECUTIVE_NIGHT = 100  # Excess consecutive night-shifts penalty
    PENALTY_SKILL_VIOLATION = 300  # Skill-level violation penalty
    PENALTY_SHIFT_RESTRICTION = 150  # Shift restriction violation penalty
    PENALTY_FORBIDDEN_TRANSITION = 100  # Forbidden shift-transition penalty

    def _compute_optimal_penalty(self, instance):
        """Compute OPT for a NurseSchedulingInstance.
        
        Returns:
            A tuple (optimal penalty, whether hard constraints are satisfied)."""
        from tasks.hnr_d.baseline import compute_optimal_penalty
        
        instance_data = {
            'nurses': instance.nurses,
            'days': instance.days,
            'required': instance.required,
            'min_work': instance.min_work,
            'max_work': instance.max_work,
            'contracts': getattr(instance, 'contracts', {}),
            'nurse_skills': getattr(instance, 'nurse_skills', {}),
            'restrictions': getattr(instance, 'restrictions', {}),
            'max_consecutive_work': getattr(instance, 'max_consecutive_work', 6),
            'max_consecutive_night_shifts': getattr(instance, 'max_consecutive_night_shifts', 2),
            'max_night_shifts_per_week': getattr(instance, 'max_night_shifts_per_week', 3),
            'min_rest_per_week': getattr(instance, 'min_rest_per_week', 1),
            'max_rest_per_week': getattr(instance, 'max_rest_per_week', 2),
            'nurse_workload_bounds': getattr(instance, 'nurse_workload_bounds', {}),
        }
        
        # Compute the optimal penalty using the baseline solver
        return compute_optimal_penalty(instance_data)
    
    def _calculate_k_from_instance(self, instance) -> Tuple[int, int, str, bool]:
        """Compute the decision threshold k from the instance.
        
        Returns:
            (k, opt_value, expected_answer, is_feasible)."""
        # Reuse a precomputed k rather than running the solver again
        if hasattr(instance, 'k') and hasattr(instance, 'opt_value'):
            opt_value = instance.opt_value
            expected_answer = "Yes" if opt_value <= instance.k else "No"
            satisfies_hard_constraints = getattr(instance, 'satisfies_hard_constraints', True)
            return instance.k, opt_value, expected_answer, satisfies_hard_constraints
        
        # Step 1: compute OPT and check hard constraints
        opt_value, satisfies_hard_constraints = self._compute_optimal_penalty(instance)
        
        # Mark the instance invalid if hard constraints are infeasible
        if not satisfies_hard_constraints:
            return 0, opt_value, "No", False
        
        # With a nonnegative threshold, OPT=0 cannot produce a No instance.
        rng = random.Random(getattr(instance, 'seed', None))
        is_yes_instance = opt_value == 0 or rng.choice([True, False])
        k = (opt_value + rng.randint(0, 10) if is_yes_instance
             else max(0, opt_value - rng.randint(1, min(10, opt_value))))
        expected_answer = "Yes" if opt_value <= k else "No"
        
        return k, opt_value, expected_answer, True

    def generate_instance_config(self, level):
        """Generate nurse rostering configuration for scale level 1 through 10."""
        num_days = min(77, 14 + (level - 1) * 7)
        num_nurses = max(6, int(num_days * 0.43))
        
        if not 1 <= level <= 10:
            raise ValueError("level must be between 1 and 10")
        return {
            "num_nurses": num_nurses,
            "num_days": num_days,
            "demand_ratio": 0.45 + (level - 1) * 0.025,
            "demand_variation": 0.05,
        }
    
    def build(self, instance):
        """Build a MedicalNPProblem from a NurseSchedulingInstance."""
        problem = MedicalNPProblem(
            name="Nurse Scheduling NP-Complete Problem",
            problem_type="scheduling"
        )
        
        # Compute threshold k and optimal penalty
        k, opt_value, expected_answer, satisfies_hard_constraints = self._calculate_k_from_instance(instance)
        
        # Mark the problem invalid if hard constraints are not satisfied
        if not satisfies_hard_constraints:
            problem.instance_data = {
                "invalid": True,
                "reason": "Hard constraints are infeasible or the solver did not prove optimality",
                "opt_value": opt_value,
            }
            return problem
        
        # Store source instance data
        problem.instance_data = {
            "nurses": instance.nurses,
            "days": instance.days,
            "required": instance.required,
            "min_work": instance.min_work,
            "max_work": instance.max_work,
            "shift_types": ["D", "E", "N", "O"],
            "max_consecutive_work": getattr(instance, 'max_consecutive_work', 6),
            "max_consecutive_night_shifts": getattr(instance, 'max_consecutive_night_shifts', 2),
            "max_night_shifts_per_week": getattr(instance, 'max_night_shifts_per_week', 3),
            "min_rest_per_week": getattr(instance, 'min_rest_per_week', 1),
            "max_rest_per_week": getattr(instance, 'max_rest_per_week', 2),
            "nurse_skills": getattr(instance, 'nurse_skills', {}),
            "contracts": getattr(instance, 'contracts', {}),
            "preferences": getattr(instance, 'preferences', {}),
            "restrictions": getattr(instance, 'restrictions', {}),
            "nurse_workload_bounds": getattr(instance, 'nurse_workload_bounds', {}),
            "k": k,
            "opt_value": opt_value,
            "expected_answer": expected_answer,
            "penalty_weights": {
                "workload_low": self.PENALTY_WORKLOAD_LOW,
                "workload_high": self.PENALTY_WORKLOAD_HIGH,
                "night_excess": self.PENALTY_NIGHT_EXCESS,
                "rest_low": self.PENALTY_REST_LOW,
                "rest_high": self.PENALTY_REST_HIGH,
                "consecutive_work": self.PENALTY_CONSECUTIVE_WORK,
                "consecutive_night": self.PENALTY_CONSECUTIVE_NIGHT,
                "skill_violation": self.PENALTY_SKILL_VIOLATION,
                "shift_restriction": self.PENALTY_SHIFT_RESTRICTION,
                "forbidden_transition": self.PENALTY_FORBIDDEN_TRANSITION,
            }
        }
        
        # Set the problem description
        problem.description = (
            f"Determine whether there exists a nurse schedule with total penalty <= {k}."
        )
        
        # Update the clinical context without repeating the threshold
        problem.clinical_context = {
            "task_type": "Nurse scheduling feasibility determination",
            "description": "Hospital nurse rostering is a safety-critical operational planning task. Adequate coverage for day, evening, and night shifts is needed to maintain ward care capacity, while fatigue-related constraints and night-shift senior supervision help reduce patient safety risks. This task is limited to determining whether the stated scheduling and penalty rules admit a schedule with total penalty at or below the given threshold. Do not infer clinical diagnoses, treatment recommendations, or staffing policies beyond the provided rules."
        }
        
        # Update the decision space without repeating the threshold
        problem.decision_space = [
            "Assign each nurse to a shift (Day, Evening, Night, or Off) for each day",
            "Ensure staffing requirements are met while respecting labor regulations",
            "Calculate penalty for violating soft constraints",
            "Answer Yes if such a schedule exists, No otherwise"
        ]
        
        # Add metadata
        problem.metadata = {
            "num_nurses": len(instance.nurses),
            "num_days": len(instance.days),
            "k": k,
            "opt_value": opt_value,
            "expected_answer": expected_answer,
            "complexity": "NP-complete",
            "problem_type": "scheduling",
            "algorithm": "Constraint Satisfaction with Penalties",
            "shift_types": ["D", "E", "N", "O"]
        }
        
        # ========= Define entities =========
        for nurse in instance.nurses:
            problem.add_entity("nurses", {"id": nurse})
        
        for day in instance.days:
            problem.add_entity("days", {"id": day})
        
        # ========= Define constraints (rules) =========
        
        # 1. Shift type definitions
        problem.add_constraint(Constraint(
            name="shift_definitions",
            description="Shift types: Day (D) = 7am-3pm, Evening (E) = 3pm-11pm, Night (N) = 11pm-7am, and Off (O) = rest day. Working shifts include D, E, and N; O represents a rest day.",
            level="base",
            weight=0.0,
            category="definition",
            scope="general"
        ))
        
        # 2. Workload calculation rules
        problem.add_constraint(Constraint(
            name="workload_definition",
            description=f"Workload: Counted as the number of working shifts (D, E, N). Nurses have different workload requirements based on their contract type.",
            level="base",
            weight=0.0,
            category="definition",
            scope="nurse"
        ))
        
        # 3. Definition of a week
        problem.add_constraint(Constraint(
            name="week_definition",
            description="Week definition: Weeks are evaluated in consecutive 7-day blocks starting from Day 0: Week 1 = Day 0 to Day 6, Week 2 = Day 7 to Day 13, and so on. If the final week is incomplete, evaluate it using the available days in that block.",
            level="base",
            weight=0.0,
            category="definition",
            scope="general"
        ))
        
        # 4. Total penalty definition
        problem.add_constraint(Constraint(
            name="total_penalty_definition",
            description="Total penalty: Total penalty is the sum of all penalties from the SOFT CONSTRAINTS section.",
            level="base",
            weight=0.0,
            category="definition",
            scope="general"
        ))
        
        # ========= Hard Constraints =========
        
        # 3. Shift assignment rules (hard constraint)
        problem.add_constraint(Constraint(
            name="one_shift_per_day",
            description="Each nurse must be assigned exactly one shift per day",
            level="hard",
            weight=1000.0,
            category="validity",
            scope="nurse-day"
        ))
        
        # 4. Staffing requirements (hard constraint)
        problem.add_constraint(Constraint(
            name="shift_coverage",
            description="Each shift type must meet or exceed the required staffing level each day to ensure adequate patient care",
            level="hard",
            weight=1000.0,
            category="validity",
            scope="day-shift"
        ))
        
        # 5. Consecutive work limit (soft constraint)
        max_consecutive_work = getattr(instance, 'max_consecutive_work', 6)
        problem.add_constraint(Constraint(
            name="max_consecutive_work",
            description=f"Consecutive Working Days: Nurses should not work more than {max_consecutive_work} consecutive days. Penalty: {self.PENALTY_CONSECUTIVE_WORK} per extra consecutive day.",
            level="soft",
            weight=self.PENALTY_CONSECUTIVE_WORK,
            category="soft",
            scope="nurse"
        ))
        
        # 6. Consecutive night-shift limit (soft constraint)
        max_consecutive_night = getattr(instance, 'max_consecutive_night_shifts', 2)
        problem.add_constraint(Constraint(
            name="max_consecutive_night",
            description=f"Consecutive Night Shifts: Because repeated overnight work may increase fatigue, nurses should not work more than {max_consecutive_night} consecutive night shifts. Penalty: {self.PENALTY_CONSECUTIVE_NIGHT} per violation.",
            level="soft",
            weight=self.PENALTY_CONSECUTIVE_NIGHT,
            category="soft",
            scope="nurse"
        ))
        
        # 7. Skill-level constraint (soft constraint)
        required_night_skill = "Senior"
        restricted_night_skill = "Junior"
        problem.add_constraint(Constraint(
            name="skill_requirements",
            description=f"Skill Level on Night Shifts: Night coverage should include sufficient clinical experience. Each night shift should have at least one {required_night_skill} nurse, and {restricted_night_skill} nurses should not work night shifts alone. Penalty: {self.PENALTY_SKILL_VIOLATION} per violation.",
            level="soft",
            weight=self.PENALTY_SKILL_VIOLATION,
            category="soft",
            scope="nurse-shift"
        ))
        
        # 8. Shift transition restriction (N->D forbidden; soft constraint)
        problem.add_constraint(Constraint(
            name="forbidden_transitions",
            description=f"Shift Rotation (N to D): Nurses should not work a Day (D) shift immediately after a Night (N) shift. Penalty: {self.PENALTY_FORBIDDEN_TRANSITION} per violation.",
            level="soft",
            weight=self.PENALTY_FORBIDDEN_TRANSITION,
            category="soft",
            scope="nurse-consecutive"
        ))
        
        # 9. Weekly night-shift limit (soft constraint)
        max_night_shifts_per_week = getattr(instance, 'max_night_shifts_per_week', 3)
        if max_night_shifts_per_week > 0:
            problem.add_constraint(Constraint(
                name="max_night_shifts_per_week",
                description=f"Weekly Night Shifts: Each nurse may work at most {max_night_shifts_per_week} night shifts per week. Penalty for excess: {self.PENALTY_NIGHT_EXCESS} per extra night.",
                level="soft",
                weight=self.PENALTY_NIGHT_EXCESS,
                category="soft",
                scope="nurse"
            ))
        
        # 10. Weekly rest-day constraint (soft constraint)
        min_rest = getattr(instance, 'min_rest_per_week', 1)
        max_rest = getattr(instance, 'max_rest_per_week', 2)
        problem.add_constraint(Constraint(
            name="weekly_rest",
            description=f"Weekly Rest Days: Each nurse should have at least {min_rest} rest day(s) and at most {max_rest} rest day(s) per week. Penalty for deficit: {self.PENALTY_REST_LOW} per missing rest day. Penalty for excess: {self.PENALTY_REST_HIGH} per extra rest day.",
            level="soft",
            weight=self.PENALTY_REST_LOW,
            category="soft",
            scope="nurse"
        ))
        
        # 11. Contract workload constraint (soft constraint)
        # Extract full-time and part-time workload bounds from nurse_workload_bounds
        # so the prompt uses the same values as the baseline evaluator.
        nurse_workload_bounds = getattr(instance, 'nurse_workload_bounds', {})
        contracts = getattr(instance, 'contracts', {})
        ft_min, ft_max = 6, 10  # Fallback default
        pt_min, pt_max = 4, 7   # Fallback default
        for nurse, wlb in nurse_workload_bounds.items():
            if contracts.get(nurse, "full-time") == "full-time":
                ft_min, ft_max = wlb
            else:
                pt_min, pt_max = wlb
        problem.add_constraint(Constraint(
            name="contract_hours",
            description=f"Contract Workload Limits: Full-time nurses should work between {ft_min} and {ft_max} working days over the planning horizon. Part-time nurses should work between {pt_min} and {pt_max} working days over the planning horizon. Workload deficit penalty: {self.PENALTY_WORKLOAD_LOW} per shift below contract minimum. Workload excess penalty: {self.PENALTY_WORKLOAD_HIGH} per shift above contract maximum.",
            level="soft",
            weight=self.PENALTY_WORKLOAD_LOW,
            category="soft",
            scope="nurse"
        ))
        
        # ========= Objective =========
        problem.add_objective(Objective(
            name="penalty_minimization",
            description="Minimize total penalty and determine feasibility",
            direction="minimize",
            weight=1.0,
            priority=1,
            semantics={
                "type": "optimization",
                "scope": "penalty",
                "threshold": k,
                "opt_value": opt_value,
                "expected_answer": expected_answer
            }
        ))
        
        return problem
