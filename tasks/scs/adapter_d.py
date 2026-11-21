#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter for the operating-room scheduling decision task."""

import random

from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective

class ORSchedulingNPCompleteAdapter(MedicalProblemAdapter):
    """Adapter for the operating-room scheduling decision task."""

    def generate_instance_config(self, level):
        """Generate the decision task instance configuration.
        
        Args:
            level: Scale level from 1 to 10.
        
        Returns:
            Instance configuration."""
        # Use the same instance configuration as OR Scheduling
        # Use the same parameters at each level; scale changes through counts
        base_surgeries = 20
        num_surgeries = base_surgeries + (level - 1) * 3
        num_or_days = max(4, (num_surgeries + 3) // 5)
        
        return {
            "num_surgeries": num_surgeries,
            "num_or_days": num_or_days,
            "load_factor": 0.75 + (level - 1) * 0.015,
            "level": level
        }
    
    def _compute_optimal_penalty(self, surgeries, or_days):
        """Compute the optimal penalty using the shared baseline solver.
        
        Args:
            surgeries: List of surgeries.
            or_days: List of operating-room days.
        
        Returns:
            The optimal penalty."""
        from tasks.scs.baseline import generate_baseline_with_ortools
        
        instance_data = {
            'surgeries': surgeries,
            'or_days': or_days
        }
        
        baseline = generate_baseline_with_ortools(instance_data)
        
        if baseline.get('status') != 'optimal':
            return None
        return baseline['total_penalty']
    
    def _calculate_k_from_instance(self, instance):
        """Choose a threshold around a certified optimum and derive its label."""
        surgeries = instance.surgeries
        or_days = instance.or_days
        
        # Step 1: compute the optimal penalty (OPT)
        opt_value = self._compute_optimal_penalty(surgeries, or_days)
        
        if opt_value is None:
            return None, None, None
        is_yes_instance = opt_value <= 0 or random.choice([True, False])
        if is_yes_instance:
            k = round(opt_value + random.uniform(0.5, 5.0), 2)
        else:
            k = round(max(0.0, opt_value - random.uniform(0.5, 5.0)), 2)
            if k >= opt_value:
                k = 0.0
        expected_answer = "Yes" if opt_value <= k else "No"
        
        return k, opt_value, expected_answer
    
    def build(self, instance):
        """Build the decision problem from an ORInstance.
        
        Returns:
            The constructed MedicalNPProblem."""
        problem = MedicalNPProblem(
            name="Operating Room Scheduling NP-Complete Problem",
            problem_type="scheduling"
        )
        
        # Reuse k and opt_value when already present in a loaded instance
        if hasattr(instance, 'k') and hasattr(instance, 'opt_value') and instance.k > 0:
            k = instance.k
            opt_value = instance.opt_value
            expected_answer = getattr(instance, 'expected_answer', "Yes")
            print(f"  使用已加载的 k={k}, opt_value={opt_value}, expected_answer={expected_answer}")
        else:
            # Calculate threshold k using the standard method
            k, opt_value, expected_answer = self._calculate_k_from_instance(instance)
            if opt_value is None:
                problem.instance_data = {"invalid": True,
                                         "reason": "Solver did not prove optimality"}
                return problem
        
        # Store source instance data
        problem.instance_data = {
            "surgeries": instance.surgeries,
            "or_days": instance.or_days,
            "k": k,
            "opt_value": opt_value,
            "expected_answer": expected_answer,
            "turnover": {
                "same_specialty": 10,
                "different_specialty": 20
            }
        }
        
        # Calculate total surgery duration and OR-day capacity
        total_surgery_time = sum(surgery['duration'] for surgery in instance.surgeries)
        total_capacity = sum(or_day['capacity'] for or_day in instance.or_days)
        excess = total_surgery_time - total_capacity
        
        # Describe the problem and total_penalty calculation
        problem.description = (
            f"Determine whether there exists a scheduling solution for the following operating room problem "
            f"with total penalty <= {k}, where: "
            f"total penalty = weighted cancellation penalty + cross-specialty mismatch penalty + idle time penalty."
        )
        
        # Set a concise clinical context
        problem.clinical_context = {
            "description": (
                f"Operating room (OR) scheduling feasibility problem involves determining whether a feasible "
                f"schedule exists that assigns surgical procedures to available OR-days while keeping the "
                f"total penalty within an acceptable threshold {k}. Each surgery has a duration, urgency level, "
                f"and specialty requirement. Each OR-day has a time capacity and supported specialties. "
                f"Cross-specialty assignment is allowed but incurs a mismatch penalty."
            )
        }
        
        # Set the decision space
        problem.decision_space = [
            "Determine whether each surgery should be assigned to an OR-day or cancelled",
            "For each OR-day, determine which surgeries are scheduled within its capacity",
            "Cross-specialty assignment is allowed but incurs a mismatch penalty",
            f"Verify if a schedule exists with total penalty <= {k}",
            "Answer Yes if such a schedule exists, No otherwise"
        ]
        
        # Add metadata
        problem.metadata = {
            "num_surgeries": len(instance.surgeries),
            "num_or_days": len(instance.or_days),
            "k": k,
            "total_surgery_time": total_surgery_time,
            "total_capacity": total_capacity,
            "excess": excess,
            "complexity": "NP-complete",
            "problem_type": "scheduling"
        }
        
        # ========= Define entities =========
        for surgery in instance.surgeries:
            problem.add_entity("surgeries", surgery)
        
        for or_day in instance.or_days:
            problem.add_entity("or_days", or_day)
        
        # ========= Define actions (decision variables) =========
        # x[s, d] = whether surgery s is assigned to OR-day d
        for surgery in instance.surgeries:
            sid = surgery["id"]
            for or_day in instance.or_days:
                did = or_day["id"]
                problem.add_action(
                    Action(
                        name=f"assign_{sid}_to_{did}",
                        description=f"Assign surgery {sid} to {did}"
                    )
                )
        
        # ========= Definitions and Rules =========
        problem.add_constraint(Constraint(
            name="capacity_definition",
            description=(
                "Occupied Time: The total occupied time of an OR-day equals: "
                "sum(surgery durations) + sum(turnover times between consecutive surgeries). "
                "The total occupied time of an OR-day must not exceed its capacity."
            ),
            level='base',
            weight=0.0,
            category="definition"
        ))
        
        problem.add_constraint(Constraint(
            name="turnover_time_rule",
            description=(
                "Turnover Time: For consecutive surgeries in the same OR-day: "
                "Same specialty: 10 minutes. "
                "Different specialties: 20 minutes."
            ),
            level='base',
            weight=0.0,
            category="definition"
        ))
        
        problem.add_constraint(Constraint(
            name="urgency_priority_rule",
            description="Urgency Priority: Urgency levels define clinical importance order: emergency > urgent > elective. Emergency surgeries have the highest priority for scheduling.",
            level='clinical',
            weight=0.0,
            category="definition",
            semantics={
                "urgency_levels": ["emergency", "urgent", "elective"],
                "priority_order": "emergency > urgent > elective"
            }
        ))
        
        problem.add_constraint(Constraint(
            name="total_penalty_definition",
            description="Total penalty: Total penalty = cancellation penalty + cross-specialty mismatch penalty + idle time penalty.",
            level='base',
            weight=0.0,
            category="definition"
        ))
        
        # ========= Validity Constraints =========
        problem.add_constraint(Constraint(
            name="only_listed_validity",
            description="Only listed surgeries and OR-days may be used in the schedule.",
            level='base',
            weight=1000.0,
            category="validity"
        ))
        
        problem.add_constraint(Constraint(
            name="assignment_once_validity",
            description="Each surgery may be assigned to at most one OR-day.",
            level='base',
            weight=1000.0,
            category="validity"
        ))
        
        problem.add_constraint(Constraint(
            name="no_split_validity",
            description="Surgeries cannot be split across OR-days.",
            level='base',
            weight=1000.0,
            category="validity"
        ))
        
        problem.add_constraint(Constraint(
            name="capacity_validity",
            description="For every OR-day, occupied time must not exceed its capacity.",
            level='base',
            weight=1000.0,
            category="validity"
        ))
        
        problem.add_constraint(Constraint(
            name="scheduled_cancelled_validity",
            description="A surgery assigned to an OR-day counts as scheduled; an unassigned surgery counts as cancelled.",
            level='base',
            weight=1000.0,
            category="validity"
        ))
        
        # ========= Soft Constraints =========
        problem.add_constraint(Constraint(
            name="cancellation_penalty",
            description="Cancellation Penalty: Each cancelled surgery incurs a penalty based on urgency: Emergency surgery = 20; Urgent surgery = 8; Elective surgery = 1",
            level='base',
            weight=1.0,
            category="soft"
        ))
        
        problem.add_constraint(Constraint(
            name="specialty_penalty",
            description="Cross-specialty Mismatch Penalty: If a surgery is assigned to an OR-day that does not support the surgery's specialty, a mismatch penalty of 3 is incurred.",
            level='base',
            weight=3.0,
            category="soft"
        ))
        
        problem.add_constraint(Constraint(
            name="idle_time_penalty",
            description="Idle Time Penalty: For each OR-day, idle time equals capacity minus occupied time. Idle time penalty equals idle time multiplied by 0.01.",
            level='base',
            weight=0.01,
            category="soft"
        ))
        
        # ========= Objective =========
        problem.add_objective(Objective(
            name="minimize_total_penalty",
            description=(
                f"Determine if there exists a schedule with total penalty <= {k}. "
                f"The total penalty is calculated as: weighted cancellation penalty + cross-specialty mismatch penalty + idle time penalty. "
                f"Refer to DEFINITIONS AND RULES for penalty weight details."
            ),
            direction="minimize",
            weight=1.0,
            priority=1,
            semantics={
                "type": "total_penalty",
                "scope": "all_surgeries",
                "max_total_penalty": k,
                "cancellation_penalties": {
                    "emergency": 20,
                    "urgent": 8,
                    "elective": 1
                },
                "cross_specialty_penalty": 3,
                "idle_time_coefficient": 0.01
            }
        ))
        
        return problem
    
