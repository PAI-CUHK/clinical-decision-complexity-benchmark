"""
Operating Room Scheduling Problem Adapter

Translates OR scheduling instance into the unified problem framework
"""

from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective

class ORSchedulingAdapter(MedicalProblemAdapter):
    """
    Operating Room Scheduling adapter
    
    This is a classic Bin Packing / Multiple Knapsack problem:
    - Assign surgeries to OR-days
    - Respect capacity constraints
    - Primary goal: minimize cancellations
    - Secondary: minimize total duration of cancelled surgeries
    - Tertiary: minimize idle OR time
    """

    def generate_instance_config(self, level):
        """Generate the operating-room scheduling instance configuration.
        
        Args:
            level: Scale level from 1 to 10.
        
        Returns:
            Instance configuration."""
        # Adjust the target load with scale; sampled durations determine actual load.
        base_surgeries = 20  # Base number of surgeries
        num_surgeries = base_surgeries + (level - 1) * 3  # Increase by three surgeries per level
        num_or_days = max(4, (num_surgeries + 3) // 5)  # About one fifth of the number of surgeries
        
        return {
            "num_surgeries": num_surgeries,
            "num_or_days": num_or_days,
            "load_factor": 0.75 + (level - 1) * 0.015,
            "level": level
        }
    
    def build(self, instance):
        problem = MedicalNPProblem(
            name="Operating Room Scheduling Problem",
            problem_type="scheduling"
        )
        
        # Store source instance data
        problem.instance_data = {
            "surgeries": instance.surgeries,
            "or_days": instance.or_days,
            "turnover": instance.turnover if hasattr(instance, 'turnover') else {
                "same_specialty": 10,
                "different_specialty": 20
            }
        }
        
        # Set the problem description
        problem.description = (
            "All surgery information and OR-day capacities are known in advance. "
            "The goal is to find a schedule that minimizes the total clinical "
            "and operational penalty under weighted objectives."
        )
        
        # Set a concise clinical context
        problem.clinical_context = {
            "description": (
                "Operating room (OR) scheduling involves assigning surgical procedures to available "
                "OR-days while respecting constraints. Each surgery has a duration, urgency level, and specialty requirement. "
                "Each OR-day has a time capacity and may be restricted to certain specialties."
            )
        }
        
        # Set the decision space
        problem.decision_space = [
            "Assign each surgery to exactly one OR-day or leave it cancelled",
            "For each OR-day, determine which surgeries are scheduled within its capacity",
            "Select which surgeries to cancel when total duration exceeds capacity",
            "Return the complete schedule in the required JSON format"
        ]
        
        # Add metadata
        problem.metadata = {
            "num_surgeries": len(instance.surgeries),
            "num_or_days": len(instance.or_days),
            "complexity": "NP-hard",
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
                "The total occupied time of an OR-day equals: "
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
                "Turnover time rules for consecutive surgeries in the same OR-day: "
                "Same specialty: 10 minutes. "
                "Different specialties: 20 minutes."
            ),
            level='base',
            weight=0.0,
            category="definition"
        ))
        
        problem.add_constraint(Constraint(
            name="assignment_rules",
            description="Each surgery must be assigned to at most one OR-day and cannot be split across multiple OR-days.",
            level='base',
            weight=0.0,
            category="definition"
        ))
        
        problem.add_constraint(Constraint(
            name="urgency_priority_rule",
            description="Urgency levels define clinical importance order: emergency > urgent > elective.",
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
            description="Cancellation Penalty: Each cancelled surgery incurs a penalty based on urgency: Emergency surgery: 20; Urgent surgery: 8; Elective surgery: 1",
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
        
        # ========= Objectives =========
        problem.add_objective(Objective(
            name="minimize_total_penalty",
            description="Minimize the total weighted penalty as defined in the penalty definitions. The total penalty is the sum of all penalties incurred under the SOFT CONSTRAINTS section.",
            direction="minimize",
            weight=1.0,
            priority=1,
            semantics={
                "type": "total_penalty",
                "formula": "cancellation_penalty + specialty_penalty + idle_time_penalty"
            }
        ))
        
        return problem
    
