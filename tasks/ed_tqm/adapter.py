from core import MedicalProblemAdapter, MedicalNPProblem, Constraint, Objective
from tasks.ed_tqm.generator import EDInstance
from tasks.ed_tqm.protocol import (
    get_protocol_config,
    MAX_ESCALATIONS,
)


class EDSchedulingAdapter(MedicalProblemAdapter):
    """
    ED Scheduling Problem — Single Doctor with Dynamic Priority Escalation

    Complexity: O(n^2). Still in P.
    """

    def generate_instance_config(self, level):
        config = get_protocol_config(level)
        return {
            "level": config["level"],
            "num_patients": config["num_patients"],
            "num_doctors": config["num_doctors"],
            "escalation_thresholds": config["escalation_thresholds"],
            "max_escalations": config["max_escalations"],
            "arrival_window": config["arrival_window"],
            "seed": config.get("seed"),
        }

    def build(self, instance: EDInstance) -> MedicalNPProblem:
        config = get_protocol_config(1)
        num_patients = len(instance.patients)

        problem = MedicalNPProblem(
            name="ED Scheduling Problem",
            problem_type="scheduling",
        )

        problem.description = (
            f"There are {num_patients} patients waiting for evaluation by "
            f"the single on-duty physician. Determine the correct processing "
            f"order under dynamic priority escalation rules."
        )

        problem.instance_data = {
            "patients": instance.patients,
            "num_doctors": 1,
            "escalation_thresholds": config["escalation_thresholds"],
            "max_escalations": config["max_escalations"],
        }

        problem.clinical_context = {
            "description": (
                "Following ED triage using the Emergency Severity Index (ESI), "
                "patients wait for physician evaluation. A single physician "
                "sees patients one at a time. Patient priority changes over "
                "time due to waiting-time escalation."
            ),
        }

        problem.decision_space = [
            "At each step, serve the arrived patient with the highest "
            "effective priority (considering escalation).",
            "Output the complete processing order of all patients.",
        ]

        problem.metadata = {
            "num_patients": num_patients,
            "num_doctors": 1,
            "escalation_thresholds": config["escalation_thresholds"],
            "max_escalations": config["max_escalations"],
            "complexity": "P",
            "problem_type": "scheduling",
            "algorithm": "Event-Driven Priority Queue with Dynamic Escalation",
            "time_complexity": "O(n^2)",
        }

        for patient in instance.patients:
            problem.add_entity("patients", patient)

        # ========= Rules (definitions) =========
        problem.add_constraint(Constraint(
            name="priority_rule_definition",
            description=(
                "Effective Priority Order Definition:\n\n"
                "A patient's EFFECTIVE triage level determines their priority. "
                "It changes as waiting time grows via level-specific safe "
                "waiting targets. After each escalation, the consumed threshold "
                "is subtracted from the wait time, and the new level's threshold "
                "is checked against the REMAINING wait — the patient may cascade "
                "further.\n\n"
                "Safe waiting time targets:\n"
                "- ESI-1: no escalation\n"
                "- ESI-2: 10 min\n"
                "- ESI-3: 30 min\n"
                "- ESI-4: 60 min\n"
                "- ESI-5: 120 min\n\n"
                "Escalation is triggered by STRICTLY GREATER THAN (>). "
                "Waiting time exactly equal to the target does NOT trigger an "
                "escalation.\n\n"
                "Cascade example: ESI-3 patient waits 45 min. "
                "45 > 30 → upgrade to ESI-2, remaining wait = 45 - 30 = 15 min. "
                "15 > 10 → upgrade to ESI-1, remaining wait = 15 - 10 = 5 min. "
                "Result: effective ESI-1.\n\n"
                "Non-cascade example: ESI-3 patient waits 35 min. "
                "35 > 30 → upgrade to ESI-2, remaining wait = 35 - 30 = 5 min. "
                "5 > 10? No → stop at ESI-2.\n\n"
                f"Maximum escalations per patient: {MAX_ESCALATIONS}\n"
                "Minimum effective level: 1"
            ),
            level="clinical",
            weight=0.0,
            category="definition",
        ))

        problem.add_constraint(Constraint(
            name="tie_breaker_rule",
            description=(
                "Tie-Breaker Hierarchy (when effective triage levels match):\n\n"
                "1. VULNERABLE POPULATION: < 2 years or >= 65 years first.\n"
                "2. PAIN: Higher NRS first.\n"
                "3. FIFO: Earlier arrival first.\n\n"
                "Full order: effective_triage (asc) -> vulnerable (yes first) "
                "-> NRS (desc) -> arrival_time (asc)"
            ),
            level="clinical",
            weight=0.0,
            category="definition",
        ))

        problem.add_constraint(Constraint(
            name="non_preemptive_rule",
            description=(
                "Service is non-preemptive: once started, a patient's "
                "evaluation runs to completion."
            ),
            level="clinical",
            weight=0.0,
            category="definition",
        ))

        # ========= Validity Constraints =========
        problem.add_constraint(Constraint(
            name="no_skipping",
            description=(
                "Cannot skip the highest-priority arrived patient. Always "
                "serve the patient with the highest effective priority first."
            ),
            level="hard",
            weight=1000.0,
            category="validity",
        ))

        problem.add_constraint(Constraint(
            name="serve_all_patients",
            description="All patients must be served exactly once.",
            level="hard",
            weight=1000.0,
            category="validity",
        ))

        problem.add_constraint(Constraint(
            name="cannot_start_before_arrival",
            description=(
                "A patient cannot start evaluation before their arrival time."
            ),
            level="hard",
            weight=1000.0,
            category="validity",
        ))

        problem.add_constraint(Constraint(
            name="no_parallel_service",
            description=(
                "The physician serves one patient at a time."
            ),
            level="hard",
            weight=1000.0,
            category="validity",
        ))

        # ========= Objective =========
        problem.add_objective(Objective(
            name="correct_ordering",
            description=(
                "Determine the correct patient processing order under "
                "dynamic priority escalation rules."
            ),
            direction="minimize",
            weight=1.0,
            priority=1,
            semantics={
                "type": "processing_order",
                "scope": "all_patients",
            },
        ))

        return problem
