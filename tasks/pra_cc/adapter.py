from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective

class PASPAdapter(MedicalProblemAdapter):
    """
    Patient Admission Scheduling Problem adapter
    Translates PASP medical instance into MedicalNPProblem
    """

    def generate_instance_config(self, level):
        """Generate PASP instance configuration for scale level 1 through 10."""
        # PASP scale levels increase the number of patients
        num_patients = 10 + level * 1
        horizon = 10 + level * 1
        return {
            "num_patients": num_patients,
            "horizon": horizon
        }

    def build(self, instance):
        problem = MedicalNPProblem(
            name="Patient Admission Scheduling Problem",
            problem_type="scheduling"
        )

        # Store source instance data
        problem.instance_data = {
            "patients": instance.patients,
            "rooms": instance.rooms,
            "horizon": instance.horizon
        }

        # Set the problem description
        problem.description = (
            "Given a set of patients requiring hospitalization and hospital rooms with varying "
            "capacities, gender policies, age restrictions, and medical specialisms, assign each "
            "patient to appropriate rooms for each day of their stay while respecting all constraints "
            "and minimizing penalty costs."
        )

        # Set a concise clinical context
        problem.clinical_context = {
            "description": (
                "Patient admission scheduling involves assigning patients to rooms based on their "
                "medical needs, demographic characteristics, and room capabilities. Each patient has "
                "specific requirements regarding gender, age compatibility, and required medical "
                "specialisms. Rooms have limited capacity and may have restrictions on patient types."
            )
        }

        # Set the decision space
        problem.decision_space = [
            "For each patient and each day of their stay, select a room assignment",
            "Group consecutive days in the same room into a single assignment segment",
            "Decide when to transfer patients between rooms based on constraints and penalties",
            "Select room assignments that minimize total weighted penalty while satisfying all constraints"
        ]

        # Add metadata
        problem.metadata = {
            "num_patients": len(instance.patients),
            "num_rooms": len(instance.rooms),
            "horizon": instance.horizon,
            "complexity": "NP-hard",
            "problem_type": "scheduling"
        }

        # ========= Define medical entity types =========
        for p in instance.patients:
            problem.add_entity("patients", p)

        for r in instance.rooms:
            problem.add_entity("rooms", r)

        problem.add_entity_type("day")
        for d in range(instance.horizon):
            problem.add_entity("day", d)

        # ==================================================
        # 1. Decision variables (actions)
        # ==================================================
        # x[p, r, d] = whether patient p is assigned to room r on day d

        for p in instance.patients:
            pid = p["id"]
            start, end = p["stay"]

            for r in instance.rooms:
                rid = r["id"]
                for d in range(start, end):
                    problem.add_action(
                        Action(
                            name=f"assign_{pid}_{rid}_day{d}",
                            description=(
                                f"Assign patient {pid} to room {rid} "
                                f"on day {d}"
                            )
                        )
                    )

        # ==================================================
        # 2. Definitions and rules
        # ==================================================

        problem.add_constraint(Constraint(
            name="assignment_continuity_definition",
            description="Each assignment object covers a continuous period in a single room; non-contiguous segments are forbidden",
            level="base",
            weight=0.0,
            category="definition"
        ))

        problem.add_constraint(Constraint(
            name="room_transfer_rule",
            description="Room transfers are allowed only when the room changes; consecutive days in the same room must be a single assignment object",
            level="base",
            weight=0.0,
            category="definition"
        ))

        problem.add_constraint(Constraint(
            name="adjacent_assignment_rule",
            description="Adjacent assignment objects for the same patient must use different rooms",
            level="base",
            weight=0.0,
            category="definition"
        ))

        # ==================================================
        # 3. Soft constraints
        # ==================================================
        
        # Gender-policy violation penalty
        problem.add_constraint(Constraint(
            name="gender_penalty",
            description="Gender policy penalty: 5.0 for each continuous room assignment segment whose gender policy does not match the patient's gender.",
            level="base",
            weight=5.0,
            category="soft"
        ))

        # Age-policy violation penalty
        problem.add_constraint(Constraint(
            name="age_penalty",
            description="Age policy penalty: 10.0 for each continuous room assignment segment whose age limits are not compatible (adult: 18-64, elderly: >=65, pediatric: <=16).",
            level="base",
            weight=10.0,
            category="soft"
        ))

        # Specialty violation penalty
        problem.add_constraint(Constraint(
            name="specialism_penalty",
            description="Specialism penalty: 5.0 for each continuous room assignment segment in a room that does not provide a required medical specialism with priority < 5.",
            level="base",
            weight=5.0,
            category="soft"
        ))

        # Room-preference violation penalty
        problem.add_constraint(Constraint(
            name="room_preference_penalty",
            description="Room preference penalty: 0.8 × max(0, assigned_capacity - preferred_capacity) × days_stayed in that room.",
            level="base",
            weight=0.8,
            category="soft"
        ))

        # Room-transfer penalty
        problem.add_constraint(Constraint(
            name="transfer_penalty",
            description="Room transfer penalty: 11.0 each time a patient moves from one room to another.",
            level="base",
            weight=11.0,
            category="soft"
        ))

        # ==================================================
        # 4. Validity constraints
        # ==================================================

        # Invalid patient or room entries (hard constraint)
        problem.add_constraint(Constraint(
            name="invalid_entry_constraint",
            description="Only valid patient and room entries may be used in assignments",
            level="base",
            weight=1000.0,
            category="validity"
        ))

        # Capacity constraint: daily occupancy cannot exceed room capacity
        problem.add_constraint(Constraint(
            name="capacity_constraint",
            description="The number of patients assigned to a room on any given day must not exceed the room's capacity",
            level="base",
            weight=1000.0,  # Hard constraint; violations are not allowed
            category="validity",
            scope="room-day"
        ))

        # All patients must be assigned
        problem.add_constraint(Constraint(
            name="all_patients_assigned",
            description="All patients must be assigned to rooms for their entire stay",
            level="base",
            weight=1000.0,
            category="validity",
            scope="patient"
        ))

        # ==================================================
        # 4. Optimization objectives
        # ==================================================

        # Minimize total penalty
        problem.add_objective(Objective(
            name="minimize_total_penalty",
            description="Minimize the total weighted penalty as defined in the penalty definitions. The total penalty is the sum of all penalties incurred under the SOFT CONSTRAINTS section.",
            direction="minimize",
            weight=1.0,
            priority=1,
            semantics={
                "type": "total_penalty"
            }
        ))

        return problem

