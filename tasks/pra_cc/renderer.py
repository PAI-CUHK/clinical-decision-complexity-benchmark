#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderer for the patient-to-room allocation task.

The prompt contains task, instance data, constraints, objectives, and output format."""

from core.renderer import BaseLLMRenderer


class PASPLLMRenderer(BaseLLMRenderer):
    """Render a MedicalNPProblem into a clinically meaningful prompt for model
    reasoning and decision-making using problem.entities and problem.instance_data."""

    SPECIALISM_MAP = {
        1: "Cardiology",
        2: "Neurology",
        3: "Orthopedics",
        4: "General Medicine"
    }

    ROOM_TYPE_MAP = {
        1: "single-bed room",
        2: "double-bed room",
        4: "multi-bed room"
    }

    def __init__(self):
        super().__init__()

    def render_soft_constraints(self, problem):
        lines = []
        lines.append("\n---")
        lines.append("\n# SOFT CONSTRAINTS")
        
        if hasattr(problem, 'constraints') and problem.constraints:
            for constraint in problem.constraints:
                if hasattr(constraint, 'category') and constraint.category == 'soft':
                    lines.append(constraint.description)
                    lines.append("")
        
        return lines

    def render(self, problem):
        lines = []

        # Render the task header using the base class
        lines.extend(self.render_task_header(problem))

        # ===== Clinical Context =====
        lines.extend(self.render_clinical_context(problem))

        # ===== Instance Data =====
        lines.append("\n---")
        lines.append("\n# INSTANCE DATA")
        lines.append("")

        # Add patient information
        patients = problem.entities.get("patients", [])
        rooms = problem.entities.get("rooms", [])

        lines.append("## Patients")
        lines.append("")
        for p in patients:
            start, end = p.get("stay", (0, 0))
            parts = [
                f"- Patient {p['id']}:",
                f"{p['age']}-year-old {p['gender']} patient",
                f"hospital stay from day {start} to day {end}"
            ]

            if "specialism_plan" in p:
                if len(p["specialism_plan"]) == 1:
                    spec_id, _ = p["specialism_plan"][0]
                    spec = self.SPECIALISM_MAP.get(spec_id, f"specialism {spec_id}")
                    parts.append(f"requires treatment in {spec}")
                else:
                    readable = []
                    for spec_id, dur in p["specialism_plan"]:
                        spec = self.SPECIALISM_MAP.get(spec_id, f"Specialism {spec_id}")
                        readable.append(f"{spec} for {dur} days")
                    parts.append("follows a multi-stage treatment plan: " + ", then ".join(readable))

            if "room_pref" in p:
                room_type = self.ROOM_TYPE_MAP.get(p["room_pref"], f"{p['room_pref']}-bed room")
                parts.append(f"prefers a {room_type}")

            parts.append("and requires continuous accommodation during the stay")
            lines.append(", ".join(parts) + ".")

        lines.append("")
        lines.append("## Hospital Rooms")
        lines.append("")
        for r in rooms:
            parts = [
                f"- Room {r['id']}:",
                f"capacity {r['capacity']} beds",
                f"gender policy {r['gender_policy']}"
            ]

            if "age_policy" in r:
                if r["age_policy"] == "Any":
                    parts.append("no age restriction")
                elif r["age_policy"] == "Elderly":
                    parts.append("accepts elderly patients (age ≥ 65)")
                elif r["age_policy"] == "Adult":
                    parts.append("accepts adult patients (age 18–64)")
                elif r["age_policy"] == "Pediatric":
                    parts.append("accepts pediatric patients (age ≤ 16)")

            if "specialisms" in r and r["specialisms"]:
                readable = []
                for k, v in r["specialisms"].items():
                    name = self.SPECIALISM_MAP.get(k, f"Specialism {k}")
                    readable.append(f"{name} (priority {v})")
                parts.append("supports medical specialisms " + ", ".join(readable))

            lines.append(", ".join(parts) + ".")

        # ===== Decision Space =====
        lines.extend(self.render_decision_space(problem))

        # ===== Definitions and Rules =====
        lines.extend(self.render_definitions_and_rules(problem))

        # ===== Validity Constraints =====
        lines.extend(self.render_validity_constraints(problem))

        # ===== Soft Constraints =====
        lines.extend(self.render_soft_constraints(problem))

        # ===== Task Objective =====
        lines.extend(self.render_task_objective(problem))

        # ===== Output Format =====
        lines.extend(self.render_output_format(problem))
        lines.append("")
        lines.append("Your output MUST be enclosed within <root></root> tags and follow this structure exactly:")
        lines.append("")
        lines.append("<root>")
        lines.append("  <reasoning>")
        lines.append("    For each patient assignment, give a brief, factually checkable reason for the selected room assignment or transfer, consistent with the given instance data and the final JSON schedule.")
        lines.append("  </reasoning>")
        lines.append("  <final_answer>")
        lines.append("    {")
        lines.append('      "assignments": {')
        lines.append('        "P0": [{"start_day":0,"end_day":7,"room":"R0"}],')
        lines.append('        "P1": [{"start_day":0,"end_day":4,"room":"R1"}],')
        lines.append("        ...")
        lines.append("      }")
        lines.append("    }")
        lines.append("  </final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- The <final_answer> section MUST contain valid JSON only (no prose).")
        lines.append("- Do NOT include any text outside <root></root>.")
        lines.append("- Do NOT add comments, markdown, or extra fields.")

        return "\n".join(lines)
