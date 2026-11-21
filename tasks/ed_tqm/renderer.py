#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED Scheduling Problem Renderer — Single Doctor with Dynamic Priority Escalation
"""

from core.renderer import BaseLLMRenderer


class EDSchedulingLLMRenderer(BaseLLMRenderer):
    """ED Scheduling Problem Renderer."""

    def __init__(self):
        super().__init__()

    def render(self, problem):
        lines = []
        lines.extend(self.render_task_header(problem))
        lines.extend(self.render_clinical_context(problem))
        lines.extend(self.render_instance_data(problem))
        lines.extend(self.render_decision_space(problem))
        lines.extend(self.render_definitions_and_rules(problem))
        lines.extend(self.render_validity_constraints(problem))
        lines.extend(self.render_task_objective(problem))
        lines.extend(self.render_output_format(problem))
        return "\n".join(lines)

    def render_task_header(self, problem):
        lines = []
        lines.append(
            "You are given a clinical decision-making task: "
            "ED Physician Scheduling with Dynamic Priority Escalation."
        )
        lines.append("")
        lines.append(problem.description)
        return lines

    def render_clinical_context(self, problem):
        lines = []
        lines.append("")
        lines.append("# CLINICAL CONTEXT")
        lines.append("")
        if hasattr(problem, 'clinical_context') and problem.clinical_context:
            if isinstance(problem.clinical_context, dict):
                lines.append(problem.clinical_context.get('description', ''))
            else:
                lines.append(problem.clinical_context)
        return lines

    def render_instance_data(self, problem):
        lines = []
        lines.append("")
        lines.append("# INSTANCE DATA")
        lines.append("")

        patients = problem.instance_data.get('patients', [])
        if patients:
            lines.append("## Patients")
            lines.append("")
            lines.append(
                "Patient ID | Arrival Time | Triage Level | "
                "Evaluation Time | Age | NRS"
            )
            lines.append("-" * 80)
            lines.append("")

            if "patients" in problem.entities:
                for patient in problem.entities["patients"]:
                    lines.append(
                        f"{patient['id']} | {patient['arrival_time']} | "
                        f"Level {patient['triage_level']} | "
                        f"{patient['processing_time']} min | "
                        f"{patient['age']} yrs | "
                        f"{patient.get('nrs', 0)}"
                    )
                    lines.append("")

        return lines

    def render_decision_space(self, problem):
        lines = []
        lines.append("")
        lines.append("# DECISION SPACE")
        lines.append("")
        if hasattr(problem, 'decision_space') and problem.decision_space:
            if isinstance(problem.decision_space, list):
                for item in problem.decision_space:
                    lines.append(item)
                    lines.append("")
            else:
                lines.append(problem.decision_space)
        return lines

    def render_definitions_and_rules(self, problem):
        lines = []
        lines.append("")
        lines.append("# DEFINITIONS AND RULES")
        lines.append("")
        if hasattr(problem, 'constraints') and problem.constraints:
            for constraint in problem.constraints:
                if hasattr(constraint, 'category') and constraint.category == 'definition':
                    lines.append(constraint.description)
                    lines.append("")
        return lines

    def render_validity_constraints(self, problem):
        lines = []
        lines.append("")
        lines.append("# VALIDITY CONSTRAINTS")
        lines.append("")
        if hasattr(problem, 'constraints') and problem.constraints:
            for constraint in problem.constraints:
                if hasattr(constraint, 'category') and constraint.category == 'validity':
                    lines.append(constraint.description)
                    lines.append("")
        return lines

    def render_task_objective(self, problem):
        lines = []
        lines.append("")
        lines.append("# TASK OBJECTIVE")
        lines.append("")
        if hasattr(problem, 'objectives') and problem.objectives:
            for obj in problem.objectives:
                lines.append(obj.description)
                lines.append("")
        return lines

    def render_output_format(self, problem):
        lines = []
        lines.append("")
        lines.append("# OUTPUT FORMAT STRICTLY REQUIRED")
        lines.append("")
        lines.append(
            "Your output MUST be enclosed within <root></root> tags:"
        )
        lines.append("")
        lines.append("<root>")
        lines.append("  <reasoning>")
        lines.append(
        "    Briefly state the key triage level and the main reason "
        "for the scheduling priority. Avoid detailed reasoning."
        )
        lines.append("  </reasoning>")
        lines.append("")
        lines.append("  <final_answer>")
        lines.append("    {")
        lines.append('      "order": ["P1", "P2", "..."]')
        lines.append("    }")
        lines.append("  </final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append(
            "The <final_answer> section MUST contain valid JSON only (no prose). "
            "Do NOT include any text outside <root></root>. "
            "Do NOT add comments, markdown, or extra fields."
        )
        return lines
