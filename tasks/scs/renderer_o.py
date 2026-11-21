#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderer for the operating-room scheduling optimization task.

The prompt includes clinical context, instance data, decision space,
definitions and rules, validity constraints, task objective, and output format."""

from core.renderer import BaseLLMRenderer


class ORSchedulingLLMRenderer(BaseLLMRenderer):
    """
    Render an OR Scheduling problem into a clinically meaningful prompt
    for LLM-based reasoning and decision-making.
    """

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

        # ===== Task Header =====
        lines.extend(self.render_task_header(problem))

        # ===== Clinical Context =====
        lines.extend(self.render_clinical_context(problem))

        # ===== Instance Data =====
        lines.append("\n---")
        lines.append("\n# INSTANCE DATA")
        lines.append("")

        lines.append("## Surgeries")
        lines.append("")
        for surgery in problem.entities.get("surgeries", []):
            urgency = surgery.get("urgency", "elective")
            specialty = surgery.get("specialty", "general")
            duration = surgery.get("duration", 0)
            lines.append(f"- {surgery['id']}: {duration} minutes, urgency={urgency}, specialty={specialty}")

        lines.append("")
        lines.append("## OR-days")
        lines.append("")
        for or_day in problem.entities.get("or_days", []):
            specialties = or_day.get("specialties", ["all"])
            or_type = or_day.get("type", "general")
            capacity = or_day.get("capacity", 0)
            lines.append(f"- {or_day['id']}: capacity {capacity} minutes, specialties={', '.join(specialties)} ({or_type})")

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
        lines.append("    For each surgery, give a brief, checkable reason for its assigned OR-day or cancellation, consistent with the given instance data and the final JSON schedule.")
        lines.append("  </reasoning>")
        lines.append("  <final_answer>")
        lines.append("    {")
        lines.append('      "schedule": {')
        lines.append('        "OR-day1": ["S1", "S3", ...],')
        lines.append('        "OR-day2": ["S2", "S5", ...]')
        lines.append('      },')
        lines.append('      "cancelled": ["S8", "S9", ...]')
        lines.append("    }")
        lines.append("  </final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- The <final_answer> section MUST contain valid JSON only.")
        lines.append("- Each surgery can be scheduled at most once.")
        lines.append("- Surgeries cannot be split across days.")
        lines.append("- Only the listed surgeries and OR-days may appear in the schedule.")
        lines.append("- Do NOT include any text outside <root></root>.")
        lines.append("- cancelled must be an array; an empty array is allowed if no surgeries are cancelled.")

        return "\n".join(lines)