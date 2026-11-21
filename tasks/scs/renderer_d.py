#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderer for the operating-room scheduling decision task.

The prompt includes clinical context, instance data, decision space,
definitions and rules, validity constraints, soft constraints, task objective,
and output format."""

from core.renderer import BaseLLMRenderer


class ORSchedulingNPCompleteRenderer(BaseLLMRenderer):
    """
    Render an OR Scheduling NP-Complete problem into a clinically meaningful prompt
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

        lines.append("## OR-days")
        lines.append("")
        for or_day in problem.entities.get("or_days", []):
            specialties = or_day.get("specialties", ["all"])
            capacity = or_day.get("capacity", 0)
            lines.append(f"- {or_day['id']}: capacity {capacity} minutes, supported specialties={', '.join(specialties)}")

        lines.append("")
        lines.append("## Surgeries")
        lines.append("")
        for surgery in problem.entities.get("surgeries", []):
            urgency = surgery.get("urgency", "elective")
            specialty = surgery.get("specialty", "general")
            duration = surgery.get("duration", 0)
            lines.append(f"- {surgery['id']}: {duration} minutes, urgency={urgency}, specialty={specialty}")

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
        lines.append("Your output MUST follow this structure exactly:")
        lines.append("")
        lines.append("<root>")
        lines.append("  <reasoning>Briefly explain the answer in the most concise way possible. State the key reason supporting the result, without providing a detailed step-by-step reasoning process.</reasoning>")
        lines.append("  <final_answer>YES</final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- The <final_answer> section MUST contain exactly 'YES' or 'NO'.")
        lines.append("- Do NOT include any text outside <root></root>.")
        lines.append("- Do NOT add comments, markdown, or extra fields.")

        return "\n".join(lines)
