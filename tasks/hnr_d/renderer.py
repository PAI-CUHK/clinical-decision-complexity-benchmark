#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Renderer for the nurse rostering decision task.

The prompt includes clinical context, instance data, decision space,
definitions and rules, validity constraints, soft constraints, objective, and
output format."""

from core.renderer import BaseLLMRenderer


class NurseSchedulingLLMRenderer(BaseLLMRenderer):
    """Render the nurse rostering decision task using problem constraints and objectives."""

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
        """Render the problem as a model prompt.
        
        Args:
            problem: Constructed problem instance.
        
        Returns:
            The prompt text."""
        lines = []

        instance_data = problem.instance_data
        nurses = instance_data.get('nurses', [])
        days = instance_data.get('days', [])
        num_days = len(days)
        required = instance_data.get('required', [])
        shift_types = instance_data.get('shift_types', ["D", "E", "N", "O"])
        nurse_skills = instance_data.get('nurse_skills', {})
        contracts = instance_data.get('contracts', {})
        preferences = instance_data.get('preferences', {})

        lines.extend(self.render_task_header(problem))

        lines.extend(self.render_clinical_context(problem))

        lines.append("\n---")
        lines.append("\n# INSTANCE DATA")
        
        lines.append("")
        lines.append("## Nurses")
        lines.append(f"Total nurses: {len(nurses)}")
        lines.append("")
        lines.append("Each nurse has a skill level and contract type:")
        lines.append("")
        for i, nurse in enumerate(nurses, 1):
            skill = nurse_skills.get(nurse, "RN")
            contract_type = contracts.get(nurse, "full-time")
            prefs = preferences.get(nurse, [])
            
            line = f"- Nurse_{i}: Skill={skill}, Contract={contract_type}"
            if prefs:
                line += f", Requests off: Days {', '.join(map(str, prefs))}"
            lines.append(line)

        lines.append("")
        lines.append("## Planning Horizon")
        lines.append(f"Total days: {num_days}")
        lines.append(f"Days are indexed from Day 0 to Day {num_days-1}.")

        lines.append("")
        lines.append("## Shift Types")
        lines.append("")
        lines.append("- D = Day shift; E = Evening shift; N = Night shift; O = Off day")

        lines.append("")
        lines.append("## Daily Shift Coverage Requirements")
        lines.append("")
        for d in range(num_days):
            day_required = required[d] if d < len(required) else {}
            if isinstance(day_required, dict):
                requirements = []
                for shift_type in ["D", "E", "N"]:
                    count = day_required.get(shift_type, 0)
                    if count > 0:
                        requirements.append(f"{shift_type}: {count}")
                if requirements:
                    lines.append(f"Day {d}: {', '.join(requirements)}")
            else:
                lines.append(f"Day {d}: D: {day_required}, E: {day_required}, N: {day_required}")

        lines.extend(self.render_decision_space(problem))

        lines.extend(self.render_definitions_and_rules(problem))

        lines.extend(self.render_validity_constraints(problem))

        lines.extend(self.render_soft_constraints(problem))

        lines.extend(self.render_task_objective(problem))

        lines.extend(self.render_output_format(problem))
        lines.append("")
        lines.append("Your output MUST follow this structure exactly:")
        lines.append("")
        lines.append("<root>")
        lines.append("  <reasoning>")
        lines.append("    Briefly justify the proposed schedule by highlighting the key")
        lines.append("    assignments and constraints that determine feasibility. If no valid")
        lines.append("    schedule exists, identify the main blocking constraint(s). Keep the")
        lines.append("    rationale concise and consistent with the final answer.")
        lines.append("  </reasoning>")  
        lines.append("  <final_answer>YES</final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- The <final_answer> section MUST contain exactly 'YES' or 'NO'.")
        lines.append("- Do NOT include any text outside <root></root>.")
        lines.append("- Do NOT add comments, markdown, or extra fields.")

        return "\n".join(lines)
