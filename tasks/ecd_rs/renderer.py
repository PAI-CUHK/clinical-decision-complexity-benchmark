#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWS2 Risk Assessment Renderer

Renders the NEWS2 task prompt entirely from the problem object:

- clinical context   <- problem.clinical_context
- instance data      <- problem.instance_data (patients + snapshots)
- decision space     <- problem.decision_space
- definitions/rules <- problem.constraints (category == "definition")
- validity           <- problem.constraints (category == "validity")
- task objective     <- problem.objectives

No scoring table, threshold, or rule is hardcoded here. Every rule is
read from the constraints the adapter attaches to the problem. Only the
prompt skeleton (section headers and the XML/JSON output envelope) is
renderer-level presentation.
"""

import json

from core.renderer import BaseLLMRenderer


class MedicalRiskRenderer(BaseLLMRenderer):
    """NEWS2 Risk Assessment Renderer (fully data-driven)."""

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def render(self, problem):
        sections = [
            self.render_task_header(problem),
            self.render_clinical_context(problem),
            self.render_instance_data(problem),
            self.render_decision_space(problem),
            self.render_definitions_and_rules(problem),
            self.render_validity_constraints(problem),
            self.render_task_objective(problem),
            self.render_output_format(problem),
        ]
        return "\n\n".join(section for section in sections if section)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _constraints_by_category(self, problem, category):
        constraints = getattr(problem, "constraints", None) or []
        return [c for c in constraints if getattr(c, "category", None) == category]

    def _get_patients(self, problem):
        instance_data = getattr(problem, "instance_data", {}) or {}
        return instance_data.get("patients", {}) or {}

    def _snapshot_fields(self, patients):
        """Derive snapshot field names from the actual instance data."""
        for pid in sorted(patients.keys()):
            snaps = patients[pid].get("snapshots") or []
            if snaps and isinstance(snaps[0], dict):
                return list(snaps[0].keys())
        return []

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------
    def render_task_header(self, problem):
        lines = []
        task_name = self._get_professional_task_name(problem.name)
        lines.append(f"You are given a clinical decision-making task: {task_name}.")
        if getattr(problem, "description", None):
            lines.append("")
            lines.append(problem.description)
        return "\n".join(lines)

    def render_clinical_context(self, problem):
        lines = ["# CLINICAL CONTEXT", ""]
        ctx = getattr(problem, "clinical_context", None)
        if ctx:
            desc = ctx.get("description", "") if isinstance(ctx, dict) else str(ctx)
            if desc:
                lines.append(desc)
        return "\n".join(lines)

    def render_instance_data(self, problem):
        patients = self._get_patients(problem)

        ordered = {}
        for pid in sorted(patients.keys()):
            p = patients[pid] or {}
            ordered[pid] = {
                "age": p.get("age"),
                "conditions": p.get("conditions", []),
                "snapshots": p.get("snapshots", []),
            }
        body = json.dumps(ordered, indent=2, ensure_ascii=False)

        lines = [
            "# INSTANCE DATA",
            "",
            "## Patients",
            "",
            body,
            "",
            "Each patient contains: age, conditions, snapshots.",
        ]

        fields = self._snapshot_fields(patients)
        if fields:
            lines.append(f"Each snapshot contains: {', '.join(fields)}.")
        return "\n".join(lines)

    def render_decision_space(self, problem):
        lines = ["# DECISION SPACE", ""]
        ds = getattr(problem, "decision_space", None)
        if ds:
            items = ds if isinstance(ds, list) else [ds]
            for item in items:
                lines.append(item)
                lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def render_definitions_and_rules(self, problem):
        lines = ["# DEFINITIONS AND RULES", ""]
        for constraint in self._constraints_by_category(problem, "definition"):
            lines.append(constraint.description)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def render_validity_constraints(self, problem):
        lines = ["# VALIDITY CONSTRAINTS", ""]
        for constraint in self._constraints_by_category(problem, "validity"):
            lines.append(constraint.description)
            lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def render_task_objective(self, problem):
        lines = ["# TASK OBJECTIVE", ""]
        objectives = getattr(problem, "objectives", None) or []
        if objectives:
            ordered = sorted(objectives, key=lambda o: getattr(o, "priority", 999))
            for obj in ordered:
                lines.append(obj.description)
                lines.append("")
        while lines and lines[-1] == "":
            lines.pop()
        return "\n".join(lines)

    def render_output_format(self, problem):
        patients = self._get_patients(problem)
        pids = sorted(patients.keys())

        sample = pids[:3]
        if len(pids) > 3:
            sample = sample + ["..."]

        example = {
            "scores": {pid: 0 for pid in sample},
            "risk_category": {pid: "Low Risk" for pid in sample},
            "urgent_review": [pids[1]] if len(pids) >= 2 else [],
        }
        example_json = json.dumps(example, indent=2, ensure_ascii=False)
        indented = "\n".join("  " + line for line in example_json.splitlines())

        lines = [
            "# STRICT OUTPUT FORMAT",
            "",
            "Your output MUST follow this structure exactly:",
            "",
            "<root>",
            "",
            "<reasoning>",
            "",
            (
                "Provide a patient-specific justification for the calculated score, "
                "risk category, and urgent review decision. Avoid generic explanations "
                "of the scoring framework."
            ),
            "</reasoning>",
            "",
            "<final_answer>",
            "",
            indented,
            "",
            "</final_answer>",
            "",
            "</root>",
            "",
            (
                "The <final_answer> section MUST contain valid JSON only (no prose). "
                "Do NOT include any text outside <root></root>. "
                "Do NOT add comments, markdown, or extra fields."
            ),
        ]
        return "\n".join(lines)
