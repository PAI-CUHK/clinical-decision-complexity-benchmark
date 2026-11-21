#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG RR-Interval Pattern Recognition Renderer

Renders the ECG RR-interval pattern recognition task entirely from the
problem object:

- clinical context   <- problem.clinical_context
- instance data      <- problem.instance_data (RR interval sequence)
- decision space     <- problem.decision_space
- definitions/rules <- problem.constraints (category == "definition")
- validity           <- problem.constraints (category == "validity")
- task objective     <- problem.objectives

No task-specific rule, threshold, or scoring table is hardcoded here.
Every rule is read from the constraints the adapter attaches to the problem.
Only the prompt skeleton (section headers and the XML/JSON output envelope)
is renderer-level presentation.
"""

from core.renderer import BaseLLMRenderer


class ECGPatternLLMRenderer(BaseLLMRenderer):
    """Renderer for ECG RR-Interval Pattern Recognition (fully data-driven)."""

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
            self.render_interpretation_limitations(problem),
            self.render_task_objective(problem),
            self.render_output_format(problem),
        ]
        return "\n\n---\n\n".join(section for section in sections if section)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_rr_intervals(self, problem):
        """Return the RR interval list from the problem (instance_data first)."""
        instance_data = getattr(problem, "instance_data", {}) or {}
        rr = instance_data.get("rr_intervals")
        if rr:
            return rr
        entities = getattr(problem, "entities", {}) or {}
        return [e["value"] for e in entities.get("rr_intervals", [])]

    @staticmethod
    def _fmt_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_numbered_rr_list(self, rr_intervals):
        lines = []
        for idx, value in enumerate(rr_intervals, start=1):
            lines.append(f"{idx}. RR_{idx} = {self._fmt_value(value)}")
        return "\n".join(lines)

    def _constraints_by_category(self, problem, category):
        constraints = getattr(problem, "constraints", None) or []
        return [c for c in constraints if getattr(c, "category", None) == category]

    def _pattern_definition_constraints(self, problem):
        result = []
        for c in self._constraints_by_category(problem, "definition"):
            sem = getattr(c, "semantics", None) or {}
            if sem.get("type") == "pattern_definition":
                result.append(c)
        return result

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
        rr_intervals = self._get_rr_intervals(problem)
        n = len(rr_intervals)
        numbered_list = self._format_numbered_rr_list(rr_intervals)

        lines = [
            "# INSTANCE DATA",
            "",
            "## RR Intervals",
            "",
            (
                f"There is a finite ordered sequence of RR intervals: "
                f"RR = [RR_1, RR_2, ..., RR_{n}]."
            ),
            "",
            "Each RR_i represents the interval, in seconds, between two consecutive heartbeats.",
            "",
            "The complete RR interval sequence is provided for this instance:",
            "",
            "RR intervals, in seconds:",
            "",
            numbered_list,
        ]
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
        rr_intervals = self._get_rr_intervals(problem)
        n = len(rr_intervals)

        lines = ["# DEFINITIONS AND RULES", ""]
        for constraint in self._constraints_by_category(problem, "definition"):
            lines.append(constraint.description)
            sem = getattr(constraint, "semantics", None) or {}
            # Append a dynamically computed valid 1-based starting-position
            # range for pattern definitions, derived from the instance size
            # and the pattern semantics. No rule text is hardcoded.
            if sem.get("type") == "pattern_definition":
                pattern_name = sem.get("pattern", constraint.name)
                scope = sem.get("scope")
                end = None
                if scope == "pair":
                    end = max(n - 1, 1)
                elif scope == "single":
                    end = n
                elif scope == "sliding_window":
                    window = int(sem.get("window", 1))
                    end = max(n - window + 1, 1) if n >= window else 0
                if end is not None and end >= 1:
                    lines.append(
                        f"Valid 1-based starting positions for {pattern_name}: 1 to {end}."
                    )
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

    def render_interpretation_limitations(self, problem):
        pattern_names = []
        for c in self._pattern_definition_constraints(problem):
            sem = getattr(c, "semantics", None) or {}
            name = sem.get("pattern")
            if name:
                pattern_names.append(name)

        lines = [
            "# INTERPRETATION LIMITATIONS",
            "",
            (
                "The detected patterns are predefined computational screening patterns and "
                "should not be interpreted as definitive cardiac rhythm diagnoses."
            ),
            "",
        ]
        if pattern_names:
            lines.append("Only report the predefined RR-interval pattern labels:")
            lines.append("")
            for name in pattern_names:
                lines.append(f"- {name};")
            lines.append("")
        lines.append(
            "Do not infer or report any cardiac rhythm diagnosis beyond these pattern labels."
        )
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
        pattern_names = []
        for c in self._pattern_definition_constraints(problem):
            sem = getattr(c, "semantics", None) or {}
            name = sem.get("pattern")
            if name:
                pattern_names.append(name)

        lines = [
            "# STRICT OUTPUT FORMAT",
            "",
            "Your output MUST follow this structure exactly:",
            "",
            "<root>",
            "",
            "<reasoning>",
            "",
            "Briefly explain the pattern detection process.",
            "",
            "The reasoning should:",
            "",
            "- mention that 1-based indexing is used;",
            "- explain the RR classification rules applied;",
            "- explain the detection rule for each pattern;",
            "- identify the detected starting positions for each pattern;",
            "- avoid making definitive arrhythmia diagnoses.",
            "",
            "</reasoning>",
            "",
            "<final_answer>",
            "",
            "{",
        ]
        for idx, name in enumerate(pattern_names):
            suffix = "," if idx < len(pattern_names) - 1 else ""
            lines.append(
                f'  "{name}": [<comma-separated list of 1-based positions>]{suffix}'
            )
        lines += [
            "}",
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
