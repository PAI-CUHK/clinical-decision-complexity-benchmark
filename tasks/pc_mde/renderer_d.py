#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the PC-MDE decision task.

The prompt includes instance data, decision space, scoring definitions,
validity constraints, the threshold objective, and the required YES/NO output.
The scoring definition is shared with the optimization task."""

from core.renderer import BaseLLMRenderer


class DiseaseDiagnosisNPCompleteRenderer(BaseLLMRenderer):
    """Render a PC-MDE decision prompt using the optimization task scoring rules."""

    def render(self, problem):
        lines = []

        # ===== Header =====
        lines.append("You are given a clinical decision-making task: "
                     "Multisystem Diagnostic Classification.")
        lines.append("")
        lines.append(problem.description)

        # ===== Clinical Context =====
        lines.extend(self.render_clinical_context(problem))

        # ===== Instance Data =====
        lines.extend(self._render_instance_data(problem))

        # ===== Decision Space =====
        lines.extend(self.render_decision_space(problem))

        # ===== Definitions and Rules =====
        lines.extend(self._render_definitions(problem))

        # ===== Validity Constraints =====
        lines.extend(self._render_validity(problem))

        # ===== Task Objective =====
        lines.extend(self._render_objectives(problem))

        # ===== Output Format =====
        lines.extend(self._render_output_format(problem))

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # INSTANCE DATA
    # ------------------------------------------------------------------

    def _render_instance_data(self, problem):
        data = problem.instance_data
        phenotypes = data.get("phenotypes", [])
        candidate_diseases = data.get("candidate_diseases", [])
        sim_pq = data.get("phenotype_similarity", [])
        term_table = data.get("disease_term_table", {})

        lines = []
        lines.append("\n---")
        lines.append("\n# INSTANCE DATA")
        lines.append("")

        # --- Patient phenotypes ---
        lines.append("## Patient Phenotypes")
        lines.append("")
        lines.append(f"The patient presents {len(phenotypes)} observed phenotypes:")
        lines.append("")
        lines.append("| Label | Phenotype |")
        lines.append("| --- | --- |")
        for i, p in enumerate(phenotypes):
            lines.append(f"| P{i + 1} | {p} |")
        lines.append("")

        # --- Pairwise phenotype similarity ---
        lines.append("## Phenotype Similarity s(p, q)")
        lines.append("")
        lines.append("Semantic similarity between the patient's phenotypes "
                     "(0 = unrelated, 1 = identical/synonymous). "
                     "The diagonal is 1 by definition and is omitted.")
        lines.append("")
        header = "| | " + " | ".join(f"P{j + 1}" for j in range(len(phenotypes))) + " |"
        lines.append(header)
        lines.append("| --- " * (len(phenotypes) + 1) + "|")
        for i in range(len(phenotypes)):
            cells = []
            for j in range(len(phenotypes)):
                if i == j:
                    cells.append("—")
                elif j < i:
                    cells.append("")
                else:
                    cells.append(f"{sim_pq[i][j]:.3f}" if sim_pq else "")
            lines.append(f"| P{i + 1} " + " | ".join(f" {c} " for c in cells) + " |")
        lines.append("")
        lines.append("(Only the upper triangle is shown; s(p, q) is symmetric.)")
        lines.append("")

        # --- Candidate diseases ---
        lines.append("## Candidate Diseases")
        lines.append("")
        lines.append(f"{len(candidate_diseases)} candidate diseases are given. For each "
                     "disease its **complete phenotype profile** is listed below: every "
                     "phenotype it can produce and its likelihood `P(phenotype | disease)` "
                     "in patients with that disease.")
        lines.append("")
        lines.append("The patient's phenotypes are drawn from these profiles. A disease "
                     "`d` explains a patient phenotype `p` **if and only if** `p` appears "
                     "in `d`'s profile, with explanation strength `A_d(p) = P(p | d)`; "
                     "otherwise `A_d(p) = 0`. Use these `A_d(p)` values as-is for scoring "
                     "(coverage / uncovered / extra / relation).")
        lines.append("")
        lines.append("In every cluster you form, the diseases you select must explain "
                     "the cluster's patient phenotypes. A selected disease whose profile "
                     "covers patient phenotypes **outside** its cluster is penalized by "
                     "the `Extra Phenotypes of a Disease` term below.")
        lines.append("")
        for d in candidate_diseases:
            rows = term_table.get(d, [])
            lines.append(f"### {d}")
            lines.append("")
            if rows:
                prose = ", ".join(f"{h} (P={p:.2f})" for h, p in rows)
                lines.append(f"Associated symptoms: {prose}")
            else:
                lines.append("Associated symptoms: (none listed)")
            lines.append("")

        return lines

    # ------------------------------------------------------------------
    # Formulas / constraints / objective
    # ------------------------------------------------------------------

    def _render_definitions(self, problem):
        lines = []
        lines.append("\n---")
        lines.append("\n# DEFINITIONS AND RULES")
        lines.append("")

        definitions = [
            c for c in getattr(problem, 'constraints', [])
            if getattr(c, 'category', '') == 'definition'
        ]
        for d in definitions:
            lines.append(d.description.rstrip())
            lines.append("")
        return lines

    def _render_validity(self, problem):
        lines = []
        lines.append("\n---")
        lines.append("\n# VALIDITY CONSTRAINTS")
        lines.append("")

        constraints = [
            c for c in getattr(problem, 'constraints', [])
            if getattr(c, 'category', '') == 'validity'
        ]
        for c in constraints:
            lines.append(c.description)
        lines.append("")
        return lines

    def _render_objectives(self, problem):
        lines = []
        lines.append("\n---")
        lines.append("\n# TASK OBJECTIVE")
        lines.append("")

        objectives = sorted(
            getattr(problem, 'objectives', []),
            key=lambda o: getattr(o, 'priority', 999)
        )
        for obj in objectives:
            lines.append(obj.description)
            lines.append("")
        return lines

    def _render_output_format(self, problem):
        lines = []
        lines.extend(self.render_output_format(problem))
        lines.append("Your output MUST follow this structure exactly:")
        lines.append("")
        lines.append("<root>")
        lines.append("  <reasoning>")
        lines.append("    Briefly justify whether some element partition and candidate")
        lines.append("    assignments can reach the threshold T, considering cluster")
        lines.append("    coherence, candidate explanation strength, uncovered elements,")
        lines.append("    and extra elements. Keep the rationale concise.")
        lines.append("  </reasoning>")
        lines.append("  <final_answer>YES</final_answer>")
        lines.append("</root>")
        lines.append("")
        lines.append("Rules:")
        lines.append("- The <final_answer> section MUST contain exactly 'YES' or 'NO'.")
        lines.append("- Answer YES if there exists a valid partition + per-cluster non-empty")
        lines.append("  disease assignment whose total score reaches at least the threshold T;")
        lines.append("  otherwise answer NO.")
        lines.append("- Do NOT include any text outside <root></root>.")
        return lines