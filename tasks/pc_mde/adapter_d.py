#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter for the PC-MDE decision task.

The decision and optimization variants share the phenotype partition,
disease assignments, scoring formula, and validity constraints. The decision
task asks whether a valid solution reaches threshold T, returning YES or NO.
YES thresholds use a feasible reference; NO thresholds exceed a conservative
objective upper bound."""

from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective
from tasks.pc_mde.generator_o import LEVEL_CONFIG
from tasks.pc_mde.adapter_o import DiseaseDiagnosisAdapter


class DiseaseDiagnosisNPCompleteAdapter(MedicalProblemAdapter):
    """Adapter for the PC-MDE decision task."""

    def generate_instance_config(self, level):
        # Use the same scale configuration as the optimization task
        cfg = LEVEL_CONFIG.get(level, LEVEL_CONFIG[5])
        return {"level": level, "m": cfg["m"], "n": cfg["n"]}

    def build(self, instance):
        problem = MedicalNPProblem(
            name="Disease Diagnosis NP-Complete Problem",
            problem_type="diagnosis"
        )

        phenotypes = list(instance.phenotypes)
        candidate_diseases = list(instance.candidate_diseases)
        disease_probs = {d: dict(instance.disease_phenotype_probs.get(d, {}))
                         for d in candidate_diseases}
        weights = dict(instance.weights or {})

        sim_pq, term_table = DiseaseDiagnosisAdapter._build_similarity_tables(
            phenotypes, candidate_diseases, disease_probs)

        threshold = float(getattr(instance, "threshold", 0.0) or 0.0)
        has_solution = bool(getattr(instance, "has_solution", False))
        opt_score = float(getattr(instance, "opt_score", 0.0) or 0.0)
        meta = instance.metadata() if hasattr(instance, "metadata") else {}

        problem.instance_data = {
            "phenotypes": phenotypes,
            "candidate_diseases": candidate_diseases,
            "disease_phenotype_probs": disease_probs,
            "weights": weights,
            "phenotype_similarity": sim_pq,
            "disease_term_table": term_table,
            # Decision task parameters
            "threshold": threshold,
            "has_solution": has_solution,
            # Reference data for validation
            "opt_score": opt_score,
        }

        problem.description = (
            "Given a patient's observed phenotypes and a list of candidate diseases "
            "with their phenotype profiles, decide whether there exists a way to "
            "partition the patient's phenotypes into clinically coherent clusters and "
            "assign each cluster a set of candidate diseases, so that the total "
            "explanation score defined below reaches at least the given threshold T. "
            "Answer YES or NO."
        )

        problem.clinical_context = {
            "description": (
                "Probabilistic differential diagnosis involves explaining observed "
                "patient phenotypes using candidate diseases. Each candidate disease "
                "is described by its phenotype profile and the likelihood of each "
                "phenotype given the disease. For each patient phenotype, a "
                "precomputed explanation strength `A_d(p)` is provided and used "
                "directly in all scoring terms."
            )
        }

        # ========= Decision space =========
        problem.decision_space = [
            "Decide whether there exists a partition of the patient's phenotype set "
            "into clusters (every phenotype belongs to exactly one cluster; the number "
            "of clusters is your choice) such that, assigning each cluster a non-empty "
            "set of candidate diseases, the total explanation score reaches at least "
            f"the threshold T = {threshold:.6f}.",
            "A disease may be selected for more than one cluster.",
            "Answer YES if such a partition and disease assignment exists, otherwise NO.",
        ]

        problem.metadata = {
            "num_phenotypes": len(phenotypes),
            "num_candidate_diseases": len(candidate_diseases),
            "threshold": round(threshold, 6),
            "opt_score": round(opt_score, 6),
            "has_solution": has_solution,
            "weights": weights,
            "complexity": "NP-complete",
            "problem_type": "diagnosis",
        }

        for disease in candidate_diseases:
            problem.add_entity("diseases", {"name": disease})
        for phenotype in phenotypes:
            problem.add_entity("phenotypes", {"name": phenotype})
        for disease in candidate_diseases:
            problem.add_action(Action(
                name=f"select_{disease}",
                description=f"Use {disease} to explain the phenotypes of a cluster",
            ))

        # ========= Scoring definitions shared with the optimization task =========
        problem.add_constraint(Constraint(
            name="coherence_definition",
            description=(
                "### Cluster Coherence\n\n"
                "coh(C) = 0                                  if |C| = 1\n"
                "coh(C) = (1/(|C|·(|C|−1))) · Σ_{p≠q∈C} s(p,q)  otherwise"
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="coverage_definition",
            description=(
                "### Coverage\n\n"
                "A_d(p) = P(p|d)  if `p` is in `d`'s profile,  and 0 otherwise.\n\n"
                "coverage(p, S) = max_{d ∈ S} A_d(p)\n"
                "Coverage(C, S) = Σ_{p ∈ C} coverage(p, S)"
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="extra_definition",
            description=(
                "### Extra Phenotypes\n\n"
                "For a disease assigned to a cluster, the extra-phenotype penalty is "
                "the fraction of its complete profile probability mass that falls "
                "outside the cluster:\n\n"
                "Extra(d, C) = Σ_{h ∉ C} P(h|d) / Σ_h P(h|d)   ∈ [0, 1]\n\n"
                "The denominator includes all phenotypes in the disease's complete "
                "profile, including phenotypes not observed in the patient.\n\n"
                "Extra(C, S) = Σ_{d ∈ S} Extra(d, C)"
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="uncovered_definition",
            description=(
                "### Uncovered Phenotypes (penalty)\n\n"
                "Uncov(C, S) = Σ_{p ∈ C} (1 − max_{d ∈ S} A_d(p))"
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="relation_definition",
            description=(
                "### Disease Relation\n\n"
                "For two selected diseases `d_a` and `d_b` within the same cluster:\n\n"
                "O(d_a, d_b) = Σ_{p ∈ C} A_{d_a}(p) · A_{d_b}(p)      (overlap)\n"
                "R(d_a, d_b) = −ρ · O(d_a, d_b)\n"
                "Relation(C, S) = Σ_{d_a < d_b ∈ S} R(d_a, d_b)\n\n"
                "Only overlapping (redundant) disease explanations are penalized; "
                "the benefit of complementary explanations is already counted by "
                "Coverage."
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="local_fit_definition",
            description=(
                "### Local Fit\n\n"
                "LocalFit = (1/m) Σ_i Coverage(C_i, S_i)\n\n"
                "where `m` is the total number of patient phenotypes."
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        problem.add_constraint(Constraint(
            name="score_definition",
            description=(
                "### Total Score\n\n"
                f"α = {weights.get('alpha', 1.0):g}, β = {weights.get('beta', 1.0):g}, "
                f"γ = {weights.get('gamma', 0.1):g}, δ = {weights.get('delta', 0.05):g}, "
                f"η = {weights.get('eta', 0.02):g}, "
                f"ρ = {weights.get('rho', 0.5):g}:\n\n"
                "Total = α Σ_i Coh(C_i) + β LocalFit\n"
                "      − δ Σ_i Uncov(C_i,S_i)\n"
                "      − γ Σ_i Extra(C_i,S_i)\n"
                "      + η Σ_i Relation(C_i,S_i)\n\n"
                "For a given partition `{ (C_i, S_i) }`, `Total` is the objective value. "
                "The decision to answer is whether there exists a valid `{ (C_i, S_i) }` "
                f"whose `Total` is at least the threshold T = {threshold:.6f}."
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        # ========= Validity constraints shared with the optimization task =========
        problem.add_constraint(Constraint(
            name="partition_constraint",
            description=(
                "* The clusters must form a partition of the patient's phenotypes: "
                "every phenotype given above appears in exactly one cluster, and no "
                "phenotype is left out. There is no limit on the number of clusters."
            ),
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="nonempty_disease_set",
            description="* Every cluster must be assigned at least one disease.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="select_from_candidate",
            description="* Select diseases only from the candidate disease list.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="reuse_allowed",
            description=(
                "* The same disease may be selected for several clusters; there is "
                "no limit on how many diseases a cluster uses."
            ),
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="no_inferred_associations",
            description=(
                "* Use only the explained strength values `A_d(p)` given in the "
                "Candidate Diseases table. Do not invent disease–phenotype "
                "associations beyond what is listed. The phenotype similarity "
                "`s(p, q)` is given explicitly for coherence; all scoring uses the "
                "provided `A_d(p)` directly."
            ),
            level="clinical", weight=0.0, category="validity",
        ))

        # ========= Objectives =========
        problem.add_objective(Objective(
            name="decide_feasibility",
            description=(
                f"Decide whether there exists a valid phenotype partition and disease "
                f"assignment whose total score is at least the threshold T = "
                f"{threshold:.6f}. Return YES if it exists, NO otherwise."
            ),
            direction="maximize",
            weight=1.0,
            priority=1,
        ))

        return problem
