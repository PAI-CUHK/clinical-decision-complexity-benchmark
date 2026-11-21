#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter for the PC-MDE optimization task.

The decision space partitions patient phenotypes into nonempty clusters and
assigns a nonempty subset of candidate diseases to each cluster. The objective
combines coherence, coverage, extra phenotypes, uncovered phenotypes, and
overlap; its implementation and weights are defined in objective.py.

The prompt supplies pairwise phenotype similarities and explanation strengths
A_d(p) = P(p|d) for phenotypes in a disease profile, or zero otherwise."""


from core import MedicalProblemAdapter, MedicalNPProblem, Action, Constraint, Objective
from tasks.pc_mde.generator_o import LEVEL_CONFIG
from utils.hpo_similarity import get_hpo_similarity


class DiseaseDiagnosisAdapter(MedicalProblemAdapter):
    """Adapter for the PC-MDE optimization task."""

    def generate_instance_config(self, level):
        cfg = LEVEL_CONFIG.get(level, LEVEL_CONFIG[5])
        return {"level": level, "m": cfg["m"], "n": cfg["n"]}

    # ------------------------------------------------------------------
    # Precompute similarity tables for the prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _build_similarity_tables(phenotypes, candidate_diseases, disease_probs):
        sim = get_hpo_similarity()

        # Pairwise patient phenotype similarities for coherence (symmetric; diagonal omitted)
        sim_pq = [[float(sim.sim(a, b)) for b in phenotypes] for a in phenotypes]

        # Complete phenotype profile for each candidate disease [(h, P(h|d)), ...]
        term_table = {}
        for d in candidate_diseases:
            probs = disease_probs.get(d, {})
            term_table[d] = [(h, float(pv)) for h, pv in probs.items()]
        return sim_pq, term_table

    # ------------------------------------------------------------------

    def build(self, instance):
        problem = MedicalNPProblem(
            name="Disease Diagnosis Problem",
            problem_type="diagnosis"
        )

        phenotypes = list(instance.phenotypes)
        candidate_diseases = list(instance.candidate_diseases)
        disease_probs = {d: dict(instance.disease_phenotype_probs.get(d, {}))
                         for d in candidate_diseases}
        weights = dict(instance.weights or {})

        sim_pq, term_table = self._build_similarity_tables(
            phenotypes, candidate_diseases, disease_probs)

        k_opt = int(getattr(instance, "k_opt", 0) or 0)
        opt_score = float(getattr(instance, "opt_score", 0.0) or 0.0)
        meta = instance.metadata() if hasattr(instance, "metadata") else {}

        problem.instance_data = {
            "phenotypes": phenotypes,
            "candidate_diseases": candidate_diseases,
            "disease_phenotype_probs": disease_probs,
            "weights": weights,
            "phenotype_similarity": sim_pq,
            "disease_term_table": term_table,
            # Reference score computed by the generator for validation
            "opt_score": opt_score,
            "k_opt": k_opt,
            "opt_clusters": [
                {"phenotypes": list(b["phenotypes"]), "diseases": list(b["diseases"])}
                for b in (getattr(instance, "opt_clusters", None) or [])
            ],
            "merge_score": float(meta.get("merge_score", 0.0) or 0.0),
            "singleton_score": float(meta.get("singleton_score", 0.0) or 0.0),
        }

        problem.description = (
            "Given a patient's observed phenotypes and a list of candidate diseases "
            "with their phenotype profiles, partition the patient's phenotypes into "
            "clinically coherent clusters and, for each cluster, choose a set of "
            "candidate diseases that best explains it, so that the total explanation "
            "score defined below is maximized."
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
            "Decide a partition of the patient's phenotype set into clusters "
            "(every phenotype belongs to exactly one cluster; the number of "
            "clusters is your choice).",
            "For every cluster, select a non-empty set of candidate diseases that "
            "explains that cluster.",
            "A disease may be selected for more than one cluster.",
            "Return the partition and the per-cluster disease sets in the required "
            "JSON format.",
        ]

        problem.metadata = {
            "num_phenotypes": len(phenotypes),
            "num_candidate_diseases": len(candidate_diseases),
            "k_opt": k_opt,
            "opt_score": round(opt_score, 6),
            "weights": weights,
            "complexity": "NP-hard",
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
                "Maximize Total over all valid `{ (C_i, S_i) }`."
            ),
            level="clinical", weight=0.0, category="definition",
        ))

        # ========= Validity Constraints =========
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
            name="maximize_total_score",
            description=(
                "Maximize the total score defined above over all valid phenotype "
                "partitions and disease assignments."
            ),
            direction="maximize",
            weight=0.1,
            priority=1,
        ))

        return problem