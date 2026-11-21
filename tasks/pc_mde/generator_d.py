#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate PC-MDE decision instances from the optimization task.

The decision asks whether a valid phenotype partition and nonempty disease
assignments achieve a score of at least T. A feasible reference certifies YES
when T is below its score. A conservative upper bound certifies NO when T
exceeds the bound; the heuristic reference is not a proof of optimality."""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List

from tasks.pc_mde.generator_o import generate_disease_instance


@dataclass
class DiseaseNPCCompleteInstance:
    """PC-MDE decision instance with phenotype partition and disease assignments.
    
    The answer is YES when a valid solution has score at least threshold, and NO
    otherwise. The has_solution field stores the expected answer."""
    # Observed patient phenotypes
    phenotypes: List[str]
    # Candidate diseases
    candidate_diseases: List[str]
    # Disease-to-phenotype probability mapping {disease: {phenotype: P(phenotype|disease)}}
    disease_phenotype_probs: Dict[str, Dict[str, float]]
    # Scoring weights (ScoreWeights.to_dict())
    weights: Dict[str, float] = field(default_factory=dict)
    # Decision threshold T for the existence of a valid solution with score >= T
    threshold: float = 0.0
    # Whether a solution exists (True -> YES, False -> NO)
    has_solution: bool = False
    # Reference solution score
    opt_score: float = 0.0
    # Feasible reference partition
    opt_clusters: List[Dict] = field(default_factory=list)
    # Decision margin epsilon
    epsilon: float = 0.0
    # Metadata
    _metadata: Dict = field(default_factory=dict)

    def metadata(self) -> Dict:
        """Return instance metadata."""
        return self._metadata


def generate_disease_npcomplete_instance(config: Dict, verbose: bool = False) -> DiseaseNPCCompleteInstance:
    """Generate a PC-MDE decision instance.
    
    Args:
        config: Level, seed, and optional optimization generator settings.
        verbose: Whether to print generation details.
    
    Returns:
        A DiseaseNPCCompleteInstance sharing the optimization task structure."""
    # Reuse the optimization instance and its feasible reference solution
    base = generate_disease_instance(config, verbose=verbose)

    opt_score = float(base.opt_score)
    rng = random.Random(int(config.get("seed", 42)))
    weights = dict(base.weights or {})
    # A feasible reference certifies Yes; an objective upper bound certifies No.
    alpha = float(weights.get("alpha", 1.0))
    beta = float(weights.get("beta", 1.0))
    gamma = float(weights.get("gamma", 0.1))
    delta = float(weights.get("delta", 0.05))
    eta = float(weights.get("eta", 0.02))
    rho = float(weights.get("rho", 0.5))
    if (not all(math.isfinite(x) for x in (opt_score, alpha, beta, gamma, delta, eta, rho))
            or gamma < 0 or delta < 0 or eta * rho < 0):
        raise ValueError("Cannot certify decision bound with these weights")
    if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1
           for profile in base.disease_phenotype_probs.values() for p in profile.values()):
        raise ValueError("Cannot certify decision bound with invalid probabilities")
    upper_bound = max(alpha, 0.0) * len(base.phenotypes) + max(beta, 0.0)
    if opt_score > upper_bound + 1e-7:
        raise ValueError("Reference score exceeds certified upper bound")
    want_yes = rng.random() < 0.5
    margin = rng.uniform(0.1, 1.0)
    threshold = opt_score - margin if want_yes else upper_bound + margin
    epsilon = abs(threshold - opt_score)
    has_solution = want_yes
    meta = base.metadata() if hasattr(base, "metadata") else {}

    instance = DiseaseNPCCompleteInstance(
        phenotypes=list(base.phenotypes),
        candidate_diseases=list(base.candidate_diseases),
        disease_phenotype_probs={d: dict(base.disease_phenotype_probs.get(d, {}))
                                 for d in base.candidate_diseases},
        weights=weights,
        threshold=float(threshold),
        has_solution=bool(has_solution),
        opt_score=float(opt_score),
        opt_clusters=[{"phenotypes": list(b.get("phenotypes", [])),
                       "diseases": list(b.get("diseases", []))}
                      for b in (base.opt_clusters or [])],
        epsilon=float(epsilon),
        _metadata={
            **meta,
            "level": int(config.get("level", 5)),
            "seed": int(config.get("seed", 42)),
            "complexity": "NP-complete",
            "decision_version": True,
            "opt_score": float(opt_score),
            "threshold": float(threshold),
            "epsilon": float(epsilon),
            "has_solution": bool(has_solution),
            "expected_answer": "Yes" if has_solution else "No",
            "want_yes": bool(want_yes),
        },
    )
    return instance


# Compatibility alias for earlier callers
def generate_disease_npcomplete_instance_data(config: Dict, verbose: bool = False) -> DiseaseNPCCompleteInstance:
    """Compatibility alias for generate_disease_npcomplete_instance."""
    return generate_disease_npcomplete_instance(config, verbose=verbose)
