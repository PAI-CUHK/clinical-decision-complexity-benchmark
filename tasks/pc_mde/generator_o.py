"""Sample PC-MDE instances from disease profiles without difficulty targeting."""

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List

from tasks.pc_mde.objective import ScoreWeights, load_global_disease_pool


LEVEL_CONFIG = {level: {"m": 19 + level, "n": 14 + level}
                for level in range(1, 11)}

_MODIFIER_PATTERNS = (
    re.compile(r".*\bonset\b$", re.IGNORECASE),
    re.compile(r"^(acute|chronic|subacute|progressive|relapsing|remitting)\b", re.IGNORECASE),
    re.compile(r"\b(late[- ]onset|early[- ]onset|slowly[- ]progressive|rapidly[- ]progressive)\b", re.IGNORECASE),
)
_MODIFIERS = {
    "Neonatal onset", "Infantile onset", "Childhood onset", "Juvenile onset",
    "Young adult onset", "Middle age onset", "Late onset", "Congenital onset",
    "Fetal onset", "Antenatal onset", "Adolescent onset", "Acute onset",
    "Chronic progressive", "Slowly progressive", "Rapidly progressive",
    "Relapsing-remitting", "Progressive",
}


def is_modifier_phenotype(phenotype: str) -> bool:
    return phenotype in _MODIFIERS or any(p.search(phenotype) for p in _MODIFIER_PATTERNS)


@dataclass
class DiseaseInstance:
    phenotypes: List[str]
    candidate_diseases: List[str]
    disease_phenotype_probs: Dict[str, Dict[str, float]]
    weights: Dict[str, float] = field(default_factory=dict)
    opt_score: float = 0.0
    opt_clusters: List[Dict] = field(default_factory=list)
    k_opt: int = 0
    _metadata: Dict = field(default_factory=dict)

    def metadata(self) -> Dict:
        return self._metadata


def generate_disease_instance(config: Dict, verbose: bool = False) -> DiseaseInstance:
    """Sample candidates and observed phenotypes, then compute one reference.

    The level controls only the numbers of phenotypes and candidate diseases.
    ``seed`` makes sampling reproducible. The reference search is heuristic and
    is used for scoring; it does not filter or select generated candidates.
    """
    level = int(config.get("level", 5))
    if level not in LEVEL_CONFIG:
        raise ValueError("level must be between 1 and 10")
    m = int(config.get("m", LEVEL_CONFIG[level]["m"]))
    n = int(config.get("n", LEVEL_CONFIG[level]["n"]))
    if m < 1 or n < 1:
        raise ValueError("m and n must be positive")
    rng = random.Random(config.get("seed"))
    pool = {
        disease: {p: float(prob) for p, prob in profile.items()
                  if not is_modifier_phenotype(p)}
        for disease, profile in load_global_disease_pool().items()
    }
    pool = {d: profile for d, profile in pool.items() if len(profile) >= 3}
    if len(pool) < n:
        raise ValueError(f"Disease pool has {len(pool)} usable profiles; {n} required")
    diseases = rng.sample(sorted(pool), n)
    phenotype_pool = sorted({phenotype for disease in diseases for phenotype in pool[disease]})
    if len(phenotype_pool) < m:
        raise ValueError(f"Sampled profiles have {len(phenotype_pool)} distinct phenotypes; {m} required")
    phenotypes = rng.sample(phenotype_pool, m)
    probs = {disease: pool[disease] for disease in diseases}
    weights = ScoreWeights(gamma=0.8)
    for name in ("alpha", "beta", "gamma", "delta", "eta", "rho", "omega_g"):
        if config.get(name) is not None:
            setattr(weights, name, float(config[name]))
    instance_data = {"phenotypes": phenotypes, "candidate_diseases": diseases,
                     "disease_phenotype_probs": probs, "weights": weights.to_dict()}
    for name in ("glb_iters", "glb_restarts", "glb_unifiers", "glb_mlp", "glb_budget"):
        if name in config:
            instance_data[name] = config[name]
    from tasks.pc_mde.baseline import compute_cluster_baseline
    baseline = compute_cluster_baseline(instance_data)
    clusters = baseline["clusters"]
    assigned = [p for block in clusters for p in block["phenotypes"]]
    if (not clusters or sorted(assigned) != sorted(phenotypes)
            or any(not block["diseases"] for block in clusters)):
        raise RuntimeError("Reference search did not return a valid solution")
    if verbose:
        print(f"[PC-MDE] level={level} phenotypes={m} diseases={n}")
    return DiseaseInstance(
        phenotypes=phenotypes, candidate_diseases=diseases,
        disease_phenotype_probs=probs, weights=weights.to_dict(),
        opt_score=baseline["opt_score"], opt_clusters=clusters,
        k_opt=baseline["k_opt"],
        _metadata={"level": level, "seed": config.get("seed"), "m": m, "n": n,
                   "opt_score": baseline["opt_score"], "k_opt": baseline["k_opt"],
                   "merge_score": baseline["merge_score"],
                   "singleton_score": baseline["singleton_score"]},
    )
