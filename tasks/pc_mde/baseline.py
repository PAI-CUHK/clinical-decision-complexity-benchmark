#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference algorithms for multi-disease explanation."""

from typing import List, Dict, Tuple

def calculate_score(symptoms: List[str], diseases: List[str], 
                   disease_symptom_probs: Dict[str, Dict[str, float]]) -> float:
    """Return the proportion of symptoms covered by the selected diseases.
    
    Args:
        symptoms: Observed symptoms.
        diseases: Selected diseases.
        disease_symptom_probs: Disease-to-symptom probability mapping.
    
    Returns:
        Coverage fraction in [0, 1]."""
    if not symptoms or not diseases:
        return 0.0
    
    covered_count = 0
    for symptom in symptoms:
        max_prob = 0.0
        for disease in diseases:
            prob = disease_symptom_probs.get(disease, {}).get(symptom, 0.0)
            if prob > max_prob:
                max_prob = prob
        if max_prob > 0:
            covered_count += 1
    
    return covered_count / len(symptoms)


def compute_cluster_baseline(instance_data: Dict) -> Dict:
    """Compute a reference solution from instance data.
    
    Search from known and alternative partitions using hill climbing, random
    perturbations, and restarts. The best solution found within the time and
    iteration budgets is returned without a proof of global optimality.
    
    Returns:
        A dictionary containing clusters, k_opt, opt_score, merge_score,
        singleton_score, and s_sizes."""
    from tasks.pc_mde.objective import (
        build_cluster_problem_data, ScoreWeights, local_global_total,
    )
    from tasks.pc_mde.solver import best_disease_set, global_optimum

    weights = instance_data.get("weights") or {}
    w = ScoreWeights(**{k: v for k, v in weights.items()
                        if k in ScoreWeights.__dataclass_fields__})
    data = build_cluster_problem_data(
        instance_data.get("phenotypes", []),
        instance_data.get("candidate_diseases", []),
        instance_data.get("disease_phenotype_probs", {}),
        w,
    )
    p_index = {p: i for i, p in enumerate(data.phenotypes)}
    d_index = {d: i for i, d in enumerate(data.diseases)}

    # Search from known partitions and keep the best solution within the budget.
    seed_blocks = []
    for blk in (instance_data.get("opt_clusters") or []):
        c_idx = [p_index[p] for p in blk.get("phenotypes", []) if p in p_index]
        if c_idx:
            seed_blocks.append(c_idx)
    go = global_optimum(
        data, w, seed_blocks,
        iters=int(instance_data.get("glb_iters", 200)),
        restarts=int(instance_data.get("glb_restarts", 6)),
        top_unifiers=int(instance_data.get("glb_unifiers", 4)),
        time_limit=float(instance_data.get("glb_mlp", 5.0)),
        budget=float(instance_data.get("glb_budget", 30.0)))
    clusters = go["clusters"]
    k_opt = len(clusters)
    opt_score = float(go["opt_score"])

    # Merge and singleton references supply the lower score for normalization.
    def _total_of(part: List[Tuple[List[int], List[int]]]) -> float:
        try:
            return float(local_global_total(data, part, w)["total"])
        except Exception:
            return -1e18

    mg = best_disease_set(data, list(range(data.m)), w, time_limit=20.0)
    merge_score = _total_of([(list(range(data.m)), list(mg["diseases"]))]) \
        if mg.get("success") else -1e18
    sing_total = 0.0
    for p in range(data.m):
        r = best_disease_set(data, [p], w, time_limit=10.0)
        if r.get("success"):
            sing_total += _total_of([([p], list(r["diseases"]))])

    return {
        "clusters": [
            {"phenotypes": [data.phenotypes[i] for i in c],
             "diseases": [data.diseases[i] for i in s]}
            for c, s in clusters
        ],
        "k_opt": k_opt,
        "opt_score": opt_score,
        "merge_score": merge_score,
        "singleton_score": sing_total,
        "s_sizes": [len(s) for _, s in clusters],
    }
