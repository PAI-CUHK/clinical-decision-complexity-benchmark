#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoring functions for PC-MDE phenotype clustering and disease assignment.

The benchmark score combines cluster coherence, phenotype coverage, profile
mismatch, uncovered phenotypes, and overlapping disease explanations. The
implementation of the final objective is ``local_global_total``.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils.hpo_similarity import get_hpo_similarity

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLUSTERS_FILE = os.path.join(_BASE_DIR, "data", "processed", "clusters_with_diseases.json")


# ======================================================================
# Weights
# ======================================================================

@dataclass
class ScoreWeights:
    alpha: float = 1.0    # cluster coherence     (Local: Σ_i Coh(C_i))
    beta: float = 1.0     # local fit            (Local: Σ_i LocalFit(C_i,S_i))
    gamma: float = 0.1    # extra phenotypes penalty (weighted support outside the cluster)
    delta: float = 0.05   # uncovered phenotypes penalty (tie-break)
    eta: float = 0.02     # disease relation (tie-break)
    rho: float = 0.5      # overlap penalty inside relation
    omega_g: float = 0.5  # Global unification: ω·G, G(C,S)=max_{d∈S} U(d,C)
    # Legacy weights retained for compatibility but excluded from the objective
    gamma_o: float = 0.4
    alpha_u: float = 0.0
    delta_g: float = 0.0

    def to_dict(self) -> Dict:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma,
                "delta": self.delta, "eta": self.eta, "rho": self.rho,
                "omega_g": self.omega_g, "gamma_o": self.gamma_o,
                "alpha_u": self.alpha_u, "delta_g": self.delta_g}


# ======================================================================
# Data
# ======================================================================

@dataclass
class ClusterProblemData:
    """Precomputed data for one instance."""
    phenotypes: List[str]
    diseases: List[str]
    A: np.ndarray                    # (n, m) A[d][p] = P(p|d) if p∈profile(d) else 0
    h_sim: List[np.ndarray]          # For each disease: s(h, p) with shape (|H_d|, m)
    h_prob: List[np.ndarray]         # For each disease: P(h|d) with shape (|H_d|,)
    h_names: List[List[str]]
    coh: np.ndarray                  # (m, m) s(p, q)
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    @property
    def n(self) -> int:
        return len(self.diseases)

    @property
    def m(self) -> int:
        return len(self.phenotypes)


def build_cluster_problem_data(
    phenotypes: Sequence[str],
    diseases: Sequence[str],
    disease_probs: Dict[str, Dict[str, float]],
    weights: Optional[ScoreWeights] = None,
) -> ClusterProblemData:
    """Precompute A, h_sim, h_prob, and coherence."""
    sim = get_hpo_similarity()
    phenotypes = list(phenotypes)
    diseases = list(diseases)
    m = len(phenotypes)

    A = np.zeros((len(diseases), m))
    h_sim: List[np.ndarray] = []
    h_prob: List[np.ndarray] = []
    h_names: List[List[str]] = []

    for i, d in enumerate(diseases):
        probs = disease_probs.get(d, {})
        names = list(probs.keys())
        h_names.append(names)
        if names:
            pv = np.array([float(probs[h]) for h in names])
            mv = np.array([[sim.sim(h, p) for p in phenotypes] for h in names])
        else:
            pv = np.zeros(0)
            mv = np.zeros((0, m))
        h_prob.append(pv)
        h_sim.append(mv)
        # Exact membership: A[d,p] = P(p|d) if p is in the disease profile, otherwise 0
        row = np.zeros(m)
        for p_idx, p in enumerate(phenotypes):
            if p in probs:
                row[p_idx] = float(probs[p])
        A[i] = row

    coh = np.zeros((m, m))
    for a in range(m):
        for b in range(a + 1, m):
            v = sim.sim(phenotypes[a], phenotypes[b])
            coh[a, b] = v
            coh[b, a] = v
    np.fill_diagonal(coh, 1.0)

    return ClusterProblemData(
        phenotypes=phenotypes, diseases=diseases, A=A,
        h_sim=h_sim, h_prob=h_prob, h_names=h_names, coh=coh,
        weights=weights or ScoreWeights(),
    )


# ======================================================================
# Component scores
# ======================================================================

def cluster_coherence(data: ClusterProblemData, c_idx: Sequence[int]) -> float:
    c = list(c_idx)
    if len(c) < 2:
        return 0.0
    sub = data.coh[np.ix_(c, c)]
    k = len(c)
    total = sub.sum() - k  # Exclude the diagonal (equal to 1)
    return float(total / (k * (k - 1)))


def cluster_coverage(data: ClusterProblemData, c_idx: Sequence[int],
                     s_idx: Sequence[int]) -> float:
    c = list(c_idx)
    s = list(s_idx)
    if not c or not s:
        return 0.0
    return float(data.A[np.ix_(s, c)].max(axis=0).sum())


def cluster_uncovered(data: ClusterProblemData, c_idx: Sequence[int],
                      s_idx: Sequence[int]) -> float:
    return float(len(list(c_idx)) - cluster_coverage(data, c_idx, s_idx))


def disease_extra(data: ClusterProblemData, d_idx: int, c_idx: Sequence[int]) -> float:
    """Normalized disease-to-cluster mismatch:
        ExtraNorm(d, C) = Σ_{h∉C} P(h|d) / Σ_h P(h|d), in [0, 1].
    
    Use the complete HPO profile of disease d. Phenotypes in C incur no penalty;
    all others contribute their probability to the numerator. The denominator
    is the total probability mass of the profile."""
    c_names = {data.phenotypes[j] for j in c_idx}
    total = 0.0
    mass = 0.0
    for h, pv in zip(data.h_names[d_idx], data.h_prob[d_idx]):
        mass += float(pv)
        if h not in c_names:      # Profile phenotypes outside this cluster, including unobserved phenotypes
            total += float(pv)
    return float(total / mass) if mass > 0.0 else 0.0


def cluster_extra(data: ClusterProblemData, c_idx: Sequence[int],
                  s_idx: Sequence[int]) -> float:
    return float(sum(disease_extra(data, d, c_idx) for d in s_idx))


def relation_pair(va: np.ndarray, vb: np.ndarray, rho: float) -> float:
    """Penalize overlapping explanations for the same phenotype."""
    O = float((va * vb).sum())
    return -rho * O


def disease_relation(data: ClusterProblemData, c_idx: Sequence[int],
                     s_idx: Sequence[int], rho: float) -> float:
    """Sum pairwise overlap penalties within a cluster:
        Relation(S_i) = Σ_{a<b} R(a,b) = -ρ Σ_{a<b} O(a,b)."""
    c = list(c_idx)
    s = list(s_idx)
    if len(s) < 2 or not c:
        return 0.0
    sub = data.A[np.ix_(s, c)]           # (|S|, |C|)
    total = 0.0
    for a in range(len(s)):
        for b in range(a + 1, len(s)):
            total += relation_pair(sub[a], sub[b], rho)
    return float(total)


@dataclass
class BlockScore:
    coherence: float
    coverage: float
    extra: float
    uncovered: float
    relation: float
    total: float


def score_block(data: ClusterProblemData, c_idx: Sequence[int], s_idx: Sequence[int],
                w: Optional[ScoreWeights] = None) -> BlockScore:
    w = w or data.weights
    coh = cluster_coherence(data, c_idx)
    cov = cluster_coverage(data, c_idx, s_idx)
    unc = float(len(list(c_idx))) - cov
    ext = cluster_extra(data, c_idx, s_idx)
    rel = disease_relation(data, c_idx, s_idx, w.rho)
    total = (w.alpha * coh + w.beta * cov - w.gamma * ext
             - w.delta * unc + w.eta * rel)
    return BlockScore(coh, cov, ext, unc, rel, total)


def total_score(data: ClusterProblemData,
                clusters: Sequence[Tuple[Sequence[int], Sequence[int]]],
                w: Optional[ScoreWeights] = None) -> float:
    """Return the total score of clusters [(c_idx, s_idx), ...]."""
    return float(sum(score_block(data, c, s, w).total for c, s in clusters))


# ======================================================================
# Scoring primitives for the nested cluster and disease-selection objective
# ======================================================================
# The module docstring describes the total; helpers here compute its components.
# The caller supplies attribution[p] as a disease index in S_i or -1 for unexplained.

def global_fit(data: ClusterProblemData, s_idx: Sequence[int]) -> float:
    """GlobalFit(P,S) = Σ_{p∈P} max_{d∈S} A(d,p)."""
    s = list(s_idx)
    if not s:
        return 0.0
    return float(data.A[np.ix_(s, range(data.m))].max(axis=0).sum())


def global_omission_value(data: ClusterProblemData, s_idx: Sequence[int]) -> float:
    """Σ_{p∈P} (1 − max_{d∈S} A(d,p))."""
    s = list(s_idx)
    if not s:
        return float(data.m)
    return float(data.m - data.A[np.ix_(s, range(data.m))].max(axis=0).sum())


def unique_contribution(data: ClusterProblemData, s_idx: Sequence[int]) -> float:
    """Compute Unique(P,S) = Σ_d [GlobalFit(S) - GlobalFit(S minus d)].
    
    For each phenotype, use the top two explanation strengths. Removing the
    strongest disease reduces its contribution by top1 - top2."""
    s = list(s_idx)
    if not s:
        return 0.0
    if len(s) == 1:
        return global_fit(data, s)   # With one disease, its marginal contribution is the total fit
    A_S = data.A[np.ix_(s, range(data.m))]                # (|S|, m)
    argmax = A_S.argmax(axis=0)                           # per-phenotype best disease (int)
    top1 = A_S.max(axis=0)
    masked = A_S.copy()
    rows = np.arange(data.m)
    masked[argmax, rows] = -np.inf
    top2 = masked.max(axis=0)
    drop = top1 - top2                                    # (m,) loss per phenotype on removing best
    u = 0.0
    for didx_local, d in enumerate(s):
        u += float(drop[argmax == didx_local].sum())
    return u


def attributed_fit(data: ClusterProblemData, attribution: Sequence[int],
                   valid: Optional[Sequence[int]] = None) -> float:
    """Sum A(E(p), p) for phenotypes attributed to a valid disease.
    
    attribution[p] = -1 denotes an unexplained phenotype. Attribution to a
    disease outside the valid disease set does not contribute."""
    vset = set(int(v) for v in valid) if valid is not None else None
    total = 0.0
    m = data.m
    for p in range(m):
        d = int(attribution[p])
        if d >= 0 and (vset is None or d in vset):
            total += float(data.A[d, p])
    return total


def local_omission_value(data: ClusterProblemData, attribution: Sequence[int],
                         s_idx: Sequence[int]) -> float:
    """Return Σ_{p:E(p)=∅} (1 - max_{d∈S} A(d,p)), the unexplained-phenotype penalty."""
    s = list(s_idx)
    m = data.m
    if not s:
        return float(sum(1.0 for p in range(m) if int(attribution[p]) < 0))
    best = data.A[np.ix_(s, range(m))].max(axis=0)        # (m,)
    total = 0.0
    for p in range(m):
        if int(attribution[p]) < 0:
            total += float(1.0 - best[p])
    return total


def nested_score_total(data: ClusterProblemData,
                       clusters: Sequence[Tuple[Sequence[int], Sequence[int]]],
                       attribution: Sequence[int],
                       w: Optional[ScoreWeights] = None) -> Dict[str, float]:
    """Compute the nested objective for a complete solution (C,S,E).
    
    clusters contains (phenotype indices, disease indices) pairs; the global
    disease set is their union. attribution maps each phenotype to a disease
    index or -1. Returns the score and its component values."""
    w = w or data.weights
    s_all: List[int] = []
    for _, s in clusters:
        for d in s:
            if int(d) not in s_all:
                s_all.append(int(d))
    unc_total = 0.0              # Uncovered penalty for weakly explained phenotypes in each cluster
    for c, s in clusters:
        if not s or not c:
            unc_total += float(len(list(c)))
            continue
        cov = float(data.A[np.ix_(s, c)].max(axis=0).sum())
        unc_total += float(len(list(c)) - cov)
    coh = sum(score_block(data, c, s2, w).coherence for c, s2 in clusters)
    afit = attributed_fit(data, attribution, valid=s_all)
    lom = local_omission_value(data, attribution, s_all)
    ext = sum(score_block(data, c, s2, w).extra for c, s2 in clusters)
    rel = sum(score_block(data, c, s2, w).relation for c, s2 in clusters)
    uni = unique_contribution(data, s_all)
    gom = global_omission_value(data, s_all)
    total = (w.alpha * coh
             + w.beta * afit
             - w.gamma_o * lom
             - w.delta * unc_total
             - w.gamma * ext
             + w.eta * rel
             + w.alpha_u * uni
             - w.delta_g * gom)
    return {"coherence": coh, "attributed_fit": afit, "local_omission": lom,
            "uncovered": unc_total, "extra": ext, "relation": rel, "unique": uni,
            "global_omission": gom, "total": total, "num_diseases": len(s_all)}


# ======================================================================
# Total objective (local components only)
# ======================================================================
#   L(C,S) = Σ_i [ α·Coh(C_i) + β·LocalFit(C_i,S_i) − γ·Extra(C_i,S_i)
#                  + η·Rel(S_i) − δ·Uncov(C_i,S_i) ]
#             LocalFit(C_i,S_i) = (1/|C_i|) Σ_{p∈C_i} max_{d∈S_i} A_d(p)
#       Partitions phenotypes into coherent modules, each with well-matched
#       disease assignments.
#
#   The former cross-module Global term has been removed:
#   the objective consists of additive block-local components.

def local_fit(data: ClusterProblemData, c_idx: Sequence[int],
              s_idx: Sequence[int]) -> float:
    """LocalFit(C_i,S_i) = (1/|C_i|) Σ_{p∈C_i} max_{d∈S_i} A_d(p)."""
    c = list(c_idx)
    s = list(s_idx)
    if not c or not s:
        return 0.0
    return float(data.A[np.ix_(s, c)].max(axis=0).mean())


def disease_support(data: ClusterProblemData, d_idx: int, c_idx: Sequence[int]) -> float:
    """Return M(d,C_i) = (1/|C_i|) Σ_{p∈C_i} A_d(p)."""
    c = list(c_idx)
    if not c:
        return 0.0
    return float(data.A[d_idx, c].mean())


def local_global_total(data: ClusterProblemData,
                       clusters: Sequence[Tuple[Sequence[int], Sequence[int]]],
                       w: Optional[ScoreWeights] = None) -> Dict[str, float]:
    """Compute the total objective for a complete cluster solution.
    
    Score = α·ΣCoh + β·LocalFit − γ·ΣExtra + η·ΣRel − δ·ΣUncov.
    clusters contains (phenotype indices, disease indices) pairs. Returns the
    component values and total; the removed global term is always zero."""
    w = w or data.weights
    coh = 0.0
    lf = 0.0
    ext = 0.0
    rel = 0.0
    unc = 0.0
    cov_sum = 0.0
    for c, s in clusters:
        c = list(c)
        s = list(s)
        if not c:
            continue
        coh += cluster_coherence(data, c)
        cov = cluster_coverage(data, c, s)
        cov_sum += cov
        ext += cluster_extra(data, c, s)
        rel += disease_relation(data, c, s, w.rho)
        unc += float(len(c) - cov)
    # The objective contains local components only (Global has been removed):
    #   Score = α·ΣCoh + β·LocalFit − γ·ΣExtra + η·ΣRel − δ·ΣUncov
    # LocalFit is normalized by m, so each phenotype contributes once.
    lf = cov_sum / float(data.m) if data.m else 0.0
    total = (w.alpha * coh + w.beta * lf
             - w.gamma * ext + w.eta * rel - w.delta * unc)
    return {"coherence": coh, "local_fit": lf, "global": 0.0,
            "extra": ext, "relation": rel, "uncovered": unc,
            "total": float(total), "num_diseases": len({int(d) for _, s in clusters for d in s})}


def block_local_value(data: ClusterProblemData, c_idx: Sequence[int],
                      s_idx: Sequence[int], w: ScoreWeights) -> float:
    """Return one block local value: α·Coh + β·LocalFit − γ·Extra + η·Rel − δ·Uncov.
    
    This block-level value excludes ω·G, which depends on all clusters."""
    c = list(c_idx)
    s = list(s_idx)
    coh = cluster_coherence(data, c)
    lf = local_fit(data, c, s)
    ext = cluster_extra(data, c, s)
    rel = disease_relation(data, c, s, w.rho)
    cov = cluster_coverage(data, c, s)
    unc = float(len(c) - cov)
    return float(w.alpha * coh + w.beta * lf - w.gamma * ext + w.eta * rel - w.delta * unc)


# ======================================================================
# Solution validity
# ======================================================================

def check_partition(m: int, clusters: Sequence[Tuple[Sequence[int], Sequence[int]]]):
    """Check that clusters partition P and every disease set is nonempty.
    
    Returns (ok, errors)."""
    errors = []
    seen = []
    for i, (c, s) in enumerate(clusters):
        if len(list(c)) == 0:
            errors.append(f"cluster {i} has no phenotypes")
        if len(list(s)) == 0:
            errors.append(f"cluster {i} has no diagnoses")
        seen.extend(list(c))
    if sorted(seen) != list(range(m)):
        dup = [x for x in set(seen) if seen.count(x) > 1]
        if dup:
            errors.append(f"phenotypes assigned to multiple clusters: {sorted(dup)}")
        missing = sorted(set(range(m)) - set(seen))
        if missing:
            errors.append(f"phenotypes not assigned: {missing}")
    return (len(errors) == 0), errors


# ======================================================================
# Global disease pool (shared with the earlier version)
# ======================================================================

def load_global_disease_pool() -> Dict[str, Dict[str, float]]:
    """Load {disease_name: {phenotype: P(phenotype|disease)}} without OMIM prefixes."""
    with open(CLUSTERS_FILE, "r", encoding="utf-8") as f:
        clusters = json.load(f)
    pool: Dict[str, Dict[str, float]] = {}
    for cluster in clusters:
        for d in cluster["diseases"]:
            name = d["disease"]
            if "|" in name:
                name = name.split("|", 1)[1]
            if name not in pool:
                pool[name] = {s["symptom"]: float(s["probability"]) for s in d["symptoms"]}
    return pool