#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MILP disease selection and heuristic partition search for PC-MDE."""

import itertools
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

from tasks.pc_mde.objective import (
    ClusterProblemData, ScoreWeights, disease_extra, relation_pair, score_block,
    local_global_total,
)


def best_disease_set(data: ClusterProblemData, c_idx: Sequence[int],
                     weights: Optional[ScoreWeights] = None,
                     time_limit: float = 30.0,
                     force: Sequence[int] = ()) -> Dict:
    """Find the best nonempty disease subset for a fixed cluster.
    
    force optionally specifies disease indices that must be selected. Returns
    success, score, coverage, extra, relation, diseases, coherence, and uncovered."""
    w = weights or data.weights
    c = list(c_idx)
    n = data.n
    mc = len(c)
    if mc == 0 or n == 0:
        return {"success": False, "message": "empty cluster or no diseases",
                "score": 0.0, "diseases": []}

    A_c = data.A[:, c]                              # (n, mc)
    extra_d = np.array([disease_extra(data, d, c) for d in range(n)])
    tau = np.zeros((n, n))                          # R(a,b) on block C
    for a in range(n):
        for b in range(a + 1, n):
            tau[a, b] = relation_pair(A_c[a], A_c[b], w.rho)

    pairs = list(itertools.combinations(range(n), 2))
    pair_idx = {p: i for i, p in enumerate(pairs)}

    n_x, n_w = n, n * mc
    n_y = len(pairs)
    num_vars = n_x + n_w + n_y
    integrality = np.zeros(num_vars, dtype=int)
    integrality[:n_x] = 1
    integrality[n_x + n_w:] = 1
    lb = np.zeros(num_vars)
    ub = np.ones(num_vars)
    for _d in force:                       # Force x_d = 1
        if 0 <= int(_d) < n:
            lb[int(_d)] = 1.0

    # Objective (minimize): γ·Extra − (β+δ)·Coverage − η·Relation
    # True score = α·coh(C) − δ·|C| − obj (constant terms are independent of S)
    cobj = np.zeros(num_vars)
    for i in range(n):
        cobj[i] = w.gamma * extra_d[i]
    for i in range(n):
        for j in range(mc):
            # Coverage weight must match local_global_total: LocalFit uses the total
            # number m of patient phenotypes (β·cov/m), plus δ·cov from Uncovered.
            # Thus each block coverage coefficient is (β/m + δ), not (β+δ).
            # The latter overstates coverage gains and can select unsuitable diseases.
            cobj[n_x + i * mc + j] = -(w.beta / data.m + w.delta) * A_c[i, j]
    for (a, b), k in pair_idx.items():
        cobj[n_x + n_w + k] = -w.eta * tau[a, b]

    rows, cols, vals, bl, bu = [], [], [], [], []

    # Σ_d w[p][d] ≤ 1
    for j in range(mc):
        r = len(bl)
        for i in range(n):
            rows.append(r); cols.append(n_x + i * mc + j); vals.append(1.0)
        bl.append(-np.inf); bu.append(1.0)

    # w[p][d] - x_d ≤ 0
    for i in range(n):
        for j in range(mc):
            r = len(bl)
            rows.append(r); cols.append(n_x + i * mc + j); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-1.0)
            bl.append(-np.inf); bu.append(0.0)

    # y_ab = x_a · x_b
    for (a, b), k in pair_idx.items():
        for u in (a, b):
            r = len(bl)
            rows.append(r); cols.append(n_x + n_w + k); vals.append(1.0)
            rows.append(r); cols.append(u); vals.append(-1.0)
            bl.append(-np.inf); bu.append(0.0)
        r = len(bl)
        rows.append(r); cols.append(n_x + n_w + k); vals.append(-1.0)
        rows.append(r); cols.append(a); vals.append(1.0)
        rows.append(r); cols.append(b); vals.append(1.0)
        bl.append(-np.inf); bu.append(1.0)

    # Σ_d x_d ≥ 1
    r = len(bl)
    for i in range(n):
        rows.append(r); cols.append(i); vals.append(1.0)
    bl.append(1.0); bu.append(np.inf)

    A_mat = coo_matrix((vals, (rows, cols)), shape=(len(bl), num_vars)).tocsr()
    try:
        res = milp(c=cobj, constraints=[LinearConstraint(A_mat, np.array(bl), np.array(bu))],
                   bounds=Bounds(lb, ub), integrality=integrality,
                   options={"time_limit": time_limit, "disp": False})
    except Exception as e:  # pragma: no cover
        return {"success": False, "message": f"MILP error: {e}", "score": 0.0, "diseases": []}
    if not res.success:
        return {"success": False, "message": f"MILP failed: {res.message}",
                "score": 0.0, "diseases": []}

    sel = [i for i in range(n) if res.x[i] > 0.5]
    bs = score_block(data, c, sel, w)
    return {
        "success": True,
        "score": bs.total,
        "milp_obj": float(res.fun),
        "coherence": bs.coherence,
        "coverage": bs.coverage,
        "extra": bs.extra,
        "uncovered": bs.uncovered,
        "relation": bs.relation,
        "diseases": sel,
        "message": f"inner k={len(sel)} score={bs.total:.4f}",
    }


# ======================================================================
# Evaluate a partition and its disease subsets
# ======================================================================

def eval_partition_global(data: ClusterProblemData, clusters, w,
                          top_unifiers: int = 4, time_limit: float = 10.0,
                          _cache=None):
    """Optimize disease assignments for each block of a fixed phenotype partition.
    
    The local objective is additive across blocks. Returns (value, best_clusters),
    where each cluster is a pair of phenotype and disease index lists."""
    blocks = [list(c) for c, _ in clusters]

    def _solve(b):
        key = (tuple(b),)
        if _cache is not None and key in _cache:
            return _cache[key]
        r = best_disease_set(data, b, w, time_limit=time_limit)
        out = (r.get("success") and list(r["diseases"])) or []
        if _cache is not None:
            _cache[key] = out
        return out

    base_sets = [_solve(b) for b in blocks]
    best_clusters = [(b, list(s)) for b, s in zip(blocks, base_sets)]
    best_val = sum(local_global_total(data, [(b, s)], w)["total"] if s else -1e18
                   for b, s in best_clusters)
    return float(best_val), best_clusters


def _first_order_neighbors(p: List[List[int]]) -> List[List[List[int]]]:
    """Enumerate first-order phenotype partitions: single-phenotype moves, merges,
    splits, and cross-block swaps. Empty source blocks are removed."""
    K = len(p)
    out: List[List[List[int]]] = []
    # Move one phenotype across blocks
    for i in range(K):
        for ph in list(p[i]):
            for j in range(K):
                if i == j:
                    continue
                nxt = [list(x) for x in p]
                nxt[i].remove(ph)
                if not nxt[i]:
                    nxt.pop(i)
                    jj = j if j < i else j - 1
                    nxt[jj].append(ph)
                else:
                    nxt[j].append(ph)
                out.append(nxt)
    # Merge block pairs
    for i in range(K):
        for j in range(i + 1, K):
            nxt = [list(x) for k, x in enumerate(p) if k not in (i, j)]
            nxt.append(p[i] + p[j])
            out.append(nxt)
    # Split any size-one or size-two subset from block i into a new block.
    # Important phenotype pairs need not be adjacent in block order,
    # so prefix-only splits may miss such blocks.
    for i in range(K):
        s = len(p[i])
        for x in range(s):
            px = p[i][x]
            nxt = [list(t) for t in p]
            nxt[i].remove(px)
            nxt.append([px])
            out.append(nxt)
            for y in range(x + 1, s):
                py = p[i][y]
                nxt2 = [list(t) for t in p]
                nxt2[i].remove(px)
                nxt2[i].remove(py)
                nxt2.append([px, py])
                out.append(nxt2)
    # Split at a contiguous cut (larger prefix/suffix blocks)
    for i in range(K):
        for cut in range(1, len(p[i])):
            nxt = [list(x) for k, x in enumerate(p) if k != i]
            nxt.append(p[i][:cut])
            nxt.append(p[i][cut:])
            out.append(nxt)
    # Swap one phenotype between two blocks.
    # Some swaps avoid worse intermediate partitions that hill climbing cannot accept.
    for i in range(K):
        for j in range(i + 1, K):
            for a in p[i]:
                for b in p[j]:
                    nxt = [list(x) for x in p]
                    nxt[i].remove(a)
                    nxt[i].append(b)
                    nxt[j].remove(b)
                    nxt[j].append(a)
                    out.append(nxt)
    # Create a new block using one phenotype from each of two blocks (K unchanged).
    # A block {p,q} made from two existing blocks cannot always be formed
    # by one ordinary split, move, or swap, so the neighborhood explicitly
    # includes this operation.
    for i in range(K):
        for j in range(i + 1, K):
            if len(p[i]) < 2 or len(p[j]) < 2:
                continue
            for a in p[i]:
                for b in p[j]:
                    nxt = [list(x) for x in p]
                    nxt[i].remove(a)
                    nxt[j].remove(b)
                    nxt.append([a, b])
                    out.append(nxt)
    return out


def global_optimum(data: ClusterProblemData, w, seed_blocks,
                   iters: int = 250, restarts: int = 6, top_unifiers: int = 4,
                   time_limit: float = 5.0, seed: int = 0, budget: float = 90.0,
                   store_cache=None) -> Dict:
    """Heuristically search phenotype partitions using hill climbing, iterated local
    search, perturbations, and multiple restarts. Each partition receives blockwise
    disease assignments. The time and iteration budgets limit the search; global
    optimality is not guaranteed."""
    import time as _time
    import math as _math
    import random as _random

    t0 = _time.time()
    _cache = {} if store_cache is None else store_cache

    def _init_partition(r):
        if not seed_blocks:
            seed_blocks.append(list(range(data.m)))
        base = [list(map(int, b)) for b in seed_blocks]
        variants = [base]                      # Known partition
        variants.append([list(range(data.m))])  # Merged partition (K=1)
        # Generate random balanced partitions by shuffled round-robin allocation
        # to explore additional regions of the search space.
        for K_ in (3, 4, 5):
            perm2 = list(range(data.m)); r.shuffle(perm2)
            variants.append([perm2[t::K_] for t in range(K_)])
        for _ in range(restarts):
            # Random partitions with one large block and several small blocks
            perm = list(range(data.m)); r.shuffle(perm)
            k_small = r.randint(1, 3)
            if k_small >= data.m:
                continue
            block = list(perm)
            out = []
            for _ in range(k_small if k_small > 1 else 0):
                sz = r.randint(2, max(2, data.m // 4))
                ex = block[:sz]; block = block[sz:]
                out.append(ex)
            out.append(block)
            variants.append(out)
        return variants

    def _eval(part):
        """Evaluate a phenotype partition and return its score and disease assignments."""
        return eval_partition_global(data, [(b, []) for b in part], w,
                                     top_unifiers, time_limit, _cache)

    def _best_improvement(val, cl):
        """Hill-climb deterministically to a first-order local optimum."""
        cur = [list(c) for c, _ in cl]
        cur_val = val
        best_cl = [(list(map(int, c)), list(map(int, s))) for c, s in cl]
        while _time.time() - t0 < budget:
            best_nv, best_nc = cur_val, None
            for nxt in _first_order_neighbors(cur):
                try:
                    nv, nc = _eval(nxt)
                except Exception:
                    continue
                if nv > best_nv + 1e-12:
                    best_nv, best_nc = nv, nc
            if best_nc is None:
                break
            cur_val, best_cl = best_nv, best_nc
            cur = [list(c) for c, _ in best_nc]
        return cur_val, [(list(map(int, c)), list(map(int, s))) for c, s in best_cl]

    def _perturb(cl, r):
        """Perturb a partition to escape a local optimum.
        
        Use one to three first-order steps or create a new block from one phenotype
        taken from each of two existing blocks."""
        part = [list(c) for c, _ in cl]
        if len(part) >= 2 and r.random() < 0.6:
            i, j = r.sample(range(len(part)), 2)
            if part[i] and part[j]:
                p = r.choice(part[i])
                q = r.choice(part[j])
                nxt = [list(x) for x in part]
                nxt[i].remove(p)
                nxt[j].remove(q)
                nxt.append([p, q])
                return nxt
        for _ in range(1 + r.randrange(3)):
            nb = _first_order_neighbors(part)
            if not nb:
                return None
            part = [list(x) for x in r.choice(nb)]
        return part

    rng = _random.Random(seed)
    best_val, best_cl = -1e18, None
    for variant in _init_partition(rng):
        if not variant or sum(len(x) for x in variant) != data.m:
            continue
        cur = [list(x) for x in variant if x]
        try:
            cv, ccl = _eval(cur)
        except Exception:
            continue
        ccl = [(list(map(int, c)), list(map(int, s))) for c, s in ccl]
        cv, ccl = _best_improvement(cv, ccl)
        if cv > best_val:
            best_val, best_cl = cv, ccl

        # ILS: random perturbation, renewed hill climbing, and annealing-based acceptance
        # Initial temperature 2.5 allows temporary score decreases of roughly 2–3.
        # At low temperatures exp(dv/T) is near zero and can trap the search;
        # higher temperature with slow cooling (0.99) helps cross such valleys.
        T = 2.5
        for _ in range(iters):
            if _time.time() - t0 > budget:
                break
            nxt = _perturb(ccl, rng)
            if nxt is None:
                break
            try:
                nv, ncl = _eval(nxt)
            except Exception:
                continue
            nv, ncl = _best_improvement(nv, ncl)
            dv = nv - cv
            if nv >= cv - 1e-12:
                cv, ccl = nv, ncl
            elif dv > -50 and rng.random() < _math.exp(dv / T):
                cv, ccl = nv, ncl
            T = max(T * 0.99, 0.05)
            if cv > best_val:
                best_val, best_cl = cv, ccl
        if _time.time() - t0 > budget:
            break
    best_cl = best_cl or [([], [])]
    return {
        "opt_score": float(best_val),
        "clusters": best_cl,
        "k_opt": len(best_cl),
        "s_sizes": [len(s) for _, s in best_cl],
        "artifacts": True,
    }