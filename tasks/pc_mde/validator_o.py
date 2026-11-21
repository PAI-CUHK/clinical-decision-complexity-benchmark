"""Validator for the PC-MDE optimization task.

A correct solution partitions observed phenotypes exactly once and assigns
a nonempty subset of candidate diseases to each cluster. Its score must be
at least the heuristic reference score minus tolerance. Scores are recomputed
using objective.py; the reference is rebuilt from instance data by the
baseline solver rather than read from stored generator values.

Partial quality is clip((score - worst) / (reference - worst), 0, 1),
where worst is the lower of the merged and singleton reference scores."""

import json
import re
from typing import Dict, List, Optional, Tuple

from core.validator import Validator, repair_truncated_json
from tasks.pc_mde.baseline import compute_cluster_baseline
from tasks.pc_mde.scorer import extract_clusters_from_output

# Floating-point tolerance for comparing score and reference score
SCORE_TOLERANCE = 1e-6


class DiseaseValidator(Validator):
    """Validator for the PC-MDE optimization task."""

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract_answer(self, output: str):
        """Extract a clusters object from model output.
        
        Try complete JSON in final_answer, repair truncated final_answer JSON,
        recover a clusters object from the full output, wrap a bare clusters array,
        and finally fall back to extract_clusters_from_output."""
        if isinstance(output, dict):
            return output
        if not isinstance(output, str):
            return output

        def try_load(raw: str):
            """Parse JSON, trying truncated JSON repair on failure; return a dictionary or None."""
            if not raw or not raw.strip():
                return None
            s = raw.strip()
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            repaired = repair_truncated_json(s)
            if repaired is not None:
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
            return None

        try:
            # 1) / 2) First extract the final_answer tag content
            fin = re.search(r'<final_answer>\s*([\s\S]*?)\s*</final_answer>',
                            output, re.DOTALL | re.IGNORECASE)
            if fin:
                parsed = try_load(fin.group(1))
                if parsed is not None:
                    return parsed
                # 2b) For an unclosed final_answer tag, take the text through the end
                fin_open = re.search(r'<final_answer>\s*([\s\S]*)$',
                                     output, re.DOTALL | re.IGNORECASE)
                if fin_open:
                    parsed = try_load(fin_open.group(1))
                    if parsed is not None:
                        return parsed

            # 3) Find a clusters object in the full output, including truncated JSON
            obj = re.search(r'\{\s*("clusters"\s*:.*)$', output, re.DOTALL)
            if obj:
                parsed = try_load(obj.group(1))
                if parsed is not None:
                    return parsed

            # 4) Wrap a bare clusters array as a clusters object
            arr = re.search(r'\[\s*\{.*$', output, re.DOTALL)
            if arr:
                repaired = repair_truncated_json(arr.group(0))
                if repaired is not None:
                    try:
                        lst = json.loads(repaired)
                        if isinstance(lst, list):
                            return {"clusters": lst}
                    except Exception:
                        pass

            # 5) Fallback
            clusters = extract_clusters_from_output(output)
            if clusters:
                return {"clusters": clusters}
            return output
        except Exception:
            return output

    # ------------------------------------------------------------------
    # Recompute score
    # ------------------------------------------------------------------

    @staticmethod
    def _rebuild_data(instance_data: Dict):
        """Rebuild ClusterProblemData from instance data using the generator definition."""
        from tasks.pc_mde.objective import (
            build_cluster_problem_data, ScoreWeights,
        )
        weights = instance_data.get("weights") or {}
        w = ScoreWeights(**{k: v for k, v in weights.items()
                            if k in ScoreWeights.__dataclass_fields__})
        return build_cluster_problem_data(
            instance_data.get("phenotypes", []),
            instance_data.get("candidate_diseases", []),
            instance_data.get("disease_phenotype_probs", {}),
            w,
        ), w

    def _parse_clusters(self, data, solution) -> Tuple[Optional[List], List[str]]:
        """Normalize a solution into (phenotype indices, disease indices) pairs.
        
        Returns (clusters, errors)."""
        errors: List[str] = []
        if not isinstance(solution, dict):
            return None, ["Solution must be a JSON object with a 'clusters' field"]

        raw = solution.get("clusters")
        if raw is None:
            return None, ["Missing required field: 'clusters'"]
        if not isinstance(raw, list) or len(raw) == 0:
            return None, ["'clusters' must be a non-empty list"]

        p_index = {p: i for i, p in enumerate(data.phenotypes)}
        d_index = {d: i for i, d in enumerate(data.diseases)}

        clusters = []
        for i, blk in enumerate(raw):
            if not isinstance(blk, dict):
                errors.append(f"cluster {i} must be an object")
                continue
            phenos = blk.get("phenotypes")
            diseases = blk.get("diseases")
            if not isinstance(phenos, list) or not phenos:
                errors.append(f"cluster {i} has no phenotypes")
                continue
            if not isinstance(diseases, list) or not diseases:
                errors.append(f"cluster {i} has no diseases")
                continue

            c_idx, s_idx = [], []
            for p in phenos:
                if p not in p_index:
                    errors.append(f"unknown phenotype '{p}' in cluster {i}")
                else:
                    c_idx.append(p_index[p])
            for d in diseases:
                if d not in d_index:
                    errors.append(f"disease '{d}' in cluster {i} is not a candidate disease")
                else:
                    s_idx.append(d_index[d])
            if c_idx and s_idx:
                clusters.append((c_idx, s_idx))

        if errors:
            return None, errors

        seen: List[int] = []
        for c_idx, _s in clusters:
            seen.extend(c_idx)
        if sorted(seen) != list(range(data.m)):
            dup = sorted({x for x in seen if seen.count(x) > 1})
            if dup:
                errors.append("phenotypes assigned to more than one cluster: "
                              + ", ".join(data.phenotypes[i] for i in dup))
            missing = sorted(set(range(data.m)) - set(seen))
            if missing:
                errors.append("phenotypes missing from the partition: "
                              + ", ".join(data.phenotypes[i] for i in missing))
            return None, errors

        return clusters, errors

    def _score(self, data, w, clusters) -> Tuple[float, List[Dict]]:
        from tasks.pc_mde.objective import (
            score_block, local_global_total,
        )
        # Per-block details are displayed but do not determine the total
        detail = []
        for c_idx, s_idx in clusters:
            bs = score_block(data, c_idx, s_idx, w)
            detail.append({
                "phenotypes": [data.phenotypes[i] for i in c_idx],
                "diseases": [data.diseases[i] for i in s_idx],
                "coherence": round(bs.coherence, 6),
                "coverage": round(bs.coverage, 6),
                "extra": round(bs.extra, 6),
                "uncovered": round(bs.uncovered, 6),
                "relation": round(bs.relation, 6),
                "score": round(bs.total, 6),   # Block local value (without Global)
            })
        # Recompute the total objective once for the complete solution
        res = local_global_total(data, clusters, w)
        total = float(res["total"])
        return total, detail

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _baseline(self, problem):
        """Compute and memoize the validation reference solution."""
        key = id(problem)
        if getattr(self, "_baseline_cache", None) is None:
            self._baseline_cache = {}
        if key not in self._baseline_cache:
            self._baseline_cache[key] = compute_cluster_baseline(problem.instance_data)
        return self._baseline_cache[key]

    def validate_solution(self, problem, solution):
        instance_data = problem.instance_data
        base = self._baseline(problem)
        opt_score = float(base["opt_score"])
        k_opt = int(base["k_opt"])

        data, w = self._rebuild_data(instance_data)
        clusters, errors = self._parse_clusters(data, solution)

        if clusters is None:
            return self._make_result(False, errors, 0, 0.0, None, k_opt, opt_score)

        score, detail = self._score(data, w, clusters)

        scale = max(abs(opt_score), 1.0)
        if opt_score > 0 and score < opt_score - SCORE_TOLERANCE * scale:
            errors.append(
                f"suboptimal score: {score:.6f} < opt_score {opt_score:.6f} "
                f"(gap = {(opt_score - score) / scale * 100:.3f}%)"
            )

        return self._make_result(len(errors) == 0, errors, len(clusters),
                                 score, detail, k_opt, opt_score)

    def _make_result(self, valid, errors, k, score, detail, k_opt, opt_score):
        return {
            "valid": bool(valid),
            "violations": list(errors),
            "details": {
                "k": k,
                "k_opt": k_opt,
                "score": round(score, 6) if score is not None else None,
                "opt_score": round(opt_score, 6),
                "blocks": detail,
            },
        }

    # ------------------------------------------------------------------
    # Baseline and partial score
    # ------------------------------------------------------------------

    def calculate_baseline(self, problem):
        """Return the reference solution searched from instance data; global optimality is not guaranteed."""
        return self._baseline(problem)

    def calculate_score(self, problem, solution):
        """Return clip((score - worst) / (reference - worst), 0, 1)."""
        instance_data = problem.instance_data
        data, w = self._rebuild_data(instance_data)
        clusters, _errors = self._parse_clusters(data, solution)
        if clusters is None:
            return 0.0

        score, _detail = self._score(data, w, clusters)
        base = self._baseline(problem)
        opt_score = float(base["opt_score"])
        if opt_score <= 0:
            return 1.0

        worst = min(float(base["merge_score"]), float(base["singleton_score"]))
        if opt_score - worst <= 1e-12:
            return 1.0 if score >= opt_score - SCORE_TOLERANCE * max(abs(opt_score), 1.0) else 0.0
        return float(min(max((score - worst) / (opt_score - worst), 0.0), 1.0))

    def evaluate_solution(self, problem, solution, baseline=None):
        validation = self.validate_solution(problem, solution)
        score = self.calculate_score(problem, solution)

        result = {
            "valid": validation["valid"],
            "score": round(score, 4),
            "violations": validation["violations"],
            "details": validation["details"],
        }
        if isinstance(baseline, dict) and baseline:
            result["baseline_comparison"] = {
                "opt_score": baseline.get("opt_score"),
                "k_opt": baseline.get("k_opt"),
                "k_model": validation["details"].get("k"),
                "score_model": validation["details"].get("score"),
            }
        return result