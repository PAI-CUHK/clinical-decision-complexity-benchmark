#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HPO semantic similarity s(p, h) in [0, 1].

The disease diagnosis score uses Lin similarity:
    s(p, h) = 2 * IC(MICA(p, h)) / (IC(p) + IC(h))
MICA is the common ancestor with maximum information content. Disease
annotations in phenotype.hpoa propagate to ancestors, with
    IC(t) = -log(freq(t) / freq(root))
where freq(t) counts diseases annotated to t or its descendants.

Primary names and synonyms from hp.obo are indexed case-insensitively.
Unknown terms fall back to exact string equality (1.0) or 0.0.

Example:
    sim = get_hpo_similarity()
    sim.sim("Developmental delay", "Intellectual disability")"""

import os
import math
import pickle
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OBO_PATH = os.path.join(_BASE_DIR, "data", "hp.obo")
DEFAULT_HPOA_PATH = os.path.join(_BASE_DIR, "data", "phenotype.hpoa")
DEFAULT_CACHE_PATH = os.path.join(_BASE_DIR, "data", "processed", "hpo_ic_cache.pkl")

ROOT_ID = "HP:0000001"


def _norm(name: str) -> str:
    return " ".join(str(name).strip().lower().split())


class HPOSimilarity:
    """Lin semantic similarity using the hp.obo DAG and phenotype.hpoa annotations."""

    def __init__(self, obo_path: str = None, hpoa_path: str = None,
                 cache_path: str = None, use_cache: bool = True):
        self.obo_path = obo_path or DEFAULT_OBO_PATH
        self.hpoa_path = hpoa_path or DEFAULT_HPOA_PATH
        self.cache_path = cache_path or DEFAULT_CACHE_PATH

        cached = self._load_cache() if use_cache else None
        if cached is not None:
            (self._ic, self._ancestors, self._name_to_id) = cached
        else:
            self._ic, self._ancestors, self._name_to_id = self._build()
            if use_cache:
                self._save_cache()

        self._pair_cache: Dict[Tuple[str, str], float] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self):
        terms = self._parse_obo()
        name_to_id = self._build_name_index(terms)
        ancestors = self._build_ancestors(terms)
        ic = self._compute_ic(ancestors)
        return ic, ancestors, name_to_id

    def _parse_obo(self) -> Dict[str, Dict]:
        terms: Dict[str, Dict] = {}
        cur: Optional[Dict] = None
        with open(self.obo_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("["):
                    if cur is not None and cur.get("id"):
                        terms[cur["id"]] = cur
                    cur = {"name": None, "parents": [], "synonyms": [],
                           "obsolete": False} if line.strip() == "[Term]" else None
                    continue
                if cur is None:
                    continue
                if line.startswith("id: "):
                    cur["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    cur["name"] = line[6:].strip()
                elif line.startswith("is_a: "):
                    parent = line[6:].split("!", 1)[0].strip()
                    if parent:
                        cur["parents"].append(parent)
                elif line.startswith("alt_id: "):
                    cur.setdefault("alts", []).append(line[8:].strip())
                elif line.startswith("synonym: "):
                    txt = line[9:].strip()
                    if txt.startswith('"'):
                        end = txt.find('"', 1)
                        if end > 0:
                            cur["synonyms"].append(txt[1:end])
                elif line.startswith("is_obsolete: true"):
                    cur["obsolete"] = True
        if cur is not None and cur.get("id"):
            terms[cur["id"]] = cur
        return terms

    @staticmethod
    def _build_name_index(terms: Dict[str, Dict]) -> Dict[str, str]:
        name_to_id: Dict[str, str] = {}
        # Prefer primary names
        for tid, t in terms.items():
            if t.get("obsolete") or not t.get("name"):
                continue
            name_to_id.setdefault(_norm(t["name"]), tid)
        # Use synonyms only if no primary name is registered
        for tid, t in terms.items():
            if t.get("obsolete"):
                continue
            for syn in t.get("synonyms", []):
                name_to_id.setdefault(_norm(syn), tid)
        return name_to_id

    @staticmethod
    def _build_ancestors(terms: Dict[str, Dict]) -> Dict[str, Set[str]]:
        """Compute ancestor closures including each term itself, using memoized DFS."""
        ancestors: Dict[str, Set[str]] = {}

        def dfs(tid: str) -> Set[str]:
            if tid in ancestors:
                return ancestors[tid]
            ancestors[tid] = {tid}  # Mark visited before recursion to prevent cycles
            acc = {tid}
            for p in terms.get(tid, {}).get("parents", []):
                acc |= dfs(p)
            ancestors[tid] = acc
            return acc

        for tid in terms:
            dfs(tid)
        return ancestors

    def _compute_ic(self, ancestors: Dict[str, Set[str]]) -> Dict[str, float]:
        disease_annotations: Dict[str, Set[str]] = defaultdict(set)
        with open(self.hpoa_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                db_id, hpo_id = parts[0].strip(), parts[3].strip()
                if not db_id or not hpo_id:
                    continue
                disease_annotations[db_id].add(hpo_id)

        freq: Dict[str, int] = defaultdict(int)
        for hpo_ids in disease_annotations.values():
            closed: Set[str] = set()
            for hid in hpo_ids:
                closed |= ancestors.get(hid, {hid})
            for t in closed:
                freq[t] += 1

        total = float(freq.get(ROOT_ID, 0)) or float(len(disease_annotations)) or 1.0
        ic: Dict[str, float] = {}
        for tid in ancestors:
            c = freq.get(tid, 0)
            ic[tid] = -math.log(max(c, 1e-9) / total)
        return ic

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _load_cache(self):
        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("version") == 1 and data.get("hpoa_size") == os.path.getsize(self.hpoa_path):
                return data["ic"], data["ancestors"], data["name_to_id"]
        except Exception:
            return None
        return None

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "version": 1,
                    "hpoa_size": os.path.getsize(self.hpoa_path),
                    "ic": self._ic,
                    "ancestors": self._ancestors,
                    "name_to_id": self._name_to_id,
                }, f, protocol=4)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def term_id(self, name: str) -> Optional[str]:
        return self._name_to_id.get(_norm(name))

    def ic(self, term_id: str) -> float:
        return self._ic.get(term_id, 0.0)

    def sim(self, name_a: str, name_b: str) -> float:
        """Return Lin similarity in [0, 1] between two phenotype names."""
        ka, kb = _norm(name_a), _norm(name_b)
        if ka == kb:
            return 1.0
        key = (ka, kb) if ka <= kb else (kb, ka)
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached

        ia = self._name_to_id.get(ka)
        ib = self._name_to_id.get(kb)
        value = 0.0
        if ia is not None and ib is not None:
            ca, cb = self._ancestors.get(ia, set()), self._ancestors.get(ib, set())
            shared = ca & cb
            if shared:
                mica = max(shared, key=lambda t: self._ic.get(t, 0.0))
                ic_m = self._ic.get(mica, 0.0)
                denom = self._ic.get(ia, 0.0) + self._ic.get(ib, 0.0)
                if denom > 1e-12:
                    value = 2.0 * ic_m / denom
                else:
                    value = 1.0 if ia == ib else 0.0

        value = float(min(max(value, 0.0), 1.0))
        self._pair_cache[key] = value
        return value

    def sim_matrix(self, names_a: List[str], names_b: List[str]):
        """Return a NumPy similarity matrix of shape (len(a), len(b))."""
        import numpy as np
        out = np.zeros((len(names_a), len(names_b)))
        for i, a in enumerate(names_a):
            for j, b in enumerate(names_b):
                out[i, j] = self.sim(a, b)
        return out


_INSTANCE: Optional[HPOSimilarity] = None


def get_hpo_similarity() -> HPOSimilarity:
    """Return the process-wide singleton."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = HPOSimilarity()
    return _INSTANCE


if __name__ == "__main__":
    sim = get_hpo_similarity()
    print("terms indexed:", len(sim._name_to_id))
    print("IC(root) =", sim.ic(ROOT_ID))
    pairs = [
        ("Developmental delay", "Intellectual disability"),
        ("Developmental delay", "Global developmental delay"),
        ("Developmental delay", "Cardiomyopathy"),
        ("Intellectual disability", "Seizure"),
        ("Seizure", "Cardiomyopathy"),
        ("Hypotonia", "Generalized hypotonia"),
        ("Short stature", "Hypertelorism"),
    ]
    for a, b in pairs:
        print(f"{sim.sim(a, b):.4f}  {a}  <->  {b}")