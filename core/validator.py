"""Interfaces for validation and evaluation of medical NP problems."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class MissingBaselineError(Exception):
    """Raised when a required baseline is missing."""
    pass


def repair_truncated_json(text: str) -> Optional[str]:
    """Repair incomplete JSON truncated by the model token limit.
    
    Try closing brackets and braces in increasing-length combinations until
    json.loads succeeds. Return the completed JSON string, or None on failure."""
    import itertools as _it
    import json as _json
    if not text or not isinstance(text, str):
        return None
    stripped = text.strip()
    try:
        _json.loads(stripped)
        return stripped
    except Exception:
        pass
    base = stripped
    for k in range(1, 9):
        for tup in _it.product(']}', repeat=k):
            cand = base + ''.join(tup)
            try:
                _json.loads(cand)
                return cand
            except Exception:
                pass
    return None


class Validator(ABC):
    """Base class for problem validators."""
    
    def _ensure_baseline(self, problem):
        """Require a baseline for the problem.
        
        Raises:
            MissingBaselineError: If the baseline is unavailable."""
        if not hasattr(problem, 'baseline') or not problem.baseline:
            raise MissingBaselineError("Problem must have a baseline")
    
    @abstractmethod
    def extract_answer(self, output: str) -> Any:
        """Extract an answer from model output."""
        pass
    
    @abstractmethod
    def validate_solution(self, problem, solution: Any) -> Dict:
        """Validate a solution for the given problem and return a result dictionary."""
        pass
    
    @abstractmethod
    def calculate_baseline(self, problem) -> Any:
        """Compute the reference solution for the given problem."""
        pass
    
    @abstractmethod
    def calculate_score(self, problem, solution: Any) -> float:
        """Compute a numerical score for a solution to the problem."""
        pass
    
    @abstractmethod
    def evaluate_solution(self, problem, solution: Any, baseline=None) -> Dict:
        """Evaluate a solution against an optional baseline and return a result dictionary."""
        pass
