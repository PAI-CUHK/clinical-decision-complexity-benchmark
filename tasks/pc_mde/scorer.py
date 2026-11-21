#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scoring utilities for multi-disease explanation."""

import json
import re
from typing import List, Dict, Optional


def extract_diseases_from_output(output: str) -> List[str]:
    """Extract the disease list from model output.
    
    Args:
        output: Model output text.
    
    Returns:
        The extracted list of diseases."""
    # First extract from the root tag
    root_match = re.search(r'<root>\s*([\s\S]*?)\s*</root>', output, re.DOTALL | re.IGNORECASE)
    if root_match:
        output = root_match.group(1)
    
    # Try to extract from the final_answer tag
    match = re.search(r'<final_answer>\s*(\[.*?\])\s*</final_answer>', output, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            diseases = json.loads(match.group(1))
            if isinstance(diseases, list):
                return diseases
        except:
            pass
    
    # Try to extract a list directly
    match = re.search(r'\[(.*?)\]', output, re.DOTALL)
    if match:
        try:
            diseases = json.loads('[' + match.group(1) + ']')
            if isinstance(diseases, list):
                return diseases
        except:
            pass
    
    return []


def extract_clusters_from_output(output: str) -> List[Dict]:
    """Extract phenotype and disease clusters from model output.
    
    Parse a clusters object or a bare list within final_answer; return an empty
    list if parsing fails."""
    if not isinstance(output, str):
        return []

    for text in (output, _extract_tag(output, "final_answer") or ""):
        if not text:
            continue
        for pattern in (r'\{\s*"clusters"\s*:\s*(\[[\s\S]*?\])\s*\}', r'\[\s*\{[\s\S]*\}\s*\]'):
            match = re.search(pattern, text)
            if not match:
                continue
            try:
                payload = json.loads(match.group(1) if match.re.groups else match.group(0))
            except Exception:
                continue
            if isinstance(payload, list) and payload and all(
                    isinstance(b, dict) and "phenotypes" in b and "diseases" in b
                    for b in payload):
                return payload
    return []


def _extract_tag(output: str, tag: str) -> Optional[str]:
    match = re.search(rf'<{tag}>\s*([\s\S]*?)\s*</{tag}>', output, re.DOTALL | re.IGNORECASE)
    return match.group(1) if match else None


def extract_yes_no_answer(output: str) -> Optional[str]:
    """Extract Yes or No from the final_answer tag.
    
    Return None if the tag is missing or does not contain a valid Yes/No answer.
    This can occur when the model output is truncated or incorrectly formatted."""
    if isinstance(output, bool):
        return "Yes" if output else "No"

    if not isinstance(output, str):
        return None

    # First extract from the root tag
    root_match = re.search(r'<root>\s*([\s\S]*?)\s*</root>', output, re.DOTALL | re.IGNORECASE)
    if root_match:
        output = root_match.group(1)

    # Extract only from the final_answer tag
    match = re.search(r'<final_answer>\s*(.*?)\s*</final_answer>', output, re.DOTALL | re.IGNORECASE)
    if match:
        answer = match.group(1).strip().lower()
        if answer in ['yes', '是']:
            return "Yes"
        elif answer in ['no', '否']:
            return "No"

    # Return None if no valid final_answer tag is found
    return None


def validate_disease_diagnosis(benchmark: Dict, output: str) -> tuple:
    """Validate a disease diagnosis result.
    
    Args:
        benchmark: Benchmark instance.
        output: Model output.
    
    Returns:
        A tuple (is_valid, errors, score)."""
    from tasks.pc_mde.baseline import calculate_score
    
    # Extract the diseases selected by the model
    selected_diseases = extract_diseases_from_output(output)
    
    # First obtain benchmark data from the instance
    if "instance" in benchmark:
        symptoms = benchmark["instance"].get("symptoms", [])
        candidate_diseases = benchmark["instance"].get("candidate_diseases", [])
        disease_symptom_probs = benchmark["instance"].get("disease_symptom_probs", {})
    else:
        # Fall back to obtaining data from the benchmark
        symptoms = benchmark.get("symptoms", [])
        candidate_diseases = benchmark.get("candidate_diseases", [])
        disease_symptom_probs = benchmark.get("disease_symptom_probs", {})
    
    # Check whether the diseases are in the candidate list
    for disease in selected_diseases:
        if disease not in candidate_diseases:
            return False, {"error": f"疾病{disease}不在候选列表中"}, 0.0
    
    # Calculate the score
    score = calculate_score(symptoms, selected_diseases, disease_symptom_probs)
    
    return True, {}, score
