#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operating Room Scheduling Scorer

Validates and scores OR scheduling solutions
"""

import json
import re
from collections import defaultdict


def calculate_penalty(cancellation_count, cancellation_time, total_unused_time):
    """Calculate the operating-room scheduling penalty.
    
    Args:
        cancellation_count: Number of canceled surgeries.
        cancellation_time: Total duration of canceled surgeries.
        total_unused_time: Total unused operating-room time.
    
    Returns:
        The penalty as a float."""
    return 1.0 * cancellation_count + 0.1 * cancellation_time + 0.01 * total_unused_time


def parse_or_prompt(prompt: str):
    """
    Parse OR scheduling prompt to extract surgeries and OR-days
    
    Returns:
        (surgeries_dict, or_days_dict)
    """
    surgeries = {}
    or_days = {}
    
    # Parse surgeries
    surgery_pattern = re.compile(
        r'-\s*(S\d+):\s*(\d+)\s*minutes',
        re.IGNORECASE
    )
    
    for match in surgery_pattern.finditer(prompt):
        sid, duration = match.groups()
        surgeries[sid] = {
            "id": sid,
            "duration": int(duration)
        }
    
    # Parse OR-days
    or_day_pattern = re.compile(
        r'-\s*(OR-day\d+):\s*capacity\s*(\d+)\s*minutes',
        re.IGNORECASE
    )
    
    for match in or_day_pattern.finditer(prompt):
        did, capacity = match.groups()
        or_days[did] = {
            "id": did,
            "capacity": int(capacity)
        }
    
    assert surgeries, "No surgeries parsed from prompt"
    assert or_days, "No OR-days parsed from prompt"
    
    return surgeries, or_days


def extract_schedule_from_output(raw_output: str) -> dict:
    """
    Extract schedule from LLM output
    
    Returns:
        dict with "schedule" and optionally "total_unused_time"
    """
    if not isinstance(raw_output, str):
        raise ValueError("raw_output must be a string")
    
    # Extract <final_answer>...</final_answer>
    match = re.search(
        r'<final_answer>\s*([\s\S]*?)\s*</final_answer>',
        raw_output,
        flags=re.IGNORECASE
    )
    
    if not match:
        raise ValueError("Missing <final_answer> block")
    
    json_text = match.group(1).strip()
    
    # Parse JSON
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in final_answer: {e}")
    
    if "schedule" not in data:
        raise ValueError("Missing 'schedule' field in output")
    
    # Optional fields:
    # - "cancelled": list of surgery IDs that are explicitly cancelled
    # - "total_unused_time": reported total unused time (for checking only)
    return data


def validate_or_schedule(benchmark, raw_output):
    """
    Validate OR scheduling solution
    
    Returns:
        (is_feasible, errors_dict, unused_time)
    """
    errors = defaultdict(int)
    is_feasible = True
    
    # Parse prompt or instance data
    if "instance" in benchmark and benchmark["instance"]:
        instance_data = benchmark["instance"]
        
        if "surgeries" in instance_data and "or_days" in instance_data:
            # Use instance data directly
            surgeries_list = instance_data["surgeries"]
            or_days_list = instance_data["or_days"]
            
            surgeries = {s["id"]: s for s in surgeries_list}
            or_days = {o["id"]: o for o in or_days_list}
        else:
            # Instance data exists but missing required fields
            missing_fields = []
            if "surgeries" not in instance_data:
                missing_fields.append("surgeries")
            if "or_days" not in instance_data:
                missing_fields.append("or_days")
            raise ValueError(f"Instance data missing required fields: {', '.join(missing_fields)}")
    elif "prompt" in benchmark and benchmark["prompt"]:
        # Fallback to parsing from prompt
        surgeries, or_days = parse_or_prompt(benchmark["prompt"])
    else:
        raise ValueError("No instance data or prompt provided for validation")
    
    # Extract schedule
    try:
        solution = extract_schedule_from_output(raw_output)
        schedule = solution["schedule"]
    except Exception as e:
        errors["parse_error"] += 1
        return False, dict(errors), None
    
    # Track which surgeries are assigned
    assigned_surgeries = set()
    # Initialize usage for all OR-days so empty days count toward unused time
    or_day_usage = {
        did: {"used": 0, "capacity": info["capacity"], "unused": info["capacity"]}
        for did, info in or_days.items()
    }
    
    # Validate each OR-day assignment
    for or_day_id, surgery_list in schedule.items():
        # Check if OR-day exists
        if or_day_id not in or_days:
            errors["invalid_or_day"] += 1
            is_feasible = False
            continue
        
        or_day = or_days[or_day_id]
        total_duration = 0
        
        for surgery_id in surgery_list:
            # Check if surgery exists
            if surgery_id not in surgeries:
                errors["invalid_surgery"] += 1
                is_feasible = False
                continue
            
            # Check for duplicate assignment
            if surgery_id in assigned_surgeries:
                errors["duplicate_assignment"] += 1
                is_feasible = False
            else:
                assigned_surgeries.add(surgery_id)
            
            # Add to total duration
            total_duration += surgeries[surgery_id]["duration"]
        
        # Check capacity constraint
        if total_duration > or_day["capacity"]:
            errors["capacity_exceeded"] += 1
            is_feasible = False
            over_capacity = total_duration - or_day["capacity"]
            errors["total_over_capacity"] += over_capacity
        
        or_day_usage[or_day_id] = {
            "used": total_duration,
            "capacity": or_day["capacity"],
            "unused": or_day["capacity"] - total_duration
        }
    
    # Determine cancelled surgeries:
    # - If solution provides explicit "cancelled" list, use it;
    # - Otherwise, any surgery not assigned is treated as cancelled.
    explicit_cancelled = set(solution.get("cancelled", []))
    all_surgeries = set(surgeries.keys())
    inferred_cancelled = all_surgeries - assigned_surgeries
    cancelled = explicit_cancelled or inferred_cancelled
    
    # Calculate unused time across all OR-days; empty days contribute full capacity
    total_unused_time = sum(usage["unused"] for usage in or_day_usage.values())
    
    # Verify reported unused time (if provided)
    if "total_unused_time" in solution:
        reported_unused = solution["total_unused_time"]
        if abs(reported_unused - total_unused_time) > 0.1:
            errors["incorrect_unused_time_calculation"] += 1
            # Note: This is not a feasibility issue, just an accuracy issue
    
    # Lexicographic objective components:
    cancellations = len(cancelled)
    cancellation_time = sum(
        surgeries[sid]["duration"] for sid in cancelled if sid in surgeries
    )
    idle_time = total_unused_time
    
    # Store objective components in errors dict for downstream consumers
    errors["cancellations"] = cancellations
    errors["cancellation_time"] = cancellation_time
    errors["idle_time"] = idle_time
    
    return is_feasible, dict(errors), total_unused_time


def score_or_results(results_file, output_file):
    """
    Score OR scheduling results from a results file
    
    Args:
        results_file: Path to JSON file with test results
        output_file: Path to save scoring results
    """
    with open(results_file, "r") as f:
        results = json.load(f)
    
    scored_results = []
    
    for result in results:
        if result.get("status") != "success":
            scored_results.append({
                "instance_id": result.get("instance_id"),
                "status": "test_failed",
                "error": result.get("error")
            })
            continue
        
        # Load benchmark
        with open(result["instance_path"], "r") as f:
            benchmark = json.load(f)
        
        # Validate
        try:
            is_feasible, errors, unused_time = validate_or_schedule(
                benchmark, result["raw_output"]
            )
            
            scored_results.append({
                "instance_id": result["instance_id"],
                "status": "evaluated",
                "feasible": is_feasible,
                "errors": errors,
                "unused_time": unused_time
            })
        except Exception as e:
            scored_results.append({
                "instance_id": result["instance_id"],
                "status": "validation_failed",
                "error": str(e)
            })
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(scored_results, f, indent=2)
    
    # Print summary
    total = len(scored_results)
    feasible = sum(1 for r in scored_results if r.get("feasible", False))
    
    print(f"\n{'='*60}")
    print("OR Scheduling Scoring Summary")
    print(f"{'='*60}")
    print(f"Total instances: {total}")
    print(f"Feasible solutions: {feasible}")
    print(f"Accuracy: {feasible/total:.2%}" if total > 0 else "N/A")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser("OR Scheduling Scorer")
    parser.add_argument("--results", required=True, help="Results JSON file")
    parser.add_argument("--output", required=True, help="Output file")
    
    args = parser.parse_args()
    
    score_or_results(args.results, args.output)
