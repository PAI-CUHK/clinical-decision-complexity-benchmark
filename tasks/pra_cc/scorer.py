#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import re
from collections import defaultdict
from .penalty import (
    CP_WEIGHT_SCALE, CP_WEIGHTS, room_preference_units, segment_penalty_components,
)

# =============================
# Penalty Weights (Soft Constraints Only)
# =============================
WEIGHTS = {
    "gender_policy": 5.0,
    "age_policy": 10.0,
    "specialism": 5.0,
    "room_preference": 0.8,
    "room_transfer": 11.0,
}

ROOM_PREF_MAP = {
    "single-bed": 1,
    "double-bed": 2,
    "multi-bed": 4
}

# =============================
# Prompt Parsing
# =============================

def parse_prompt(prompt: str):
    patients = {}
    rooms = {}

    # -----------------------------
    # Parse patients
    # -----------------------------
    patient_pattern = re.compile(
        r"-?\s*Patient\s+(P\d+):,\s*"
        r"(\d+)-year-old\s+([MF])\s+patient,\s*"
        r"hospital stay from day\s+(\d+)\s+to\s+day\s+(\d+),\s*"
        r"requires treatment in specialism_plan\s+([A-Za-z ]+),\s*"
        r"prefers a\s+([a-z\-]+)\s+room",
        re.IGNORECASE
    )

    for m in patient_pattern.finditer(prompt):
        pid, age, gender, start, end, spec, pref = m.groups()
        start = int(start)
        end = int(end)
        stay_len = end - start + 1
        patients[pid] = {
            "age": int(age),
            "gender": gender,
            "start_day": start,
            "end_day": end,
            "specialism_plan": [(spec.strip(), stay_len)],
            "room_pref": ROOM_PREF_MAP.get(pref.lower(), 0)
        }

    # -----------------------------
    # Parse rooms
    # -----------------------------
    spec_pattern = re.compile(r"Specialism\s+([A-Za-z ]+)\s*\(priority\s+(\d+)\)", re.IGNORECASE)

    for line in prompt.splitlines():
        line = line.strip()
        if not line.startswith("- Room"):
            continue

        # Match the main fields without relying on comma or spacing details
        room_match = re.match(
            r"- Room (R\d+):, capacity (\d+) beds, gender policy (\w+),.*supports medical specialisms (.+?)(?:\.|$)",
            line,
            re.IGNORECASE
        )

        if not room_match:
            print("Failed to parse room line:", line)
            continue

        rid, cap, gender, spec_blob = room_match.groups()

        # Parse age_policy
        age_policy = "any"
        if "elderly" in line.lower():
            age_policy = "elderly"
        elif "adult" in line.lower():
            age_policy = "adult"
        elif "pediatric" in line.lower() or "child" in line.lower():
            age_policy = "pediatric"

        # Parse specialisms
        spec_map = {}
        for sm in spec_pattern.finditer(spec_blob):
            s, p = sm.groups()
            spec_map[s.strip()] = int(p)

        rooms[rid] = {
            "capacity": int(cap),
            "gender_policy": gender.upper(),
            "specialisms": spec_map,
            "age_policy": age_policy
        }

    assert patients, "No patients parsed — prompt format mismatch"
    assert rooms, "No rooms parsed — prompt format mismatch"

    return patients, rooms



# =============================
# Core Validator
# =============================



def validate_instance_segments(benchmark, raw_output, patients=None, rooms=None):
    """Validate patient allocation segments against hard constraints and penalties.
    
    Hard constraints require every patient to have a room each day, use only
    listed patients and rooms, have continuous assignments, and respect capacity.
    Any hard violation makes the solution infeasible without computing penalties.
    
    Soft penalties cover gender and age policy, specialty, room preference, and
    room transfers."""
    error_counts = defaultdict(int)
    total_penalty = 0.0
    feasible = True

    # ---------- Parse JSON ----------
    try:
        assignment_json = json.loads(raw_output)
    except Exception:
        error_counts["format_error"] += 1
        return False, dict(error_counts), 0.0

    if "assignments" not in assignment_json:
        error_counts["format_error"] += 1
        return False, dict(error_counts), 0.0

    assignments = assignment_json["assignments"]

    # ---------- Get patients and rooms ----------
    if patients is not None and rooms is not None:
        pass
    else:
        patients, rooms = parse_prompt(benchmark["prompt"])

    # ==================================================
    # Phase 1: check hard constraints and return early on failure
    # ==================================================

    # ---------- Hard constraint: assign every patient ----------
    if not assignments:
        error_counts["no_assignments"] += 1
        return False, dict(error_counts), 0.0

    # ---------- Occupancy & segment tracking ----------
    occupancy = defaultdict(lambda: defaultdict(list))  # room -> day -> [pid]
    valid_segments = defaultdict(list)

    for pid, segments in assignments.items():
        # Hard constraint 2: only listed patients are allowed
        if pid not in patients:
            error_counts["invalid_patient"] += 1
            return False, dict(error_counts), 0.0

        p = patients[pid]
        covered_days = set()

        segments = sorted(segments, key=lambda s: s["start_day"])
        prev_end = None

        for seg in segments:
            rid = seg.get("room")
            s = seg.get("start_day")
            e = seg.get("end_day")

            # Hard constraint 3: assignments must be continuous
            if prev_end is not None and s != prev_end + 1:
                error_counts["non_contiguous_segment"] += 1
                return False, dict(error_counts), 0.0

            # Hard constraint 3: only listed rooms are allowed
            if rid not in rooms:
                error_counts["invalid_room"] += 1
                return False, dict(error_counts), 0.0

            for d in range(s, e + 1):
                occupancy[rid][d].append(pid)
                covered_days.add(d)

            valid_segments[pid].append(seg)
            prev_end = e

        # Hard constraint 1: every day must be covered
        expected_days = set(range(p["start_day"], p["end_day"] + 1))
        if covered_days != expected_days:
            error_counts["coverage_days"] = len(expected_days ^ covered_days)
            return False, dict(error_counts), 0.0

    # ---------- Check for unassigned patients ----------
    assigned_patients = set(assignments.keys())
    all_patients = set(patients.keys())
    missing_patients = all_patients - assigned_patients
    if missing_patients:
        error_counts["missing_patients"] = len(missing_patients)
        return False, dict(error_counts), 0.0

    # ---------- Hard constraint 4: check capacity ----------
    for rid, days in occupancy.items():
        room = rooms[rid]
        for d, plist in days.items():
            if len(plist) > room["capacity"]:
                error_counts["capacity"] += len(plist) - room["capacity"]
                return False, dict(error_counts), 0.0

    # ==================================================
    # Phase 2: Soft Constraint Penalty Calculation
    # ==================================================

    # Count room preferences per patient-day, other room policies per continuous segment, and transfers per occurrence.
    for pid, segments in valid_segments.items():
        p = patients[pid]
        prev_room = None

        for seg in segments:
            room = rooms[seg["room"]]
            length = seg["end_day"] - seg["start_day"] + 1

            # Penalize gender, age, and specialty conflicts once per continuous segment.
            for name, count in segment_penalty_components(p, room).items():
                if count:
                    total_penalty += CP_WEIGHTS[name] / CP_WEIGHT_SCALE
                    error_counts[name] += count

            # Room preference
            pref_units = room_preference_units(p, room) * length
            if pref_units:
                total_penalty += CP_WEIGHTS["room_preference"] * pref_units / CP_WEIGHT_SCALE
                error_counts["room_preference"] += pref_units

            # Room transfers
            if prev_room is not None and prev_room != seg["room"]:
                total_penalty += CP_WEIGHTS["room_transfer"] / CP_WEIGHT_SCALE
                error_counts["room_transfer"] += 1
            prev_room = seg["room"]

    return feasible, dict(error_counts), round(total_penalty, 1)



def extract_assignments_from_raw_output(raw_output: str) -> str:
    """Extract JSON from the model final_answer tag.
    
    Returns a dictionary that must contain assignments."""

    if not isinstance(raw_output, str):
        raise ValueError("raw_output must be a string")

    # 1. Extract <final_answer>...</final_answer>
    match = re.search(
        r'<final_answer>\s*([\s\S]*?)\s*</final_answer>',
        raw_output,
        flags=re.IGNORECASE
    )
    if not match:
        raise ValueError("Missing <final_answer> block")

    json_text = match.group(1).strip()

    return json_text



# =============================
# Main
# =============================
def main():
    parser = argparse.ArgumentParser("PASP Official Scorer")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.predictions) as f:
        instances = json.load(f)

    results = []

    for inst in instances:
        with open(inst["benchmark_file"]) as bf:
            benchmark = json.load(bf)

        parsed_output = extract_assignments_from_raw_output(inst["raw_output"])

        feasible, errors, penalty = validate_instance_segments(
            benchmark, parsed_output
        )

        results.append({
            "instance_id": inst["instance_id"],
            "feasible": feasible,
            "errors": dict(errors),
            "penalty": penalty
        })

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Scoring finished: {args.output}")


if __name__ == "__main__":
    main()
