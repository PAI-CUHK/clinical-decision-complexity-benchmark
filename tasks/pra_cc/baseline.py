"""PASP baseline solved with CP-SAT using the penalties in the task prompt."""

import json

from .penalty import (
    CP_WEIGHT_SCALE, CP_WEIGHTS, room_preference_units, segment_penalty_components,
)
from .scorer import validate_instance_segments


def _pid_sort_key(pid):
    suffix = pid[1:] if pid.startswith("P") else pid
    return (0, int(suffix)) if suffix.isdigit() else (1, pid)


def _segment_penalty(patient, room):
    """Penalty paid once when entering a room, regardless of stay length."""
    return sum(
        CP_WEIGHTS[name] * count
        for name, count in segment_penalty_components(patient, room).items()
    )


def _daily_room_preference_penalty(patient, room):
    return CP_WEIGHTS["room_preference"] * room_preference_units(patient, room)


def generate_baseline_with_ortools(patients, rooms, max_time_in_seconds=60.0,
                                   num_workers=16):
    """Minimize the prompt's PASP penalty with OR-Tools CP-SAT.

    A room entry starts one assignment segment. Gender, age, and specialism
    penalties are charged once per segment; transfer is charged for every
    subsequent entry. Only room preference is charged for each occupied day.
    Stay endpoints are inclusive, as in the PASP prompt and scorer.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise ImportError("PASP baseline generation requires ortools (pip install ortools)") from exc

    pids = sorted(patients, key=_pid_sort_key)
    rids = list(rooms)
    if not rids:
        raise ValueError("PASP instance has no rooms")

    model = cp_model.CpModel()
    x = {}
    enters = {}
    patients_by_day = {}
    objective = []

    for pid in pids:
        patient = patients[pid]
        first, last = patient["start_day"], patient["end_day"]
        if last < first:
            raise ValueError(f"Invalid stay for {pid}: {first}..{last}")
        entry_vars = []
        for day in range(first, last + 1):
            patients_by_day.setdefault(day, []).append(pid)
            day_vars = []
            for rid in rids:
                assigned = model.NewBoolVar(f"x_{pid}_{rid}_{day}")
                entered = model.NewBoolVar(f"enter_{pid}_{rid}_{day}")
                x[pid, rid, day] = assigned
                enters[pid, rid, day] = entered
                day_vars.append(assigned)
                entry_vars.append(entered)

                if day == first:
                    model.Add(entered == assigned)
                else:
                    previous = x[pid, rid, day - 1]
                    model.Add(entered >= assigned - previous)
                    model.Add(entered <= assigned)
                    model.Add(entered <= 1 - previous)

                fixed_cost = _segment_penalty(patient, rooms[rid])
                daily_cost = _daily_room_preference_penalty(patient, rooms[rid])
                if fixed_cost:
                    objective.append(fixed_cost * entered)
                if daily_cost:
                    objective.append(daily_cost * assigned)
            model.AddExactlyOne(day_vars)

        # The first room entry is admission, not a transfer.
        objective.append(CP_WEIGHTS["room_transfer"] * (sum(entry_vars) - 1))

    for day, day_pids in patients_by_day.items():
        for rid in rids:
            model.Add(sum(x[pid, rid, day] for pid in day_pids) <= rooms[rid]["capacity"])

    model.Minimize(sum(objective))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_in_seconds
    if num_workers is not None:
        solver.parameters.num_workers = num_workers
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": "infeasible" if status == cp_model.INFEASIBLE else "unknown",
            "assignments": {},
            "solve_time_seconds": solver.WallTime(),
        }

    assignments = {}
    for pid in pids:
        first, last = patients[pid]["start_day"], patients[pid]["end_day"]
        segments = []
        for day in range(first, last + 1):
            rid = next(rid for rid in rids if solver.Value(x[pid, rid, day]))
            if segments and segments[-1]["room"] == rid:
                segments[-1]["end_day"] = day
            else:
                segments.append({"start_day": day, "end_day": day, "room": rid})
        assignments[pid] = segments

    objective_scaled = round(solver.ObjectiveValue())
    return {
        "assignments": assignments,
        "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
        "objective_scaled": objective_scaled,
        "best_bound_scaled": round(solver.BestObjectiveBound()),
        "total_penalty": objective_scaled / CP_WEIGHT_SCALE,
        "solve_time_seconds": solver.WallTime(),
    }


def generate_baseline_with_validation(patients, rooms, max_time_in_seconds=60.0,
                                      num_workers=16):
    """Solve and verify that the persisted score matches the PASP scorer."""
    result = generate_baseline_with_ortools(
        patients, rooms, max_time_in_seconds=max_time_in_seconds,
        num_workers=num_workers,
    )
    if result["status"] not in ("optimal", "feasible"):
        raise RuntimeError(f"PASP baseline solver returned {result['status']}")
    feasible, errors, penalty = validate_instance_segments(
        {"prompt": ""}, json.dumps({"assignments": result["assignments"]}),
        patients=patients, rooms=rooms,
    )
    if not feasible or abs(penalty - result["total_penalty"]) > 1e-6:
        raise RuntimeError(
            f"PASP baseline/scorer mismatch: feasible={feasible}, "
            f"CP-SAT={result['total_penalty']}, scorer={penalty}, errors={errors}"
        )
    return {**result, "feasible": True, "errors": errors}


def generate_baseline(patients, rooms):
    """Compatibility wrapper for callers expecting assignments and occupancy."""
    assignments = generate_baseline_with_validation(patients, rooms)["assignments"]
    occupancy = {}
    for pid, segments in assignments.items():
        for segment in segments:
            room_days = occupancy.setdefault(segment["room"], {})
            for day in range(segment["start_day"], segment["end_day"] + 1):
                room_days.setdefault(day, []).append(pid)
    return assignments, occupancy
