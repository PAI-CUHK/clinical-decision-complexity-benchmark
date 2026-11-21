#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Baseline solvers for operating-room scheduling."""

import re
from tasks.scs.scorer import calculate_penalty


def best_fit_decreasing(surgeries, or_days):
    """Schedule surgeries using best-fit decreasing.
    
    Returns (schedule, cancelled, total_unused_time)."""
    # Sort surgeries by descending duration
    sorted_surgeries = sorted(surgeries, key=lambda x: x['duration'], reverse=True)
    
    # Initialize OR-day capacity
    or_day_capacities = {or_day['id']: or_day['capacity'] for or_day in or_days}
    schedule = {or_day['id']: [] for or_day in or_days}
    cancelled = []
    
    for surgery in sorted_surgeries:
        # Find the OR-day with most remaining capacity that fits the surgery
        best_or_day = None
        max_remaining = -1
        
        for or_day_id, capacity in or_day_capacities.items():
            if capacity >= surgery['duration'] and capacity > max_remaining:
                best_or_day = or_day_id
                max_remaining = capacity
        
        if best_or_day:
            schedule[best_or_day].append(surgery['id'])
            or_day_capacities[best_or_day] -= surgery['duration']
        else:
            cancelled.append(surgery['id'])
    
    # Calculate total unused time
    total_unused_time = sum(or_day_capacities.values())
    
    return schedule, cancelled, total_unused_time


def best_baseline_solver(surgeries, or_days, max_iter=50):
    # ---------- Preprocessing ----------
    duration_map = {s['id']: s['duration'] for s in surgeries}
    capacity_map = {d['id']: d['capacity'] for d in or_days}

    def compute_unused(schedule):
        total = 0
        for d_id, s_ids in schedule.items():
            used = sum(duration_map[s] for s in s_ids)
            total += capacity_map[d_id] - used
        return total

    def evaluate(schedule, cancelled):
        total_unused = compute_unused(schedule)
        cancellation_count = len(cancelled)
        cancellation_time = sum(duration_map[s] for s in cancelled)

        penalty = calculate_penalty(
            cancellation_count,
            cancellation_time,
            total_unused
        )
        return penalty, cancellation_count, cancellation_time, total_unused

    # ---------- Step 1: initial solution (best-fit decreasing) ----------
    schedule, cancelled, _ = best_fit_decreasing(surgeries, or_days)

    current_penalty, _, _, _ = evaluate(schedule, cancelled)

    # ---------- Step 2: local search ----------
    for _ in range(max_iter):
        improved = False

        # ========= 1. INSERT (insert a canceled surgery) =========
        for s_id in cancelled[:]:
            s_dur = duration_map[s_id]

            for d_id in schedule:
                used = sum(duration_map[s] for s in schedule[d_id])
                cap = capacity_map[d_id]

                if used + s_dur <= cap:
                    new_schedule = {k: v[:] for k, v in schedule.items()}
                    new_cancelled = cancelled[:]

                    new_schedule[d_id].append(s_id)
                    new_cancelled.remove(s_id)

                    new_penalty, *_ = evaluate(new_schedule, new_cancelled)

                    if new_penalty < current_penalty:
                        schedule, cancelled = new_schedule, new_cancelled
                        current_penalty = new_penalty
                        improved = True
                        break
            if improved:
                break

        if improved:
            continue

        # ========= 2. SWAP (replace a shorter surgery with a longer one) =========
        for s_id in cancelled[:]:
            s_dur = duration_map[s_id]

            for d_id in schedule:
                for c_id in schedule[d_id]:
                    c_dur = duration_map[c_id]

                    used = sum(duration_map[s] for s in schedule[d_id])
                    cap = capacity_map[d_id]

                    if used - c_dur + s_dur <= cap:
                        new_schedule = {k: v[:] for k, v in schedule.items()}
                        new_cancelled = cancelled[:]

                        new_schedule[d_id].remove(c_id)
                        new_schedule[d_id].append(s_id)

                        new_cancelled.remove(s_id)
                        new_cancelled.append(c_id)

                        new_penalty, *_ = evaluate(new_schedule, new_cancelled)

                        if new_penalty < current_penalty:
                            schedule, cancelled = new_schedule, new_cancelled
                            current_penalty = new_penalty
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if improved:
            continue

        # ========= 3. MOVE (move between OR-days) =========
        for d1 in schedule:
            for s_id in schedule[d1]:
                s_dur = duration_map[s_id]

                for d2 in schedule:
                    if d1 == d2:
                        continue

                    used_d2 = sum(duration_map[s] for s in schedule[d2])
                    cap_d2 = capacity_map[d2]

                    if used_d2 + s_dur <= cap_d2:
                        new_schedule = {k: v[:] for k, v in schedule.items()}
                        new_schedule[d1].remove(s_id)
                        new_schedule[d2].append(s_id)

                        new_penalty, *_ = evaluate(new_schedule, cancelled)

                        if new_penalty < current_penalty:
                            schedule = new_schedule
                            current_penalty = new_penalty
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

        if not improved:
            break  # Stop early after convergence

    # ---------- Final result ----------
    final_penalty, c_count, c_time, total_unused = evaluate(schedule, cancelled)

    return {
        "schedule": schedule,
        "cancelled": cancelled,
        "total_unused_time": total_unused,
        "cancellation_count": c_count,
        "cancellation_time": c_time,
        "penalty": final_penalty,
        "method": "strong_baseline_penalty_local_search"
    }

def generate_baseline(prompt):
    """Generate an operating-room scheduling baseline from the model prompt."""
    # Extract surgery and OR-day information from the prompt
    surgeries = []
    or_days = []
    
    # Extract OR-days
    or_day_pattern = r"- OR-day(\d+): capacity (\d+) minutes"
    or_day_matches = re.findall(or_day_pattern, prompt)
    for match in or_day_matches:
        or_days.append({
            'id': f'OR-day{match[0]}',
            'capacity': int(match[1])
        })
    
    # Extract surgeries
    surgery_pattern = r"- S(\d+): (\d+) minutes"
    surgery_matches = re.findall(surgery_pattern, prompt)
    for match in surgery_matches:
        surgeries.append({
            'id': f'S{match[0]}',
            'duration': int(match[1])
        })
    
    # Use best-fit decreasing
    result = best_baseline_solver(surgeries, or_days)
    
    return result


def generate_baseline_with_ortools(instance_data):
    """Generate an operating-room scheduling reference using OR-Tools CP-SAT.
    
    Consider capacity and turnover, soft specialty mismatch, urgency-weighted
    cancellations, and total unused time. instance_data supplies surgeries,
    or_days, and optional turnover settings. Returns a baseline and penalty."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        raise ImportError("OR-Tools is not installed. Please install ortools to generate optimal baseline solutions.")
    
    surgeries = instance_data.get('surgeries', [])
    or_days = instance_data.get('or_days', [])
    turnover_config = instance_data.get('turnover', {'same_specialty': 10, 'different_specialty': 20})
    penalties = instance_data.get('penalties', {})
    cancellation_penalties = penalties.get('cancellation', {})
    
    # Read penalty settings with defaults matching the instance generator
    emergency_penalty_value = cancellation_penalties.get('emergency', 20)
    urgent_penalty_value = cancellation_penalties.get('urgent', 8)
    elective_penalty_value = cancellation_penalties.get('elective', 1)
    specialty_mismatch_penalty_value = penalties.get('specialty_mismatch', 3)
    idle_time_coefficient = penalties.get('idle_time', 0.01)
    
    if not surgeries or not or_days:
        return {
            "type": "or_scheduling_baseline",
            "status": "error",
            "error": "Missing surgeries or or_days"
        }
    
    num_surgeries = len(surgeries)
    num_or_days = len(or_days)
    
    durations = [s['duration'] for s in surgeries]
    capacities = [od['capacity'] for od in or_days]
    
    # Get all specialty types
    all_specialties = list(set(s['specialty'] for s in surgeries if s.get('specialty')))
    specialty_to_idx = {s: i for i, s in enumerate(all_specialties)}
    num_specialty_types = len(all_specialties)
    
    # Prepare specialty constraints
    specialty_compatible = []
    for i in range(num_surgeries):
        compatible = []
        surgery_specialty = surgeries[i].get('specialty')
        for j in range(num_or_days):
            or_specialties = or_days[j].get('specialties', [])
            # Assume compatibility when specialty information is absent
            if not surgery_specialty or not or_specialties:
                compatible.append(True)
            else:
                compatible.append(surgery_specialty in or_specialties)
        specialty_compatible.append(compatible)
    
    # Surgery priority and cancellation penalty
    urgency_penalty = []
    for s in surgeries:
        urgency = s.get('urgency', 'elective')
        if urgency == 'emergency':
            urgency_penalty.append(emergency_penalty_value)
        elif urgency == 'urgent':
            urgency_penalty.append(urgent_penalty_value)
        else:  # elective
            urgency_penalty.append(elective_penalty_value)
    
    model = cp_model.CpModel()
    
    # Decision variable: whether surgery i is assigned to OR-day j
    x = {}
    for i in range(num_surgeries):
        for j in range(num_or_days):
            x[(i, j)] = model.NewBoolVar(f'x_{i}_{j}')
    
    # Decision variable: whether surgery i is canceled
    cancelled = {}
    for i in range(num_surgeries):
        cancelled[i] = model.NewBoolVar(f'cancelled_{i}')
    
    # Specialty mismatch penalty variables
    specialty_mismatch = {}
    for i in range(num_surgeries):
        specialty_mismatch[i] = model.NewBoolVar(f'mismatch_{i}')
    
    # Track surgery counts and distinct specialty counts per OR-day
    n_j_vars = {}  # Surgery count
    distinct_spec_j_vars = {}  # Distinct specialty count
    turnover_vars = {}  # Turnover time
    total_dur_vars = {}  # Total surgery duration
    total_occupied_vars = {}  # Total occupied time
    
    # Create tracking variables for each OR-day
    for j in range(num_or_days):
        # Surgery count n_j
        n_j_vars[j] = model.NewIntVar(0, num_surgeries, f'n_{j}')
        model.Add(n_j_vars[j] == sum(x[(i, j)] for i in range(num_surgeries)))
        
        # Distinct specialty count s_j
        distinct_spec_j_vars[j] = model.NewIntVar(0, num_specialty_types, f'distinct_spec_{j}')
        
        # Track whether each specialty occurs on the OR-day
        has_specialty = {}
        for spec_idx in range(num_specialty_types):
            has_specialty[spec_idx] = model.NewBoolVar(f'has_spec_{j}_{spec_idx}')
            model.AddMaxEquality(
                has_specialty[spec_idx],
                [x[(i, j)] for i in range(num_surgeries) 
                 if specialty_to_idx.get(surgeries[i].get('specialty', ''), -1) == spec_idx]
            )
        model.Add(distinct_spec_j_vars[j] == sum(has_specialty[spec_idx] for spec_idx in range(num_specialty_types)))
        
        # Turnover time = 10*n + 10*s - 20 when n >= 1
        turnover_vars[j] = model.NewIntVar(0, max(capacities[j], 0), f'turnover_{j}')
        
        # If n_j == 0 turnover is zero; otherwise it is 10*n_j + 10*s_j - 20
        n_j_is_zero = model.NewBoolVar(f'nj_zero_{j}')
        model.Add(n_j_vars[j] == 0).OnlyEnforceIf(n_j_is_zero)
        model.Add(n_j_vars[j] >= 1).OnlyEnforceIf(n_j_is_zero.Not())
        model.Add(turnover_vars[j] == 0).OnlyEnforceIf(n_j_is_zero)
        model.Add(turnover_vars[j] == 10 * n_j_vars[j] + 10 * distinct_spec_j_vars[j] - 20).OnlyEnforceIf(n_j_is_zero.Not())
        
        # Total surgery duration
        total_dur_vars[j] = model.NewIntVar(0, sum(durations), f'total_dur_{j}')
        model.Add(total_dur_vars[j] == sum(x[(i, j)] * durations[i] for i in range(num_surgeries)))
        
        # Total occupied time = surgery duration + turnover
        total_occupied_vars[j] = model.NewIntVar(0, capacities[j] + max(capacities[j], 0), f'total_occupied_{j}')
        model.Add(total_occupied_vars[j] == total_dur_vars[j] + turnover_vars[j])
        
        # Constraint: total occupied time cannot exceed capacity
        model.Add(total_occupied_vars[j] <= capacities[j])
    
    # Constraint 1: each surgery is assigned to one OR-day or canceled
    for i in range(num_surgeries):
        model.Add(sum(x[(i, j)] for j in range(num_or_days)) + cancelled[i] == 1)
    
    # Constraint 2: OR-day occupied time does not exceed capacity (added above)
    
    # Constraint 3: detect specialty mismatch
    for i in range(num_surgeries):
        for j in range(num_or_days):
            if not specialty_compatible[i][j]:
                # Flag assignments to an incompatible OR-day as specialty mismatches
                model.AddImplication(x[(i, j)], specialty_mismatch[i])
    
    # Objective 1: weighted cancellation penalty
    weighted_cancellation_penalty = model.NewIntVar(0, sum(urgency_penalty), 'cancel_penalty')
    model.Add(weighted_cancellation_penalty == sum(cancelled[i] * urgency_penalty[i] for i in range(num_surgeries)))
    
    # Objective 2: specialty mismatch penalty
    specialty_penalty = model.NewIntVar(0, num_surgeries * specialty_mismatch_penalty_value, 'specialty_penalty')
    model.Add(specialty_penalty == sum(specialty_mismatch[i] * specialty_mismatch_penalty_value for i in range(num_surgeries)))
    
    # Objective 3: total unused time
    # Unused time = total capacity - surgery duration - turnover time
    total_dur_all = sum(total_dur_vars.values())
    total_turnover_all = sum(turnover_vars.values())
    total_unused = model.NewIntVar(0, sum(capacities), 'total_unused')
    model.Add(total_unused == sum(capacities) - total_dur_all - total_turnover_all)
    
    # Idle-time penalty is 0.01 per minute, scaled by 100
    # Scale decimal coefficients and the final penalty by 100
    idle_time_penalty = model.NewIntVar(0, sum(capacities), 'idle_time_penalty')
    model.Add(idle_time_penalty == total_unused)  # Idle-time penalty is one point per minute after scaling
    
    # Minimize scaled cancellation, specialty mismatch, and idle-time penalties
    # Scaled cancellation penalties: emergency=2000, urgent=800, elective=100
    # Scaled specialty mismatch penalty: 3×100=300
    # Idle time is already scaled to one point per minute
    total_penalty = model.NewIntVar(0, sum(urgency_penalty) * 100 + num_surgeries * 300 + sum(capacities), 'total_penalty')
    model.Add(total_penalty == weighted_cancellation_penalty * 100 + specialty_penalty * 100 + idle_time_penalty)
    
    model.Minimize(total_penalty)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0  # Set the time limit
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        schedule = {od['id']: [] for od in or_days}
        cancelled_list = []
        
        for i in range(num_surgeries):
            if solver.Value(cancelled[i]) == 1:
                cancelled_list.append(surgeries[i]['id'])
            else:
                for j in range(num_or_days):
                    if solver.Value(x[(i, j)]) == 1:
                        schedule[or_days[j]['id']].append(surgeries[i]['id'])
                        break
        
        # Divide the scaled penalty by 100 to recover the actual penalty
        actual_cancel_penalty = solver.Value(weighted_cancellation_penalty) / 100.0
        actual_specialty_penalty = solver.Value(specialty_penalty) / 100.0
        actual_idle_time_penalty = solver.Value(idle_time_penalty) / 100.0
        actual_total_penalty = solver.Value(total_penalty) / 100.0
        
        # Get details for each OR-day
        or_day_details = {}
        for j in range(num_or_days):
            or_day_id = or_days[j]['id']
            num_surgeries_in_or = solver.Value(n_j_vars[j])
            num_distinct_spec = solver.Value(distinct_spec_j_vars[j])
            turnover_time = solver.Value(turnover_vars[j])
            total_duration = solver.Value(total_dur_vars[j])
            total_occupied = solver.Value(total_occupied_vars[j])
            
            or_day_details[or_day_id] = {
                "num_surgeries": int(num_surgeries_in_or),
                "num_distinct_specialties": int(num_distinct_spec),
                "total_duration": int(total_duration),
                "turnover_time": int(turnover_time),
                "total_occupied": int(total_occupied),
                "capacity": capacities[j],
                "unused": int(capacities[j] - total_occupied)
            }
        
        return {
            "schedule": schedule,
            "cancelled": cancelled_list,
            "total_unused_time": solver.Value(total_unused),
            "cancellation_penalty": actual_cancel_penalty,
            "specialty_penalty": actual_specialty_penalty,
            "idle_time_penalty": actual_idle_time_penalty,
            "total_penalty": actual_total_penalty,
            "or_day_details": or_day_details,
            "method": "OR-Tools CP-SAT",
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible"
        }
    else:
        return {
            "type": "or_scheduling_baseline",
            "status": "error",
            "error": "No feasible solution found"
        }


def generate_baseline_from_instance(instance_data):
    """Generate a baseline from instance_data without OR-Tools.
    
    Use cancellation and specialty mismatch penalties from the optional penalties
    configuration. Returns a baseline solution and penalty values."""
    surgeries = instance_data.get('surgeries', [])
    or_days = instance_data.get('or_days', [])
    penalties = instance_data.get('penalties', {})
    cancellation_penalties = penalties.get('cancellation', {})
    
    # Read penalty settings with defaults matching the instance generator
    emergency_penalty = cancellation_penalties.get('emergency', 20)
    urgent_penalty = cancellation_penalties.get('urgent', 8)
    elective_penalty = cancellation_penalties.get('elective', 1)
    specialty_mismatch_penalty = penalties.get('specialty_mismatch', 3)
    
    # Get the baseline assignment
    result = best_baseline_solver(surgeries, or_days)
    schedule = result.get('schedule', {})
    cancelled = result.get('cancelled', [])
    
    # Build a dictionary for lookup
    surgery_dict = {s['id']: s for s in surgeries}
    or_day_dict = {od['id']: od for od in or_days}
    
    # Calculate cancellation penalty
    cancellation_penalty = 0
    for surgery_id in cancelled:
        surgery = surgery_dict.get(surgery_id)
        if surgery:
            urgency = surgery.get('urgency', 'elective')
            cancellation_penalty += {
                'emergency': emergency_penalty,
                'urgent': urgent_penalty,
                'elective': elective_penalty
            }.get(urgency, elective_penalty)
    
    # Calculate specialty mismatch penalty
    specialty_penalty = 0
    for or_day_id, surgery_ids in schedule.items():
        or_day = or_day_dict.get(or_day_id)
        if or_day:
            or_specialties = set(or_day.get('specialties', []))
            for surgery_id in surgery_ids:
                surgery = surgery_dict.get(surgery_id)
                if surgery:
                    surgery_specialty = surgery.get('specialty')
                    if surgery_specialty and surgery_specialty not in or_specialties:
                        specialty_penalty += specialty_mismatch_penalty
    
    # Calculate idle-time penalty
    idle_time_coefficient = penalties.get('idle_time', 0.01)
    total_unused_time = result.get('total_unused_time', 0)
    idle_time_penalty = total_unused_time * idle_time_coefficient
    
    # Calculate total penalty, including idle time
    total_penalty = cancellation_penalty + specialty_penalty + idle_time_penalty
    
    # Update the result dictionary
    result.update({
        'method': 'best_fit_decreasing_with_penalty',
        'cancellation_penalty': cancellation_penalty,
        'specialty_penalty': specialty_penalty,
        'idle_time_penalty': idle_time_penalty,
        'total_penalty': total_penalty,
        'status': 'feasible'
    })
    
    return result