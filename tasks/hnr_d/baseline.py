from typing import Dict, Tuple

SHIFT_WORK = ["D", "E", "N"]


def _solver_timeout(num_nurses: int, num_days: int) -> float:
    """Set the CP-SAT time limit based on instance size."""
    size = num_nurses * num_days
    if size <= 120:
        return 100.0
    if size <= 400:
        return 120.0
    if size <= 800:
        return 180.0
    return 200.0


def check_hard_feasibility(instance: Dict, max_time_seconds: float = 2.0) -> bool:
    """Quickly check hard feasibility during instance generation: exactly one shift
    per nurse per day and sufficient shift coverage. Soft constraints are excluded."""
    from ortools.sat.python import cp_model

    nurses = instance.get("nurses", [])
    num_nurses = len(nurses)
    num_days = len(instance.get("days", []))
    required = instance.get("required", [])

    model = cp_model.CpModel()
    x = {}
    for i in range(num_nurses):
        for d in range(num_days):
            for s in ["D", "E", "N", "O"]:
                x[(i, d, s)] = model.NewBoolVar(f"x_{i}_{d}_{s}")

    for i in range(num_nurses):
        for d in range(num_days):
            model.Add(sum(x[(i, d, s)] for s in ["D", "E", "N", "O"]) == 1)

    for d in range(num_days):
        req = required[d] if d < len(required) else {}
        if isinstance(req, dict):
            for s in SHIFT_WORK:
                model.Add(sum(x[(i, d, s)] for i in range(num_nurses)) >= req.get(s, 0))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    status = solver.Solve(model)
    return status == cp_model.OPTIMAL or status == cp_model.FEASIBLE


def compute_optimal_penalty(instance: Dict) -> Tuple[int, bool]:
    """Compute the optimal penalty (OPT) and whether hard constraints are satisfied.
    
    The instance dictionary may include nurses, days, required daily shifts
    ({D, E, N}), forced_off dates, max_consecutive_work (default 6), nurse_skills
    (Senior or Junior), contracts (full-time or part-time), min_work (default 7),
    max_work (default 12), max_night_shifts_per_week (default 3),
    min_rest_per_week (default 1), and max_rest_per_week (default 2).
    
    Optional penalty_weights include workload_low=10, workload_high=20,
    night_excess=50, rest_low=10, rest_high=15, consecutive_work=50,
    consecutive_night=100, forced_off=200, skill_violation=300, and
    forbidden_transition=100. Missing weights use solver defaults.
    
    The documented contract workload ranges are 75–90% of the horizon for
    full-time and 40–60% for part-time, subject to min_work and max_work; without
    a contract, min_work and max_work apply. forced_off specifies days off, and
    weekly rest is bounded by min_rest_per_week and max_rest_per_week.
    
    Returns:
        A tuple (optimal penalty, whether hard constraints are satisfied)."""
    from ortools.sat.python import cp_model
    
    nurses = instance.get("nurses", [])
    num_nurses = len(nurses)
    days = instance.get("days", [])
    num_days = len(days)
    if num_nurses == 0 or num_days == 0:
        return float("inf"), False

    required = instance.get("required", [])
    forced_off = instance.get("forced_off", {})
    max_consecutive_work = instance.get("max_consecutive_work", 6)
    max_consecutive_night_shifts = instance.get("max_consecutive_night_shifts", 2)
    nurse_skills = instance.get("nurse_skills", {})
    contracts = instance.get("contracts", {})
    min_work = instance.get("min_work", 7)
    max_work = instance.get("max_work", 12)
    max_night_shifts_per_week = instance.get("max_night_shifts_per_week", 3)
    min_rest_per_week = instance.get("min_rest_per_week", 1)
    max_rest_per_week = instance.get("max_rest_per_week", 2)
    
    # Read penalty weights from the instance, using defaults when absent
    penalty_weights = instance.get("penalty_weights", {})
    PENALTY_WORKLOAD_LOW = penalty_weights.get('workload_low', 10)
    PENALTY_WORKLOAD_HIGH = penalty_weights.get('workload_high', 20)
    PENALTY_NIGHT_EXCESS = penalty_weights.get('night_excess', 50)
    PENALTY_REST_LOW = penalty_weights.get('rest_low', 10)
    PENALTY_REST_HIGH = penalty_weights.get('rest_high', 15)
    PENALTY_CONSECUTIVE_WORK = penalty_weights.get('consecutive_work', 50)
    PENALTY_CONSECUTIVE_NIGHT = penalty_weights.get('consecutive_night', 100)
    PENALTY_SKILL_VIOLATION = penalty_weights.get('skill_violation', 300)
    PENALTY_SHIFT_RESTRICTION = penalty_weights.get('shift_restriction', 150)
    PENALTY_FORBIDDEN_TRANSITION = penalty_weights.get('forbidden_transition', 100)
    
    model = cp_model.CpModel()
    
    # Create variables
    x = {}
    for i in range(num_nurses):
        for d in range(num_days):
            for s in ['D', 'E', 'N', 'O']:
                x[(i, d, s)] = model.NewBoolVar(f"x_{i}_{d}_{s}")
    
    work_day = {}
    for i in range(num_nurses):
        for d in range(num_days):
            work_day[(i, d)] = sum(x[(i, d, s)] for s in SHIFT_WORK)
    
    # ========= Hard constraints =========
    
    # 1. Exactly one shift per person per day
    for i in range(num_nurses):
        for d in range(num_days):
            model.Add(sum(x[(i, d, s)] for s in ['D', 'E', 'N', 'O']) == 1)
    
    # 2. Shift coverage requirements
    for d in range(num_days):
        req = required[d] if d < len(required) else {}
        if isinstance(req, dict):
            for s in SHIFT_WORK:
                model.Add(sum(x[(i, d, s)] for i in range(num_nurses)) >= req.get(s, 0))
    
    # ========= Soft constraints (penalties)=========
    
    penalty_vars = {
        "consecutive_work": [],
        "consecutive_night": [],
        "forbidden_transition": [],
        "skill_violation": [],
        "workload_low": [],
        "workload_high": [],
        "night_excess": [],
        "rest_low": [],
        "rest_high": []
    }
    
    # 1. Consecutive-work soft constraint using sliding windows
    # At most six consecutive working days; each excess day incurs 50
    for i in range(num_nurses):
        for start in range(num_days - max_consecutive_work):
            window_work = sum(work_day[(i, start + t)] for t in range(max_consecutive_work + 1))
            excess = model.NewIntVar(0, 1, f"work_excess_{i}_{start}")
            model.Add(excess >= window_work - max_consecutive_work)
            model.Add(excess <= 1)
            penalty_vars["consecutive_work"].append((excess, 1, PENALTY_CONSECUTIVE_WORK))
    
    # 2. Consecutive-night soft constraint using sliding windows
    # Allow at most max_consecutive_night_shifts consecutive nights
    night_window = max_consecutive_night_shifts + 1
    for i in range(num_nurses):
        for d in range(num_days - max_consecutive_night_shifts):
            violation = model.NewBoolVar(f"night_violation_{i}_{d}")
            window_sum = sum(x[(i, d + k, "N")] for k in range(night_window))
            model.Add(window_sum - max_consecutive_night_shifts <= violation)
            penalty_vars["consecutive_night"].append((violation, 1, PENALTY_CONSECUTIVE_NIGHT))
    
    # 3. Forbidden N->D transition (soft constraint)
    for i in range(num_nurses):
        for d in range(num_days - 1):
            violation = model.NewBoolVar(f"nd_violation_{i}_{d}")
            model.Add(x[(i, d, "N")] + x[(i, d + 1, "D")] - 1 <= violation)
            penalty_vars["forbidden_transition"].append((violation, 1, PENALTY_FORBIDDEN_TRANSITION))
    
    # 4. Skill constraint (soft constraint)
    for d in range(num_days):
        req = required[d] if d < len(required) else {}
        n_need = req.get('N', 0) if isinstance(req, dict) else 0
        if n_need > 0:
            senior_work_night = sum(
                x[(i, d, "N")] for i, nurse in enumerate(nurses) 
                if nurse_skills.get(nurse) == "Senior"
            )
            
            no_senior = model.NewBoolVar(f"no_senior_{d}")
            model.Add(senior_work_night == 0).OnlyEnforceIf(no_senior)
            model.Add(senior_work_night >= 1).OnlyEnforceIf(no_senior.Not())
            penalty_vars["skill_violation"].append((no_senior, 1, PENALTY_SKILL_VIOLATION))
    
    # 5. Workload penalty using precomputed workload bounds
    nurse_workload_bounds = instance.get('nurse_workload_bounds', {})
    
    for i, nurse in enumerate(nurses):
        total_work = sum(work_day[(i, d)] for d in range(num_days))
        
        if nurse_workload_bounds:
            nurse_min, nurse_max = nurse_workload_bounds[nurse]
        else:
            raise ValueError(f"Nurse {nurse} not found in nurse_workload_bounds")
        
        under_work = model.NewIntVar(0, num_days, f"under_work_{nurse}")
        model.Add(under_work >= nurse_min - total_work)
        model.Add(under_work >= 0)
        penalty_vars["workload_low"].append((under_work, 1, PENALTY_WORKLOAD_LOW))
        
        over_work = model.NewIntVar(0, num_days, f"over_work_{nurse}")
        model.Add(over_work >= total_work - nurse_max)
        model.Add(over_work >= 0)
        penalty_vars["workload_high"].append((over_work, 1, PENALTY_WORKLOAD_HIGH))
    
    # 7. Night-shift penalty
    if max_night_shifts_per_week > 0:
        for i in range(num_nurses):
            for week_start in range(0, num_days, 7):
                week_end = min(week_start + 7, num_days)
                weekly_nights = sum(x[(i, d, "N")] for d in range(week_start, week_end))
                excess = model.NewIntVar(0, 7, f"excess_nights_{i}_{week_start}")
                model.Add(excess >= weekly_nights - max_night_shifts_per_week)
                model.Add(excess >= 0)
                penalty_vars["night_excess"].append((excess, 1, PENALTY_NIGHT_EXCESS))
    
    # 8. Rest penalty
    if min_rest_per_week > 0 or max_rest_per_week > 0:
        for i in range(num_nurses):
            for week_start in range(0, num_days, 7):
                week_end = min(week_start + 7, num_days)
                weekly_work = sum(work_day[(i, d)] for d in range(week_start, week_end))
                weekly_rest = (week_end - week_start) - weekly_work
                
                if min_rest_per_week > 0:
                    under_rest = model.NewIntVar(0, 7, f"under_rest_{i}_{week_start}")
                    model.Add(under_rest >= min_rest_per_week - weekly_rest)
                    model.Add(under_rest >= 0)
                    penalty_vars["rest_low"].append((under_rest, 1, PENALTY_REST_LOW))
                
                if max_rest_per_week > 0:
                    over_rest = model.NewIntVar(0, 7, f"over_rest_{i}_{week_start}")
                    model.Add(over_rest >= weekly_rest - max_rest_per_week)
                    model.Add(over_rest >= 0)
                    penalty_vars["rest_high"].append((over_rest, 1, PENALTY_REST_HIGH))
    
    # Build the total penalty
    penalties = []
    for constraint_type, vars_list in penalty_vars.items():
        for var, excess, weight in vars_list:
            penalties.append(var * excess * weight)
    
    # Minimize the penalty
    if penalties:
        objective = model.NewIntVar(0, 1000000, "objective")
        model.Add(objective == sum(penalties))
        model.Minimize(objective)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _solver_timeout(num_nurses, num_days)
    status = solver.Solve(model)
    
    # A feasible incumbent alone cannot certify a No answer below its value.
    if status == cp_model.OPTIMAL:
        print("=" * 60)
        print("各约束惩罚值明细:")
        print("=" * 60)
        
        total_penalty = 0
        constraint_names = {
            "consecutive_work": "连续工作惩罚",
            "consecutive_night": "连续夜班惩罚",
            "forbidden_transition": "N->D班次转换惩罚",
            "skill_violation": "技能约束惩罚",
            "workload_low": "工作量不足惩罚",
            "workload_high": "工作量超额惩罚",
            "night_excess": "夜班超额惩罚",
            "rest_low": "休息不足惩罚",
            "rest_high": "休息超额惩罚"
        }
        
        for constraint_type, vars_list in penalty_vars.items():
            penalty_sum = 0
            for var, excess, weight in vars_list:
                penalty_sum += solver.Value(var) * excess * weight
            total_penalty += penalty_sum
            print(f"{constraint_names[constraint_type]:<20} : {penalty_sum}")
        
            print("=" * 60)
        print(f"{'总惩罚值':<20} : {total_penalty}")
        print("=" * 60)
        
        # Generate the roster
        schedule = {}
        for i, nurse in enumerate(nurses):
            schedule[nurse] = {}
            for d in range(num_days):
                for s in SHIFT_WORK:
                    if solver.Value(x[(i, d, s)]) == 1:
                        schedule[nurse][d] = s
                        break
                if d not in schedule[nurse]:
                    schedule[nurse][d] = "O"
        
        # Print the roster
        print("\n" + "=" * 60)
        print("排班表:")
        print("=" * 60)
        
        # Print the table header
        header = "护士".ljust(12)
        for d in range(num_days):
            header += f" D{d}".ljust(4)
        print(header)
        print("-" * (12 + num_days * 4))
        
        # Print each row
        for nurse in nurses:
            row = f"{nurse}".ljust(12)
            for d in range(num_days):
                row += f"  {schedule[nurse][d]}".ljust(4)
            print(row)
        
        print("=" * (12 + num_days * 4))
        
        return int(solver.ObjectiveValue()), True
    else:
        return float('inf'), False


def compute_optimal_schedule(instance: Dict) -> Tuple[int, Dict, bool]:
    """Compute (optimal penalty, schedule, hard-constraint feasibility) from an instance."""
    from ortools.sat.python import cp_model
    
    nurses = instance.get("nurses", [])
    num_nurses = len(nurses)
    days = instance.get("days", [])
    num_days = len(days)
    if num_nurses == 0 or num_days == 0:
        return float("inf"), {}, False

    required = instance.get("required", [])
    max_consecutive_work = instance.get("max_consecutive_work", 6)
    max_consecutive_night_shifts = instance.get("max_consecutive_night_shifts", 2)
    nurse_skills = instance.get("nurse_skills", {})
    contracts = instance.get("contracts", {})
    min_work = instance.get("min_work", 7)
    max_work = instance.get("max_work", 12)
    max_night_shifts_per_week = instance.get("max_night_shifts_per_week", 3)
    min_rest_per_week = instance.get("min_rest_per_week", 1)
    max_rest_per_week = instance.get("max_rest_per_week", 2)
    
    penalty_weights = instance.get("penalty_weights", {})
    PENALTY_WORKLOAD_LOW = penalty_weights.get('workload_low', 10)
    PENALTY_WORKLOAD_HIGH = penalty_weights.get('workload_high', 20)
    PENALTY_NIGHT_EXCESS = penalty_weights.get('night_excess', 50)
    PENALTY_REST_LOW = penalty_weights.get('rest_low', 10)
    PENALTY_REST_HIGH = penalty_weights.get('rest_high', 15)
    PENALTY_CONSECUTIVE_WORK = penalty_weights.get('consecutive_work', 50)
    PENALTY_CONSECUTIVE_NIGHT = penalty_weights.get('consecutive_night', 100)
    PENALTY_SKILL_VIOLATION = penalty_weights.get('skill_violation', 300)
    PENALTY_FORBIDDEN_TRANSITION = penalty_weights.get('forbidden_transition', 100)
    
    model = cp_model.CpModel()
    
    x = {}
    for i in range(num_nurses):
        for d in range(num_days):
            for s in ['D', 'E', 'N', 'O']:
                x[(i, d, s)] = model.NewBoolVar(f"x_{i}_{d}_{s}")
    
    work_day = {}
    for i in range(num_nurses):
        for d in range(num_days):
            work_day[(i, d)] = sum(x[(i, d, s)] for s in SHIFT_WORK)
    
    # ========= Hard constraints =========
    for i in range(num_nurses):
        for d in range(num_days):
            model.Add(sum(x[(i, d, s)] for s in ['D', 'E', 'N', 'O']) == 1)
    
    for d in range(num_days):
        req = required[d] if d < len(required) else {}
        if isinstance(req, dict):
            for s in SHIFT_WORK:
                model.Add(sum(x[(i, d, s)] for i in range(num_nurses)) >= req.get(s, 0))
    
    # ========= Soft constraints (penalties)=========
    penalties = []
    
    # 1. Consecutive-work limit using sliding windows
    for i in range(num_nurses):
        for start in range(num_days - max_consecutive_work):
            window_work = sum(work_day[(i, start + t)] for t in range(max_consecutive_work + 1))
            excess = model.NewIntVar(0, 1, f"work_excess_{i}_{start}")
            model.Add(excess >= window_work - max_consecutive_work)
            model.Add(excess >= 0)
            penalties.append(excess * PENALTY_CONSECUTIVE_WORK)
    
    # 2. Consecutive-night limit using sliding windows
    night_window = max_consecutive_night_shifts + 1
    for i in range(num_nurses):
        for d in range(num_days - max_consecutive_night_shifts):
            violation = model.NewBoolVar(f"night_violation_{i}_{d}")
            window_sum = sum(x[(i, d + k, "N")] for k in range(night_window))
            model.Add(window_sum - max_consecutive_night_shifts <= violation)
            penalties.append(violation * PENALTY_CONSECUTIVE_NIGHT)
    
    # 3. Forbidden N->D transition (soft constraint)
    for i in range(num_nurses):
        for d in range(num_days - 1):
            violation = model.NewBoolVar(f"nd_violation_{i}_{d}")
            model.Add(x[(i, d, "N")] + x[(i, d + 1, "D")] - 1 <= violation)
            penalties.append(violation * PENALTY_FORBIDDEN_TRANSITION)
    
    # 4. Skill constraint (soft constraint)
    for d in range(num_days):
        req = required[d] if d < len(required) else {}
        n_need = req.get('N', 0) if isinstance(req, dict) else 0
        if n_need > 0:
            senior_work_night = sum(
                x[(i, d, "N")] for i, nurse in enumerate(nurses) 
                if nurse_skills.get(nurse) == "Senior"
            )
            no_senior = model.NewBoolVar(f"no_senior_{d}")
            model.Add(senior_work_night == 0).OnlyEnforceIf(no_senior)
            model.Add(senior_work_night >= 1).OnlyEnforceIf(no_senior.Not())
            penalties.append(no_senior * PENALTY_SKILL_VIOLATION)
    
    # 5. Workload penalty
    nurse_workload_bounds = instance.get('nurse_workload_bounds', {})
    
    for i, nurse in enumerate(nurses):
        total_work = sum(work_day[(i, d)] for d in range(num_days))
        
        if nurse_workload_bounds:
            nurse_min, nurse_max = nurse_workload_bounds[nurse]
        else:
            contract_type = contracts.get(nurse, "full-time")
            if contract_type == "full-time":
                nurse_min, nurse_max = int(num_days * 0.7), int(num_days * 0.85)
            else:
                nurse_min, nurse_max = int(num_days * 0.3), int(num_days * 0.5)
        
        under_work = model.NewIntVar(0, num_days, f"under_work_{nurse}")
        model.Add(under_work >= nurse_min - total_work)
        model.Add(under_work >= 0)
        penalties.append(under_work * PENALTY_WORKLOAD_LOW)
        
        over_work = model.NewIntVar(0, num_days, f"over_work_{nurse}")
        model.Add(over_work >= total_work - nurse_max)
        model.Add(over_work >= 0)
        penalties.append(over_work * PENALTY_WORKLOAD_HIGH)
    
    # 6. Night-shift penalty
    if max_night_shifts_per_week > 0:
        for i in range(num_nurses):
            for week_start in range(0, num_days, 7):
                week_end = min(week_start + 7, num_days)
                weekly_nights = sum(x[(i, d, "N")] for d in range(week_start, week_end))
                excess = model.NewIntVar(0, 7, f"excess_nights_{i}_{week_start}")
                model.Add(excess >= weekly_nights - max_night_shifts_per_week)
                model.Add(excess >= 0)
                penalties.append(excess * PENALTY_NIGHT_EXCESS)
    
    # 7. Rest penalty
    if min_rest_per_week > 0 or max_rest_per_week > 0:
        for i in range(num_nurses):
            for week_start in range(0, num_days, 7):
                week_end = min(week_start + 7, num_days)
                weekly_work = sum(work_day[(i, d)] for d in range(week_start, week_end))
                weekly_rest = (week_end - week_start) - weekly_work
                
                if min_rest_per_week > 0:
                    under_rest = model.NewIntVar(0, 7, f"under_rest_{i}_{week_start}")
                    model.Add(under_rest >= min_rest_per_week - weekly_rest)
                    model.Add(under_rest >= 0)
                    penalties.append(under_rest * PENALTY_REST_LOW)
                
                if max_rest_per_week > 0:
                    over_rest = model.NewIntVar(0, 7, f"over_rest_{i}_{week_start}")
                    model.Add(over_rest >= weekly_rest - max_rest_per_week)
                    model.Add(over_rest >= 0)
                    penalties.append(over_rest * PENALTY_REST_HIGH)
    
    # Minimize the penalty
    if penalties:
        objective = model.NewIntVar(0, 1000000, "objective")
        model.Add(objective == sum(penalties))
        model.Minimize(objective)
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = _solver_timeout(num_nurses, num_days)
    status = solver.Solve(model)
    
    if status == cp_model.OPTIMAL:
        # Generate the roster
        schedule = {}
        for i, nurse in enumerate(nurses):
            schedule[nurse] = {}
            for d in range(num_days):
                for s in SHIFT_WORK:
                    if solver.Value(x[(i, d, s)]) == 1:
                        schedule[nurse][d] = s
                        break
                if d not in schedule[nurse]:
                    schedule[nurse][d] = "O"
        
        return int(solver.ObjectiveValue()), schedule, True
    else:
        return float('inf'), {}, False


def generate_nurse_baseline(instance: Dict) -> Dict:
    """Assemble a baseline from instance fields without running CP-SAT. OPT must
    already have been computed by the adapter using compute_optimal_penalty."""
    if instance.get("invalid"):
        return {
            "baseline_answer": "No",
            "opt_value": instance.get("opt_value", float("inf")),
            "satisfies_hard_constraints": False,
            "invalid": True,
            "reason": instance.get("reason", ""),
        }

    if "opt_value" not in instance:
        raise ValueError(
            "instance 缺少 opt_value，请先调用 adapter.build() 计算 OPT"
        )

    opt_value = instance["opt_value"]
    k = instance.get("k", 0)
    if "expected_answer" in instance:
        baseline_answer = instance["expected_answer"]
    else:
        baseline_answer = "Yes" if (opt_value <= k if k > 0 else opt_value == 0) else "No"

    baseline = {
        "baseline_answer": baseline_answer,
        "opt_value": opt_value,
        "satisfies_hard_constraints": True,
    }
    if "k" in instance:
        baseline["k"] = k
    return baseline
