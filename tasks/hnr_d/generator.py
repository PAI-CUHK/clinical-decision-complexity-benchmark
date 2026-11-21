"""Generate nurse rostering instances with controllable, natural demand.

The generator samples daily coverage needs independently around a target load.
It does not inject predetermined conflicts or decide the Yes/No answer; the
adapter derives the latter from the certified optimum and decision threshold.
"""
import random


SHIFT_TYPES = ("D", "E", "N")


class NurseSchedulingInstance:
    def __init__(self, nurses, days, required, min_work, max_work,
                 max_consecutive_work=6, max_consecutive_night_shifts=2,
                 max_night_shifts_per_week=3, min_rest_per_week=1,
                 max_rest_per_week=2, nurse_skills=None, contracts=None,
                 nurse_workload_bounds=None):
        self.nurses = nurses
        self.days = days
        self.required = required
        self.min_work = min_work
        self.max_work = max_work
        self.max_consecutive_work = max_consecutive_work
        self.max_consecutive_night_shifts = max_consecutive_night_shifts
        self.max_night_shifts_per_week = max_night_shifts_per_week
        self.min_rest_per_week = min_rest_per_week
        self.max_rest_per_week = max_rest_per_week
        self.nurse_skills = nurse_skills or {}
        self.contracts = contracts or {}
        self.nurse_workload_bounds = nurse_workload_bounds or {}

    def metadata(self):
        return {"num_nurses": len(self.nurses), "num_days": len(self.days),
                "min_work": self.min_work, "max_work": self.max_work,
                "max_consecutive_work": self.max_consecutive_work,
                "max_consecutive_night_shifts": self.max_consecutive_night_shifts,
                "max_night_shifts_per_week": self.max_night_shifts_per_week,
                "min_rest_per_week": self.min_rest_per_week,
                "max_rest_per_week": self.max_rest_per_week}


def generate_nurse_scheduling_instance(config, max_retries=5):
    """Sample coverage and staff attributes; do not force feasibility outcomes.

    ``demand_ratio`` controls mean daily staffing relative to staff count and
    ``demand_variation`` controls day-to-day variation. ``seed`` makes the
    instance and its subsequent threshold selection reproducible. The
    ``max_retries`` argument remains for compatibility; no rejection loop is
    needed because each day's demand is bounded by the staff count.
    """
    seed = config.get("seed")
    rng = random.Random(seed)
    num_nurses = config.get("num_nurses", 8)
    num_days = config.get("num_days", 14)
    ratio = config.get("demand_ratio", 0.55)
    variation = config.get("demand_variation", 0.06)
    if not isinstance(num_nurses, int) or isinstance(num_nurses, bool) or num_nurses < 3:
        raise ValueError("num_nurses must be an integer >= 3")
    if not isinstance(num_days, int) or isinstance(num_days, bool) or num_days < 1:
        raise ValueError("num_days must be a positive integer")
    if not 0 < ratio < 1 or not 0 <= variation < 1 or ratio + variation >= 1:
        raise ValueError("demand_ratio must be positive and ratio + variation < 1")

    nurses = [f"Nurse_{i + 1}" for i in range(num_nurses)]
    days = list(range(num_days))
    num_senior = max(1, round(num_nurses * 0.3))
    num_junior = max(1, round(num_nurses * 0.15))
    skills = (["Senior"] * num_senior + ["Junior"] * num_junior +
              ["RN"] * (num_nurses - num_senior - num_junior))
    rng.shuffle(skills)
    nurse_skills = dict(zip(nurses, skills))
    contracts = {nurse: rng.choices(("full-time", "part-time"), weights=(0.7, 0.3))[0]
                 for nurse in nurses}
    min_rest = config.get("min_rest_per_week", 1)
    max_rest = config.get("max_rest_per_week", 2)
    if not 0 <= min_rest <= max_rest <= 6:
        raise ValueError("rest bounds must satisfy 0 <= min <= max <= 6")
    for key in ("max_consecutive_work", "max_consecutive_night_shifts"):
        value = config.get(key, 6 if key == "max_consecutive_work" else 2)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    min_work = max(1, int(num_days * (7 - max_rest) / 7))
    max_work = max(1, min(num_days, int(num_days * (7 - min_rest) / 7)))
    workload = {}
    for nurse, contract in contracts.items():
        if contract == "full-time":
            low = max(min_work, int(num_days * 0.70))
            high = min(max_work, int(num_days * 0.90))
        else:
            low = max(1, int(num_days * 0.30))
            high = min(max_work, int(num_days * 0.50))
        workload[nurse] = (min(low, high), high)

    required = []
    for _ in days:
        daily_load = rng.uniform(max(0, ratio - variation), ratio + variation)
        total = max(1, min(num_nurses, round(num_nurses * daily_load)))
        counts = {shift: 0 for shift in SHIFT_TYPES}
        for shift in rng.choices(SHIFT_TYPES, weights=(0.45, 0.30, 0.25), k=total):
            counts[shift] += 1
        required.append(counts)

    instance = NurseSchedulingInstance(
        nurses, days, required, min_work, max_work,
        max_consecutive_work=config.get("max_consecutive_work", 6),
        max_consecutive_night_shifts=config.get("max_consecutive_night_shifts", 2),
        max_night_shifts_per_week=config.get("max_night_shifts_per_week", 3),
        min_rest_per_week=min_rest, max_rest_per_week=max_rest,
        nurse_skills=nurse_skills, contracts=contracts, nurse_workload_bounds=workload)
    instance.seed = seed
    return instance
