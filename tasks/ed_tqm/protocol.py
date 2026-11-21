"""Parameters and sampling helpers for the emergency queue task."""
ESCALATION_THRESHOLDS = {1: None, 2: 10, 3: 30, 4: 60, 5: 120}
MAX_ESCALATIONS = 2
NUM_DOCTORS = 1
TRIAGE_WEIGHTS = [0.08, 0.27, 0.30, 0.25, 0.10]
AGE_GROUPS = ((0, 17, 0.15), (18, 64, 0.60), (65, 100, 0.25))
PROCESSING_TIME_RANGE = (5, 20)


def get_protocol_config(level):
    if not 1 <= level <= 10:
        raise ValueError('level must be between 1 and 10')
    count = 20 + (level - 1) * 2
    return {'level': level, 'num_patients': count, 'num_doctors': NUM_DOCTORS,
            'escalation_thresholds': ESCALATION_THRESHOLDS,
            'max_escalations': MAX_ESCALATIONS, 'arrival_window': count * 6}


def _sample_triage_level(rng):
    return rng.choices((1, 2, 3, 4, 5), weights=TRIAGE_WEIGHTS, k=1)[0]


def _sample_processing_time(rng):
    return rng.randint(*PROCESSING_TIME_RANGE)


def _sample_age(rng):
    low, high, _ = rng.choices(AGE_GROUPS, weights=[g[2] for g in AGE_GROUPS], k=1)[0]
    return rng.randint(low, high)


def _sample_nrs(rng, triage_level):
    ranges = {1: (6, 10), 2: (6, 10), 3: (4, 8), 4: (2, 6), 5: (0, 3)}
    return rng.randint(*ranges[triage_level])
