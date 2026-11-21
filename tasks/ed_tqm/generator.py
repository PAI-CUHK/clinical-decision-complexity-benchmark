"""Sample emergency queue patients without prescribed decision traps."""
import random
from tasks.ed_tqm.protocol import (
    _sample_triage_level, _sample_processing_time, _sample_age, _sample_nrs,
)


class EDInstance:
    def __init__(self, patients, num_doctors=1):
        self.patients = patients
        self.num_doctors = num_doctors

    def metadata(self):
        return {'num_patients': len(self.patients), 'num_doctors': self.num_doctors}


def generate_ed_instance(config):
    rng = random.Random(config.get('seed'))
    n = config.get('num_patients', 20)
    window = config.get('arrival_window', n * 6)
    if not isinstance(n, int) or n < 1 or window < 0:
        raise ValueError('num_patients must be positive and arrival_window nonnegative')
    patients = []
    for i in range(n):
        triage = _sample_triage_level(rng)
        patients.append({'id': f'P{i+1}', 'arrival_time': rng.randint(0, int(window)),
                         'triage_level': triage, 'processing_time': _sample_processing_time(rng),
                         'age': _sample_age(rng), 'nrs': _sample_nrs(rng, triage)})
    return EDInstance(patients)
