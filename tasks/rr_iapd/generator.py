"""Generate varied RR sequences without threshold-targeted traps."""
import random
from typing import List, Dict
from tasks.rr_iapd.baseline import detect_patterns


class ECGInstance:
    def __init__(self, rr_intervals: List[float]):
        self.rr_intervals = rr_intervals
        self.patterns = {
            "Pattern 1": {
                "name": "Short-Long RR Pattern",
                "description": (
                    "A Short RR interval immediately followed by a Long RR interval. "
                    "Pattern 1 occurs if RR_i is Short and RR_{i+1} is Long."
                )
            },
            "Pattern 2": {
                "name": "Periodic Short-Long RR Pattern",
                "description": (
                    "A repeated Short-Long RR timing pattern with constant periodicity. "
                    "Each unit is one Short RR immediately followed by one Long RR; "
                    "the number of consecutive Normal RR intervals between neighboring "
                    "Short RR intervals remains constant and the pattern repeats at "
                    "least three consecutive times."
                )
            },
            "Pattern 3": {
                "name": "Prolonged RR Interval",
                "description": (
                    "A prolonged RR interval meeting the predefined pause criterion. "
                    "Pattern 3 occurs if RR_i >= 2.0 seconds."
                )
            },
            "Pattern 4": {
                "name": "Relative RR Difference Episode",
                "description": (
                    "A local episode characterized by frequent large relative "
                    "differences between successive RR intervals. A sliding window "
                    "of 10 consecutive RR intervals containing at least 8 relative "
                    "RR difference events (relative difference >= 3.25%) is detected."
                )
            }
        }

    def metadata(self):
        return {
            "num_rr_intervals": len(self.rr_intervals),
            "patterns": list(self.patterns.keys())
        }



def generate_ecg_instance(config: Dict) -> ECGInstance:
    rng = random.Random(config.get("seed"))
    count = config.get("num_intervals", 30)
    baseline = config.get("baseline_rr", 0.9)
    max_rr = config.get("max_rr", 2.6)
    if not isinstance(count, int) or count < 5 or baseline <= 0 or max_rr < 2:
        raise ValueError("Require at least five intervals, positive baseline, max_rr >= 2")
    rr = []
    for index in range(count):
        if index < 5:
            value = rng.gauss(baseline, baseline * 0.035)
        else:
            kind = rng.choices(("regular", "short", "long", "pause"),
                               weights=(0.82, 0.07, 0.07, 0.04), k=1)[0]
            factor = {"regular": (1.0, 0.07), "short": (0.70, 0.08),
                      "long": (1.30, 0.12), "pause": (2.4 / baseline, 0.15)}[kind]
            value = rng.gauss(baseline * factor[0], baseline * factor[1])
        rr.append(round(max(0.25, min(max_rr, value)), 4))
    return ECGInstance(rr)


def compute_ground_truth(rr_intervals: List[float]) -> Dict:
    detections = detect_patterns(rr_intervals)
    return {
        "Pattern 1": detections["Pattern 1"]["positions"],
        "Pattern 2": detections["Pattern 2"]["positions"],
        "Pattern 3": detections["Pattern 3"]["positions"],
        "Pattern 4": detections["Pattern 4"]["positions"],
    }
