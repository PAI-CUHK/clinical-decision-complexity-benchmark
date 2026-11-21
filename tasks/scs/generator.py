"""
Operating Room Scheduling Problem (OR Scheduling) Instance Generator

This is a classic Bin Packing / Multiple Knapsack problem:
- Items = Surgeries (with durations)
- Bins = OR-days (with capacity)
- Goal = Minimize unused time

Complexity: NP-hard
"""

import random
from typing import Dict, List


class ORInstance:
    """Operating Room Scheduling Instance"""
    
    def __init__(self, surgeries: List[Dict], or_days: List[Dict], level: int = 1, penalties: Dict = None):
        self.surgeries = surgeries
        self.or_days = or_days
        self.level = level
        self.penalties = penalties or {
            "cancellation": {
                "emergency": 20,
                "urgent": 8,
                "elective": 1
            },
            "specialty_mismatch": 3,
            "idle_time": 0.01
        }
        # Turnover time between consecutive surgeries in the same OR-day
        self.turnover = {
            "same_specialty": 10,
            "different_specialty": 20
        }
    
    def metadata(self):
        return {
            "num_surgeries": len(self.surgeries),
            "num_or_days": len(self.or_days),
            "total_surgery_time": sum(s["duration"] for s in self.surgeries),
            "total_capacity": sum(d["capacity"] for d in self.or_days),
            "level": self.level
        }


from typing import Dict

# Medical semantic elements
URGENCY_LEVELS = ["emergency", "urgent", "elective"]
URGENCY_WEIGHTS = {
    "emergency": 20,
    "urgent": 8,
    "elective": 1
}
SPECIALTIES = ["general", "orthopedic", "cardiac", "neurosurgery"]


def generate_or_instance(config: Dict) -> ORInstance:
    """
    Generate an Operating Room Scheduling instance.

    Surgery durations vary around the configured average load. All surgery
    specialties are covered by at least one OR-day.

    Args:
        config:
            - num_surgeries
            - num_or_days
            - capacity_per_day
            - duration_range
            - load_factor
            - seed
    """

    seed = config.get("seed", random.randint(0, 10000))
    random.seed(seed)

    num_surgeries = config.get("num_surgeries", 10)
    num_or_days = config.get("num_or_days", 3)
    capacity_per_day = config.get("capacity_per_day", 480)
    # Use the same higher load factor as the decision variant
    load_factor = config.get("load_factor", 0.95)

    # ----- Step 1: Compute target total duration -----

    total_capacity = num_or_days * capacity_per_day
    target_total_duration = int(total_capacity * load_factor)

    # Independent, varied durations around the requested average load.
    # The realized total is allowed to fluctuate naturally.
    mean_duration = target_total_duration / num_surgeries
    durations = [max(30, min(240, round(random.gauss(mean_duration, mean_duration * 0.25))))
                 for _ in range(num_surgeries)]

    # ----- Step 5: Create surgeries with medical semantics -----

    surgeries = []
    for i in range(num_surgeries):
        # Sample urgency at random
        urgency = random.choices(URGENCY_LEVELS, weights=[0.2, 0.35, 0.45])[0]
        
        # Sample a specialty at random
        specialty = random.choice(SPECIALTIES)
        
        surgeries.append({
            "id": f"S{i+1}",
            "duration": durations[i],
            "urgency": urgency,
            "specialty": specialty
        })

    # ----- Step 6: Determine required specialties from surgeries -----
    
    # Collect specialties required by all surgeries
    required_specialties = set(s["specialty"] for s in surgeries)
    
    # Ensure at least one OR-day supports each required specialty
    num_required = len(required_specialties)
    
    # Adjust if required specialties outnumber OR-days
    if num_required > num_or_days:
        pass  # Handle silently, as in the decision variant

    # ----- Step 7: Create OR days with specialty configurations (Ensure all required specialties are covered) -----
    
    or_days = []
    required_list = sorted(required_specialties)
    
    # First allocate available OR-days to required specialties.
    for i, specialty in enumerate(required_list[:num_or_days]):
        # Use 50% specialized and 50% general OR-days
        if random.random() < 0.5 or i == len(required_list) - 1:
            specialties = [specialty]
            or_type = "specialist"
        else:
            other_specialties = [s for s in SPECIALTIES if s != specialty]
            second_specialty = random.choice(other_specialties)
            specialties = [specialty, second_specialty]
            or_type = "general"
        
        or_days.append({
            "id": f"OR-day{i+1}",
            "capacity": capacity_per_day,
            "specialties": specialties,
            "type": or_type
        })
    
    # Create additional OR-days if there are more days than required specialties
    for i in range(num_required, num_or_days):
        # Use 40% specialized and 60% general OR-days
        if random.random() < 0.4:
            specialties = [random.choice(SPECIALTIES)]
            or_type = "specialist"
        else:
            s1 = random.choice(SPECIALTIES)
            s2 = random.choice([s for s in SPECIALTIES if s != s1])
            specialties = [s1, s2]
            or_type = "general"
        
        or_days.append({
            "id": f"OR-day{i+1}",
            "capacity": capacity_per_day,
            "specialties": specialties,
            "type": or_type
        })

    # Verify that all required specialties are covered
    covered_specialties = set()
    for or_day in or_days:
        covered_specialties.update(or_day["specialties"])
    
    missing_specialties = required_specialties - covered_specialties
    if missing_specialties:
        print(f"Error: Missing coverage for specialties: {missing_specialties}")
        # Add support for any missing specialty
        for missing in missing_specialties:
            # Choose an OR-day at random to support that specialty
            or_day = random.choice(or_days)
            if missing not in or_day["specialties"]:
                or_day["specialties"].append(missing)
                or_day["type"] = "general"  # Update the OR-day to general

    # ----- Step 6: Shuffle (increase randomness) -----

    random.shuffle(surgeries)
    random.shuffle(or_days)

    # print(f"Total surgery duration: {sum(durations)}")
    # print(f"Target duration: {target_total_duration}")

    return ORInstance(
        surgeries=surgeries,
        or_days=or_days,
        level=config.get("level", 1),
        penalties={
            "cancellation": {
                "emergency": 20,
                "urgent": 8,
                "elective": 1
            },
            "specialty_mismatch": 3,
            "idle_time": 0.01
        }
    )


# Example usage
if __name__ == "__main__":
    config = {
        "num_surgeries": 15,
        "num_or_days": 4,
        "capacity_per_day": 480,
        "seed": 42
    }
    
    instance = generate_or_instance(config)
    
    print("OR Scheduling Instance:")
    print(f"Surgeries: {len(instance.surgeries)}")
    print(f"OR-days: {len(instance.or_days)}")
    print(f"\nMetadata: {instance.metadata()}")
    
    print("\nSurgeries:")
    for s in instance.surgeries:
        print(f"  {s['id']}: {s['duration']} minutes")
    
    print("\nOR-days:")
    for d in instance.or_days:
        print(f"  {d['id']}: {d['capacity']} minutes capacity")
