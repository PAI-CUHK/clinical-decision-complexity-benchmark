import random
from typing import List, Dict

# Constants
DEFAULT_SPECIALISMS = ["Cardiology", "Neurology", "Orthopedics"]
ROOM_CAPACITY_OPTIONS = (1, 2, 4)
GENDER_POLICY_OPTIONS = ["M", "F", "Any"]
AGE_POLICY_OPTIONS = ["Pediatric", "Adult", "Elderly", "Any"]
SPECIALISM_PRIORITY_OPTIONS = [1, 2, 99]
MIN_AGE = 5
MAX_AGE = 90
DEFAULT_NUM_PATIENTS = 10
DEFAULT_HORIZON = 8

# Mappings
SPECIALISM_MAP = {
    1: "Cardiology",
    2: "Neurology",
    3: "Orthopedics",
    4: "General Medicine"
}

ROOM_TYPE_MAP = {
    1: "single-bed room",
    2: "double-bed room",
    4: "multi-bed room"
}


class PASPInstance:
    """Patient allocation instance containing patients, rooms, and the time horizon."""
    def __init__(self, patients, rooms, horizon):
        self.patients = patients
        self.rooms = rooms
        self.horizon = horizon

    def metadata(self):
        """Return a dictionary of instance metadata."""
        return {
            "num_patients": len(self.patients),
            "num_rooms": len(self.rooms),
            "horizon": self.horizon
        }


def random_room_partition(n, allowed=ROOM_CAPACITY_OPTIONS, max_rooms=None):
    """Generate a random list of room capacities.
    
    Args:
        n: Required total capacity.
        allowed: Allowed room capacities.
        max_rooms: Maximum number of rooms."""
    while True:
        remaining = n
        rooms = []

        while remaining > 0:
            choices = [s for s in allowed if s <= remaining]
            s = random.choice(choices)
            rooms.append(s)
            remaining -= s

        if max_rooms is None or len(rooms) <= max_rooms:
            random.shuffle(rooms)
            return rooms


def generate_patients(
    n_patients: int,
    horizon: int,
    specialisms: List[str]
) -> List[Dict]:
    """Generate patient data for the specified count, horizon, and specialties."""
    patients = []

    for i in range(n_patients):
        start = random.randint(0, horizon - 2)
        end = random.randint(start + 1, horizon - 1)
        spec = random.choice(specialisms)

        patients.append({
            "id": f"P{i}",
            "age": random.randint(MIN_AGE, MAX_AGE),
            "gender": random.choice(GENDER_POLICY_OPTIONS[:2]),
            "stay": (start, end),
            "specialism_plan": [(spec, end - start + 1)],
            "room_pref": random.choice(ROOM_CAPACITY_OPTIONS),
            "required_props": [random.randint(0, 1) for _ in range(4)],
            "specialism": spec
        })

    return patients


def generate_rooms(n_patients: int, specialisms: List[str]) -> List[Dict]:
    """Generate room data for the patient count and available specialties."""
    max_rooms = max(1, n_patients // 2)
    room_sizes = random_room_partition(
        n_patients,
        allowed=ROOM_CAPACITY_OPTIONS,
        max_rooms=max_rooms
    )

    rooms = []
    room_id = 0

    for size in room_sizes:
        # Random specialty priority
        spec_map = {
            s: random.choice(SPECIALISM_PRIORITY_OPTIONS)
            for s in specialisms
        }

        rooms.append({
            "id": f"R{room_id}",
            "capacity": size,
            "gender_policy": random.choice(GENDER_POLICY_OPTIONS),
            "age_policy": random.choice(AGE_POLICY_OPTIONS),
            "specialisms": spec_map,
            "props": [random.randint(0, 1) for _ in range(4)]
        })

        room_id += 1

    return rooms


def validate_instance(
    patients: List[Dict],
    rooms: List[Dict],
    max_rooms: int
) -> bool:
    """Check instance validity for the patients, rooms, and room-count limit."""
    n_patients = len(patients)
    assert sum(r["capacity"] for r in rooms) >= n_patients
    assert all(r["capacity"] in (1, 2, 4) for r in rooms)
    assert len(rooms) <= max_rooms + 1
    return True


def generate_pasp_instance(config: Dict) -> "PASPInstance":
    """Generate a PASP instance.
    
    The configuration supplies seed, num_patients, horizon, and specialisms.
    Raises ValueError when configuration parameters are invalid."""
    # Validate configuration parameters
    if not isinstance(config, dict):
        raise ValueError("配置参数必须是字典类型")

    # Set the random seed
    seed = config.get("seed", random.randint(0, 10000))
    if not isinstance(seed, int):
        raise ValueError("随机种子必须是整数")
    random.seed(seed)

    # Read configuration parameters
    n_patients = config.get("num_patients", DEFAULT_NUM_PATIENTS)
    if not isinstance(n_patients, int) or n_patients <= 0:
        raise ValueError("患者数量必须是正整数")

    horizon = config.get("horizon", DEFAULT_HORIZON)
    if not isinstance(horizon, int) or horizon < 2:
        raise ValueError("时间范围必须是至少为2的整数")

    specialisms = config.get("specialisms", DEFAULT_SPECIALISMS)
    if not isinstance(specialisms, list) or not specialisms:
        raise ValueError("专科列表必须是非空列表")

    # Generate patient and room data
    patients = generate_patients(n_patients, horizon, specialisms)
    rooms = generate_rooms(n_patients, specialisms)

    # Validate the instance
    max_rooms = max(1, n_patients // 2)
    validate_instance(patients, rooms, max_rooms)

    # Return the instance
    return PASPInstance(
        patients=patients,
        rooms=rooms,
        horizon=horizon
    )
