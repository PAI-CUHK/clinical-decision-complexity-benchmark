"""Shared integer penalty rules for the PASP solver and answer validator."""

CP_WEIGHT_SCALE = 10
CP_WEIGHTS = {
    "gender_policy": 50,
    "age_policy": 100,
    "specialism": 50,
    "room_preference": 8,
    "room_transfer": 110,
}

ROOM_PREF_MAP = {"single-bed": 1, "double-bed": 2, "multi-bed": 4}


def segment_penalty_components(patient, room):
    """Return the policy violations charged once per continuous room segment."""
    gender_policy = str(room["gender_policy"]).upper()
    age_policy = str(room.get("age_policy", "any")).lower()
    age = patient["age"]
    specialisms = room["specialisms"]
    return {
        "gender_policy": int(gender_policy not in ("ANY", "D", patient["gender"].upper())),
        "age_policy": int(
            (age_policy == "adult" and not 18 <= age <= 64)
            or (age_policy == "elderly" and age < 65)
            or (age_policy in ("child", "pediatric") and age > 16)
        ),
        "specialism": int(not any(
            specialisms.get(specialism, float("inf")) < 5
            for specialism, _ in patient["specialism_plan"]
        )),
    }


def room_preference_units(patient, room):
    """Excess beds charged for every day of this patient's room assignment."""
    preference = patient.get("room_pref", 1)
    if isinstance(preference, str):
        preference = ROOM_PREF_MAP[preference.lower()]
    return max(0, room["capacity"] - preference)
