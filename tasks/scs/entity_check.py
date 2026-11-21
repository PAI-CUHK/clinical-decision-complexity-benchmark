"""Entity consistency of clinical and symbolic SCS answers, without solving."""


def or_entity_errors(instance, answer):
    """Return HE diagnostics; an explicit cancellation list must be exhaustive.

    Omitted declaration fields retain the benchmark's inferred-cancellation
    convention. An explicit empty list, however, asserts no cancellations.
    Schedule multiplicity/capacity remain hard-constraint checks elsewhere.
    """
    if not isinstance(answer, dict) or not isinstance(answer.get("schedule"), dict):
        return []
    surgeries = {s["id"] for s in instance.get("surgeries", instance.get("tasks", []))}
    days = {d["id"] for d in instance.get("or_days", instance.get("resources", []))}
    errors, scheduled = [], set()
    for day, ids in answer["schedule"].items():
        if day not in days:
            errors.append(f"unknown resource: {day}")
        if isinstance(ids, list):
            for sid in ids:
                if isinstance(sid, str):
                    scheduled.add(sid)
                    if sid not in surgeries:
                        errors.append(f"unknown scheduled entity: {sid}")
    for field in ("cancelled", "unassigned"):
        if field not in answer:
            continue
        values = answer[field]
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
            continue  # A malformed field is a format error, not an invented ID.
        declared = set(values)
        for sid in sorted(surgeries - scheduled - declared):
            errors.append(f"missing entity from schedule and {field}: {sid}")
        for sid in sorted(scheduled & declared):
            errors.append(f"scheduled and {field}: {sid}")
        for sid in sorted(declared - surgeries):
            errors.append(f"unknown {field} entity: {sid}")
        if len(values) != len(declared):
            errors.append(f"duplicate entities in {field}")
    return errors
