"""Benchmark task abbreviations, titles, complexity classes, and internal identifiers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    abbreviation: str
    title: str
    legacy_name: str
    data_key: str
    complexity: str


TASKS = (
    Task('RR-IAPD', 'RR-interval pattern detection', 'ECG Pattern Detection', 'ECG_Pattern_Detection', 'P'),
    Task('ED-TQM', 'Emergency triage queue management', 'ED Scheduling', 'ED_Scheduling', 'P'),
    Task('ECD-RS', 'Early deterioration risk stratification', 'Medical Risk', 'Medical_Risk', 'P'),
    Task('PC-MDE (D)', 'Multi-disease explanation decision', 'Disease Diagnosis NP-complete', 'Disease_Diagnosis_NP-complete', 'NP-complete'),
    Task('HNR (D)', 'Nurse rostering feasibility', 'Nurse Scheduling', 'Nurse_Scheduling', 'NP-complete'),
    Task('SCS (D)', 'Surgical scheduling decision', 'OR Scheduling NP-complete', 'OR_Scheduling_NP-complete', 'NP-complete'),
    Task('PC-MDE (O)', 'Multi-disease explanation optimization', 'Disease Diagnosis', 'Disease_Diagnosis', 'NP-hard'),
    Task('SCS (O)', 'Surgical scheduling optimization', 'OR Scheduling', 'OR_Scheduling', 'NP-hard'),
    Task('PRA-CC', 'Patient-to-room allocation', 'PASP', 'PASP', 'NP-hard'),
)

_BY_NAME = {name.casefold(): task for task in TASKS
            for name in (task.abbreviation, task.legacy_name, task.data_key)}


def resolve_task(name: str) -> Task:
    """Resolve a task identifier; fail for unknown names."""
    try:
        return _BY_NAME[name.strip().casefold()]
    except KeyError:
        raise ValueError(f'Unknown task: {name!r}') from None


def legacy_task_name(name: str) -> str:
    """Return the internal identifier used by benchmark storage and dispatch."""
    return resolve_task(name).legacy_name
