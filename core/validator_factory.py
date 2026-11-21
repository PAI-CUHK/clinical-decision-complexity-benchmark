"""
Validator factory

Return the validator implementation for a problem type
"""

from core.validator import Validator
from core.tasks import legacy_task_name


def get_validator(problem_name: str) -> Validator:
    """
    Return the validator for a problem name
    
    Args:
        problem_name: Problem name
        
    Returns:
        Validator: Validator instance
    """
    problem_name = legacy_task_name(problem_name)
    if problem_name == "PASP":
        from tasks.pra_cc.validator import PASPValidator
        return PASPValidator()
    elif problem_name == "ECG Pattern Detection":
        from tasks.rr_iapd.validator import ECGValidator
        return ECGValidator()
    elif problem_name == "Disease Diagnosis":
        from tasks.pc_mde.validator_o import DiseaseValidator
        return DiseaseValidator()
    elif problem_name == "Disease Diagnosis NP-complete":
        from tasks.pc_mde.validator_d import DiseaseDiagnosisNPCompleteValidator
        return DiseaseDiagnosisNPCompleteValidator()
    elif problem_name == "OR Scheduling":
        from tasks.scs.validator_o import ORValidator
        return ORValidator()
    elif problem_name == "OR Scheduling NP-complete":
        from tasks.scs.validator_d import ORSchedulingNPCompleteValidator
        return ORSchedulingNPCompleteValidator()
    elif problem_name == "ED Scheduling":
        from tasks.ed_tqm.validator import EDValidator
        return EDValidator()
    elif problem_name == "Nurse Scheduling":
        from tasks.hnr_d.validator import NurseValidator
        return NurseValidator()
    elif problem_name == "Medical Risk":
        from tasks.ecd_rs.validator import MedicalRiskValidator
        return MedicalRiskValidator()
    else:
        raise ValueError(f"Unknown problem name: {problem_name}")
