#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWS2 Risk Assessment Adapter

Translates a NEWS2 instance into the unified problem framework.
"""

from core import MedicalProblemAdapter, MedicalNPProblem, Constraint, Objective


class MedicalRiskAdapter(MedicalProblemAdapter):
    """NEWS2 Risk Assessment Adapter."""

    def generate_instance_config(self, level):
        """
        Generate instance configuration for a given difficulty level.

        Args:
            level: Difficulty level (1-10).

        Returns:
            dict: Instance configuration.
        """
        # Only num_patients varies with level; all other factors are fixed
        # across levels (see data_generation_protocol.py).
        num_patients = 10 + level * 3
        return {
            "num_patients": num_patients,
            "level": level,
        }

    def build(self, instance):
        problem = MedicalNPProblem(
            name="Medical Risk",
            problem_type="risk_assessment"
        )

        problem.instance_data = {
            "patients": instance.patients,
        }

        problem.description = (
            "Given physiological observations for one or more patients, calculate the "
            "National Early Warning Score 2 (NEWS2), determine the corresponding clinical "
            "risk category, and identify patients requiring urgent clinical review."
        )

        problem.clinical_context = {
            "description": (
                "The National Early Warning Score 2 (NEWS2) is a standardized physiological "
                "scoring system used to identify patients at risk of acute clinical deterioration. "
                "Each patient's NEWS2 score is calculated by assigning scores to physiological "
                "variables and summing the component scores. "
                "All required scoring rules are explicitly provided. "
                "Do not infer diagnoses, treatments, disease severity, or clinical conditions "
                "beyond the provided scoring rules."
            )
        }

        problem.decision_space = [
            "For each patient, calculate the NEWS2 score using the provided physiological observations and scoring rules.",
            "Determine exactly one clinical risk category for each patient.",
            "Determine whether urgent clinical review is required for each patient.",
            "Return the calculated NEWS2 scores, risk categories, and urgent review decisions.",
        ]

        num_patients = len(instance.patients)
        problem.metadata = {
            "num_patients": num_patients,
            "complexity": "P",
            "problem_type": "risk_assessment",
        }

        for patient_id, patient_info in instance.patients.items():
            problem.add_entity("patients", {
                "id": patient_id,
                "info": patient_info,
            })

        # ========= Definitions and Rules =========
        problem.add_constraint(Constraint(
            name="rule_rr",
            description="Respiratory Rate scoring: RR ≤ 8 → +3; RR 9-11 → +1; RR 12-20 → +0; RR 21-24 → +2; RR ≥ 25 → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "respiratory_rate"}
        ))
        problem.add_constraint(Constraint(
            name="rule_spo2_scale1",
            description="SpO₂ Scale 1 scoring: SpO₂ ≥96 → +0; SpO₂ 94-95 → +1; SpO₂ 92-93 → +2; SpO₂ ≤91 → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "spo2", "scale": 1}
        ))
        problem.add_constraint(Constraint(
            name="rule_spo2_scale2",
            description="SpO₂ Scale 2 scoring: SpO₂ ≥97 → +3; SpO₂ 95-96 → +2; SpO₂ 93-94 → +1; SpO₂ 88-92 → +0; SpO₂ 86-87 → +1; SpO₂ 84-85 → +2; SpO₂ ≤83 → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "spo2", "scale": 2}
        ))
        problem.add_constraint(Constraint(
            name="rule_spo2_scale_selection",
            description=(
                "SpO₂ Scale Selection: the scale is determined by the patient's `conditions` field, "
                "NOT given separately. If `conditions` contains COPD or hypercapnic_resp_failure, "
                "use SpO₂ Scale 2; otherwise use SpO₂ Scale 1. The chosen scale is patient-level "
                "and applies to every snapshot of that patient."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scale_selection", "feature": "spo2"}
        ))
        problem.add_constraint(Constraint(
            name="rule_oxygen",
            description="Supplemental Oxygen scoring: Room Air → +0; Supplemental Oxygen → +2.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "supplemental_oxygen"}
        ))
        problem.add_constraint(Constraint(
            name="rule_temp",
            description="Temperature scoring: ≤35.0 → +3; 35.1-36.0 → +1; 36.1-38.0 → +0; 38.1-39.0 → +1; ≥39.1 → +2.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "temperature"}
        ))
        problem.add_constraint(Constraint(
            name="rule_sbp",
            description="Systolic Blood Pressure scoring: ≤90 → +3; 91-100 → +2; 101-110 → +1; 111-219 → +0; ≥220 → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "systolic_bp"}
        ))
        problem.add_constraint(Constraint(
            name="rule_hr",
            description="Heart Rate scoring: ≤40 → +3; 41-50 → +1; 51-90 → +0; 91-110 → +1; 111-130 → +2; ≥131 → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "heart_rate"}
        ))
        problem.add_constraint(Constraint(
            name="rule_cons",
            description="Consciousness (ACVPU) scoring: Alert → +0; New Confusion, Voice, Pain, Unresponsive → +3.",
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "consciousness"}
        ))
        problem.add_constraint(Constraint(
            name="rule_total",
            description=(
                "Total NEWS2 score = sum of the seven component scores, where each component "
                "score is the MAXIMUM (worst) score that component attains across ALL of the "
                "patient's snapshots."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "scoring_rule", "feature": "total"}
        ))
        problem.add_constraint(Constraint(
            name="rule_multisnapshot_aggregation",
            description=(
                "Multi-snapshot aggregation: each patient has K timestamped snapshots. For each "
                "of the seven physiological components, compute the score at every snapshot and "
                "take the maximum (worst) across snapshots; the total NEWS2 score is the sum of "
                "these seven component maxima: total = Σ_components max_across_snapshots(score). "
                "This is NOT the same as picking the single worst snapshot and summing its "
                "components (max_snapshots Σ_components score). Deterioration may be spread across "
                "different snapshots and different components — each component's worst value must "
                "be accumulated independently."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "aggregation", "feature": "multisnapshot"}
        ))
        problem.add_constraint(Constraint(
            name="rule_risk_category",
            description=(
                "Clinical Risk Category rules: "
                "NEWS2 = 0 → Low Risk; "
                "NEWS2 = 1-4 → Low Risk; "
                "NEWS2 = 3 in any single physiological parameter → Low-Medium Risk; "
                "NEWS2 = 5-6 → Medium Risk; "
                "NEWS2 >= 7 → High Risk. "
                "If multiple rules apply, assign the highest applicable risk category."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "risk_category"}
        ))
        problem.add_constraint(Constraint(
            name="rule_urgent",
            description=(
                "Urgent Clinical Review: Required if NEWS2 score >= 5 OR any single "
                "physiological variable (RR, SpO2, Temperature, SBP, HR) receives +3. "
                "Supplemental Oxygen and Consciousness are NOT physiological variables for this criterion."
            ),
            level="clinical", weight=0.0, category="definition",
            semantics={"type": "urgent_review"}
        ))

        # ========= Validity Constraints =========
        problem.add_constraint(Constraint(
            name="validity_use_rules",
            description="Use only the scoring rules explicitly provided.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_spo2_scale",
            description=(
                "Determine the SpO₂ scale from the patient's `conditions` field: if conditions "
                "contain COPD or hypercapnic_resp_failure, use Scale 2; otherwise use Scale 1. "
                "Do not mix scales within a patient."
            ),
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_multisnapshot_aggregation",
            description=(
                "For each patient, aggregate snapshots by taking the per-component maximum "
                "(worst) score across all snapshots, then sum the seven component maxima. "
                "Do NOT sum components within a single snapshot first."
            ),
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_no_age",
            description="Age must not contribute to the NEWS2 score.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_one_score",
            description="Every patient must receive exactly one NEWS2 score.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_one_category",
            description="Every patient must receive exactly one clinical risk category.",
            level="clinical", weight=0.0, category="validity",
        ))
        problem.add_constraint(Constraint(
            name="validity_no_diagnosis",
            description="Do not infer diagnoses, treatments, or disease severity.",
            level="clinical", weight=0.0, category="validity",
        ))

        # ========= Objectives =========
        problem.add_objective(Objective(
            name="calculate_news2_accurately",
            description=(
                "Primary objective: Accurately calculate the NEWS2 score for every patient. "
                "Secondary objectives: Determine the correct clinical risk category; "
                "Correctly identify all patients requiring urgent clinical review."
            ),
            direction="maximize", weight=1.0, priority=1,
            semantics={"type": "accuracy", "scope": "all_patients"}
        ))

        return problem
