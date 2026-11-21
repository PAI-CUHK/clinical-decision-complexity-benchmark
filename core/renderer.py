#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base renderer interface shared by all task renderers."""


class BaseLLMRenderer:
    """Base model prompt renderer with a shared problem-rendering interface."""

    def _get_professional_task_name(self, problem_name):
        """Return the professional medical task name for a problem name."""
        # Professional medical name mapping
        name_mapping = {
            "ED Scheduling": "Emergency Department Patient Triage",
            "OR Scheduling": "Operating Room Scheduling",
            "OR Scheduling NP-complete": "Operating Room Scheduling Optimization",
            "Nurse Scheduling": "Nurse Shift Scheduling",
            "Nurse Scheduling NP-complete": "Advanced Nurse Shift Scheduling",
            "Disease Diagnosis": "Disease Diagnosis",
            "Disease Diagnosis NP-complete": "Multiple Disease Differential Diagnosis",
            "PASP": "Patient Assignment and Bed Planning",
            "ECG Pattern Detection": "ECG RR-Interval Pattern Recognition",
            "ECG RR Interval Pattern Detection": "ECG RR-Interval Pattern Recognition",
            "Medical Risk": "Medical Risk Assessment"
        }
        
        # Remove level information
        base_name = problem_name.split(" Level ")[0]
        
        # Return the mapped professional name, or the original if unmapped
        return name_mapping.get(base_name, base_name)
    
    def render_task_header(self, problem):
        """Render the task header as a list of lines."""
        lines = []
        
        # Task name using the professional medical name
        task_name = self._get_professional_task_name(problem.name)
        lines.append(f"You are given a clinical decision-making task: {task_name}.")
        lines.append("")
        
        # Problem description from problem.description
        if hasattr(problem, 'description') and problem.description:
            lines.append(problem.description)
        
        return lines
    
    def render_clinical_context(self, problem):
        """Render the clinical context as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# CLINICAL CONTEXT")
        
        if hasattr(problem, 'clinical_context') and problem.clinical_context:
            if isinstance(problem.clinical_context, dict):
                lines.append(problem.clinical_context.get('description', ''))
            else:
                lines.append(problem.clinical_context)
        
        return lines
    
    def render_instance_data(self, problem):
        """Render the instance data as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# INSTANCE DATA")
        lines.append("")
        lines.append("[Instance-specific data]")
        return lines
    
    def render_decision_space(self, problem):
        """Render the decision space as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# DECISION SPACE")
        
        # Get decision space information from the problem
        if hasattr(problem, 'decision_space') and problem.decision_space:
            if isinstance(problem.decision_space, list):
                for item in problem.decision_space:
                    lines.append(item)
                    lines.append("")
            else:
                lines.append(problem.decision_space)
        
        return lines
    
    def render_definitions_and_rules(self, problem):
        """Render definitions and rules as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# DEFINITIONS AND RULES")
        
        # Extract definitions and rules, including penalties, from problem.constraints
        definitions = []
        if hasattr(problem, 'constraints') and problem.constraints:
            for constraint in problem.constraints:
                if hasattr(constraint, 'category') and (constraint.category == 'definition' or constraint.category == 'penalty'):
                    definitions.append(constraint)
        
        if definitions:
            for constraint in definitions:
                if hasattr(constraint, 'weight') and constraint.weight > 0 and constraint.category == 'penalty':
                    if constraint.name == 'cancellation_penalty':
                        lines.append(constraint.description)
                    else:
                        lines.append(f"{constraint.description} (penalty weight: {constraint.weight})")
                else:
                    lines.append(constraint.description)
                lines.append("")
        
        return lines
    
    def render_validity_constraints(self, problem):
        """Render validity constraints as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# VALIDITY CONSTRAINTS")
        
        # Extract validity constraints from problem.constraints
        constraints = []
        if hasattr(problem, 'constraints') and problem.constraints:
            for constraint in problem.constraints:
                if hasattr(constraint, 'category') and constraint.category == 'validity':
                    constraints.append(constraint.description)
        
        if constraints:
            for const in constraints:
                lines.append(const)
                lines.append("")
        
        return lines
    
    def render_task_objective(self, problem):
        """Render the task objective as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# TASK OBJECTIVE")
        
        # Extract task objectives from problem.objectives
        objectives = []
        if hasattr(problem, 'objectives') and problem.objectives:
            # Sort by priority
            obj_list = []
            for obj in problem.objectives:
                priority = getattr(obj, 'priority', 999)
                obj_list.append((priority, obj.description))
            obj_list.sort(key=lambda x: x[0])
            objectives = [desc for _, desc in obj_list]
        
        if objectives:
            for obj in objectives:
                lines.append(obj)
                lines.append("")
        
        return lines
    
    def render_output_format(self, problem):
        """Render the required output format as a list of lines."""
        lines = []
        lines.append("\n---")
        lines.append("\n# OUTPUT FORMAT STRICTLY REQUIRED")
        lines.append("")
        return lines

    def render(self, problem):
        """Render a MedicalNPProblem as a model prompt string."""
        raise NotImplementedError("Subclasses must implement render()")
