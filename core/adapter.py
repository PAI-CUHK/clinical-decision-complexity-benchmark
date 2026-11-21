from abc import ABC, abstractmethod
from .problem import MedicalNPProblem


class MedicalProblemAdapter(ABC):
    """Base class for medical problem adapters"""
    
    @abstractmethod
    def build(self, instance) -> MedicalNPProblem:
        """
        Build a medical problem
        
        Args:
            instance: Source problem instance
            
        Returns:
            MedicalNPProblem: Constructed problem
        """
        pass
    
    
    