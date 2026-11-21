class Action:
    """
    Abstract decision variable.
    Represents whether a medical action is taken.
    """

    def __init__(self, name, description, metadata=None):
        self.name = name                  # Unique identifier for the NP structure
        self.description = description    # Medical semantics supplied to the LLM
        self.metadata = metadata or {}    # Optional indices (p, r, t)

    def serialize(self):
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }
