class Constraint:
    """
    Domain-agnostic constraint with semantic structure.
    """

    def __init__(
        self,
        name: str,
        description: str,
        level: str = "base",   # base | hard | soft
        weight: float = 1.0,
        category: str = None,
        scope: str = None,
        semantics: dict = None
    ):
        """
        level:
          - base : independent feasibility constraints (P)
          - hard : global feasibility constraints (NP-complete)
          - soft : optimization objectives (NP-hard)
          - clinical : clinical priority rules and medical constraints
        """
        assert level in {"base", "hard", "soft", "clinical"}

        self.name = name
        self.description = description
        self.level = level
        self.weight = weight
        self.category = category
        self.scope = scope
        self.semantics = semantics or {}

    @property
    def hard(self):
        return self.level in {"base", "hard"}

    def describe(self):
        tag = self.level.upper()
        return f"[{tag}] {self.description}"
