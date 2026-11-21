class Objective:
    def __init__(
        self,
        name: str,
        description: str,
        direction: str = "minimize",
        weight: float = 1.0,
        priority: int = 1,
        semantics: dict = None
    ):
        """
        direction: minimize / maximize
        priority: objective priority (lower number = higher priority)
        semantics: penalty definition, trade-off meaning, etc.
        """
        self.name = name
        self.description = description
        self.direction = direction
        self.weight = weight
        self.priority = priority
        self.semantics = semantics or {}

    def describe(self):
        return f"{self.direction.upper()} {self.description} (w={self.weight}, priority={self.priority})"
