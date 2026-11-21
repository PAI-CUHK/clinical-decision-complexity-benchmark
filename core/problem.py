class MedicalNPProblem:
    def __init__(self, name, problem_type=None, validator=None):
        self.name = name
        self.problem_type = problem_type  # Problem type, such as "scheduling", "diagnosis", "risk", etc.
        self.description = ""  # Problem description

        # Do not predefine medical roles
        # Keys are strings; values are lists of entities
        self.entities = {}

        self.actions = []
        self.constraints = []
        self.objectives = []
        self.objective_mode = "single"
        self.metadata = {}  # Problem metadata
        self.instance_data = {}  # Source instance data
        self.validator = validator  # Validator

    def add_entity_type(self, entity_type):
        if entity_type not in self.entities:
            self.entities[entity_type] = []

    def add_entity(self, entity_type, entity):
        self.add_entity_type(entity_type)
        self.entities[entity_type].append(entity)

    def add_action(self, action):
        self.actions.append(action)

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def add_objective(self, objective):
        self.objectives.append(objective)