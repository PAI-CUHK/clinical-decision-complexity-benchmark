"""Plot labels indexed by historical result keys; names come from core.tasks."""
from core.tasks import TASKS

TASK_FULL_NAMES = {task.data_key: task.title for task in TASKS}
TASK_SHORT_NAMES = {task.data_key: task.abbreviation for task in TASKS}
