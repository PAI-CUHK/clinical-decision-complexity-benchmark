"""Load the intermediate inputs used by the final figures."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reports/plot_data"


def load_data(name, path=None):
    path = Path(path) if path is not None else DATA_DIR / f"{name}.json"
    with path.open(encoding="utf-8") as stream:
        prepared = json.load(stream)
    if prepared.get("schema_version") != 1 or prepared.get("figure") != name:
        raise ValueError(f"Unsupported intermediate data schema: {path}")
    return prepared["data"]
