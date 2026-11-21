"""Render exactly the figures currently retained in plot/ from saved data.

Only the saved intermediate data are read.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .plot_data import DATA_DIR, ROOT


def rendering_interpreters(args, env, required=("3.9.2", "3.11.0")):
    """The retained WA figures use 3.11; the other figures use 3.9.2.

    Match the original renderers before writing any figure. Allow explicit
    overrides for other machines, or discover the current Python environment.
    """
    candidates = dict.fromkeys(filter(None, (
        sys.executable, shutil.which("python"),
    )))
    versions = {}
    for executable in candidates:
        if not Path(executable).is_file():
            continue
        probe = subprocess.run(
            [executable, "-c", "from importlib.metadata import version; "
             "print(version('matplotlib'))"],
            cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        if probe.returncode == 0:
            versions.setdefault(probe.stdout.strip(), executable)
    for version, executable in (("3.9.2", args.python_mpl39),
                                ("3.11.0", args.python_mpl311)):
        if version not in required:
            continue
        if executable:
            probe = subprocess.check_output(
                [executable, "-c", "from importlib.metadata import version; "
                 "print(version('matplotlib'))"], cwd=ROOT, env=env, text=True).strip()
            if probe != version:
                raise ValueError(f"{executable}: expected Matplotlib {version}, found {probe}")
            versions[version] = executable
        if version not in versions:
            flag = "--python-mpl39" if version == "3.9.2" else "--python-mpl311"
            raise RuntimeError(f"Matplotlib {version} is required to reproduce the retained "
                               f"figures exactly. Supply its Python interpreter with {flag}.")
    return versions


def commands(data_dir, output_dir):
    semantics = output_dir / "medical_semantics_wa"
    return [
        ["analysis.plot_model_performance", "--data", str(data_dir / "model_performance.json"),
         "--out", str(output_dir / "model_performance.png")],
        ["analysis.plot_problem_level_accuracy", "--data", str(data_dir / "problem_level_accuracy.csv"),
         "--out", str(output_dir / "problem_level_accuracy.png")],
        ["analysis.plot_error_composition", "--data", str(data_dir / "error_composition.json"),
         "--out", str(output_dir / "error_composition.png")],
        ["analysis.build_medical_metrics", "--data", str(data_dir / "medical_metrics.json"),
         "--out", str(output_dir / "medical_metrics.png")],
        ["analysis.plot_pc_mde_semantics_sd", "--data",
         str(data_dir / "pc_mde_semantics_sd.json"), "--output-dir", str(output_dir)],
        ["analysis.plot_medical_semantics_wa", "--data", str(data_dir / "medical_semantics_wa.json"),
         "--output-dir", str(semantics)],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "plot")
    parser.add_argument("--python-mpl39", help="Python with Matplotlib 3.9.2")
    parser.add_argument("--python-mpl311", help="Python with Matplotlib 3.11.0")
    args = parser.parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/mednp-mpl")
    interpreters = rendering_interpreters(args, env)
    for command in commands(data_dir, output_dir):
        version = "3.11.0" if command[0] == "analysis.plot_medical_semantics_wa" else "3.9.2"
        executable = interpreters[version]
        print("Running:", executable, "-m", *command, flush=True)
        subprocess.run([executable, "-m", *command], cwd=ROOT, env=env, check=True)
    print("Rendered the retained figures from intermediate data.")


if __name__ == "__main__":
    main()
