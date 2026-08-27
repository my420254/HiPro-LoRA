#!/usr/bin/env python3
"""Optional sequential entry point for all strict experiment groups."""

import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ORDER = [
    "run_table2_strict.py",
    "run_table3_strict.py",
    "run_sensitivity_gate_strict.py",
]


def main():
    for script_name in ORDER:
        script = RUN_DIR / script_name
        print(f"[run_all] start {script_name}", flush=True)
        subprocess.run([sys.executable, str(script)], cwd=str(RUN_DIR.parents[1]), check=True)
        print(f"[run_all] done  {script_name}", flush=True)


if __name__ == "__main__":
    main()
