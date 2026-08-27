#!/usr/bin/env python3
"""Run strict sensitivity and gate scripts for all three datasets."""

from pathlib import Path

from _runner import PROJECT_DIR, Task, run_tasks


SCRIPTS = [
    ("sst5_sensitivity",     "sensitivity_gate_strict/sst5_sensitivity.py"),
    ("sst5_gate",            "sensitivity_gate_strict/sst5_gate.py"),
    ("smp2020_sensitivity",  "sensitivity_gate_strict/smp2020_sensitivity.py"),
    ("smp2020_gate",         "sensitivity_gate_strict/smp2020_gate.py"),
    ("tweeteval_sensitivity","sensitivity_gate_strict/tweeteval_sensitivity.py"),
    ("tweeteval_gate",       "sensitivity_gate_strict/tweeteval_gate.py"),
]


def main():
    tasks = [Task(name, PROJECT_DIR / Path(script)) for name, script in SCRIPTS]
    run_tasks(tasks, "sensitivity_gate_strict")


if __name__ == "__main__":
    main()
