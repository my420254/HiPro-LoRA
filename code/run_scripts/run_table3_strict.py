#!/usr/bin/env python3
"""Run all Table 3 strict scripts from the self-contained project directory."""

from pathlib import Path

from _runner import PROJECT_DIR, Task, run_tasks


SCRIPTS = [
    ("smp2020_table3", "table3_strict/smp2020_table3_strict.py"),
    ("sst5_table3", "table3_strict/sst5_table3_strict.py"),
    ("tweeteval_table3", "table3_strict/tweeteval_table3_strict.py"),
]


def main():
    tasks = [Task(name, PROJECT_DIR / Path(script)) for name, script in SCRIPTS]
    run_tasks(tasks, "table3_strict")


if __name__ == "__main__":
    main()
