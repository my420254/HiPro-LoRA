#!/usr/bin/env python3
"""Run all Table 2 strict scripts from the self-contained project directory."""

from pathlib import Path

from _runner import PROJECT_DIR, Task, run_tasks


SCRIPTS = [
    ("smp2020_main", "table2_strict/smp2020_main_table2_strict.py"),
    ("smp2020_lora_adv", "table2_strict/smp2020_lora_adv_table2_strict.py"),
    ("sst5_main", "table2_strict/sst5_main_table2_strict.py"),
    ("sst5_lora_adv", "table2_strict/sst5_lora_adv_table2_strict.py"),
    ("tweeteval_main", "table2_strict/tweeteval_main_table2_strict.py"),
    ("tweeteval_lora_adv", "table2_strict/tweeteval_lora_adv_table2_strict.py"),
]


def main():
    tasks = [Task(name, PROJECT_DIR / Path(script)) for name, script in SCRIPTS]
    run_tasks(tasks, "table2_strict")


if __name__ == "__main__":
    main()
