#!/usr/bin/env python3
"""Generate paper figures from markdown inputs collected under HiPro-loRA/."""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def main():
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(
        [sys.executable, str(PROJECT_DIR / "create_Image.py")],
        cwd=str(PROJECT_DIR),
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
