#!/usr/bin/env python3
"""Small multi-GPU runner used by project startup scripts."""

import os
import subprocess
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
MAIN_CODE_DIR = PROJECT_DIR.parent


@dataclass(frozen=True)
class Task:
    name: str
    script: Path


def parse_gpu_slots(default="0:2,1:2,2:3,3:3"):
    raw = os.environ.get("HIPRO_RUN_GPU_SLOTS", default)
    slots = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        gpu, cap = item.split(":", 1)
        slots[gpu.strip()] = int(cap.strip())
    gpus_raw = os.environ.get("HIPRO_RUN_GPUS")
    if gpus_raw:
        gpus = [x.strip() for x in gpus_raw.split(",") if x.strip()]
        slots = {gpu: slots.get(gpu, 1) for gpu in gpus}
    return slots


def make_env(gpu):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_tasks(tasks, suite_name, gpu_slots=None):
    gpu_slots = gpu_slots or parse_gpu_slots()
    max_parallel = int(os.environ.get("HIPRO_RUN_MAX_PARALLEL", str(sum(gpu_slots.values()))))
    log_dir = Path(os.environ.get("HIPRO_RUN_LOG_DIR", str(PROJECT_DIR / "run_logs" / suite_name)))
    log_dir.mkdir(parents=True, exist_ok=True)

    queue = deque(tasks)
    running = []
    active = Counter()
    completed = []
    failed = []

    print(f"[runner:{suite_name}] tasks={len(tasks)} gpu_slots={gpu_slots} max_parallel={max_parallel}", flush=True)
    while queue or running:
        while queue and len(running) < max_parallel:
            available = [gpu for gpu, cap in gpu_slots.items() if active[gpu] < cap]
            if not available:
                break
            gpu = min(available, key=lambda item: active[item])
            task = queue.popleft()
            if not task.script.exists():
                raise FileNotFoundError(task.script)
            log_path = log_dir / f"{task.name}.log"
            log_fh = log_path.open("w")
            process = subprocess.Popen(
                [sys.executable, str(task.script)],
                cwd=str(MAIN_CODE_DIR),
                env=make_env(gpu),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[gpu] += 1
            running.append((process, log_fh, task, gpu, log_path))
            print(f"[runner:{suite_name}] start {task.name} GPU{gpu} pid={process.pid}", flush=True)

        still_running = []
        for process, log_fh, task, gpu, log_path in running:
            rc = process.poll()
            if rc is None:
                still_running.append((process, log_fh, task, gpu, log_path))
                continue
            log_fh.close()
            active[gpu] -= 1
            record = {"task": task.name, "gpu": gpu, "returncode": rc, "log": str(log_path)}
            if rc == 0:
                completed.append(record)
                print(f"[runner:{suite_name}] done {task.name} GPU{gpu}", flush=True)
            else:
                failed.append(record)
                print(f"[runner:{suite_name}] FAIL {task.name} GPU{gpu} rc={rc} log={log_path}", flush=True)
        running = still_running

        if queue or running:
            time.sleep(5)

    print(f"[runner:{suite_name}] completed={len(completed)} failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)
