#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
MAIN_CODE_DIR = THIS_DIR.parents[1]
RUNNER = THIS_DIR / "run_llm_fewshot.py"
DEFAULT_OUTPUT_DIR = THIS_DIR / "few_shot_results"
DEFAULT_LOG_DIR = THIS_DIR / "run_logs" / "llm_aligned"


def make_env(gpu: str | None) -> dict:
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def build_cmd(args: argparse.Namespace, model: str) -> list[str]:
    cmd = [
        sys.executable,
        str(RUNNER),
        "--model",
        model,
        "--datasets",
        args.datasets,
        "--shots",
        args.shots,
        "--prompt-style",
        args.prompt_style,
        "--output-dir",
        str(args.output_dir),
        "--local-files-only",
    ]
    if args.skip_existing:
        cmd.append("--skip-existing")
    return cmd


def run_one(args: argparse.Namespace, model: str, gpu: str | None) -> subprocess.Popen:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"{model}_aligned.log"
    log_fh = log_path.open("w")
    cmd = build_cmd(args, model)
    print(f"[llm-aligned] start model={model} gpu={gpu} log={log_path}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=str(MAIN_CODE_DIR),
        env=make_env(gpu),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen,llama")
    parser.add_argument("--datasets", default="SMP2020,SST-5,TweetEval")
    parser.add_argument("--shots", default="0,1,3,5")
    parser.add_argument("--prompt-style", choices=["no-cot", "cot", "both"], default="both")
    parser.add_argument("--gpus", default=os.environ.get("HIPRO_LLM_GPUS", "2,3"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    models = [item.strip() for item in args.models.split(",") if item.strip()]
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.sequential or len(models) == 1:
        for i, model in enumerate(models):
            gpu = gpus[i % len(gpus)] if gpus else None
            proc = run_one(args, model, gpu)
            rc = proc.wait()
            if rc != 0:
                raise SystemExit(rc)
        return

    running = []
    for i, model in enumerate(models):
        gpu = gpus[i % len(gpus)] if gpus else None
        running.append((model, run_one(args, model, gpu)))

    failed = []
    for model, proc in running:
        rc = proc.wait()
        if rc != 0:
            failed.append((model, rc))
    if failed:
        print(f"[llm-aligned] failed={failed}", flush=True)
        raise SystemExit(1)
    print("[llm-aligned] all done", flush=True)


if __name__ == "__main__":
    main()
