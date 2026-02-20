#!/usr/bin/env python3
"""Run eval_matrix with live output for one-model debugging.

This wrapper mirrors the normal matrix runner path by invoking:
    scripts/eval_matrix.py run

It streams stdout/stderr live (including tqdm progress), writes logs/artifacts
to a temporary run directory, and removes that directory at the end unless
--keep-artifacts is provided.
"""

from __future__ import annotations

import argparse
import os
import pty
import select
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_DATASET_TYPES = ["mmlu_pro", "gpqa", "arc_easy", "arc_challenge"]
DEFAULT_DISTRACTOR_SOURCES = ["scratch", "dhuman", "dmodel"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live debug wrapper around scripts/eval_matrix.py run",
    )
    parser.add_argument("--model", required=True, help="Evaluation model name")
    parser.add_argument("--dataset-path", required=True, help="Unified dataset path")
    parser.add_argument(
        "--generator-dataset-label",
        required=True,
        help="Generator label (e.g. opus, gpt-4.1, gpt-5.2)",
    )
    parser.add_argument("--preset", default="core16", help="Matrix preset")
    parser.add_argument(
        "--dataset-types",
        nargs="+",
        default=DEFAULT_DATASET_TYPES,
        choices=DEFAULT_DATASET_TYPES,
    )
    parser.add_argument(
        "--distractor-sources",
        nargs="+",
        default=DEFAULT_DISTRACTOR_SOURCES,
        choices=DEFAULT_DISTRACTOR_SOURCES,
    )
    parser.add_argument("--eval-mode", default="behavioral", choices=["accuracy", "behavioral"])
    parser.add_argument("--limit", type=int, default=10, help="Entries per config")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--workpack-format", default="parquet", choices=["none", "parquet", "arrow"])
    parser.add_argument("--workpack-root", default="", help="Shared workpack root path")
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--choices-only", action="store_true")
    parser.add_argument("--vllm-max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=None)
    parser.add_argument("--vllm-enable-chunked-prefill", type=int, choices=[0, 1], default=None)
    parser.add_argument(
        "--run-root",
        default="",
        help="Parent directory for temporary run dir (default: ./results/debug_live)",
    )
    parser.add_argument("--keep-artifacts", action="store_true", help="Do not delete run dir")
    parser.add_argument(
        "--keep-on-failure",
        action="store_true",
        help="Keep artifacts when command fails",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    cmd: list[str] = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/eval_matrix.py",
        "run",
        "--preset",
        args.preset,
        "--model",
        args.model,
        "--dataset-path",
        args.dataset_path,
        "--generator-dataset-label",
        args.generator_dataset_label,
        "--dataset-types",
        *args.dataset_types,
        "--distractor-sources",
        *args.distractor_sources,
        "--eval-mode",
        args.eval_mode,
        "--limit",
        str(args.limit),
        "--max-tokens",
        str(args.max_tokens),
        "--save-interval",
        str(args.save_interval),
        "--keep-checkpoints",
        str(args.keep_checkpoints),
        "--workpack-format",
        args.workpack_format,
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--output-dir",
        str(output_dir),
    ]
    if args.workpack_root:
        cmd.extend(["--workpack-root", args.workpack_root])
    if args.choices_only:
        cmd.append("--choices-only")
    if args.vllm_max_num_batched_tokens is not None:
        cmd.extend(["--vllm-max-num-batched-tokens", str(args.vllm_max_num_batched_tokens)])
    if args.vllm_max_num_seqs is not None:
        cmd.extend(["--vllm-max-num-seqs", str(args.vllm_max_num_seqs)])
    if args.vllm_enable_chunked_prefill is not None:
        cmd.extend(["--vllm-enable-chunked-prefill", str(args.vllm_enable_chunked_prefill)])
    return cmd


def stream_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["UV_NO_SYNC"] = "1"

    master_fd, slave_fd = pty.openpty()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=slave_fd,
        stderr=slave_fd,
        text=False,
        close_fds=True,
    )
    os.close(slave_fd)

    with open(log_path, "wb") as logfile:
        while True:
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in readable:
                try:
                    chunk = os.read(master_fd, 8192)
                except OSError:
                    chunk = b""
                if chunk:
                    os.write(sys.stdout.fileno(), chunk)
                    logfile.write(chunk)
                elif proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                break

    os.close(master_fd)
    return int(proc.wait())


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if args.run_root:
        run_root = Path(args.run_root).expanduser().resolve()
    else:
        run_root = (repo_root / "results" / "debug_live").resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f"matrix_live_{ts}_",
            dir=str(run_root),
        )
    )
    log_path = temp_dir / "live_output.log"
    cmd = build_command(args, temp_dir)

    print("=== Matrix Live Debug Run ===")
    print(f"Repo root: {repo_root}")
    print(f"Run dir: {temp_dir}")
    print(f"Log path: {log_path}")
    print("Command:")
    print("  " + " ".join(cmd))
    print()

    rc = stream_subprocess(cmd, repo_root, log_path)
    print()
    print(f"Exit code: {rc}")

    should_keep = args.keep_artifacts or (args.keep_on_failure and rc != 0)
    if should_keep:
        print(f"Keeping artifacts: {temp_dir}")
    else:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Deleted artifacts: {temp_dir}")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

