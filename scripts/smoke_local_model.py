#!/usr/bin/env python3
"""Minimal smoke test for local vLLM model load + single generation."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Allow direct script execution (`python scripts/smoke_local_model.py`) to import repo modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODEL_CACHE_DIR
from models import get_client

DEFAULT_PROMPT = (
    "Answer with exactly one letter (A/B/C/D).\n"
    "Question: What color is the clear daytime sky?\n"
    "A: Green\n"
    "B: Blue\n"
    "C: Red\n"
    "D: Yellow"
)


def _safe_name(value: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in value)


def _resolve_cached_snapshot(model_id: str, cache_root: Path) -> Path | None:
    """Resolve newest snapshot dir under HF cache for model_id, if present."""
    repo_dir = cache_root / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Model alias/name (e.g., Nanbeige/Nanbeige4.1-3B)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tokenizer-mode", default=None, choices=["auto", "slow"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--use-local-snapshot", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = _parser().parse_args()

    cache_root = Path(os.getenv("MODEL_CACHE_DIR", str(MODEL_CACHE_DIR))).expanduser()
    model_override = None
    if args.use_local_snapshot and "/" in args.model and not Path(args.model).exists():
        snapshot = _resolve_cached_snapshot(args.model, cache_root)
        if snapshot is not None:
            model_override = str(snapshot)
            print(f"Resolved cached snapshot for {args.model}: {snapshot}")
        else:
            print(
                f"No cached snapshot found for {args.model} under {cache_root}/hub. "
                "Will use model id directly."
            )

    client_kwargs: dict[str, object] = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
    }
    if model_override is not None:
        client_kwargs["model_id"] = model_override
    if args.dtype is not None:
        client_kwargs["dtype"] = args.dtype
    if args.max_model_len is not None:
        client_kwargs["max_model_len"] = args.max_model_len
    if args.tokenizer_mode is not None:
        client_kwargs["tokenizer_mode"] = args.tokenizer_mode
    if args.trust_remote_code is not None:
        client_kwargs["trust_remote_code"] = args.trust_remote_code
    if args.stop_token_id:
        client_kwargs["stop_token_ids"] = list(args.stop_token_id)

    print("=== Local Model Smoke Test ===")
    print(f"Model key: {args.model}")
    print(f"Client kwargs: {client_kwargs}")
    print(f"Cache root: {cache_root}")

    t0 = time.perf_counter()
    client = get_client(args.model, **client_kwargs)
    t1 = time.perf_counter()
    print(f"Client created in {t1 - t0:.2f}s: {client.name}")

    try:
        t2 = time.perf_counter()
        result = client.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        t3 = time.perf_counter()

        print(f"Generation done in {t3 - t2:.2f}s")
        print(f"Finish reason: {result.finish_reason}")
        answer = result.extract_answer()
        print(f"Extracted answer: {answer}")
        out = (result.text or "").strip()
        preview = out[:500]
        print("--- Output Preview ---")
        print(preview)
        print("--- End Output Preview ---")
    finally:
        unload = getattr(client, "unload", None)
        if callable(unload):
            unload()

    print("SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
