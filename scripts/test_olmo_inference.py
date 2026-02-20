#!/usr/bin/env python3
"""
Diagnostic: test OLMo-3-7B-Instruct inference via vLLM.

Tests both raw .generate() and .chat() approaches on a few MCQA prompts,
printing raw outputs, finish reasons, and token counts so we can see why
the model is producing empty or near-empty responses.

Usage (on a GPU node):
  uv run --no-sync python scripts/test_olmo_inference.py
"""

import os, sys, textwrap
from pathlib import Path

# ── env ──────────────────────────────────────────────────────────────────────
CACHE = os.environ.get("MODEL_CACHE_DIR", "/fs/nexus-scratch/adesai10/hub")
os.environ.setdefault("HF_HOME", CACHE)
os.environ.setdefault("MODEL_CACHE_DIR", CACHE)

MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MAX_TOKENS = 150

# A couple of MCQA prompts representative of what eval uses
PROMPTS = [
    textwrap.dedent("""\
        Answer the following multiple-choice question by selecting the letter \
of the correct answer.

Question: What is the capital of France?
A) Berlin
B) Madrid
C) Paris
D) Rome

Answer:"""),
    textwrap.dedent("""\
        Answer the following multiple-choice question by selecting the letter \
of the correct answer.

Question: Which element has atomic number 1?
A) Helium
B) Hydrogen
C) Lithium
D) Oxygen

Answer:"""),
]


def sep(title=""):
    w = 70
    if title:
        print(f"\n{'─'*3} {title} {'─'*(w-5-len(title))}")
    else:
        print("─" * w)


def show_output(label, outputs):
    for i, out in enumerate(outputs):
        tok = out.outputs[0]
        sep(f"{label} | prompt {i}")
        print(f"  finish_reason : {tok.finish_reason}")
        print(f"  token_ids     : {list(tok.token_ids)[:20]}{'...' if len(tok.token_ids)>20 else ''}")
        print(f"  output_tokens : {len(tok.token_ids)}")
        raw = repr(tok.text)
        print(f"  text          : {raw}")


def main():
    from vllm import LLM, SamplingParams

    print(f"Loading {MODEL_ID} ...")
    llm = LLM(
        model=MODEL_ID,
        download_dir=CACHE,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        trust_remote_code=True,
        seed=42,
    )
    print("Model loaded.\n")

    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # ── 1. Raw generate ───────────────────────────────────────────────────────
    sep("TEST 1: raw .generate() — no chat template")
    outputs_raw = llm.generate(PROMPTS, sp)
    show_output("raw generate", outputs_raw)

    # ── 2. Chat API ───────────────────────────────────────────────────────────
    sep("TEST 2: .chat() — applies chat template")
    messages = [[{"role": "user", "content": p}] for p in PROMPTS]
    if hasattr(llm, "chat"):
        outputs_chat = llm.chat(messages, sp)
        show_output("chat", outputs_chat)
    else:
        print("  llm.chat() not available on this vLLM version")

    # ── 3. Tokenizer check ────────────────────────────────────────────────────
    sep("TEST 3: tokenizer & chat template sanity")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE, trust_remote_code=True)
    print(f"  chat_template present: {tok.chat_template is not None}")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}],
        tokenize=False, add_generation_prompt=True,
    )
    print(f"  rendered (first 400 chars):\n{rendered[:400]}")

    sep("DONE")


if __name__ == "__main__":
    main()
