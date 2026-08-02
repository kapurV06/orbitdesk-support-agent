"""
Response-generation node.

Model: Qwen/Qwen2.5-1.5B-Instruct (revision: main).
Picked over a smaller 0.5B model because it follows the "only use
supplied evidence, cite source IDs" instruction far more reliably in
testing, and over a larger 3B+ model because 1.5B already loads in a
few seconds on CPU and comfortably fits the ~4GB VRAM budget of the
target machine (RTX 3050 laptop GPU) in fp16 -- a deliberate
hardware-aware trade-off. device_map="auto" lets it use CUDA if
available and fall back to CPU otherwise.

Design decision: this node is ONLY reached on the "answerable" path
(and on escalation, for summarizing evidence into a human-readable
note). Clarification / out-of-scope / safe-failure responses are
deterministic templates (see graph.py) -- letting a small local model
freely generate those risks exactly the kind of unsupported, invented
instruction the verification node has to catch, so we don't ask it to.
"""
from __future__ import annotations
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.state import AgentState

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
GENERATION_MODEL_REVISION = "main"

SYSTEM_PROMPT = (
    "You are the OrbitDesk support assistant. Answer ONLY using the evidence "
    "passages provided below. Every factual claim must be traceable to one of "
    "the passages. If the evidence does not fully support an instruction, say so "
    "instead of inventing steps. Do not claim to have performed any account "
    "action. Cite the source_id (e.g. KB-004 or CASE-1041) for each key claim "
    "inline in parentheses. Keep the answer concise and practical."
)

_pipe = None
_load_time_s: Optional[float] = None


def get_generator():
    global _pipe, _load_time_s
    if _pipe is None:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, revision=GENERATION_MODEL_REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            GENERATION_MODEL_NAME,
            revision=GENERATION_MODEL_REVISION,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        _pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        _load_time_s = time.time() - t0
    return _pipe


def _format_evidence(retrieved: list[dict]) -> str:
    lines = []
    for r in retrieved:
        lines.append(f"[{r['source_id']}] {r['passage']}")
    return "\n\n".join(lines) if lines else "(no evidence retrieved)"


def _build_messages(question: str, retrieved: list[dict], revision_feedback: Optional[str] = None):
    evidence_block = _format_evidence(retrieved)
    user_content = f"Evidence passages:\n{evidence_block}\n\nUser question:\n{question}"
    if revision_feedback:
        user_content += (
            f"\n\nYour previous answer failed verification for this reason: "
            f"{revision_feedback}\nProduce a corrected answer that fixes this."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generation_node(state: AgentState) -> dict:
    t0 = time.time()
    logs = list(state.get("logs", []))
    retrieved = state.get("retrieved", [])
    revision_count = state.get("revision_count", 0)

    # Demo hook for required test case 5 ("a case where the initial generated
    # answer fails verification"). A real ~1.5B model's failures are not
    # reliably reproducible on demand, so this flag deterministically strips
    # citations from the FIRST draft only, forcing verification to fail and
    # the retry path to fire -- the retry itself still calls the real model
    # normally. This is documented here and in README.md, not hidden.
    force_demo_failure = state.get("_demo_force_verification_failure", False) and revision_count == 0

    revision_feedback = None
    if revision_count > 0 and state.get("verification"):
        revision_feedback = "; ".join(state["verification"]["issues"])

    pipe = get_generator()
    messages = _build_messages(state["question"], retrieved, revision_feedback)
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    out = pipe(
        prompt,
        max_new_tokens=300,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )
    generated = out[0]["generated_text"][len(prompt):].strip()

    if force_demo_failure:
        for r in retrieved:
            generated = generated.replace(f"({r['source_id']})", "").replace(r["source_id"], "")
        logs.append("generation: demo hook stripped citations from first draft to force verification failure")

    latency_s = time.time() - t0
    logs.append(
        f"generation: revision={revision_count} latency={latency_s:.2f}s "
        f"chars={len(generated)}"
    )

    return {"draft_answer": generated, "logs": logs}
