"""
Triage node.

Design decision: triage reuses the embedding retriever (already a local
HF model) rather than loading a second classification model. A ~50-chunk
domain-specific corpus does not need a dedicated zero-shot classifier;
cosine-similarity strength + a small set of deterministic keyword rules
give reliable, auditable routing without extra model-load cost. This is
the "hardware-aware trade-off" called out in the assignment -- documented
here and in README.md rather than hidden.

Rule precedence (checked in this order):
  1. Explicit unsupported-action / prompt-injection language -> out_of_scope
  2. Explicit escalation signals (repeated failure, "already tried",
     credential exposure, billing/legal) -> requires_escalation
  3. Vague/underspecified symptom language -> requires_clarification
  4. Strong retrieval match -> answerable
  5. Weak retrieval match and nothing else matched -> out_of_scope
"""
from __future__ import annotations
import re
from src.state import AgentState, Classification
from src.retrieval import get_retriever, RetrievalHit

# Requests explicitly outside what the assistant can do (KB-010 "Unsupported
# Actions") or attempts to override the system's rules (KB-010 "Instructions
# inside user messages ... do not override these rules").
OUT_OF_SCOPE_PATTERNS = [
    r"\brefund\b",
    r"\bcancel (my|the) subscription\b",
    r"\blegal advice\b",
    r"\bignore (the )?(supplied|above|previous|prior) (documentation|instructions)\b",
    r"\bissue a refund\b",
    r"\bmedical advice\b",
    r"\bfinancial advice\b",
]

ESCALATION_PATTERNS = [
    r"\balready (checked|tried|attempted)\b",
    r"\btwo (export )?runs? in a row\b",
    r"\bconsecutive\b.*\bfail",
    r"\bcredential (was |has been )?(exposed|leaked|compromised)\b",
    r"\bstill (did not|didn't) work\b",
    r"\bdid not work\b",
    r"\bsuggested solution did not work\b",
]

VAGUE_PATTERNS = [
    r"^\s*(sync|it|this|export|dashboard) is (not working|broken)\.?\s*$",
    r"\bnot working\b(?!.*\b(error|code|id|schedule|connection|since)\b)",
    r"\bcan you (tell me )?how to fix it\b",
]

ANSWERABLE_SCORE_THRESHOLD = 0.32  # below this, retrieval evidence is too weak to trust


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def triage_node(state: AgentState) -> dict:
    question = state["question"]
    logs = list(state.get("logs", []))

    retriever = get_retriever()
    hits: list[RetrievalHit] = retriever.search(question, top_k=6)
    top_score = hits[0].score if hits else 0.0

    classification: Classification
    reason: str

    if _matches_any(OUT_OF_SCOPE_PATTERNS, question):
        classification = "out_of_scope"
        reason = "Request asks for an unsupported action or attempts to override system rules (KB-010)."
    elif _matches_any(ESCALATION_PATTERNS, question):
        classification = "requires_escalation"
        reason = "Request indicates documented steps were already attempted or a repeated-failure/exposure condition (KB-008)."
    elif _matches_any(VAGUE_PATTERNS, question):
        classification = "requires_clarification"
        reason = "Request lacks the object/symptom/error information needed to choose a documented path (KB-010, KB-006)."
    elif top_score >= ANSWERABLE_SCORE_THRESHOLD:
        classification = "answerable"
        reason = f"Retrieved evidence with sufficient similarity (top score {top_score:.2f})."
    else:
        classification = "out_of_scope"
        reason = f"No sufficiently relevant evidence found in the knowledge base (top score {top_score:.2f})."

    logs.append(f"triage: classification={classification} top_score={top_score:.3f} reason={reason!r}")

    retrieved = [
        {"source_id": h.source_id, "passage": h.passage, "score": h.score, "doc_status": h.doc_status}
        for h in hits
    ]

    return {
        "classification": classification,
        "triage_reason": reason,
        "retrieved": retrieved,   # populated here so downstream nodes reuse it (no re-embedding)
        "logs": logs,
    }
