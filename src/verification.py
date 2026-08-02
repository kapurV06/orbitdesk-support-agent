"""
Verification node.

Deliberately implemented as deterministic checks rather than "ask the
model if its own answer is good" -- a small local model grading itself
is unreliable, and the assignment wants verification that is auditable
and doesn't depend on model wording. Checks:

  1. Evidence support   - does the answer semantically overlap with the
                           retrieved passages? (reuses the embedding
                           model already loaded for retrieval; no third
                           model needed)
  2. Source references  - does the answer cite at least one retrieved
                           source_id?
  3. Schema shape        - will this survive JSON-schema validation once
                           assembled into the final output?
  4. Unsupported actions - does the answer claim to have performed an
                           action the assistant cannot perform (KB-010)?
"""
from __future__ import annotations
import re
from typing import List

import numpy as np

from src.state import AgentState
from src.retrieval import get_retriever

MIN_EVIDENCE_SIMILARITY = 0.30

UNSUPPORTED_CLAIM_PATTERNS = [
    r"\bi (have|'ve) (issued|processed|created|refunded|cancelled|canceled|revoked|reset)\b",
    r"\bi (have|'ve) (changed|updated|reset) your (role|password|workspace)\b",
    r"\byour refund (has been|is) (issued|processed)\b",
    r"\bthe credential (has been|is now) created\b",
]


def _cite_pattern(source_ids: List[str]) -> str:
    escaped = [re.escape(sid) for sid in source_ids]
    return r"(" + "|".join(escaped) + r")" if escaped else r"$^"  # never matches if empty


def verification_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    answer = state.get("draft_answer") or ""
    retrieved = state.get("retrieved", [])
    source_ids = [r["source_id"] for r in retrieved]

    issues: list[str] = []

    # 1. Evidence support via embedding similarity between the answer and
    #    the concatenated evidence block.
    if retrieved:
        retriever = get_retriever()
        evidence_text = " ".join(r["passage"] for r in retrieved)
        emb = retriever.model.encode(
            [answer, evidence_text], convert_to_numpy=True, normalize_embeddings=True
        )
        similarity = float(np.dot(emb[0], emb[1]))
        if similarity < MIN_EVIDENCE_SIMILARITY:
            issues.append(
                f"Answer has low semantic similarity to retrieved evidence ({similarity:.2f})."
            )
    else:
        similarity = 0.0
        issues.append("No evidence was retrieved to support this answer.")

    # 2. Source references present
    if not re.search(_cite_pattern(source_ids), answer):
        issues.append("Answer does not cite any retrieved source_id.")

    # 3. Non-empty, reasonable length (schema minLength=1; also catch empty/garbage generation)
    if len(answer.strip()) < 10:
        issues.append("Answer is empty or too short to be useful.")

    # 4. Unsupported action claims
    for pattern in UNSUPPORTED_CLAIM_PATTERNS:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            issues.append(f"Answer appears to claim an unsupported action was performed: matched {pattern!r}.")
            break

    passed = len(issues) == 0
    logs.append(f"verification: passed={passed} similarity={similarity:.3f} issues={issues}")

    return {"verification": {"passed": passed, "issues": issues}, "logs": logs}
