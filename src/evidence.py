"""
Retrieval / evidence-assembly node.

Triage already ran the embedding search (src/retrieval.py) to make its
routing decision, so this node does not re-embed anything -- it applies
the source-priority rule from README.md / KB-001 to the hits already in
state:

  - current KB docs are primary evidence
  - resolved cases are secondary evidence
  - superseded cases are kept ONLY as a flagged warning (useful for
    testing verification), never presented as current guidance

This keeps "retrieval" a clearly separate responsibility from "triage"
even though they share one underlying model call, and keeps the
precedence rule in one deterministic place rather than folded into
prompt text (where the model could ignore it).
"""
from __future__ import annotations
from src.state import AgentState

MAX_EVIDENCE_ITEMS = 4


def evidence_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    hits = list(state.get("retrieved", []))

    current = [h for h in hits if h["doc_status"] in ("current", "resolved", "escalated")]
    superseded = [h for h in hits if h["doc_status"] == "superseded"]

    selected = current[:MAX_EVIDENCE_ITEMS]

    warnings = []
    if superseded and not selected:
        # Only superseded material was retrieved -- do not let generation
        # treat it as current guidance.
        warnings.append(
            "Only superseded historical material matched this question; "
            "it is not presented as current guidance."
        )
    elif superseded:
        warnings.append(
            f"Superseded case(s) {[h['source_id'] for h in superseded]} were retrieved "
            "but excluded as evidence (historical only)."
        )

    logs.append(
        f"evidence: selected={[h['source_id'] for h in selected]} "
        f"excluded_superseded={[h['source_id'] for h in superseded]}"
    )

    all_warnings = list(state.get("warnings", [])) + warnings
    return {"retrieved": selected, "logs": logs, "warnings": all_warnings}
