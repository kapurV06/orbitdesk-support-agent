"""
Shared typed state for the OrbitDesk support-agent graph.

This is the single object LangGraph threads through every node. Every
node reads from it and returns a partial dict of updates (LangGraph
merges these into the running state). Keeping it typed makes routing
functions and tests easy to reason about, and keeps model reasoning
(generation.py) cleanly separated from deterministic control flow
(graph.py / triage.py / verification.py).
"""

from __future__ import annotations
from typing import TypedDict, List, Dict, Optional, Literal

Classification = Literal[
    "answerable",
    "requires_clarification",
    "requires_escalation",
    "out_of_scope",
    "safe_failure",
]


class RetrievedPassage(TypedDict):
    source_id: str          # KB-003, CASE-1041, etc.
    passage: str             # excerpt text
    score: float              # similarity score, 0-1
    doc_status: str            # "current" | "resolved" | "superseded" | "escalated"


class VerificationResult(TypedDict):
    passed: bool
    issues: List[str]


class FinalOutput(TypedDict, total=False):
    classification: Classification
    answer: str
    sources: List[Dict[str, str]]
    confidence: float
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: List[str]


class AgentState(TypedDict, total=False):
    # input
    question_id: Optional[str]
    question: str

    # triage
    classification: Optional[Classification]
    triage_reason: Optional[str]

    # retrieval
    retrieved: List[RetrievedPassage]

    # generation
    draft_answer: Optional[str]
    draft_confidence: float

    # verification / control flow
    verification: Optional[VerificationResult]
    revision_count: int          # loop guard, capped at MAX_REVISIONS in graph.py

    # accumulated warnings surfaced by any node (e.g. superseded evidence excluded)
    warnings: List[str]

    # output
    final_output: Optional[FinalOutput]

    # traceability
    logs: List[str]              # ordered list of "node_name: message" entries

    # optional demo hook (see generation.py) -- forces the first draft to fail
    # verification so the retry path can be reliably demonstrated live
    _demo_force_verification_failure: bool
