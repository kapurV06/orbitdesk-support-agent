"""
Graph assembly.

  START -> triage -[route_after_triage]-> {clarification, out_of_scope,
                                             evidence(->escalation),
                                             evidence(->generation)}
  evidence -[route_after_evidence]-> {escalation, generation}
  generation -> verification -[route_after_verification]-> {finalize,
                                             generation (1 retry), safe_failure}

Loop guard: revision_count is incremented every time verification fails
and routes back to generation. MAX_REVISIONS caps this at 1 retry, so
the worst case is triage -> evidence -> generation -> verification ->
generation -> verification -> safe_failure -- a bounded, finite path.
This is enforced in code (route_after_verification), not by hoping the
model asks to stop.
"""
from __future__ import annotations
from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.triage import triage_node
from src.evidence import evidence_node
from src.generation import generation_node
from src.verification import verification_node
from src.terminal_nodes import (
    clarification_node,
    escalation_node,
    out_of_scope_node,
    safe_failure_node,
    finalize_node,
)

MAX_REVISIONS = 1


def route_after_triage(state: AgentState) -> str:
    c = state["classification"]
    if c == "requires_clarification":
        return "clarification"
    if c == "out_of_scope":
        return "out_of_scope"
    # both "answerable" and "requires_escalation" need evidence assembled first
    return "evidence"


def route_after_evidence(state: AgentState) -> str:
    return "escalation" if state["classification"] == "requires_escalation" else "generation"


def route_after_verification(state: AgentState) -> str:
    verification = state["verification"]
    if verification["passed"]:
        return "finalize"
    if state.get("revision_count", 0) < MAX_REVISIONS:
        return "revise"
    return "safe_failure"


def _increment_revision(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    count = state.get("revision_count", 0) + 1
    logs.append(f"control: routing back to generation for revision #{count}")
    return {"revision_count": count, "logs": logs}


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("triage", triage_node)
    g.add_node("evidence", evidence_node)
    g.add_node("generation", generation_node)
    g.add_node("verify", verification_node)
    g.add_node("clarification", clarification_node)
    g.add_node("escalation", escalation_node)
    g.add_node("out_of_scope", out_of_scope_node)
    g.add_node("safe_failure", safe_failure_node)
    g.add_node("finalize", finalize_node)
    g.add_node("increment_revision", _increment_revision)

    g.set_entry_point("triage")

    g.add_conditional_edges(
        "triage",
        route_after_triage,
        {"clarification": "clarification", "out_of_scope": "out_of_scope", "evidence": "evidence"},
    )
    g.add_conditional_edges(
        "evidence",
        route_after_evidence,
        {"escalation": "escalation", "generation": "generation"},
    )
    g.add_edge("generation", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verification,
        {"finalize": "finalize", "revise": "increment_revision", "safe_failure": "safe_failure"},
    )
    g.add_edge("increment_revision", "generation")

    for terminal in ("clarification", "escalation", "out_of_scope", "safe_failure", "finalize"):
        g.add_edge(terminal, END)

    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    print(graph.get_graph().draw_mermaid())
