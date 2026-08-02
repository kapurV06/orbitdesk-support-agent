"""
Deterministic terminal nodes: clarification, escalation, out_of_scope,
safe_failure, and the finalize node that assembles/validates the
schema-shaped output for the answerable path.

These are template-driven, not model-generated -- see generation.py's
docstring for why. Escalation still surfaces retrieved evidence (so the
human team gets context) but the write-up itself is deterministic.
"""
from __future__ import annotations
import json
import jsonschema
from pathlib import Path

from src.state import AgentState

SCHEMA = json.loads(
    (Path(__file__).resolve().parent.parent / "data" / "output_schema.json").read_text()
)


def _sources_from_retrieved(retrieved: list[dict]) -> list[dict]:
    return [{"source_id": r["source_id"], "passage": r["passage"][:200]} for r in retrieved]


def clarification_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("clarification: returning clarification request")
    output = {
        "classification": "requires_clarification",
        "answer": (
            "I need a bit more detail before I can point you to the right steps. "
            "Could you share the specific object involved (e.g. schedule, connection or "
            "credential), any visible error code, and when the issue started?"
        ),
        "sources": [],
        "confidence": 0.0,
        "requires_human": False,
        "reason": state.get("triage_reason", "Request lacks specific details."),
        "clarification_question": (
            "What is the affected object (schedule/connection/dashboard/credential) "
            "and what error code or message do you see, if any?"
        ),
        "warnings": state.get("warnings", []),
    }
    return {"final_output": output, "logs": logs}


def escalation_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    retrieved = state.get("retrieved", [])
    logs.append(f"escalation: routing to human team, evidence={[r['source_id'] for r in retrieved]}")
    output = {
        "classification": "requires_escalation",
        "answer": (
            "This looks like it needs a human team. Based on the documented checks, "
            "please collect: workspace ID, the affected object ID (schedule/dashboard/"
            "connection/credential), the exact error code, timestamps with timezone, "
            "and the troubleshooting steps already attempted (KB-008). Do not include "
            "passwords, API secrets, OAuth tokens or exported customer data. Share this "
            "with the appropriate OrbitDesk support team."
        ),
        "sources": _sources_from_retrieved(retrieved),
        "confidence": 0.6,
        "requires_human": True,
        "reason": state.get("triage_reason", "Escalation criteria met."),
        "clarification_question": None,
        "warnings": state.get("warnings", []),
    }
    return {"final_output": output, "logs": logs}


def out_of_scope_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("out_of_scope: declining, request unrelated to or unsupported by KB")
    output = {
        "classification": "out_of_scope",
        "answer": (
            "That request is outside the OrbitDesk support knowledge base available to me. "
            "I can't process refunds, cancellations, or provide legal/financial advice, and I "
            "only answer from the supplied product documentation. If this is a billing or "
            "account matter, please contact OrbitDesk's billing/support team directly."
        ),
        "sources": [],
        "confidence": 0.0,
        "requires_human": True,
        "reason": state.get("triage_reason", "Outside available knowledge base."),
        "clarification_question": None,
        "warnings": state.get("warnings", []),
    }
    return {"final_output": output, "logs": logs}


def safe_failure_node(state: AgentState) -> dict:
    logs = list(state.get("logs", []))
    issues = (state.get("verification") or {}).get("issues", [])
    logs.append(f"safe_failure: verification failed twice, issues={issues}")
    output = {
        "classification": "safe_failure",
        "answer": (
            "I wasn't able to produce a response I'm confident is fully supported by the "
            "available documentation. Rather than risk giving inaccurate guidance, please "
            "rephrase your question with more detail, or contact OrbitDesk support directly."
        ),
        "sources": _sources_from_retrieved(state.get("retrieved", [])),
        "confidence": 0.0,
        "requires_human": True,
        "reason": f"Verification failed after retry: {'; '.join(issues) if issues else 'unspecified'}",
        "clarification_question": None,
        "warnings": state.get("warnings", []) + ["Answer withheld after failing verification twice."],
    }
    return {"final_output": output, "logs": logs}


def finalize_node(state: AgentState) -> dict:
    """Assembles the schema-shaped output once verification has passed."""
    logs = list(state.get("logs", []))
    retrieved = state.get("retrieved", [])
    answer = state.get("draft_answer", "")

    output = {
        "classification": "answerable",
        "answer": answer,
        "sources": _sources_from_retrieved(retrieved),
        "confidence": round(min(0.95, 0.5 + 0.1 * len(retrieved)), 2),
        "requires_human": False,
        "reason": "Answer supported by retrieved evidence and passed verification.",
        "clarification_question": None,
        "warnings": state.get("warnings", []),
    }

    try:
        jsonschema.validate(instance=output, schema=SCHEMA)
        logs.append("finalize: schema validation passed")
    except jsonschema.ValidationError as e:
        logs.append(f"finalize: schema validation FAILED: {e.message}")
        # Fall back to a safe response rather than return an invalid payload
        output = {
            "classification": "safe_failure",
            "answer": "Response failed output-schema validation and was withheld.",
            "sources": [],
            "confidence": 0.0,
            "requires_human": True,
            "reason": f"Schema validation error: {e.message}",
            "clarification_question": None,
            "warnings": ["Schema validation failure at finalize step."],
        }

    return {"final_output": output, "logs": logs}
