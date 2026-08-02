"""
Loads the supplied OrbitDesk material into a flat list of retrievable
chunks. Each chunk carries a source_id (KB-00X or CASE-XXXX) and a
doc_status so downstream nodes can apply the "current KB beats
resolved cases; superseded cases are historical only" rule from
README.md / KB-010.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List, TypedDict

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class Chunk(TypedDict):
    source_id: str
    doc_status: str      # current | resolved | escalated | superseded
    title: str
    text: str


def _parse_frontmatter(raw: str) -> tuple[dict, str]:
    """Split a KB markdown file into (frontmatter dict, body)."""
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.DOTALL)
    if not m:
        return {}, raw
    fm_raw, body = m.group(1), m.group(2)
    fm = {}
    for line in fm_raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip().strip("[]")
    return fm, body


def _split_sections(body: str) -> List[str]:
    """Chunk a KB doc by '##' headers so retrieval returns focused passages
    instead of whole documents."""
    parts = re.split(r"\n(?=## )", body.strip())
    return [p.strip() for p in parts if p.strip()]


def load_knowledge_base() -> List[Chunk]:
    chunks: List[Chunk] = []
    kb_dir = DATA_DIR / "knowledge_base"
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        fm, body = _parse_frontmatter(raw)
        doc_id = fm.get("document_id", path.stem)
        status = fm.get("status", "current")
        title = fm.get("title", path.stem)
        for section in _split_sections(body):
            chunks.append(
                Chunk(source_id=doc_id, doc_status=status, title=title, text=section)
            )
    return chunks


def load_resolved_cases() -> List[Chunk]:
    chunks: List[Chunk] = []
    raw = json.loads((DATA_DIR / "resolved_cases.json").read_text(encoding="utf-8"))
    for case in raw["cases"]:
        text_lines = [f"Title: {case['title']}"]
        text_lines += ["Symptoms:"] + [f"- {s}" for s in case.get("symptoms", [])]
        text_lines += ["Resolution steps:"] + [f"- {r}" for r in case.get("resolution", [])]
        if "important_limit" in case:
            text_lines.append(f"Important limit: {case['important_limit']}")
        if "superseded_reason" in case:
            text_lines.append(f"Superseded reason: {case['superseded_reason']}")
        chunks.append(
            Chunk(
                source_id=case["case_id"],
                doc_status=case["status"],   # resolved | escalated | superseded
                title=case["title"],
                text="\n".join(text_lines),
            )
        )
    return chunks


def load_corpus() -> List[Chunk]:
    """Full retrievable corpus: current docs first, then resolved cases."""
    return load_knowledge_base() + load_resolved_cases()


def load_sample_questions() -> list[dict]:
    raw = json.loads((DATA_DIR / "sample_questions.json").read_text(encoding="utf-8"))
    return raw["questions"]


if __name__ == "__main__":
    corpus = load_corpus()
    print(f"Loaded {len(corpus)} chunks")
    for c in corpus[:3]:
        print(c["source_id"], c["doc_status"], "-", c["text"][:60].replace("\n", " "))
