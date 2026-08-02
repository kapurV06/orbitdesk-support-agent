# OrbitDesk Support Agent Network

A local-first, graph-orchestrated support agent for the fictional OrbitDesk
product, built with LangGraph and local Hugging Face models. No remote
LLM API is used anywhere in this repository.

## Architecture

```
START -> triage
  |-- requires_clarification -> clarification_node -> END
  |-- out_of_scope            -> out_of_scope_node  -> END
  `-- (answerable | requires_escalation) -> evidence
                                              |-- requires_escalation -> escalation_node -> END
                                              `-- answerable -> generation -> verify
                                                                   |-- pass -> finalize -> END
                                                                   |-- fail, revision 0 -> generation (retry) -> verify
                                                                   `-- fail, revision 1 -> safe_failure_node -> END
```

See `diagram/graph_diagram.png` for the rendered diagram.

| Node | Responsibility | Model? |
|---|---|---|
| `triage` | Classifies the request (answerable / clarification / escalation / out-of-scope) using a hybrid of deterministic keyword rules and embedding-similarity strength | reuses embedding model |
| `evidence` | Applies source-priority rules (current KB > resolved cases; superseded excluded from current guidance) to triage's retrieved hits | none (deterministic) |
| `generation` | Generates an answer strictly from retrieved evidence | local LLM |
| `verify` | Checks evidence support (embedding similarity), citation presence, non-empty answer, and unsupported-action claims | reuses embedding model |
| `clarification` / `escalation` / `out_of_scope` / `safe_failure` / `finalize` | Deterministic template assembly + JSON-schema validation | none |

**Why generation is template-free on non-answerable paths:** letting a small
local model freely write clarification/escalation/refusal text risks exactly
the kind of unsupported or invented instruction the verification node exists
to catch. Those paths are deterministic templates instead; only the
"answerable" path touches the LLM, and everything it says is checked before
being returned.

## Models Used

| Purpose | Model | Revision |
|---|---|---|
| Embeddings (retrieval + verification support-check) | `sentence-transformers/all-MiniLM-L6-v2` | `main` |
| Response generation | `Qwen/Qwen2.5-1.5B-Instruct` | `main` |

Both load via Hugging Face `transformers` / `sentence-transformers`. Device
placement is explicit (`cuda` if available, else `cpu`) rather than
`device_map="auto"` — that was tried first but unpredictably triggered
`accelerate`'s disk-offload path on this machine and stalled generation, so
explicit placement replaced it for reliability.

**Load time / latency** (measured on this machine, CPU-only — see Hardware
section below):

- Embedding model load time: ~1-2s (`sentence-transformers/all-MiniLM-L6-v2`,
  ~90MB, confirmed via `python src/retrieval.py`)
- Generation model load + first-call latency: ~110s combined (model load is
  not isolated from first inference in the current timing — the first
  generation call in a process includes both). See "What I'd improve" below.
- Generation latency once the model is warm (same process, subsequent
  calls): ~20-30s for a ~150-300 char answer on CPU (measured: 31.04s for a
  242-char answer, 21-22s for ~135-280 char answers)

**Hardware used:** ASUS Vivobook M6500QF — AMD Ryzen 7 5800H (8c/16t), 16GB
RAM, NVIDIA GeForce RTX 3050 4GB (laptop). **The RTX 3050 was not used for
this run** — `torch` installed via plain `pip install torch` resolved to a
CPU-only build (no CUDA), so all inference shown in `outputs/` ran on CPU.
Both models were chosen to comfortably fit this budget either way: the
embedding model is ~90MB and the 1.5B generation model is ~3GB in fp32 on
CPU, well inside 16GB RAM, and would fit the 4GB VRAM ceiling in fp16 if a
CUDA-enabled torch build were installed instead.

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First run downloads both models from Hugging Face Hub and caches them
locally (`~/.cache/huggingface`). After that, the app runs fully offline —
disable networking and re-run to confirm, per the assignment's requirement.

## Running

```bash
# Run all 5 sample questions, print logs + schema-shaped output for each
python -m src.cli --samples

# Save results to a file
python -m src.cli --samples --out outputs/sample_run_outputs.json

# Ask a new question
python -m src.cli --question "Can a read-only user create API credentials?"

# Reliably demonstrate the verification-failure/retry path (test case 5)
python -m src.cli --question "Can a read-only user create API credentials?" --demo-verification-failure
```

### Demo hook disclosure

A real ~1.5B model's failures aren't reliably reproducible on demand. The
`--demo-verification-failure` flag deterministically strips citations from
the *first* draft only (documented in `src/generation.py`), forcing
verification to fail and the retry path to fire. The retry itself still
calls the real model normally — this is only used to reliably exercise
required test case 5 ("a case where the initial generated answer fails
verification") live, not to fake the mechanism itself.

## Tests

```bash
pytest tests/ -v
```

`tests/test_graph_routing.py` verifies graph routing (which path each
classification takes, the loop guard, retry-then-succeed, retry-then-fail)
using a mocked retriever and a mocked generation node — it does **not**
depend on the exact wording a real model produces, and runs fully offline
with no model download.

## Required Test Cases (mapped to `sample_questions.json`)

1. **Directly answerable:** Q-002 (Viewer + API credentials, single doc)
2. **Requires two documents:** Q-001 (timezone change + missed export, spans KB-003 + KB-004)
3. **Ambiguous, requires clarification:** Q-003 ("sync is not working")
4. **Out-of-scope:** Q-005 (refund request + prompt-injection attempt)
5. **Initial answer fails verification:** run any question with `--demo-verification-failure` (see above)

## Design Trade-offs / Known Limitations

- **Triage reuses the embedding model instead of a dedicated classifier.**
  For a ~50-chunk domain-specific corpus, cosine-similarity strength plus a
  small set of deterministic keyword rules gives auditable, fast routing
  without the load-time cost of a second model. A larger, more varied
  corpus would likely need a real zero-shot classifier instead.
- **Verification is fully deterministic** (embedding similarity + regex),
  not model-graded. This is more auditable and doesn't depend on a small
  model correctly judging its own output, but it can't catch subtler
  factual errors a semantic-entailment model might.
- **Inference ran on CPU, not the available GPU.** `pip install torch`
  resolved to a CPU-only build, so the RTX 3050 sat unused; generation
  latency (~20-110s per call) reflects that. Installing a CUDA-enabled torch
  build would likely cut this to a few seconds per call.
- **Even after being told the exact verification failure, the retry doesn't
  always fix it.** In one run, the model's revised answer still failed to
  cite a source, landing on `safe_failure` after both attempts — a real
  limitation of a small (1.5B) instruction-following model, not a bug in the
  verification logic. This is exactly the case the loop guard and
  `safe_failure` path exist for.
- **What I'd improve with more time:** a small NLI-based entailment check
  (e.g. a distilled MNLI model) instead of raw embedding similarity for the
  evidence-support check in `verify`, a proper reranker
  (e.g. `cross-encoder/ms-marco-MiniLM`) between retrieval and generation to
  improve passage precision on ambiguous questions, isolating model load
  time from first-inference time in the latency logging, and installing a
  CUDA-enabled torch build to actually use the RTX 3050.

## AI Assistant Disclosure

This repository was built with assistance from Claude (Anthropic). Claude
was used to scaffold the LangGraph structure, write the node implementations,
and draft this README, based on my own reading of the assignment brief and
design decisions (model choices, triage rules, verification checks). I
understand the full implementation and can explain or modify any part of it.
