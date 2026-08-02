"""
Local retrieval over the OrbitDesk corpus.

Model: sentence-transformers/all-MiniLM-L6-v2 (revision: main, ~90MB,
384-dim). Chosen because it is small enough to load in seconds on CPU,
has no external API dependency once cached, and is more than adequate
for a ~50-chunk corpus -- a bigger embedding model would not meaningfully
improve retrieval here and would cost load time for no benefit
(hardware-aware trade-off called out in the assignment).

This module builds an in-memory cosine-similarity index. No managed
vector DB is used, per the assignment's instructions.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_loader import Chunk, load_corpus

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "main"


@dataclass
class RetrievalHit:
    source_id: str
    passage: str
    score: float
    doc_status: str


class Retriever:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        t0 = time.time()
        self.model = SentenceTransformer(model_name, revision=EMBEDDING_MODEL_REVISION)
        self.load_time_s = time.time() - t0

        self.corpus: List[Chunk] = load_corpus()
        texts = [c["text"] for c in self.corpus]
        self.embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def search(self, query: str, top_k: int = 4) -> List[RetrievalHit]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.embeddings @ q_emb  # cosine sim, since both are normalized
        top_idx = np.argsort(-sims)[:top_k]
        hits = []
        for i in top_idx:
            c = self.corpus[i]
            hits.append(
                RetrievalHit(
                    source_id=c["source_id"],
                    passage=c["text"],
                    score=float(sims[i]),
                    doc_status=c["doc_status"],
                )
            )
        return hits


# module-level singleton so the graph doesn't reload the model / re-embed
# the corpus on every node call
_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


if __name__ == "__main__":
    r = get_retriever()
    print(f"Embedding model load time: {r.load_time_s:.2f}s, corpus size: {len(r.corpus)}")
    for hit in r.search("Can a read-only Viewer create API credentials?"):
        print(f"{hit.score:.3f}  {hit.source_id}  {hit.passage[:70]!r}")
