"""
Carrega o índice vetorial pré-computado (gerado por scripts/ingest.py) e
oferece a função `retrieve(query, k)` usada pelo nó de recuperação do grafo
LangGraph.

O índice é carregado uma única vez por processo (cache em nível de módulo),
o que é importante em ambiente serverless: o "cold start" paga o custo de
carregar os arquivos pequenos (alguns MBs) uma vez, e invocações seguintes
no mesmo container reutilizam tudo em memória.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import PROCESSED_DIR, TOP_K


@dataclass
class RetrievedChunk:
    id: int
    text: str
    title: str
    source: str
    score: float


class VectorIndex:
    def __init__(self, processed_dir):
        with open(processed_dir / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(processed_dir / "svd.pkl", "rb") as f:
            self.svd = pickle.load(f)
        self.embeddings = np.load(processed_dir / "embeddings.npy")
        with open(processed_dir / "chunks.json", "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        if len(self.chunks) != self.embeddings.shape[0]:
            raise RuntimeError(
                "Índice inconsistente: número de chunks difere do número de "
                "embeddings. Rode `python scripts/ingest.py` novamente."
            )

    def query_vector(self, query: str) -> np.ndarray:
        tfidf_vec = self.vectorizer.transform([query])
        dense = self.svd.transform(tfidf_vec)
        norm = np.linalg.norm(dense, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return (dense / norm).astype(np.float32)[0]

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        qvec = self.query_vector(query)
        # como tudo está L2-normalizado, o produto escalar == similaridade de cosseno
        scores = self.embeddings @ qvec
        top_idx = np.argsort(-scores)[:k]
        results = []
        for idx in top_idx:
            chunk = self.chunks[int(idx)]
            results.append(RetrievedChunk(
                id=chunk["id"],
                text=chunk["text"],
                title=chunk["title"],
                source=chunk.get("source", ""),
                score=float(scores[idx]),
            ))
        return results


@lru_cache(maxsize=1)
def get_index() -> VectorIndex:
    if not (PROCESSED_DIR / "chunks.json").exists():
        raise RuntimeError(
            f"Índice não encontrado em {PROCESSED_DIR}. Rode primeiro: "
            "python scripts/ingest.py"
        )
    return VectorIndex(PROCESSED_DIR)


def retrieve(query: str, k: int | None = None) -> list[RetrievedChunk]:
    index = get_index()
    return index.search(query, k or TOP_K)
