from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from rdflib import URIRef

from kg_query_pipeline.graph import local_name, normalize_label


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return values / norms


@dataclass
class Match:
    value: str
    score: float


class EmbeddingMatcher:
    """Small wrapper around SentenceTransformer cosine matching."""

    def __init__(self, labels: Sequence[str], model_name: str) -> None:
        if not labels:
            raise ValueError("EmbeddingMatcher requires at least one label.")

        from sentence_transformers import SentenceTransformer

        self.labels = list(labels)
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode([normalize_label(label) for label in self.labels], convert_to_numpy=True)
        self.embeddings = _normalize_embeddings(embeddings.astype("float32"))

    def best(self, query: str) -> Match:
        query_embedding = self.model.encode([normalize_label(query)], convert_to_numpy=True).astype("float32")
        query_embedding = _normalize_embeddings(query_embedding)
        scores = self.embeddings @ query_embedding[0]
        index = int(np.argmax(scores))
        return Match(value=self.labels[index], score=float(scores[index]))

    def top_k(self, query: str, k: int) -> list[Match]:
        query_embedding = self.model.encode([normalize_label(query)], convert_to_numpy=True).astype("float32")
        query_embedding = _normalize_embeddings(query_embedding)
        scores = self.embeddings @ query_embedding[0]
        indexes = np.argsort(scores)[::-1][:k]
        return [Match(value=self.labels[int(index)], score=float(scores[int(index)])) for index in indexes]


class PredicateMatcher:
    def __init__(self, predicate_uris: Sequence[URIRef], model_name: str) -> None:
        self.predicate_uris = list(predicate_uris)
        labels = [local_name(uri) for uri in self.predicate_uris]
        self.matcher = EmbeddingMatcher(labels, model_name)

    def top_k_uris(self, phrase: str, k: int) -> list[URIRef]:
        matches = self.matcher.top_k(phrase, k)
        by_label = {local_name(uri): uri for uri in self.predicate_uris}
        return [by_label[match.value] for match in matches if match.value in by_label]
