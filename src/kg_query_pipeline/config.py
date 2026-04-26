from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    """Runtime configuration for the local graph QA pipeline."""

    graph_path: Path = PROJECT_ROOT / "data" / "output_graph.ttl"
    output_dir: Path = PROJECT_ROOT / "results" / "subgraphs"
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    resource_namespace: str = "http://example.org/resource/"
    predicate_namespace: str = "http://example.org/predicate/"
    max_depth: int = 2
    top_k_predicates: int = 8

    def resolved(self) -> "PipelineConfig":
        return PipelineConfig(
            graph_path=self.graph_path.expanduser().resolve(),
            output_dir=self.output_dir.expanduser().resolve(),
            openai_model=self.openai_model,
            embedding_model=self.embedding_model,
            resource_namespace=self.resource_namespace,
            predicate_namespace=self.predicate_namespace,
            max_depth=self.max_depth,
            top_k_predicates=self.top_k_predicates,
        )
