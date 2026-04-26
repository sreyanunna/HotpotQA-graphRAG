from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import networkx as nx
from rdflib import Graph, Literal, Namespace, URIRef


def local_name(term: object) -> str:
    value = str(term)
    if "/" in value:
        value = value.rsplit("/", 1)[-1]
    return value


def normalize_label(value: str) -> str:
    return value.replace("_", " ").strip()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned or "subgraph"


@dataclass
class SubgraphResult:
    start_node: str
    triples: list[tuple[str, str, str]]
    rdf_graph: Graph
    nx_graph: nx.DiGraph
    turtle_path: Path | None = None
    json_path: Path | None = None


class RDFGraphStore:
    """RDFLib-backed graph helper for entity lookup and local subgraphs."""

    def __init__(self, graph_path: Path, resource_ns: str, predicate_ns: str) -> None:
        self.graph_path = Path(graph_path)
        if not self.graph_path.exists():
            raise FileNotFoundError(f"RDF graph not found: {self.graph_path}")

        self.resource_ns = Namespace(resource_ns)
        self.predicate_ns = Namespace(predicate_ns)
        self.graph = Graph()
        self.graph.parse(self.graph_path, format="turtle")

    def entity_labels(self) -> list[str]:
        labels: set[str] = set()
        for subject, predicate, obj in self.graph:
            labels.add(local_name(subject))
            labels.add(local_name(predicate))
            labels.add(local_name(obj))
        return sorted(label for label in labels if label)

    def predicate_uris(self) -> list[URIRef]:
        predicates = {
            predicate
            for _, predicate, _ in self.graph
            if str(predicate).startswith(str(self.predicate_ns))
        }
        return sorted(predicates, key=str)

    def term_variants(self, value: str) -> Iterable[URIRef | Literal]:
        underscored = value.replace(" ", "_")
        quoted = quote(underscored, safe="")
        yield URIRef(self.resource_ns + underscored)
        if quoted != underscored:
            yield URIRef(self.resource_ns + quoted)
        yield Literal(value)
        if underscored != value:
            yield Literal(underscored)

    def create_subgraph(self, start_node: str, max_depth: int = 2) -> SubgraphResult:
        subgraph = Graph()
        subgraph.bind("ex", self.resource_ns)
        subgraph.bind("ns1", self.predicate_ns)

        queue: deque[tuple[URIRef | Literal, int]] = deque()
        for term in self.term_variants(start_node):
            if self._term_exists(term):
                queue.append((term, 0))

        if not queue:
            queue.append((Literal(start_node), 0))

        visited: set[URIRef | Literal] = set()
        triples: list[tuple[str, str, str]] = []
        nx_graph = nx.DiGraph()

        while queue:
            node, depth = queue.popleft()
            if depth > max_depth or node in visited:
                continue
            visited.add(node)

            for subject, predicate in self.graph.subject_predicates(object=node):
                self._append_triple(subgraph, nx_graph, triples, subject, predicate, node)
                if depth + 1 <= max_depth:
                    queue.append((subject, depth + 1))
                    queue.append((predicate, depth + 1))

            if isinstance(node, URIRef):
                for predicate, obj in self.graph.predicate_objects(subject=node):
                    self._append_triple(subgraph, nx_graph, triples, node, predicate, obj)
                    if depth + 1 <= max_depth:
                        queue.append((predicate, depth + 1))
                        if isinstance(obj, URIRef):
                            queue.append((obj, depth + 1))

            if isinstance(node, URIRef):
                for subject, obj in self.graph.subject_objects(predicate=node):
                    self._append_triple(subgraph, nx_graph, triples, subject, node, obj)
                    if depth + 1 <= max_depth:
                        queue.append((subject, depth + 1))
                        if isinstance(obj, URIRef):
                            queue.append((obj, depth + 1))

        return SubgraphResult(start_node=start_node, triples=triples, rdf_graph=subgraph, nx_graph=nx_graph)

    def save_subgraph(self, result: SubgraphResult, output_dir: Path) -> SubgraphResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = safe_filename(result.start_node)
        turtle_path = output_dir / f"{stem}_subgraph.ttl"
        json_path = output_dir / f"{stem}_triples.json"

        result.rdf_graph.serialize(destination=turtle_path, format="turtle")
        json_path.write_text(
            json.dumps(
                [{"subject": s, "predicate": p, "object": o} for s, p, o in result.triples],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        result.turtle_path = turtle_path
        result.json_path = json_path
        return result

    def _term_exists(self, term: URIRef | Literal) -> bool:
        return any(self.graph.triples((term, None, None))) or any(self.graph.triples((None, term, None))) or any(
            self.graph.triples((None, None, term))
        )

    @staticmethod
    def _append_triple(
        subgraph: Graph,
        nx_graph: nx.DiGraph,
        triples: list[tuple[str, str, str]],
        subject: URIRef | Literal,
        predicate: URIRef,
        obj: URIRef | Literal,
    ) -> None:
        subgraph.add((subject, predicate, obj))
        subject_label = local_name(subject)
        predicate_label = local_name(predicate)
        object_label = local_name(obj)
        triples.append((subject_label, predicate_label, object_label))
        nx_graph.add_edge(subject_label, object_label, label=predicate_label)
