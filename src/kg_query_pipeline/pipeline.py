from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from rdflib import Literal, URIRef

from kg_query_pipeline.config import PipelineConfig
from kg_query_pipeline.graph import RDFGraphStore, SubgraphResult
from kg_query_pipeline.matching import EmbeddingMatcher, PredicateMatcher
from kg_query_pipeline.query_parser import QueryTriple, parse_question_with_openai
from kg_query_pipeline.sparql import build_sparql, materialize_answers, run_sparql


@dataclass
class PipelineResult:
    question: str
    parsed_query: dict[str, str]
    matched_entity: str | None
    entity_match_score: float | None
    sparql: str
    answers: list[str]
    subgraph_turtle_path: str | None
    subgraph_json_path: str | None
    error: str | None = None


class KGQAPipeline:
    """End-to-end local runner for natural-language QA over an RDF graph."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config.resolved()
        self.store = RDFGraphStore(
            self.config.graph_path,
            resource_ns=self.config.resource_namespace,
            predicate_ns=self.config.predicate_namespace,
        )
        self.entity_matcher = EmbeddingMatcher(self.store.entity_labels(), self.config.embedding_model)
        self.predicate_matcher = PredicateMatcher(self.store.predicate_uris(), self.config.embedding_model)

    def run(self, question: str, save_subgraph: bool = True) -> PipelineResult:
        try:
            parsed_query = parse_question_with_openai(question, self.config.openai_model)
            matched_query, entity_name, entity_score = self._match_known_entity(parsed_query)

            subgraph = self.store.create_subgraph(entity_name or "", max_depth=self.config.max_depth)
            if save_subgraph:
                subgraph = self.store.save_subgraph(subgraph, self.config.output_dir)

            predicate_candidates = self._predicate_candidates(matched_query)
            entity_terms = list(self.store.term_variants(entity_name or ""))
            object_terms = None
            if matched_query.unknown_component == "predicate" and matched_query.object != "?":
                object_terms = list(self.store.term_variants(matched_query.object))
            sparql = build_sparql(matched_query, predicate_candidates, entity_terms, object_terms=object_terms)
            rows = run_sparql(self.store.graph, sparql)
            answers = materialize_answers(self.store.graph, rows)

            return self._result(
                question=question,
                parsed_query=matched_query,
                matched_entity=entity_name,
                entity_match_score=entity_score,
                sparql=sparql,
                answers=answers,
                subgraph=subgraph,
            )
        except Exception as exc:
            return PipelineResult(
                question=question,
                parsed_query={},
                matched_entity=None,
                entity_match_score=None,
                sparql="",
                answers=[],
                subgraph_turtle_path=None,
                subgraph_json_path=None,
                error=str(exc),
            )

    def _match_known_entity(self, query: QueryTriple) -> tuple[QueryTriple, str | None, float | None]:
        known_entity = query.known_entity
        if known_entity is None:
            return query, None, None

        match = self.entity_matcher.best(known_entity)
        return query.with_replaced_entity(match.value), match.value, match.score

    def _predicate_candidates(self, query: QueryTriple) -> list[URIRef]:
        if query.predicate == "?":
            return self.store.predicate_uris()[: self.config.top_k_predicates]
        return self.predicate_matcher.top_k_uris(query.predicate, self.config.top_k_predicates)

    @staticmethod
    def _result(
        question: str,
        parsed_query: QueryTriple,
        matched_entity: str | None,
        entity_match_score: float | None,
        sparql: str,
        answers: list[str],
        subgraph: SubgraphResult,
    ) -> PipelineResult:
        return PipelineResult(
            question=question,
            parsed_query=asdict(parsed_query),
            matched_entity=matched_entity,
            entity_match_score=entity_match_score,
            sparql=sparql,
            answers=answers,
            subgraph_turtle_path=str(subgraph.turtle_path) if subgraph.turtle_path else None,
            subgraph_json_path=str(subgraph.json_path) if subgraph.json_path else None,
        )


def result_to_row(result: PipelineResult) -> dict[str, object]:
    return {
        "question": result.question,
        "parsed_query": result.parsed_query,
        "matched_entity": result.matched_entity,
        "entity_match_score": result.entity_match_score,
        "answers": "; ".join(result.answers),
        "error": result.error,
    }


def graph_path_from_cli(path: str | Path | None) -> Path:
    return Path(path) if path else PipelineConfig().graph_path
