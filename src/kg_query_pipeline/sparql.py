from __future__ import annotations

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDFS, SKOS

from kg_query_pipeline.graph import local_name
from kg_query_pipeline.query_parser import QueryTriple


def build_sparql(
    query: QueryTriple,
    predicate_candidates: list[URIRef],
    entity_terms: list[URIRef | Literal],
    object_terms: list[URIRef | Literal] | None = None,
    limit: int = 20,
) -> str:
    entity_values = " ".join(_format_term(term) for term in entity_terms)
    predicate_values = " ".join(f"<{predicate}>" for predicate in predicate_candidates)

    if query.unknown_component == "subject":
        return f"""SELECT DISTINCT ?answer WHERE {{
  VALUES ?entity {{ {entity_values} }}
  VALUES ?p {{ {predicate_values} }}
  ?answer ?p ?entity .
}} LIMIT {limit}"""

    if query.unknown_component == "object":
        return f"""SELECT DISTINCT ?answer WHERE {{
  VALUES ?entity {{ {entity_values} }}
  VALUES ?p {{ {predicate_values} }}
  ?entity ?p ?answer .
}} LIMIT {limit}"""

    if object_terms:
        object_values = " ".join(_format_term(term) for term in object_terms)
        return f"""SELECT DISTINCT ?answer WHERE {{
  VALUES ?left {{ {entity_values} }}
  VALUES ?right {{ {object_values} }}
  {{ ?left ?answer ?right . }}
  UNION
  {{ ?right ?answer ?left . }}
}} LIMIT {limit}"""

    return f"""SELECT DISTINCT ?answer WHERE {{
  VALUES ?left {{ {entity_values} }}
  ?left ?answer ?right .
}} LIMIT {limit}"""


def run_sparql(graph: Graph, sparql: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in graph.query(sparql):
        value = row.get("answer")
        if value is not None:
            rows.append({"answer": value})
    return rows


def label_for(graph: Graph, term: object) -> str:
    if isinstance(term, URIRef):
        for predicate in (RDFS.label, SKOS.prefLabel):
            for label in graph.objects(term, predicate):
                return str(label)
        return local_name(term).replace("_", " ")
    return str(term)


def materialize_answers(graph: Graph, rows: list[dict[str, object]]) -> list[str]:
    answers: list[str] = []
    for row in rows:
        value = row.get("answer")
        if value is not None:
            answers.append(label_for(graph, value))
    return answers


def _format_term(term: URIRef | Literal) -> str:
    if isinstance(term, URIRef):
        return f"<{term}>"
    escaped = str(term).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
