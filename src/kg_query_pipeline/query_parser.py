from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QueryTriple:
    """Structured representation of a natural-language graph question."""

    subject: str
    predicate: str
    object: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "QueryTriple":
        missing = {"subject", "predicate", "object"} - set(data)
        if missing:
            raise ValueError(f"Missing query fields: {sorted(missing)}")

        triple = cls(
            subject=str(data["subject"]).strip(),
            predicate=str(data["predicate"]).strip(),
            object=str(data["object"]).strip(),
        )
        unknown_count = [triple.subject, triple.predicate, triple.object].count("?")
        if unknown_count != 1:
            raise ValueError("Parsed query must contain exactly one '?' component.")
        return triple

    def as_dict(self) -> dict[str, str]:
        return asdict(self)

    @property
    def unknown_component(self) -> str:
        for field in ("subject", "predicate", "object"):
            if getattr(self, field) == "?":
                return field
        raise ValueError("QueryTriple has no unknown component.")

    @property
    def known_entity(self) -> str | None:
        if self.subject != "?":
            return self.subject
        if self.object != "?":
            return self.object
        return None

    def with_replaced_entity(self, entity: str) -> "QueryTriple":
        if self.subject != "?":
            return QueryTriple(entity, self.predicate, self.object)
        if self.object != "?":
            return QueryTriple(self.subject, self.predicate, entity)
        return self


QUERY_PARSER_PROMPT = """Extract a subject, predicate, and object from the question.

Rules:
1. Return valid JSON only, with keys: subject, predicate, object.
2. Exactly one component must be "?".
3. Use underscores for multi-word entities.
4. Keep predicates concise and graph-like.

Examples:
Question: Who directed The Family Man?
Answer: {"subject": "?", "predicate": "directed_by", "object": "The_Family_Man"}

Question: What did Ed Wood write?
Answer: {"subject": "Ed_Wood", "predicate": "wrote", "object": "?"}

Question: What is the relationship between Maine Nordiques and NAHL?
Answer: {"subject": "Maine_Nordiques", "predicate": "?", "object": "NAHL"}

Question: {question}
Answer:"""


def parse_query_response(content: str) -> QueryTriple:
    """Parse model output without using eval."""

    cleaned = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        data = ast.literal_eval(cleaned)

    if not isinstance(data, dict):
        raise ValueError("Query parser did not return an object.")
    return QueryTriple.from_mapping(data)


def parse_question_with_openai(question: str, model: str) -> QueryTriple:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for natural-language query parsing.")

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": QUERY_PARSER_PROMPT.replace("{question}", question)}],
        temperature=0,
    )
    return parse_query_response(response.choices[0].message.content or "")
