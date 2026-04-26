"""Knowledge graph question-answering pipeline."""

__all__ = ["KGQAPipeline", "PipelineConfig", "PipelineResult", "QueryTriple"]


def __getattr__(name: str):
    if name == "PipelineConfig":
        from kg_query_pipeline.config import PipelineConfig

        return PipelineConfig
    if name == "QueryTriple":
        from kg_query_pipeline.query_parser import QueryTriple

        return QueryTriple
    if name in {"KGQAPipeline", "PipelineResult"}:
        from kg_query_pipeline.pipeline import KGQAPipeline, PipelineResult

        return {"KGQAPipeline": KGQAPipeline, "PipelineResult": PipelineResult}[name]
    raise AttributeError(f"module 'kg_query_pipeline' has no attribute {name!r}")
