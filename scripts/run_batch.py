#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a batch of questions against the RDF graph.")
    parser.add_argument("--questions", default="data/questions_set_easy.json", help="JSON file containing questions.")
    parser.add_argument("--graph", default="data/output_graph.ttl", help="Path to the Turtle graph.")
    parser.add_argument("--output", default="results/batch_results.csv", help="CSV output path.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from kg_query_pipeline import KGQAPipeline, PipelineConfig
    from kg_query_pipeline.pipeline import result_to_row
    import pandas as pd

    question_items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    questions = [item["Question"] if isinstance(item, dict) else str(item) for item in question_items]

    pipeline = KGQAPipeline(PipelineConfig(graph_path=Path(args.graph)))
    rows = [result_to_row(pipeline.run(question)) for question in questions]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
