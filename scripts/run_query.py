#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one natural-language question against the RDF graph.")
    parser.add_argument("--question", required=True, help="Natural-language question to answer.")
    parser.add_argument("--graph", default="data/output_graph.ttl", help="Path to the Turtle graph.")
    parser.add_argument("--output-dir", default="results/subgraphs", help="Directory for generated subgraphs.")
    parser.add_argument("--no-save-subgraph", action="store_true", help="Do not write subgraph artifacts.")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from kg_query_pipeline import KGQAPipeline, PipelineConfig

    config = PipelineConfig(graph_path=Path(args.graph), output_dir=Path(args.output_dir))
    result = KGQAPipeline(config).run(args.question, save_subgraph=not args.no_save_subgraph)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 1 if result.error else 0


if __name__ == "__main__":
    raise SystemExit(main())
