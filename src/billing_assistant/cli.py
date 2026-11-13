import argparse
import asyncio
import json
from pathlib import Path

from .claims import get_synthetic_claim
from .fixtures import load_fixture
from .retrieval import search_bm25


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the local billing assistant fixtures.")
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser("search", help="Search the pinned CMS excerpts.")
    search.add_argument("query")
    search.add_argument("--limit", type=int, choices=range(1, 6), default=5)

    claim = commands.add_parser("claim", help="Look up one synthetic claim.")
    claim.add_argument("claim_id")
    claim.add_argument("--allow-lookup", action="store_true", help="Enable lookup for this local invocation.")

    ask = commands.add_parser("ask", help="Answer a question within the synthetic Q0091 workflow.")
    ask.add_argument("question")
    ask.add_argument("--allow-lookup", action="store_true")
    ask.add_argument("--local-model", action="store_true", help="Use the pinned Qwen model.")
    ask.add_argument("--download-model", action="store_true", help="Allow the public model download on first use.")
    ask.add_argument("--trace-db", type=Path, help="Write metadata-only MLflow spans to this SQLite database.")

    args = parser.parse_args(argv)
    if args.command == "search":
        corpus = load_fixture("cms_ch18.json")
        result = {
            "corpus_version": corpus["version"],
            "corpus_status": corpus["status"],
            "source_url": corpus["source_url"],
            "hits": search_bm25(args.query, corpus["passages"], limit=args.limit),
        }
        status = 0
    elif args.command == "ask":
        from .assistant import Assistant
        from .model import LocalModel
        from .tracing import local_tracer

        if args.download_model and not args.local_model:
            parser.error("--download-model requires --local-model.")
        model = LocalModel(Path(".model-cache"), local_only=not args.download_model) if args.local_model else None
        trace = local_tracer(args.trace_db) if args.trace_db else None
        result = asyncio.run(Assistant(model=model, trace=trace).ask(
            args.question, allowed=args.allow_lookup,
        ))
        status = 0 if result["status"] in {"answer", "partial", "lookup_result", "abstain"} else 1
    else:
        result = get_synthetic_claim(args.claim_id, allowed=args.allow_lookup)
        status = 0 if result["status"] == "ok" else 1

    print(json.dumps(result, indent=2))
    return status
