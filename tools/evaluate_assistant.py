"""Run the frozen labels against the configured assistant and real local MCP transport."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from billing_assistant.assistant import Assistant
from billing_assistant.mcp_claims import lookup_claim
from billing_assistant.model import PROMPT_VERSION, FixtureModel, LocalModel
from billing_assistant.retrieval import HybridRetriever
from billing_assistant.tracing import local_tracer

ROOT = Path(__file__).resolve().parents[1]


async def run(args):
    suite = json.loads((ROOT / "evaluation/cases.json").read_text())
    model = LocalModel(ROOT / ".model-cache", local_only=True) if args.local_model else FixtureModel()
    trace = local_tracer(args.trace_db) if args.trace_db else None
    indexes = {}

    def hybrid_search(question, passages, *, limit):
        key = tuple(p["id"] for p in passages)
        if key not in indexes:
            indexes[key] = HybridRetriever(passages, cache_directory=str(ROOT / ".model-cache"))
        return indexes[key].search(question, limit=limit)

    assistant = Assistant(model=model, trace=trace, search=hybrid_search if args.hybrid else None)
    if suite["corpus_version"] != assistant.corpus["version"]:
        raise ValueError("Suite and corpus versions differ.")
    rows = []
    for case in suite["cases"]:
        context = case.get("context_fixture", {})
        started = time.perf_counter()
        result = await assistant.ask(
            case["question"], allowed=context.get("lookup_permission") != "denied",
            test_context=context,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        cited = [citation["id"] for citation in result["citations"]]
        rows.append({
            "id": case["id"], "category": case["category"], "split": case["split"],
            "status_match": result["status"] == case["expected_outcome"],
            "citation_id_match": set(cited) == set(case["required_passage_ids"]),
            "tool_match": all(result["tool"].get(key) == value
                              for key, value in case["expected_tool"].items()),
            "injected_passage_rejected": (
                result.get("rejected_passages", 0) == 1 if case["category"] == "injection" else None
            ),
            "latency_ms": elapsed_ms,
            **result,
        })
    denied = await lookup_claim("SYN-0001", server_allowed=False)
    return {
        "suite": suite["version"], "corpus": assistant.corpus["version"], "model": model.name,
        "prompt": PROMPT_VERSION, "retrieval": "hybrid" if args.hybrid else "bm25",
        "provider_api_cost_usd": 0,
        "model_calls": sum(row["usage"]["calls"] for row in rows),
        "input_tokens": sum(row["usage"]["input_tokens"] for row in rows),
        "output_tokens": sum(row["usage"]["output_tokens"] for row in rows),
        "median_latency_ms": statistics.median(row["latency_ms"] for row in rows),
        "p95_latency_ms": statistics.quantiles([row["latency_ms"] for row in rows], n=20, method="inclusive")[18],
        "status_matches": sum(row["status_match"] for row in rows),
        "citation_id_matches": sum(row["citation_id_match"] for row in rows),
        "tool_matches": sum(row["tool_match"] for row in rows),
        "injections_rejected": sum(row["injected_passage_rejected"] is True for row in rows),
        "direct_mcp_denial": denied,
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-db", type=Path)
    parser.add_argument("--local-model", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
