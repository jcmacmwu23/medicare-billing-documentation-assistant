"""Compare top-three retrieval strategies on one development and one replay case."""

import argparse
import asyncio
import json
import time
from pathlib import Path

import mlflow

from billing_assistant.assistant import Assistant
from billing_assistant.model import PROMPT_VERSION, FixtureModel, LocalModel
from billing_assistant.retrieval import search_bm25, search_clause_bm25
from billing_assistant.tracing import local_tracer

ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ("retrieval-11", "retrieval-10")


async def replay(args):
    suite = json.loads((ROOT / "evaluation/cases.json").read_text())
    cases = {case["id"]: case for case in suite["cases"]}
    trace = local_tracer(args.trace_db)
    model = LocalModel(ROOT / ".model-cache", local_only=True) if args.local_model else FixtureModel()
    records = []
    for case_id in CASE_IDS:
        case = cases[case_id]
        for search in (search_bm25, search_clause_bm25):
            assistant = Assistant(model=model, trace=trace, search=search, limit=3)
            if suite["corpus_version"] != assistant.corpus["version"]:
                raise RuntimeError("Replay cases and corpus use different versions.")
            started = time.perf_counter()
            result = await assistant.ask(case["question"])
            records.append({
                "case": case_id, "split": case["split"], "strategy": search.__name__,
                "status": result["status"], "trace_id": result["trace_id"],
                "citation_ids": [citation["id"] for citation in result["citations"]],
                "required_ids": case["required_passage_ids"], "answer": result["answer"],
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "usage": result["usage"],
            })

    mlflow.flush_trace_async_logging()
    for record in records:
        persisted = mlflow.get_trace(record["trace_id"])
        if persisted is None:
            raise RuntimeError("Replay trace was not persisted.")
        spans = {span.name: span for span in persisted.data.spans}
        attrs = spans["assistant"].attributes
        if attrs["model"] != model.name or attrs["prompt_version"] != PROMPT_VERSION:
            raise RuntimeError("Replay model or prompt differs from the persisted trace.")
        record.update(spans["retrieval"].outputs)
        record["spans"] = list(spans)
    return {"suite": suite["version"], "corpus": suite["corpus_version"], "k": 3,
            "model": model.name, "prompt": PROMPT_VERSION, "replays": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-db", type=Path, default=Path("mlflow.db"))
    parser.add_argument("--local-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(asyncio.run(replay(args)), indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
