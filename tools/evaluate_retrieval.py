"""Measure clean, answerable retrieval cases against draft evidence labels."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from billing_assistant.fixtures import load_fixture
from billing_assistant.retrieval import (
    EMBEDDING_MODEL,
    EMBEDDING_REVISION,
    HybridRetriever,
    search_bm25,
)

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (1, 3, 5)


def evaluate(cases, search):
    results = []
    for case in cases:
        required = set(case["required_passage_ids"])
        hits = [hit["id"] for hit in search(case["question"])]
        results.append({
            "id": case["id"],
            "split": case["split"],
            "required_passage_ids": sorted(required),
            "ranked_passage_ids": hits,
            "relevant_at_k": {k: len(required.intersection(hits[:k])) for k in CUTOFFS},
        })

    by_split = {}
    for split in ("dev", "held_out"):
        rows = [result for result in results if result["split"] == split]
        by_split[split] = {
            "case_count": len(rows),
            "required_evidence_count": sum(len(row["required_passage_ids"]) for row in rows),
            "metrics": {
                k: {
                    "macro_recall_at_k": sum(
                        row["relevant_at_k"][k] / len(row["required_passage_ids"]) for row in rows
                    ) / len(rows),
                    "all_required_at_k": sum(
                        row["relevant_at_k"][k] == len(row["required_passage_ids"]) for row in rows
                    ),
                }
                for k in CUTOFFS
            },
        }
    return {"by_split": by_split, "cases": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid", action="store_true", help="Also run local MiniLM and BM25 rank fusion.")
    parser.add_argument("--output", type=Path, help="Write the JSON report to this path.")
    args = parser.parse_args()

    corpus = load_fixture("cms_ch18.json")
    suite = json.loads((ROOT / "evaluation/cases.json").read_text(encoding="utf-8"))
    if suite["corpus_version"] != corpus["version"]:
        parser.error("Evaluation suite and corpus versions differ.")

    passages = corpus["passages"]
    passage_ids = {passage["id"] for passage in passages}
    for case in suite["cases"]:
        if not set(case["required_passage_ids"]) <= passage_ids:
            parser.error(f"Unknown evidence ID in case {case['id']}.")

    cases = [
        case for case in suite["cases"]
        if case["category"] in {"retrieval", "citation"} and case["required_passage_ids"]
    ]
    report = {
        "corpus_version": corpus["version"],
        "suite_version": suite["version"],
        "review_status": suite["review_status"],
        "scored_categories": ["retrieval", "citation"],
        "methods": {"bm25": evaluate(cases, lambda query: search_bm25(query, passages))},
    }
    if args.hybrid:
        hybrid = HybridRetriever(passages, cache_directory=str(ROOT / ".model-cache"))
        report["embedding"] = {
            "model": EMBEDDING_MODEL,
            "revision": EMBEDDING_REVISION,
            "sentence_transformers": version("sentence-transformers"),
            "torch": version("torch"),
        }
        report["methods"]["hybrid"] = evaluate(cases, hybrid.search)

    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
