# Medicare billing documentation assistant

A local retrieval-augmented assistant that helps billing analysts find and explain CMS guidance for frequency denials on Medicare Part B screening Pap specimen collection claims (HCPCS Q0091). All claims are synthetic.

[Open the public portfolio demo](https://d3t0pbgyrbj00q.cloudfront.net/) for the recorded claim walkthrough and retrieval replay.

## Run locally

Requires Python 3.11 or newer. Run from the project root. These commands use the standard library and local fixtures.

```sh
PYTHONPATH=src python3 -m billing_assistant search 'CARC 119 M83' --limit 1
PYTHONPATH=src python3 -m billing_assistant claim SYN-0001 --allow-lookup
PYTHONPATH=src python3 -m billing_assistant claim SYN-0001
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Search returns section-tagged BM25 hits with source provenance. The third command demonstrates the default denied path and exits with code 1. The lookup flag grants this local invocation access to the synthetic fixture. The analyst workflow below uses a separate MCP process with its own default-deny check.

## Source scope

The pinned [CMS Pub. 100-04, Chapter 18 PDF](sources/cms-100-04-ch18.pdf) is the source for the five indexed excerpts across §§30.2, 30.5.A, 30.6.A, and the Pap smear portion of §30.9. The [corpus manifest](src/billing_assistant/fixtures/cms_ch18.json) records the SHA-256, section revisions, one-based PDF page anchors, and evidence IDs. The indexed diagnosis list and pointer instruction each have their own evidence ID. Only the manifest passages are indexed.

To verify the source checksum and regenerate the manifest, install the optional PDF dependency and run the extractor from the project root.

```sh
python3 -m pip install '.[pdf]'
python3 tools/build_corpus.py
```

`SYN-0001` is the routine 119/M83 frequency-denial fixture. `SYN-0002` has missing prior screening history and demonstrates a partial answer that abstains on the claim-specific conclusion.

## Reviewed retrieval evaluation

The [evaluation suite](evaluation/README.md) contains 40 user-reviewed cases frozen as `reviewed-1`. Run BM25 without dependencies or compare it with hybrid retrieval using pinned MiniLM embeddings on the CPU. The first hybrid run downloads public model files into `.model-cache/`. Later runs can set `HF_HUB_OFFLINE=1` after that cache is populated. Use Python 3.12 or another Python with a compatible PyTorch wheel for the optional hybrid path.

```sh
PYTHONPATH=src python3 tools/evaluate_retrieval.py
python3 -m pip install '.[hybrid]'
HF_HOME="$PWD/.model-cache" HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=src python3 tools/evaluate_retrieval.py --hybrid
```

The retrieval runner scores the 20 retrieval and citation cases against the reviewed evidence IDs. It does not score answer correctness, tool authorization, or prompt-injection resistance.

## Run the analyst workflow

Install the local MCP and MLflow dependencies into a virtual environment. Use the project root for each command.

```sh
python3 -m pip install '.[runtime]'
python3 -m billing_assistant ask 'Look up SYN-0001. Why might Q0091 show 119/M83, given the last covered smear and service month?' --allow-lookup --trace-db mlflow.db
python3 -m billing_assistant ask 'Look up SYN-0002 and tell me if the routine screening was too early.' --allow-lookup --trace-db mlflow.db
python3 -m billing_assistant ask 'Look up SYN-0001.' --trace-db mlflow.db
```

The CLI returns page-linked citations. The third command demonstrates an authorization denial and exits with status 1. Authorization and validated ID routing happen before MCP dispatch. The MCP server exposes one read-only tool and checks process-level authorization before opening its synthetic store.

The default path uses deterministic statement ordering for fixture tests. To use the pinned Qwen2.5-0.5B-Instruct model on CPU, install the optional hybrid dependencies and allow the initial public weights download. Subsequent runs use the cache. The model orders statements selected by scoped rules and supported by authenticated retrieved passages. Only validated statement numbers can affect the answer. The model is never given MCP credentials, tool access, raw retrieved instructions, or permission to write claim data.

```sh
python3 -m pip install '.[hybrid]'
HF_HOME="$PWD/.model-cache" HF_HUB_DISABLE_TELEMETRY=1 python3 -m billing_assistant ask 'What does Q0091 describe?' --local-model --download-model
```

Run the reviewed end-to-end suite with the fixture path or add `--local-model` after caching Qwen. Add `--hybrid` to compare retrieval using cached MiniLM weights. The model cap is 96 calls per process, 2,800 input tokens per call and 80 generated tokens per call. Local CPU inference has zero provider API charges. It uses local compute and disk.

```sh
HF_HOME="$PWD/.model-cache" HF_HUB_OFFLINE=1 PYTHONPATH=src python3 tools/evaluate_assistant.py --trace-db mlflow.db
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1
```

SQLite traces capture timing, model and prompt versions, token totals, evidence IDs, and tool and response status. Trace payloads use allowlisted metadata. Raw questions, claim values, prompts, and answer text stay out of traces. Keep the SQLite database and MLflow artifacts local and out of version control.

## Reproduce the retrieval failure and fix

The replay uses a three-passage cutoff, one pinned corpus, and the same answer configuration in both arms. Plain BM25 can miss the Q0091 definition in a two-part query. The alternative searches each explicit question clause, prefers literal HCPCS matches for clauses that name a code, and deduplicates before filling the remaining slots from the full-query ranking. This experimental ranking is available through the replay. The analyst CLI uses its original BM25 configuration.

```sh
HF_HOME="$PWD/.model-cache" HF_HUB_OFFLINE=1 PYTHONPATH=src python3 tools/replay_retrieval.py --local-model
```

The runner returns trace IDs after reading the persisted spans back from MLflow. Open those IDs in the local experiment. See [the recorded comparison](evaluation/retrieval-replay.json) for the previously measured rankings and outcomes.
