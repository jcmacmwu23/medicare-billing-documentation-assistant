# Reviewed evaluation suite

[cases.json](cases.json) contains 40 cases approved by the user on November 13, 2026 and frozen as `reviewed-1`. The suite has 12 development cases and 28 held-out cases.

For each case, review the pinned corpus passages and synthetic fixtures, then confirm:

- The question is in scope or has the intended reason to abstain.
- Every required evidence ID is necessary and together the IDs support the expected policy facts.
- The expected response identifies any missing facts and limits claim-specific conclusions to facts supplied by the user or returned by an authorized lookup.
- The expected tool behavior includes exact arguments, whether a call is dispatched, and whether the fixture store is read. For the denied-access case, verify the future caller and server separately.
- Perturbations apply only to the test context. Review whether the remaining authentic evidence supports the expected answer or abstention.

Record review decisions with the case ID, reviewer, and date. For future label changes, record the decision and increment the suite version before scoring. Keep scenario families together when revising the split.

The retrieval runner scores only the 20 clean retrieval and citation prompts. It calculates macro recall as the average fraction of required IDs found per case. The all-required count measures how often every labeled passage is present at each cutoff. The end-to-end run below exercises behavior beyond retrieval.

## Reviewed retrieval baseline

See [per-case rankings and aggregate results](reviewed-retrieval-results.json). Of the 28 held-out cases, 13 clean prompts are scored for retrieval here.

| Method | Recall@1 | Recall@3 | Recall@5 | All required at 3 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 73.1% | 96.2% | 100% | 12/13 |
| BM25 + MiniLM | 65.4% | 96.2% | 100% | 12/13 |

Five passages make recall@5 a weak discriminator. Keep BM25 as the CLI default for this corpus. In case `retrieval-10`, the query needs both the Q0091 definition and the diagnosis pointer. BM25 ranks the definition fourth, and hybrid ranks the pointer fourth. The [retrieval replay](retrieval-replay.json) captures the top-three failure and its correction.

The frozen case file has SHA-256 `843b47b6382827e60bb33e546560be7525d971df58badbde7289db7598d324f2`.

## End-to-end runs

Both runs used the pinned Qwen model with prompt version `statement-order-v1` and actual local MCP and MLflow. Latencies cover the request path after initial Qwen loading. The hybrid measurement includes lazy index initialization.

| Retrieval | Outcome matches | Citation-ID matches | Tool matches | Median | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [BM25](assistant-bm25-results.json) | 40/40 | 40/40 | 40/40 | 2.17 s | 2.88 s |
| [Hybrid](assistant-hybrid-results.json) | 40/40 | 40/40 | 40/40 | 2.17 s | 4.07 s |

Each run exercised all eight abstention cases, rejected all six injected passages, and confirmed zero fixture reads for a direct unauthorized MCP call. Each used 27 model calls, 3,430 input tokens, 108 generated tokens, and $0 in provider API charges. Qwen returned 11 invalid orderings in each run. The assistant used the validated default order for those responses. Citation-ID matching compares the exact sets against the reviewed labels. The output reports retain the corresponding statements for inspection. These examples were used while developing the scoped rules, so the 40-case outcome is a regression result.


## Retrieval replay at three passages

Plain BM25 omitted the Q0091 definition on the development case (`retrieval-11`) and on the original two-part query (`retrieval-10`). Clause-level ranking resolved the development miss. The original query's short code clause also needed literal HCPCS matching. With both changes, all required passages were present for 7/7 development and 13/13 previously held-out clean prompts, up from 6/7 and 12/13. The original miss informed the final change, so this comparison measures the known regression. The five-passage baseline and the reviewed labels are preserved. The MLflow replay kept the model, prompt, corpus, and cutoff fixed.
