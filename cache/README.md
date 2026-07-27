# Cache contents

This directory contains only request-bound responses from externally deployed
services. It does not contain datasets, retrieval indexes, fixed experiment
inputs, paper metric tables, or complete historical result traces.

- `deepseek/`: request-bound cached responses for attack generation, query
  rewriting, target-model answering, and SIGMA answer judging.
- `lakera/`: request-bound cached Lakera decisions.

Every external response record includes a SHA-256 digest of the request rebuilt
by the released code. Released response files are immutable inputs. The lookup
order for RQ1 target answers is:

1. matching released response;
2. matching response from a previous runtime fallback;
3. live DeepSeek request using the current retrieval and prompt.

Steps 1 and 2 make no network request. Step 3 writes the result under
`deepseek/runtime_target_responses/`; if a new SIGMA answer also changes its
verifier request, that judgement is written under
`deepseek/runtime_sigma_judgements/`. Runtime files use the same flat
`unit_id -> response record` shape as the released files. Every record carries
the current request digest, so a stale response is never used for a new prompt.
The released files are never overwritten.

The released archive may not initially contain either `runtime_*` directory;
they are created on the first live fallback. DeepSeek credentials are required
only for that fallback. Attack generation, query rewriting, and Lakera remain
strict request-bound release caches.

## TREC-COVID / BM25 / DCMI

The original paper-time response trace for this configuration was accidentally
overwritten. We therefore reran all 400 target-model queries with the released
queries, BM25 retrieval, prompts, model parameters, and fallback rule. The
exact responses from that complete rerun are stored in
`deepseek/target_responses/dcmi/trec-covid_bm25.json`, with provenance and
request digests retained for every experiment unit.

This cache reproduces the updated reported accuracy of 0.675 (135/200) and is
read through the same cache interface as every other RQ1 configuration.
