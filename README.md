# Protocol Risk and Exploit Intelligence Agent

Retrieval and agentic diagnosis over smart-contract audit evidence. Ask a
protocol security question, get a memo where every claim carries a citation that
resolves to a passage the retriever actually returned — or an explicit
abstention when the corpus cannot support an answer.

```bash
pip install -e ".[dev]"
make ingest      # build the SQLite manifest with dedup + chunking
make eval        # golden-set evaluation, fails on a regression
make ablate      # the 17-row ablation matrix
pria ask "How does an ERC777 hook enable reentrancy in a vault withdraw?"
```

> **Scope.** This is a deliberately small, fully reproducible system: 73
> documents, 98 chunks, a 25-question golden set. Every number below is produced
> by `make eval` on this machine from the committed corpus, not quoted from
> anywhere. Where a design choice does not pay off at this scale, the README
> says so rather than shipping the mechanism switched on.

---

## What it does

```
question
   │
   ├─ guard ─────────────► refuse            (query-channel prompt injection,
   │                                          or a deployable-exploit request)
   ├─ plan               (archetype + metadata filters)
   ├─ retrieve           (BM25 ∪ dense → RRF → top-k)
   ├─ ground ───────────► abstain            (the corpus does not cover this)
   ├─ code_analysis      (Solidity span extraction + pattern rules)
   ├─ synthesise ◄──┐
   ├─ critic ───────┘    (citations: attributed? resolvable? supported?)
   └─ finalise           (memo + evidence table + citation report)
```

`pria graph` prints this as mermaid; the edges are asserted in
[`tests/test_agent.py`](tests/test_agent.py).

---

## Results

`make eval`, 25-question golden set frozen before any retrieval work:

| | |
|---|---|
| nDCG@10 (19 answerable questions, graded labels) | **0.890** |
| MRR | 0.895 |
| Recall@10 | 0.921 |
| lexical / code MRR (`code_diagnostic` + `mitigation`) | 0.875 |
| citation attribution / resolvable / supported | 1.00 / 1.00 / 1.00 |
| adversarial split passed | **6 / 6** |
| mean critic loops per answer | 0.95 |
| p50 query latency, CPU, no GPU anywhere in the path | **1.1 ms** |
| dense store after int8 quantization | 9,898 B (3.8× smaller) |

By archetype:

| archetype | n | nDCG@10 | MRR |
|---|---|---|---|
| code_diagnostic | 4 | 1.000 | 1.00 |
| vuln_pattern | 5 | 0.898 | 1.00 |
| report_page | 2 | 0.898 | 0.75 |
| incident_lookup | 4 | 0.868 | 0.88 |
| mitigation | 4 | 0.788 | 0.75 |

The split matters: a change that lifts prose questions while breaking identifier
lookup shows up here and nowhere in the headline average.

---

## Orchestration: LangGraph, and a control to prove it

The agent runs on **LangGraph**. The topology - nodes, edges, routers, the
cycle bound - is declared once in `agent/spec.py` and compiled into a
`StateGraph`. Four things made the library worth the dependency, and each one
replaced something this codebase was maintaining by hand:

**Reducers put the merge rule in the type.** `_path` and `_checkpoints` are
`Annotated[list, operator.add]` on the state schema, so "a node returns a
partial update that is merged, never substituted" stops being a convention the
engine enforces and becomes a property a node *cannot* violate. It could not
overwrite the record of what ran if it tried.

**The checkpointer is the replay log.** Every super-step is persisted, so a run
can be replayed, resumed or diffed - the property the original engine
advertised, now durable rather than held in a list this module appends to.
State types crossing the checkpointer are registered explicitly; LangGraph
blocks deserialising arbitrary classes out of a checkpoint, and it is right to.

**Interrupts are the review gate.** A memo that fails the citation critic on
its last permitted loop is exactly the case a human should see before it ships.
With `human_in_the_loop=True` the graph stops before `finalise`, and `resume()`
continues from the persisted checkpoint - so the reviewer reads the answer and
the critique, not a finished document.

**`recursion_limit` bounds the cycle.** `synthesise -> critic -> synthesise` is
a real cycle. A critic that never passes terminates with a graph error instead
of spinning.

Span tracing is preserved either way: each node still runs inside a
`node:<name>` span, so a Phoenix trace is identical under both engines.

### The conformance test

`agent/graph.py` keeps a dependency-free walker over the same spec. It is not a
fallback anyone is expected to run - it is the **control**. Both engines call
the same router functions, so the test asserting one question produces an
identical path, memo, citation set and audit trail under both is asserting that
the two executors agree, not that two hand-written copies of a graph happen to
have been edited in step. The prompt-injection refusal path is asserted the
same way, because a security short-circuit that behaves differently per engine
is worse than no short-circuit.

```
pria ask "..." --engine langgraph
pria ask "..." --engine reference     # must produce the same memo
```

---

## Four RAG upgrades, all measured, all default off

Four techniques that are standard advice were implemented, measured on the
frozen golden set, and **turned off** - which is the finding, not a failure to
land them.

| upgrade | what it does | result |
|---|---|---|
| **Contextual chunk headers** | prepend title / source / severity / SWC to each chunk before indexing | nDCG unchanged (0.8900) |
| **MMR diversification** | trade relevance for non-redundancy, lambda-weighted | redundancy 0.104 -> 0.084 at l=0.5, but nDCG 0.890 -> 0.816 |
| **Multi-query decomposition** | split a compound question into facets, retrieve each, fuse with RRF | nDCG 0.890 -> 0.858; helps the question it targets (G-12 recall 0.50 -> 0.75) |
| **HyDE expansion** | hypothetical-document expansion before retrieval | nDCG 0.890 -> 0.853 |

**They lose for one reason, and it is a reason about scale rather than about
the techniques.** This corpus is 59 documents and the first stage already
returns recall@10 = 0.921. The entire achievable gain over BM25-only is 0.014
nDCG. Every one of these methods works by *adding candidates* or *trading
relevance for coverage*, and with no headroom left that only dilutes a ranking
that was already right. Each would earn its place on a corpus where the first
stage is noisy - which is exactly the condition the harness will detect,
because the rows stay in the matrix.

A new metric was added to make one of these trade-offs visible at all:

**`redundancy@10`** - the mean pairwise similarity of the returned passages.
The corpus contains genuine near-duplicates on purpose (the same finding
written up by two firms), so a configuration can win on nDCG while handing the
model ten restatements of one piece of evidence. No relevance metric says so.
MMR is the knob that moves it, and the ablation shows the curve.

---

## The ablation matrix

Full table in [`reports/ablation.md`](reports/ablation.md), regenerated by
`make ablate`. Every row is the same frozen golden set, so the deltas are
attributable.

| config | nDCG@10 | recall@10 | p50 ms |
|---|---|---|---|
| R01 BM25 only | 0.8756 | 0.9211 | 0.25 |
| R02 dense only, feature-hashed embedder | 0.5575 | 0.6623 | 0.31 |
| R03 dense only, LSA embedder | 0.8870 | 0.9211 | 0.69 |
| R08 BM25 + dense, RRF k=60 | 0.8900 | 0.9211 | 0.68 |
| R11 …+ feature reranker | 0.8837 | 0.8947 | 17.64 |
| R15 …+ HyDE expansion | 0.8529 | 0.8684 | 1.33 |
| R17 **final** (RRF + metadata filter + int8 + cache) | **0.8900** | 0.9211 | 1.55 |

The p50 column is wall-clock on one laptop core and moves a few tenths of a
millisecond between runs; the only row where it carries information is R11,
where reranking costs roughly 25x the fused retrieval it is refining.

### Three negative results, kept in

**HyDE costs 0.037 nDCG@10 and was removed.** Audit queries are already dense
with the exact identifiers that appear in the target passage — `latestRoundData`,
`get_virtual_price`, `SWC-107`. A generated pseudo-answer dilutes those with
generic security vocabulary that matches every finding in the corpus equally
well. `use_hyde` still exists so the row stays runnable, and
`test_hyde_expansion_is_a_documented_regression` is what stops it coming back.

**The reranker costs 0.006 nDCG@10 and 0.026 recall@10, and is off by
default.** At 73 documents the candidate list after RRF is already almost pure,
so there is nothing left for a reranker to fix and its own errors dominate —
reranking the top 40 down to 10 drops relevant documents that fusion had ranked
correctly. Raising the weight on the fused rank from 1.0 to 4.0 recovers most of
the loss but never beats not reranking at all. The mechanism is the one that
matters once fusion returns a noisy top-40, so it stays behind `rerank=True`
with the ablation row as the reason it is not the default. Shipping it on
because the architecture diagram calls for it would be shipping a regression.

**Late interaction loses to the single-vector baseline on this corpus.**
Page Recall@1: 0.50 for MaxSim over layout blocks, 0.83 for one vector per page.
At Recall@3 both saturate at 1.00. The reason is page length: these pages
average 148 words, so their first 40 words — the "caption" — already carry the
title and the topic sentence, and pooling loses almost nothing. Late interaction
earns its cost when the answer is one row of a table on a 600-word page and the
pooled vector averages it away. The mechanism is implemented properly (MaxSim,
one vector per block, both indexes sharing a single fitted latent space so the
comparison is apples to apples) and it is measured honestly; the corpus is
simply not the regime where it pays.

The one clear win from the embedding side: **R02 vs R03**. A feature-hashed
embedder scores 0.5575; the same pipeline with a corpus-fitted LSA projection
scores 0.8870. Dense retrieval is worth having, but only if the embedding
actually models the corpus.

---

## Corpus

`make ingest` builds `artifacts/manifest.sqlite`:

| source | documents |
|---|---|
| Code4rena contest findings | 22 |
| Sherlock contest findings | 10 |
| rekt.news incident post-mortems | 12 |
| Spearbit review pages | 8 |
| Cantina competition report pages | 4 |
| Solidity sources (diagnostic fixtures) | 4 |
| SWC taxonomy entries | 13 |
| **active total** | **73** (98 chunks) |

Rejected as duplicates and excluded from every index:

```
c4r-0003b        near   of c4r-0003   jaccard 0.555
c4r-0011b        near   of c4r-0011   jaccard 0.680
c4r-0017-mirror  exact  of c4r-0017   jaccard 1.000
```

### Why the near-duplicate threshold is 0.50

The usual 0.8 MinHash threshold is tuned for web-scale near-exact dedup. Two
contests reporting the same root cause reuse the structure but paraphrase the
prose: the hand-labelled duplicate pairs in this corpus sit at 0.53 and 0.77.
At 0.8 the detector finds only the byte-identical pair, which content hashing
already catches for free, and the near-duplicate machinery earns nothing.

### Why ingestion is resumable, and one bug that fell out of it

Every unit of work is keyed by content hash and written to the manifest before
the next document is touched, so a second pass over an unchanged corpus reports
zero new documents. Building that surfaced a real bug: `known_hashes()`
originally mapped every content hash to a document regardless of status, so on a
resumed run the *original* of an exact-duplicate pair could resolve to its own
duplicate and be demoted instead. Which document survived depended on SQLite row
order. Only active documents may own a hash now, and
`test_ingest_is_resumable_and_idempotent` asserts the duplicate classification is
identical across runs.

---

## Three indexes, three evidence types

| index | unit | why |
|---|---|---|
| BM25 (`index/bm25.py`) | chunk | Audit queries carry rare exact tokens a 128-dim projection blurs. The tokenizer emits `latestRoundData` **and** latest/round/data, so a query written either way hits. |
| Dense (`index/dense.py`) | chunk | TF-IDF + truncated SVD, fitted on the corpus. Generalises over synonymy; int8-quantized with float32 rescoring of the top candidates. |
| Late interaction (`index/multivector.py`) | report page | One vector per layout block, scored `Σᵢ maxⱼ qᵢ·dⱼ`. Compared against a single-vector caption baseline. |

Solidity does not get a token window. `solidity/ast_lite.py` cuts each file into
declaration-aligned spans — contract, function, modifier, constructor — so a
retrieved hit is always a whole callable with its signature attached. The
brace-matching parser blanks string literals and comments *character for
character*, keeping every offset and line number exact, which is what makes the
reported line numbers trustworthy. `tree-sitter-solidity` is used instead when
installed; CI runs the fallback so the default install has no native build step.

---

## Grounding, and refusing to answer

Retrieval always returns something. On a question the corpus cannot answer it
returns the ten least-irrelevant passages, and any synthesiser handed ten
passages will write a fluent answer out of them. `agent/grounding.py` sits
between retrieval and synthesis with two signals:

- **Unknown entity** — the question names a proper noun the corpus has never
  seen. High precision, and the signal that actually fires. Months, weekdays and
  sentence-initial capitals are excluded, or "How was the Ronin bridge
  compromised in **March 2022**?" would abstain on a perfectly answerable
  question. Both cases are tested.
- **IDF-weighted coverage** — a backstop floor set below the minimum coverage
  observed on the answerable split (0.207), so it is deliberately conservative.

Without this gate the "what was the loss in the Hyperliquid bridge exploit"
question produced a confident, well-cited, entirely wrong answer assembled from
Ronin and Wormhole passages.

### Prompt injection

`security/injection.py` separates two threats. Query-channel injection is
refused before retrieval runs, by pattern match rather than a model call — the
defence must not itself be steerable by the input it is defending against.
Corpus-channel injection is handled by treating every retrieved passage as
untrusted data and defanging instruction-shaped text before it reaches a prompt.
All 6 adversarial questions pass, covering instruction override, role
escalation, prompt exfiltration, a deployable-exploit request against a named
live address, a false-premise question, and an instruction to drop citations.

---

## The citation critic

Three checks, in increasing strictness:

1. **Attribution** — does every factual sentence carry a citation?
2. **Resolution** — does the cited id appear in the evidence retrieved *for this
   question*? A citation to a chunk the retriever never returned is fabricated
   even when the chunk exists in the corpus.
3. **Support** — does the cited passage lexically support the sentence? This
   catches a citation attached to the wrong passage, which a bare id check
   misses entirely.

Two bugs this caught during development, both of which produced a memo that
looked correct:

- Citations were emitted **after** the sentence terminator (`… balance. [c4r-0001]`).
  Every sentence splitter then attributed each citation to the *following*
  sentence, silently mis-attributing the whole memo. Measured support rate:
  0.586. Citations now sit inside the sentence.
- The citation regex was `\[([\w.:-]+)\]`, so Solidity array syntax —
  `shares[msg.sender]`, `stakers[i]` — parsed as citations to non-existent
  chunks. The marker is `[cite:<id>]` now, which Solidity cannot produce.
  `test_solidity_index_syntax_is_not_mistaken_for_a_citation` pins it.

**Read the citation numbers correctly.** The default synthesiser is extractive:
it selects sentences from retrieved passages and carries their citation with
them, so attribution, resolution and support are near-ceiling by construction.
That is not evidence about an LLM's honesty. It is a regression test on the
citation plumbing and the critic loop — exactly the two things that were broken
above — and it is the reason CI can gate on it deterministically. Point
`PRIA_LLM=anthropic` at the Claude backend and the same three numbers become a
real measurement of the model.

---

## Configuration and backends

Everything an ablation touches lives in `config.py`. Heavy dependencies are all
optional and nothing in the default install or in CI imports torch:

| extra | what it swaps in |
|---|---|
| `[models]` | `bge-small-en-v1.5` embeddings, `ms-marco-MiniLM` cross-encoder |
| `[solidity]` | tree-sitter Solidity grammar instead of the brace-matching parser |
| `[pdf]` | `pdfplumber` page extraction instead of the pre-extracted page corpus |
| `[serving]` | Qdrant, Phoenix/OTel span export |
| `[llm]` | Anthropic Messages API synthesis (`claude-opus-5`, adaptive thinking) |

`tracing.py` writes OTel-shaped spans to `artifacts/spans.jsonl` always, and
forwards to Phoenix when `PRIA_PHOENIX_ENDPOINT` is set.

---

## Commands

```
pria ingest [--rebuild]     build or refresh the manifest
pria stats                  corpus statistics and detected duplicates
pria query "<q>" [--explain]  retrieval only, with reranker features
pria ask "<q>" [--trace]    full agent, prints the memo and graph path
pria eval [--no-gate]       golden-set evaluation; non-zero exit on regression
pria ablate                 regenerate reports/ablation.md
pria analyze <file.sol>     span extraction and pattern rules on one file
pria graph                  print the agent graph as mermaid
```

## Layout

```
data/raw/          corpus: findings, post-mortems, report pages, Solidity, taxonomy
data/golden/       the frozen 25-question golden set
src/pria/ingest/   manifest, MinHash/LSH, loaders, resumable pipeline
src/pria/index/    bm25, embeddings, dense store, fusion, reranking, late interaction,
                   mmr diversification, query decomposition
src/pria/solidity/ span extraction and vulnerability pattern rules
src/pria/agent/    spec (topology), langgraph engine, reference walker, grounding gate, critic
src/pria/eval/     metrics, eval harness, ablation matrix
tests/             69 tests, including the cross-engine conformance test; `make test`
```

## License

MIT.
