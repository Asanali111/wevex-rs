# Chunk-size benchmark

Which chunk size and format gives the best recall per token for Wevex, using
the same embedder the Rust port ships (`crates/embed`, BAAI/bge-small-en-v1.5,
512-token cap)?

## Setup

- **Corpus:** 119 Python files from the legacy Wevex (Skein) codebase,
  rebuilt from the `chunks` table of a copy of the live DB.
- **Queries:** 201 queries, each the first sentence of a function or class
  docstring. The docstrings are stripped from the indexed text, so the answer
  is never in the chunk verbatim.
- **Scoring:** a hit means a top-k chunk contains the target `def` line.
- **Retrieval modes:** BM25 (SQLite FTS5, porter), dense (bge-small, cosine),
  and hybrid (reciprocal-rank fusion of the top 50 from each, k=60).
- **Token counts:** the bge tokenizer, not a character estimate.
- **Preamble:** `File: <path>. Module: <module docstring line>. Defines: <symbols in chunk>.`
  It is built from the AST with no LLM, and costs about 30–60 tokens per chunk.
- **Overlap:** 15% between chunks.
- **`current`:** the legacy indexer's 80-line windows.

Run: `python bench.py <copy-of-wevex.db> [cache.npz]` using a Python
environment with `fastembed` (the onnx-quantized bge-small).

## Results (hybrid unless noted)

| Config | Mean tok | Over 512 | BM25 r@3 | Vec r@3 | r@1 | r@3 | r@5 | MRR | Top-3 tokens |
|---|---|---|---|---|---|---|---|---|---|
| current | 797 | 84% | 0.537 | 0.318 | 0.209 | 0.423 | 0.537 | 0.328 | 2486 |
| 150 | 120 | 0% | 0.254 | 0.398 | 0.199 | 0.363 | 0.458 | 0.288 | 363 |
| 300 | 251 | 0% | 0.343 | 0.363 | 0.234 | 0.408 | 0.507 | 0.333 | 749 |
| 400 | 334 | 0% | 0.388 | 0.408 | 0.289 | 0.458 | 0.542 | 0.380 | 1006 |
| 500 | 415 | 0% | 0.398 | 0.393 | 0.259 | 0.458 | 0.537 | 0.367 | 1246 |
| 800 | 642 | 85% | 0.493 | 0.383 | 0.264 | 0.433 | 0.532 | 0.357 | 1999 |
| 150 + pre | 158 | 0% | 0.323 | 0.408 | 0.244 | 0.433 | 0.527 | 0.346 | 472 |
| 300 + pre | 293 | 0% | 0.403 | 0.423 | 0.224 | 0.483 | 0.602 | 0.362 | 870 |
| **400 + pre** | 379 | 0% | 0.453 | 0.498 | 0.323 | 0.517 | 0.637 | 0.439 | **1129** |
| **500 + pre** | 462 | 15% | 0.493 | 0.502 | **0.353** | **0.542** | **0.667** | **0.463** | 1386 |
| 800 + pre | 698 | 87% | 0.527 | 0.468 | 0.308 | **0.542** | 0.657 | 0.440 | 2136 |

The size label is the budget for the body. The preamble comes on top of it.

## Findings

1. **The 512-token embedder cap is the hard ceiling.** The current chunks
   (mean 797 tokens) and the 800 setup are 84–85% over 512, so their tails are
   never embedded. Vector recall@3 falls to 0.32–0.38, and hybrid recall drops
   with it. BM25 does best on big chunks only because a bigger chunk is a
   bigger target.
2. **The preamble is the biggest single gain.** At every size it adds 4–10
   points of hybrid recall@3 and 6–13 points of recall@5, for about 120 extra
   tokens across the top 3.
3. **Hybrid beats either mode alone** once chunks fit under the cap. With the
   preamble, BM25 and vectors are about equal on their own, and fusion adds
   4–5 points.
4. **The best setups are 400 + pre and 500 + pre.** Both beat `current` on
   every hybrid metric while returning roughly half the tokens.
   - 500 + pre has the best raw recall (r@1 0.353, r@5 0.667), but 15% of its
     chunks spill past 512.
   - 400 + pre trails it by 1–3 points (2–6 of 201 queries, within noise) and
     fits under the cap with headroom.
5. **Going past 512 buys nothing.** 800 + pre ties 500 + pre on r@3 (0.542)
   but is lower on r@1, r@5 and MRR, and returns about 54% more tokens. Most
   of each chunk is truncated before embedding.
6. **Small chunks are the most token-efficient but miss more.** 150 + pre gets
   0.433 r@3 on about 470 tokens, which is good for a tight budget, but r@1
   and MRR are weak.

## Recommendation for wevex-rs

- **Code and doc chunks:** a 400–450 token body plus a structural preamble,
  with the whole chunk hard-capped at 512 bge tokens. Split on AST or heading
  boundaries where possible. Use 10–15% overlap.
- **Retrieval:** hybrid (BM25 + bge, RRF), returning the top 3–5. That is about
  1.1–1.9k tokens per recall.
- **Preamble:** always add one. Embed and index the preamble together with the
  body. Optionally strip it from the text returned to the agent.
- **Memory tiers** (proposed in the same discussion, not measured here):
  - Theses: 10–40 tokens, one conclusion each.
  - Fragments (decisions): 50–300 tokens, one decision each. Merge anything
    under about 40 tokens and split anything over about 400.
  - Chunks: as above.
  - Every tier carries a preamble, and each thesis links to its fragment and
    its chunks.

## Caveats

- Code-only corpus with synthetic docstring queries. Prose fragments and real
  user questions may behave differently.
- 201 queries means differences of 2–3 points are noise.
- fastembed runs the onnx-quantized bge-small, while wevex-rs runs full-precision
  candle. Scores should be close but not identical.
- The preamble is structural (path, module, symbols). An LLM-written preamble
  (Anthropic's contextual retrieval) would likely gain more, at a per-chunk
  cost.
