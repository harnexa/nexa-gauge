# NexaGauge — Intelligent LLM Evaluation That Works the Way You Think

## What Is NexaGauge?

**NexaGauge** is an agentic, graph-based LLM evaluation pipeline built for teams who need reliable, reproducible, and cost-aware evaluation of AI-generated outputs — from RAG systems to chat agents to red-teaming pipelines.

It turns raw LLM outputs into structured, multi-dimensional evaluation reports using a declarative node graph — without requiring you to write evaluation infrastructure from scratch.

---

## The Problem It Solves

- **LLM outputs are hard to trust.** Is your model hallucinating? Is it grounded in the context you gave it? Is it relevant to the question? Toxic? Biased?
- **Evaluation is expensive and slow.** Running full evaluations repeatedly burns LLM credits with no way to know cost upfront.
- **One-size-fits-all eval tools don't fit.** Different use cases (RAG, chat, red-teaming) need different metrics — and most tools force you to choose one framework.
- **There's no good middle ground between "vibe checks" and writing a full eval harness.**

NexaGauge bridges that gap.

---

## What Makes It Distinctive

**Graph-based pipeline architecture**
- Evaluation flows through a typed node graph: `scan → chunk → claims → dedup → [grounding | relevance | geval | redteam | reference] → eval → report`
- Each node does exactly one job. You can target any node and the system automatically resolves its prerequisites.
- No black boxes — every step is inspectable, reproducible, and independently testable.

**Multi-dimensional metrics out of the box**
- **Grounding** — Is the output factually grounded in the provided context? (Critical for RAG systems)
- **Relevance** — Are the claims in the output relevant to the question asked?
- **GEval** — Custom LLM-as-judge evaluation with configurable criteria and step-by-step reasoning
- **Red Team** — Automated bias and toxicity detection
- **Reference** — Classical NLP metrics (ROUGE, BLEU, METEOR) against ground-truth answers
- Run one metric or all of them in a single command

**Cost estimation before you spend a dollar**
- `nexagauge estimate <target>` gives you a full per-node cost breakdown *before* executing
- Shows cached vs. uncached eligible records, model routing, and projected spend
- No more surprise LLM bills from running evals on large datasets

**Cache-aware execution that saves real money**
- Every node result is cached with a fingerprint that includes: input content, model routing config, and the full dependency path
- Rerun only what changed — if your context changed but the generation didn't, only the relevant nodes re-execute
- Estimate mode can reuse run-mode cache, so you pay once

**Flexible data ingestion**
- Local files: `.json`, `.jsonl`, `.csv`, or plain text — no ETL setup required
- Hugging Face datasets: `hf://...` source with split/slice support
- `--start`, `--end`, `--limit` flags for controlled sampling

**Per-node model routing and fallbacks**
- Set a global model or override per node: `LLM_GROUNDING_MODEL`, `LLM_GEVAL_FALLBACK_MODEL`, etc.
- Route expensive nodes to cheaper models; use your best model only where it matters
- Fallback chains prevent silent failures when primary models are unavailable

**CLI-first, zero boilerplate**
- Two commands cover everything: `nexagauge run` and `nexagauge estimate`
- Works on your laptop, in CI, or in batch pipelines
- Reports are plain JSON — easy to pipe into dashboards, notebooks, or downstream systems

---

## Why It's Trustworthy

- **Deterministic caching** — same inputs + same model config = same cache hit, always
- **Typed contracts throughout** — `EvalCase` state, `Inputs`, and `NodeSpec` are fully typed; schema violations surface early
- **Fail-fast and continue-on-error modes** — you control whether one bad record halts the run
- **Result ordering preserved** — even with parallel execution, output order matches input order
- **MIT licensed, open source** — no vendor lock-in, no opaque scoring black boxes

---

## Why It's Effective

- **Claim-level evaluation** — outputs are chunked into discrete claims before scoring, making metrics more precise and interpretable than document-level scoring
- **Deduplication before scoring** — duplicate claims are removed before metric computation, preventing inflated scores
- **Parallel metric fan-out** — when running full eval, all metric nodes execute concurrently, dramatically reducing wall-clock time
- **Declarative reports** — `REPORT_VISIBILITY` controls exactly what surfaces in output; unused metric branches simply don't appear

---

## Why It's Adaptable

- Add new metrics as graph nodes — the topology and registry are designed for extension
- Swap LLMs globally or per-node with env vars — no code changes required
- Works with any dataset shape — the scanner normalizes field aliases automatically
- Runs on local files or cloud datasets with the same CLI interface

---

## Who Should Use It

| Use Case | What NexaGauge Gives You |
|---|---|
| **RAG system teams** | Grounding + relevance scores per chunk, per claim |
| **Chatbot developers** | GEval with custom rubrics, toxicity + bias checks |
| **ML researchers** | ROUGE/BLEU/METEOR against reference with cache-aware batching |
| **AI safety teams** | Automated red-teaming for bias and toxicity at scale |
| **Anyone running LLM evals** | Cost estimation + caching that actually saves money |

---

## One-Line Pitch

> **NexaGauge: Know what your LLM is doing, know what it will cost, and never re-run what hasn't changed.**

