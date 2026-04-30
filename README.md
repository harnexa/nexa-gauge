<div align="center">

<img src="nexagauge-banner.svg" alt="nexa-gauge" width="760" />

# nexa-gauge - Graph-Based Evaluation for LLM and RAG Systems

**A cache-aware evaluation engine for measuring LLM and RAG output quality with repeatable metrics, cost estimates, and structured reports.**

<p>
  <a href="https://github.com/harnexa/nexa-gauge/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/Sardhendu/nexa-gauge/ci.yml?branch=main&label=CI&style=flat-square" alt="CI status" /></a>
  <a href="https://pypi.org/project/nexa-gauge/"><img src="https://img.shields.io/pypi/v/nexa-gauge?style=flat-square&color=blue" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/nexa-gauge/"><img src="https://img.shields.io/pypi/pyversions/nexa-gauge?style=flat-square" alt="Python versions" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License" /></a>
  <a href="https://harnexa.dev/nexa-gauge/docs/introduction"><img src="https://img.shields.io/badge/docs-harnexa.dev-brightgreen?style=flat-square" alt="Documentation" /></a>
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/badge/uv-ready-purple.svg?style=flat-square" alt="uv ready" /></a>
</p>

<p>
  <strong><a href="https://harnexa.dev/nexa-gauge/docs/introduction">Read the Documentation</a></strong>
  &nbsp;&middot;&nbsp;
  <a href="#quickstart">Quickstart</a>
  &nbsp;&middot;&nbsp;
  <a href="#cli-usage">CLI Usage</a>
  &nbsp;&middot;&nbsp;
  <a href="https://github.com/harnexa/nexa-gauge/issues">Report Bug</a>
  &nbsp;&middot;&nbsp;
  <a href="https://github.com/harnexa/nexa-gauge/issues">Request Feature</a>
</p>

</div>

---

## What is nexa-gauge?

`nexa-gauge` is a Python package and command-line toolkit for evaluating generated outputs from LLM, RAG, and agentic systems. It replaces ad-hoc manual checks with a typed evaluation graph that can estimate cost, execute only the required nodes, reuse cached artifacts, and emit structured per-case reports.

It is designed for prompt iteration, benchmark runs, regression testing, release gates, and production evaluation workflows where teams need measurable quality and safety signals.

Core evaluation coverage includes:

- **Relevance** - measures whether generated claims answer the user question.
- **Grounding** - checks whether generated claims are supported by supplied context.
- **Red team scoring** - evaluates safety and risk behavior with configurable rubrics.
- **GEval / LLM-as-a-judge** - scores outputs against explicit criteria or evaluation steps.
- **Reference metrics** - computes overlap-based metrics against known reference answers.

---

## Quickstart

Install the package from PyPI:

```bash
pip install nexa-gauge
```

Install optional Hugging Face dataset support:

```bash
pip install "nexa-gauge[huggingface]"
```

Set your provider key:

```bash
export OPENAI_API_KEY="<your-key>"
```

Estimate cost before running billable evaluation work:

```bash
nexagauge estimate eval --input sample.json --limit 10
```

Run the full evaluation graph and write per-case reports:

```bash
nexagauge run eval --input sample.json --limit 10 --output-dir ./report
```

---

## Core Capabilities

- **Graph-based evaluation pipeline** - predictable node topology for scanning, chunking, claim extraction, metric execution, aggregation, and reporting.
- **Estimate-first execution** - preview uncached eligible cost before making LLM-backed calls.
- **Cache-aware runs** - avoid duplicate LLM spend and recomputation when inputs, prompts, and model routes are unchanged.
- **Structured JSON reports** - write per-case report files for CI, dashboards, notebooks, and downstream analytics.
- **Per-node model routing** - configure global models, node-specific models, fallback models, and temperatures.
- **Scalable CLI execution** - tune case-level workers, in-flight cases, and global LLM concurrency.
- **Local and hosted datasets** - evaluate JSON, JSONL, CSV, text files, and Hugging Face datasets.

---

## Evaluation Pipeline

`nexa-gauge` runs evaluations through a deterministic node graph. Each target executes only its required upstream dependencies.

| Category | Nodes | Purpose |
| --- | --- | --- |
| Input and orchestration | `scan`, `eval`, `report` | Normalize records, aggregate metric branches, and project final reports. |
| Utility | `chunk`, `refiner`, `claims`, `geval_steps` | Prepare generated text, select top-k chunks via configurable refinement, extract claims, and resolve GEval steps. |
| Metrics | `relevance`, `grounding`, `redteam`, `geval`, `reference` | Score answer quality, evidence support, safety, rubric alignment, and reference overlap. |

Typical execution paths:

```text
grounding: scan -> chunk -> refiner -> claims -> grounding
relevance: scan -> chunk -> refiner -> claims -> relevance
geval:     scan -> geval_steps -> geval
eval:      full graph execution and aggregate metric summary
```

For a full architecture diagram, see [docs/architecture.md](docs/architecture.md).

---

## CLI Usage

The CLI entry point is `nexagauge`.

```bash
nexagauge --help
nexagauge run --help
nexagauge estimate --help
nexagauge cache --help
```

Primary commands:

| Command | Purpose |
| --- | --- |
| `nexagauge estimate <target_node> --input <source>` | Estimate uncached cost for a target branch before execution. |
| `nexagauge run <target_node> --input <source>` | Execute a target branch and optionally write reports. |
| `nexagauge cache dir` | Print the resolved cache root directory. |
| `nexagauge cache delete` | Inspect or clear cached node outputs. |

Common examples:

```bash
# Estimate full evaluation cost for a dataset slice
nexagauge estimate eval --input sample.json --limit 100

# Run full evaluation and write JSON reports
nexagauge run eval --input sample.json --limit 100 --output-dir ./report

# Run full evaluation with explicit chunk/refiner strategies
nexagauge run eval --input sample.json --limit 100 --chunker semchunk --refiner mmr --refiner-top-k 3

# Run only the grounding metric branch
nexagauge run grounding --input sample.json --limit 25

# Preview cache cleanup
nexagauge cache delete --dry-run
```

Common flags:

| Area | Flags |
| --- | --- |
| Data selection | `--input`, `--adapter`, `--start`, `--end`, `--limit` |
| Model routing | `--model`, `--llm-model`, `--llm-fallback` |
| Strategy routing | `--chunker`, `--refiner`, `--refiner-top-k` |
| Caching | `--force`, `--no-cache` |
| Execution | `--max-workers`, `--max-in-flight`, `--llm-concurrency`, `--continue-on-error` |
| Debugging | `--debug` |
| Reports | `--output-dir` |

See [docs/cli-code-flow.md](docs/cli-code-flow.md) and the hosted [CLI documentation](https://harnexa.dev/nexa-gauge/docs/introduction) for deeper usage details.

---

## Input Formats

Use `--input` with local files or hosted datasets.

| Source | Example | Notes |
| --- | --- | --- |
| JSON | `sample.json` | Object or array of records. |
| JSONL | `dataset.jsonl` | One JSON object per line. |
| CSV | `dataset.csv` | Rows are loaded as dictionaries. |
| Text | `generation.txt` | Treated as a single generated output. |
| Hugging Face | `hf://org/dataset` | Requires `pip install "nexa-gauge[huggingface]"`. |

Canonical record fields include:

| Field | Used by |
| --- | --- |
| `generation` | Required for all metric branches. |
| `question` | Relevance and some GEval configurations. |
| `context` | Grounding and context-aware GEval checks. |
| `reference` | Reference metrics and reference-aware GEval checks. |
| `geval` | Rubric-driven GEval metrics. |
| `redteam` | Custom safety and risk rubrics. |

Common aliases such as `response`, `answer`, `output`, `completion`, `query`, `prompt`, `ground_truth`, and `label` are normalized during scanning.

---

## Metrics

`nexa-gauge` combines deterministic metrics with LLM-as-a-judge evaluation.

| Metric node | What it evaluates |
| --- | --- |
| `relevance` | Whether generated claims directly answer the question. |
| `grounding` | Whether generated claims are supported by the provided context. |
| `redteam` | Bias, toxicity, and custom risk behavior using rubrics. |
| `geval` | Criteria-based LLM judging with generated or provided evaluation steps. |
| `reference` | BLEU, METEOR, ROUGE-1, ROUGE-2, and ROUGE-L against reference answers. |

GEval is split into two phases:

1. `geval_steps` resolves reusable evaluation steps from criteria or accepts provided steps.
2. `geval` scores each case against those resolved steps and selected input fields.

This design makes rubric-based evaluation repeatable and cache-friendly across datasets.

---

## Caching and Cost Estimation

Cost control is a first-class part of the runtime.

```bash
# Preview uncached work before execution
nexagauge estimate eval --input sample.json --limit 50

# Reuse cache during normal runs
nexagauge run eval --input sample.json --limit 50 --output-dir ./report

# Ignore cache reads but still write fresh outputs
nexagauge run eval --input sample.json --limit 50 --force

# Disable cache reads and writes for debugging
nexagauge run eval --input sample.json --limit 50 --no-cache
```

The cache is deterministic and route-aware. Inputs, evaluation criteria, model routing, prompt versions, parser versions, and relevant upstream artifacts are included in cache keys so stale outputs are not reused across incompatible runs.

For `run`, cache location can be controlled with:

```bash
export NEXAGAUGE_CACHE_DIR="./.nexagauge-cache"
```

Inspect the active cache root:

```bash
nexagauge cache dir
```

Clear cached node outputs:

```bash
nexagauge cache delete --dry-run
nexagauge cache delete --yes
```

---

## Configuration

For local development or repeatable runs, copy the environment template:

```bash
cp .env.example .env
```

Minimum configuration for OpenAI-backed runs:

```bash
OPENAI_API_KEY=<your-key>
LLM_MODEL=gpt-4o-mini
```

Supported per-node overrides follow this pattern:

```bash
LLM_CLAIMS_MODEL=openai/gpt-4o-mini
LLM_CLAIMS_FALLBACK_MODEL=openai/gpt-4o
LLM_GROUNDING_TEMPERATURE=0.0
```

Runtime overrides can also be passed through the CLI:

```bash
nexagauge run eval \
  --input sample.json \
  --llm-model openai/gpt-4o-mini \
  --llm-model grounding=openai/gpt-4o \
  --llm-fallback openai/gpt-4o
```

---

## Development

Clone the repository and install it from source:

```bash
git clone https://github.com/harnexa/nexa-gauge.git
cd nexa-gauge
pip install -e .
```

Contributor workflow with `uv`:

```bash
uv sync
make lint
make test
make ci
```

Build distributions:

```bash
uv build
```

Expected artifacts:

```text
dist/nexa_gauge-<version>-py3-none-any.whl
dist/nexa_gauge-<version>.tar.gz
```

Releases use `release-please`. Conventional Commit titles such as `feat:`, `fix:`, `docs:`, `deps:`, and `chore:` are recommended for cleaner generated release notes, but they are not required by CI.

---

## Documentation

- Hosted documentation: [harnexa.dev/nexa-gauge/docs/introduction](https://harnexa.dev/nexa-gauge/docs/introduction)
- Local getting started guide: [docs/get-started.md](docs/get-started.md)
- Architecture: [docs/architecture.md](docs/architecture.md)
- CLI code flow: [docs/cli-code-flow.md](docs/cli-code-flow.md)
- Product summary: [docs/PRODUCT_SUMMARY.md](docs/PRODUCT_SUMMARY.md)

---

## Project Standards

- License: [MIT](LICENSE)
- Security policy: [SECURITY.md](SECURITY.md)
- Contributing guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.
