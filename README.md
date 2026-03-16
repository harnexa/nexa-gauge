# LumisEval

> Agentic LLM evaluation pipeline — orchestrate RAGAS, DeepEval, and Giskard from a single command, with upfront cost estimates and per-claim source citations.

```
  ██╗     ██╗   ██╗███╗   ███╗██╗███████╗    ███████╗██╗   ██╗ █████╗ ██╗
  ██║     ██║   ██║████╗ ████║██║██╔════╝    ██╔════╝██║   ██║██╔══██╗██║
  ██║     ██║   ██║██╔████╔██║██║███████╗    █████╗  ██║   ██║███████║██║
  ██║     ██║   ██║██║╚██╔╝██║██║╚════██║    ██╔══╝  ╚██╗ ██╔╝██╔══██║██║
  ███████╗╚██████╔╝██║ ╚═╝ ██║██║███████║    ███████╗ ╚████╔╝ ██║  ██║███████╗
  ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚══════╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
```

## What it does

LumisEval is an agentic evaluation pipeline for LLM-generated content. It decomposes any
output into atomic claims, routes each claim to the cheapest verification source (local files →
enterprise LanceDB → web search), and dispatches the right eval tool — RAGAS for RAG quality,
DeepEval for LLM judges, Giskard for adversarial probes — rather than reimplementing them.
Every run starts with a cost estimate you must acknowledge, and every verdict cites its source.

## Project structure

```
lumis-eval/
├── packages/
│   ├── lumiseval-core/         # Shared types, config, errors (Pydantic)
│   ├── lumiseval-ingest/       # Metadata scanner + semchunk chunker
│   ├── lumiseval-evidence/     # Evidence router, LanceDB indexer, MMR dedup
│   └── lumiseval-agent/        # LangGraph orchestration + metric nodes
│       └── lumiseval_agent/
│           ├── graph.py
│           └── nodes/
│               ├── claim_extractor.py
│               ├── cost_estimator.py
│               ├── aggregation.py
│               └── metrics/
│                   ├── ragas_node.py
│                   ├── deepeval_node.py
│                   ├── giskard_node.py
│                   └── rubric_node.py
├── apps/
│   ├── lumiseval-api/          # FastAPI REST API
│   └── lumiseval-cli/          # Typer CLI
├── infra/                  # Placeholder for Docker / Terraform
├── docs/
│   └── architecture.md
├── .github/workflows/ci.yml
├── pyproject.toml          # Root workspace config
├── Makefile
└── setup.sh
```

## Packages

| Package | Purpose |
|---------|---------|
| `lumiseval-core` | Shared Pydantic types, pydantic-settings config, custom errors |
| `lumiseval-ingest` | Token-accurate metadata scanner (tiktoken) + semantic chunker (semchunk) |
| `lumiseval-evidence` | Evidence router (local LanceDB → MCP → Tavily), MMR deduplicator, LanceDB indexer |
| `lumiseval-agent` | LangGraph orchestration graph, claim extractor (instructor), RAGAS / DeepEval / Giskard / Rubric metric nodes, cost estimator, aggregation |
| `lumiseval-api` | FastAPI REST interface (`POST /jobs`, `GET /jobs/{id}/report`) |
| `lumiseval-cli` | Typer CLI (`lumiseval eval`, `lumiseval estimate`, `lumiseval batch`) |

## Quick start

```bash
# 1. Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and set up
git clone <repo>
cd lumis-eval
make install

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env: add OPENAI_API_KEY (and TAVILY_API_KEY if using web search)

# 4. Evaluate a generation
lumiseval eval --input my_blog_post.txt

# 5. Get a cost estimate first (no LLM calls)
lumiseval estimate --input my_blog_post.txt

# 6. Start the REST API
make api
# Then: POST http://localhost:8080/jobs
```

## Development

```bash
make lint        # ruff linter
make format      # ruff formatter
make typecheck   # mypy
make test        # pytest
make test-verbose # pytest -v
make clean       # remove build artifacts
```

## Evaluation modes

| Mode | Command | Description |
|------|---------|-------------|
| Single generation | `lumiseval eval --input blog.txt` | Claim extraction → evidence routing → RAGAS + DeepEval |
| Cost estimate only | `lumiseval estimate --input blog.txt` | No LLM calls; returns cost breakdown with ±20% band |
| Rubric evaluation | Pass `RubricRule` list to API or rubric file to CLI | G-Eval per rule; compliance rate + composite adherence |
| Adversarial probes | `lumiseval eval --adversarial` | Giskard scan + DeepEval Privacy/Bias metrics |
| Batch dataset | `lumiseval batch dataset.jsonl` | Cost estimate first; each record needs a `generation` field |
| REST API | `POST /jobs` | Programmatic integration |

## Roadmap

- [ ] Async batch processing via TaskIQ with streamed CLI progress
- [ ] SQLite persistence (SQLModel) for job records, reports, and cost actuals
- [ ] MCP LanceDB retrieval for enterprise knowledge bases
- [ ] Rubric Extractor agent — auto-derive rules from reference documents (UC-3)
- [ ] OpenTelemetry trace export (Arize Phoenix compatible)
- [ ] Cost feedback loop — estimate vs. actual tracking for improved heuristics

## License

MIT
