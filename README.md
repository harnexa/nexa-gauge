<p align="center">
  <img src="lumiseval-banner.svg" alt="Lumis Eval" width="720" />
</p>



# LumisEval

> Agentic LLM evaluation pipeline — decompose any LLM output into claims, verify each one with custom LLM judges and DeepEval, with upfront cost estimates and per-claim verdicts.

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
output into atomic claims and evaluates them across four independent metric branches —
faithfulness (grounding), answer relevancy, adversarial probes (bias + toxicity via DeepEval),
and custom GEval metrics — all via batched LLM-judge calls. Cost estimation is available
as a dedicated preflight command, and every verdict includes per-claim ACCEPTED/REJECTED details.

## Project structure

```
lumis-eval/
├── packages/
│   ├── lumiseval-core/         # Shared types, config, errors (Pydantic)
│   ├── lumiseval-ingest/       # Metadata scanner + semchunk chunker
│   ├── lumiseval-evidence/     # Evidence router, LanceDB indexer, MMR dedup
│   └── lumiseval-graph/        # LangGraph orchestration + metric nodes
│       └── lumiseval_graph/
│           ├── graph.py
│           └── nodes/
│               ├── claim_extractor.py
│               ├── cost_estimator.py
│               ├── eval.py
│               └── metrics/
│                   ├── relevance.py
│                   ├── grounding.py
│                   ├── redteam.py
│                   └── geval/
├── apps/
│   ├── lumiseval-api/          # FastAPI REST API
│   └── lumiseval-cli/          # Typer CLI
├── infra/                  # Placeholder for Docker / Terraform
├── docs/
│   ├── architecture.md
│   ├── cli-code-flow.md
│   ├── execution-model.md
│   └── get-started.md
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
| `lumiseval-graph` | LangGraph orchestration graph, node runners, claim extraction, relevance/grounding/redteam/geval metrics, and final eval |
| `lumiseval-api` | FastAPI REST interface (`POST /jobs`, `GET /jobs/{id}/report`) |
| `lumiseval-cli` | Typer CLI (`lumiseval run <node_name> --input <source>`) |

## Docs

- `docs/get-started.md` — setup, node list, and CLI usage
- `docs/cli-code-flow.md` — end-to-end call flow for `estimate` and `run`
- `docs/execution-model.md` — traced execution model, including which parts run sequentially vs in parallel

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

# 4. Estimate a branch
lumiseval estimate eval --input sample.json

# 5. Run pipeline to final eval
lumiseval run eval --input sample.json

# 6. Or run to an intermediate node
lumiseval run grounding --input sample.json

# 7. Start the REST API
make api
# Then: POST http://localhost:8080/jobs
```

## Development

```bash
make ci          # full local CI: format check → lint → test
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
| Preflight only | `lumiseval estimate eval --input sample.json` | Scans selected cases, prints estimate, and exits |
| Claims stage | `lumiseval run claims --input sample.json` | Runs scan/chunk/claims |
| Full scoring | `lumiseval run eval --input sample.json` | Runs complete dependency chain and final scoring |
| REST API | `POST /jobs` | Programmatic integration |

## Roadmap

- [ ] Async batch processing via TaskIQ with streamed CLI progress
- [ ] SQLite persistence (SQLModel) for job records, reports, and cost actuals
- [ ] MCP LanceDB retrieval for enterprise knowledge bases
- [ ] GEval Criteria Extractor agent — auto-derive judge criteria from reference documents (UC-3)
- [x] Langfuse observability — traces, scores, and cost tracking per run
- [ ] OpenTelemetry trace export (Arize Phoenix compatible)
- [ ] Cost feedback loop — estimate vs. actual tracking for improved heuristics

## License

MIT
