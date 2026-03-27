# Get Started with LumisEval

LumisEval is an LLM evaluation pipeline that scores AI-generated text for faithfulness, hallucination, and rubric adherence. It uses a LangGraph orchestration graph, local vector search (LanceDB), and pluggable judge models via LiteLLM.

---

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/getting-started/installation/)** — fast Python package manager
- An **OpenAI API key** (minimum requirement for the judge model)

---

## 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

---

## 2. Clone and Install

```bash
git clone <repo-url>
cd lumis-eval

# Create a virtual environment and install all workspace packages
make install
```

This runs `uv venv` + `uv pip install -e` for every package in the workspace. After this you will have two binaries on your PATH (inside `.venv`):

| Binary | Purpose |
|--------|---------|
| `lumiseval` | CLI for running evaluations |
| `lumiseval-api` | FastAPI server |

Activate the environment:

```bash
source .venv/bin/activate
```

---

## 3. Configure Environment

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional — enable web search evidence
TAVILY_API_KEY=tvly-...
WEB_SEARCH_ENABLED=false

# Optional — override the judge model (default: gpt-4o-mini)
LLM_MODEL=gpt-4o-mini
```

All other values have sensible defaults. Embeddings run locally via `sentence-transformers` (no API key needed).

---

## 4. Sample Dataset

The pipeline accepts four input formats: `.txt`, `.json`, `.jsonl`, and `.csv`. The minimum required field is `generation` — the LLM output you want evaluated. Optionally include `question` (the original prompt) and `context` (reference passages).

### Single record — plain text

Save as `sample.txt`:

```
The Eiffel Tower is located in Paris, France. It was built between 1887 and 1889
as the entrance arch for the 1889 World's Fair. The tower stands 330 metres tall
and is the most-visited paid monument in the world.
```

### Single record — JSON

Save as `sample.json`:

```json
{
  "question": "What is the Eiffel Tower and where is it located?",
  "generation": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France. It was constructed between 1887 and 1889 and served as the entrance arch to the 1889 World's Fair. Standing at 330 metres, it is one of the most recognisable structures in the world.",
  "context": "The Eiffel Tower (/ˈaɪfəl/ EYE-fəl; French: Tour Eiffel) is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889 as the centerpiece of the 1889 World's Fair."
}
```

### Batch — JSONL (one record per line)

Save as `sample_batch.jsonl`:

```jsonl
{"question": "What is the Eiffel Tower?", "generation": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France, built in 1889.", "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."}
{"question": "Who invented the telephone?", "generation": "Alexander Graham Bell is widely credited with inventing the first practical telephone in 1876.", "context": "Alexander Graham Bell was awarded the first US patent for the telephone on March 7, 1876."}
{"question": "What is the speed of light?", "generation": "The speed of light in a vacuum is approximately 300,000 kilometres per second.", "context": "The speed of light in vacuum, commonly denoted c, is a universal physical constant equal to 299,792,458 metres per second."}
```

### Batch — CSV

Save as `sample_batch.csv`:

```csv
question,generation,context
"What is the Eiffel Tower?","The Eiffel Tower is a wrought-iron lattice tower in Paris, France.","The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris."
"Who invented the telephone?","Alexander Graham Bell invented the telephone in 1876.","Bell was awarded the first US patent for the telephone on March 7, 1876."
```

---

## 5. Running an Evaluation

### Step 1 — Estimate cost first (no LLM calls)

Before committing API spend, run a dry-run cost estimate:

```bash
lumiseval estimate --input sample.json
```

Output shows estimated judge calls, token usage, and USD cost band (±20%).

### Step 2 — Run a single evaluation

```bash
lumiseval eval --input sample.json
```

This runs the full pipeline:

1. Scans metadata (tokens, estimated chunks and claims)
2. Shows cost estimate and prompts for confirmation
3. Chunks the generation into 512-token semantic chunks
4. Extracts atomic claims
5. Deduplicates claims via MMR
6. Routes each claim to evidence sources (local LanceDB → web if enabled)
7. Runs RAGAS (faithfulness + answer relevancy) and DeepEval (hallucination)
8. Aggregates a composite score

Save the report to a file:

```bash
lumiseval eval --input sample.json --output report.json
```

Skip the cost confirmation prompt (useful for scripts):

```bash
lumiseval eval --input sample.json --yes
```

### Step 3 — Run a batch evaluation

```bash
lumiseval batch sample_batch.jsonl --yes --output-dir ./results
```

Each record produces a separate report JSON in `./results/`.

---

## 6. Optional Flags

| Flag | Description |
|------|-------------|
| `--model gpt-4o` | Override the judge model (any LiteLLM-supported model) |
| `--web-search` | Enable Tavily web search as an evidence source |
| `--adversarial` | Enable Giskard adversarial probes (prompt injection, PII, bias) |
| `--rubric rules.json` | Evaluate against a custom rubric (see below) |
| `--budget 0.10` | Abort if estimated cost exceeds $0.10 |
| `--yes` | Skip cost confirmation prompt |
| `--output report.json` | Write JSON report to file |

---

## 7. Using a Custom Rubric

Create a `rubric.json` file with rules to evaluate against:

```json
[
  "The response must cite specific dates or years when making historical claims.",
  "The response must not use hedging language such as 'might' or 'possibly' for established facts.",
  "The response must be written in formal English without contractions."
]
```

Run with rubric:

```bash
lumiseval eval --input sample.json --rubric rubric.json
```

Each rule is evaluated independently via G-Eval and rolled into the composite score (default weight: 20%).

---

## 8. Running the REST API

Start the API server:

```bash
make api
# or
lumiseval-api
```

The server starts at `http://localhost:8080`. Submit a job:

```bash
curl -X POST http://localhost:8080/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Eiffel Tower?",
    "generation": "The Eiffel Tower is a wrought-iron lattice tower located in Paris, France, built in 1889.",
    "judge_model": "gpt-4o-mini",
    "enable_ragas": true,
    "enable_deepeval": true,
    "web_search": false,
    "acknowledge_cost": true
  }'
```

Check the API is running:

```bash
curl http://localhost:8080/health
```

---

## 9. Understanding the Report

The output JSON report has this shape:

```json
{
  "job_id": "...",
  "composite_score": 0.84,
  "confidence_band": 0.06,
  "evaluation_incomplete": false,
  "warnings": [],
  "cost_actual_usd": 0.0031,
  "ragas": {
    "faithfulness": 0.91,
    "answer_relevancy": 0.88
  },
  "deepeval": {
    "hallucination_score": 0.12,
    "geval_score": null
  },
  "claim_verdicts": [
    {
      "claim_text": "The Eiffel Tower is located in Paris, France.",
      "verdict": "SUPPORTED",
      "source": "LOCAL",
      "passages": [...]
    }
  ]
}
```

**Composite score** is a weighted average (configurable in `.env`):

| Metric | Default weight |
|--------|---------------|
| Faithfulness | 40% |
| Hallucination (inverted) | 30% |
| Rubric adherence | 20% |
| Safety | 10% |

---

## 10. Configuration Reference

Key `.env` settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `gpt-4o-mini` | Judge model (any LiteLLM provider) |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `LANCEDB_PATH` | `./.lancedb` | Local vector store directory |
| `EVIDENCE_THRESHOLD` | `0.75` | Minimum similarity score for evidence retrieval |
| `WEB_SEARCH_ENABLED` | `false` | Enable Tavily web search |
| `BUDGET_CAP_USD` | _(none)_ | Reject jobs exceeding this cost |
| `SCORE_WEIGHT_FAITHFULNESS` | `0.4` | Composite score weight |
| `SCORE_WEIGHT_HALLUCINATION` | `0.3` | Composite score weight |
| `SCORE_WEIGHT_RUBRIC` | `0.2` | Composite score weight |
| `SCORE_WEIGHT_SAFETY` | `0.1` | Composite score weight |

---

## 11. Development Commands

```bash
make lint        # Run ruff linter
make format      # Auto-format with ruff
make typecheck   # mypy type checking
make test        # Run pytest
```

---

## Troubleshooting

**`lumiseval: command not found`** — Make sure the venv is activated: `source .venv/bin/activate`

**`OPENAI_API_KEY not set`** — Ensure `.env` exists in the repo root with a valid key.

**`BudgetExceededError`** — Your estimated cost exceeds `BUDGET_CAP_USD`. Either raise the cap or reduce your input size.

**`InputParseError`** — Check that your input file has the required `generation` field and is valid JSON/JSONL/CSV.
