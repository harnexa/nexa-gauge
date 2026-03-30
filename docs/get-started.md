# Get Started with LumisEval (Run-Only CLI)

LumisEval now exposes one primary CLI workflow:

```bash
lumiseval run <node_name> --input <source> ...
```

And one API endpoint:

```http
POST /jobs
```

`POST /jobs` accepts either a single record or an array of records.

## 1) Prerequisites

- Python `>=3.10`
- `uv`
- `OPENAI_API_KEY` for judge/model-powered nodes
- Optional: `TAVILY_API_KEY` when using `--web-search`

## 2) Setup

```bash
cd lumis-eval
uv sync
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
WEB_SEARCH_ENABLED=false
```

If Hugging Face adapter is missing:

```bash
uv add datasets
```

## 3) CLI Contract

### 3.1 Command shape

```bash
uv run lumiseval run <node_name> --input <source> [options]
```

Where:
- `<source>` is either:
  - local path (`.json`, `.jsonl`, `.csv`, `.txt`)
  - `hf://<dataset-id>`
- `<node_name>` is one of:
  - `scan`
  - `estimate`
  - `approve`
  - `chunk`
  - `claims`
  - `dedupe`
  - `relevance`
  - `grounding`
  - `redteam`
  - `rubric`
  - `eval`

### 3.2 Strict target semantics

`run <node_name>` executes prerequisites and stops at that target node.
Preflight cost estimation is target-aware, so it only estimates the branch required for that target.

Examples:
- `run estimate` stops at `estimate`
- `run claims` stops at `claims`
- `run eval` runs the full required path to final eval

### 3.3 Two-stage execution model

Every CLI run does:

1. Dataset preflight (for selected cases):
- scan records
- print scan statistics table
- estimate total cost
- prompt for confirmation (unless `--yes`)

2. Per-case node execution:
- run each record up to target node
- reuse cache by default
- print executed/cached counts per case

## 4) Examples

### 4.1 Run to `estimate` on local file

```bash
uv run lumiseval run estimate \
  --input sample.json \
  --limit 10 \
  --yes
```

### 4.2 Run to `eval` on local file

```bash
uv run lumiseval run eval \
  --input sample.json \
  --limit 10 \
  --yes \
  --output-dir ./runs/eval
```

### 4.3 Run to a node on Hugging Face dataset

```bash
uv run lumiseval run relevance \
  --input hf://openai/gsm8k \
  --adapter huggingface \
  --hf-config main \
  --split train \
  --limit 10 \
  --yes
```

## 5) Input Fields

### 5.1 Canonical record (`EvalCase`)

Required:
- `generation`

Optional:
- `case_id`
- `question`
- `ground_truth`
- `context`
- `reference_files`
- `rubric_rules`
- additional metadata fields

### 5.2 Local adapter aliases

- Generation: `generation | response | answer | output | completion`
- Case id: `case_id | id | uuid | prompt_id`
- Question: `question | query | prompt`
- Ground truth: `ground_truth | reference | gold_answer`
- Context: `context | contexts | documents`
- Reference files: `reference_files | reference_paths`
- Rubric rules: `rubric_rules | rubric`

### 5.3 Hugging Face adapter aliases

- Generation: `generation | response | answer | output | completion`
- Case id: `case_id | id | prompt_id`
- Question: `question | query | prompt`
- Ground truth: `ground_truth | reference`
- Context: `context | contexts | documents`
- Reference files: `reference_files | reference_paths`
- Rubric rules: `rubric_rules | rubric`

## 6) Cache Behavior

Cache is enabled by default.

Controls:
- `--no-cache`: disable cache reads/writes
- `--force`: ignore cache reads but still write outputs
- `--cache-dir`: custom cache location

Case hash includes:
- `generation`
- `question`
- `ground_truth`
- `rubric_rules` (`id`, `statement`, `pass_condition`)
- `reference_files`

Config hash includes:
- `judge_model`
- `enable_hallucination`
- `enable_faithfulness`
- `enable_answer_relevancy`
- `enable_adversarial`
- `enable_rubric`
- `web_search`
- `evidence_threshold`

This enables incremental behavior:
- unchanged cases hit cache
- new/changed cases execute

## 7) Important CLI Options

- Source / selection:
  - `--input`
  - `--split`
  - `--limit`
  - `--adapter`
  - `--hf-config`
  - `--hf-revision`
- Model / retrieval:
  - `--model`
  - `--web-search`
  - `--evidence-threshold`
- Metric toggles:
  - `--enable-hallucination/--disable-hallucination`
  - `--enable-faithfulness/--disable-faithfulness`
  - `--enable-answer-relevancy/--disable-answer-relevancy`
  - `--enable-adversarial/--disable-adversarial`
  - `--enable-rubric/--disable-rubric`
- Control:
  - `--yes`
  - `--continue-on-error/--fail-fast`
  - `--force`
  - `--no-cache`
  - `--cache-dir`
  - `--output-dir` (eval output JSONs)

## 8) API Usage

### 8.1 Start API

```bash
uv run uvicorn lumiseval_api.main:app --reload --port 8080
```

### 8.2 Single-record request

```bash
curl -X POST "http://localhost:8080/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "generation": "The Eiffel Tower is in Paris, France.",
    "question": "Where is the Eiffel Tower?",
    "ground_truth": "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "judge_model": "gpt-4o-mini",
    "web_search": false,
    "enable_hallucination": true,
    "enable_faithfulness": true,
    "enable_answer_relevancy": true
  }'
```

### 8.3 Multi-record request (JSON array)

```bash
curl -X POST "http://localhost:8080/jobs" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "generation": "The Eiffel Tower is in Paris, France.",
      "question": "Where is the Eiffel Tower?",
      "ground_truth": "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
      "judge_model": "gpt-4o-mini",
      "web_search": false
    },
    {
      "generation": "Photosynthesis converts light into chemical energy.",
      "question": "How does photosynthesis work?",
      "judge_model": "gpt-4o-mini",
      "web_search": false
    }
  ]'
```

### 8.4 API request fields

Required:
- `generation: str`

Optional:
- `question: str | null`
- `ground_truth: str | null`
- `rubric_rules: RubricRule[]`
- `reference_files: list[str]`
- `judge_model: str`
- `web_search: bool`
- `enable_hallucination: bool`
- `enable_faithfulness: bool`
- `enable_answer_relevancy: bool`
- `enable_adversarial: bool`
- `enable_rubric: bool`
- `evidence_threshold: float`
- `budget_cap_usd: float | null`
- `acknowledge_cost: bool` (accepted; not enforced)

## 9) Troubleshooting

- `Invalid dataset source ...`
  - check path, adapter, or `hf://` format
- `Unknown node ...`
  - use one of supported node names (including `eval`)
- Hugging Face import error
  - run `uv add datasets`
- API key errors
  - set `OPENAI_API_KEY`
- No Tavily fallback
  - set `TAVILY_API_KEY` or disable `--web-search`
