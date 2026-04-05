# Get Started with LumisEval

LumisEval exposes two CLI interfaces:

```bash
lumiseval estimate <target_node> --input <source> ...
lumiseval run <target_node> --input <source> ...
```

- `estimate` only scans/plans/prices.
- `run` executes directly (no confirmation gate in graph).

## 1) Prerequisites

- Python `>=3.10`
- `uv`
- `OPENAI_API_KEY`
- Optional: `TAVILY_API_KEY` when using web search

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

If Hugging Face adapter support is missing:

```bash
uv add datasets
```

## 3) Nodes and Topology

Canonical target nodes:

- `scan`
- `chunk`
- `claims`
- `dedupe`
- `geval_steps`
- `geval`
- `relevance`
- `grounding`
- `redteam`
- `reference`
- `eval`

Strict prerequisite paths:

- `chunk <- scan`
- `claims <- scan, chunk`
- `dedupe <- scan, chunk, claims`
- `geval_steps <- scan`
- `geval <- scan, geval_steps`
- `relevance <- scan, chunk, claims, dedupe`
- `grounding <- scan, chunk, claims, dedupe`
- `redteam <- scan`
- `reference <- scan`
- `eval <- scan + all branches`

Examples:

- `lumiseval run redteam ...` runs `scan -> redteam`
- `lumiseval run relevance ...` runs `scan -> chunk -> claims -> dedupe -> relevance`
- `lumiseval run eval ...` runs full graph branch set

## 4) Input Records

Minimal valid record:

```json
{
  "generation": "The Eiffel Tower is in Paris."
}
```

Recommended full record:

```json
{
  "case_id": "eiffel-1",
  "question": "Where is the Eiffel Tower?",
  "generation": "The Eiffel Tower is in Paris, France.",
  "context": ["The Eiffel Tower is a wrought-iron tower in Paris."],
  "reference": "The Eiffel Tower is located in Paris.",
  "geval": {
    "metrics": [
      {
        "name": "location_accuracy",
        "record_fields": ["question", "generation"],
        "criteria": "The answer must mention Paris as the location."
      }
    ]
  }
}
```

Eligibility by field:

- `generation` required for all nodes
- `context` required for: `chunk`, `claims`, `dedupe`, `relevance`, `grounding`
- `geval.metrics` required for: `geval_steps`, `geval`
- `reference` required for: `reference`

GEval contract rules:
- `geval.metrics[]` must include `name`, `record_fields`, and exactly one of `criteria` or `evaluation_steps`
- `record_fields` must be from: `question`, `generation`, `reference`, `context`
- `generation` is auto-included if omitted

## 5) CLI Usage

### 5.1 Estimate only

```bash
uv run lumiseval estimate grounding \
  --input sample.json \
  --limit 10
```

Output includes:

- scan statistics table
- node eligibility table
- execution plan table (`to_run` / `cached` / `skipped`)
- uncached delta cost table for the selected branch

### 5.2 Execute branch

```bash
uv run lumiseval run grounding \
  --input sample.json \
  --limit 10
```

Behavior:

- directly executes the selected branch
- no estimate/approval prompt
- uses cache unless `--force` or `--no-cache`

`--yes` is accepted but deprecated on `run` (no-op).

### 5.3 Full eval run

```bash
uv run lumiseval run eval \
  --input sample.json \
  --limit 10 \
  --output-dir ./runs/eval
```

For `eval` runs, `report.cost_estimate` is still populated by runner-level pre-eval estimation.

### 5.4 Hugging Face dataset source

```bash
uv run lumiseval estimate relevance \
  --input hf://openai/gsm8k \
  --adapter huggingface \
  --hf-config main \
  --split train \
  --limit 20
```

## 6) Important CLI Options

- source selection:
  - `--input`
  - `--split`
  - `--limit`
  - `--adapter`
  - `--hf-config`
  - `--hf-revision`
- model/runtime:
  - `--model`
  - `--web-search`
  - `--evidence-threshold`
- execution controls:
  - `--continue-on-error/--fail-fast` (`run` only)
  - `--force`
  - `--no-cache`
  - `--cache-dir`
  - `--output-dir` (`run eval` only)

## 7) Cache Semantics

Cache is node-level and keyed by:

- case hash: content fields (`generation`, `question`, `reference`, `context`, `geval`, `reference_files`)
- config hash: execution config (`judge_model`, enable flags, web/evidence settings)

Implications:

- rerunning same target on unchanged data reuses cached nodes
- extending from one branch to another reuses shared prerequisites
- adding new rows only computes uncached rows

`estimate` reflects **delta cost** (uncached work only).

## 8) API Contract (Design, Handlers Later)

Planned endpoints:

- `POST /estimate`
- `POST /run`

Shared request shape:

```json
{
  "target_node": "grounding",
  "cases": [
    {
      "case_id": "eiffel-1",
      "question": "Where is the Eiffel Tower?",
      "generation": "The Eiffel Tower is in Paris, France.",
      "context": ["The Eiffel Tower is in Paris."],
      "reference": "The Eiffel Tower is located in Paris.",
      "geval": {
        "metrics": []
      }
    }
  ],
  "job_config": {
    "judge_model": "gpt-4o-mini",
    "web_search": false,
    "evidence_threshold": 0.75,
    "enable_grounding": true,
    "enable_relevance": true,
    "enable_redteam": true,
    "enable_geval": true,
    "enable_reference": true
  },
  "force": false
}
```

Planned `/estimate` response shape:

```json
{
  "target_node": "grounding",
  "scan": { "record_count": 10, "total_tokens": 12345 },
  "plan": {
    "planned_nodes": ["scan", "chunk", "claims", "dedupe", "grounding"],
    "to_run_case_ids_by_node": {},
    "cached_case_ids_by_node": {},
    "skipped_case_ids_by_node": {}
  },
  "cost": {
    "rows": [],
    "target_delta_cost_usd": 0.1234
  }
}
```

Planned `/run` response shape:

```json
{
  "target_node": "grounding",
  "results": [
    {
      "case_id": "eiffel-1",
      "executed_nodes": ["scan", "chunk", "claims", "dedupe", "grounding"],
      "cached_nodes": [],
      "output": {}
    }
  ]
}
```

For `target_node=eval`, each result includes `report` with `cost_estimate`.

## 9) Troubleshooting

- `Unknown node ...`
  - use one of the canonical nodes listed above
- `Invalid dataset source ...`
  - verify local path or `hf://<dataset-id>` format
- no HF adapter
  - install `datasets`
- model key errors
  - set `OPENAI_API_KEY`
