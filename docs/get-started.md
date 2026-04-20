# Get Started with neXa-gauge

## 1) Install

Python 3.10+ is required.

Install from PyPI:

```bash
pip install nexa-gauge
```

If you plan to use Hugging Face datasets (`hf://...`), install extras:

```bash
pip install "nexa-gauge[huggingface]"
```

## 2) Configure Environment

Set at least one provider key before LLM-backed runs.

macOS/Linux:

```bash
export OPENAI_API_KEY="<your-key>"
# optional
export LLM_MODEL="openai/gpt-4o-mini"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="<your-key>"
# optional
$env:LLM_MODEL="openai/gpt-4o-mini"
```

You can also use a local `.env` file; see `.env.example` in this repository.

## 3) Create Input Data

Create a sample dataset:

```json
[
  {
    "case_id": "tower-1",
    "question": "Where is the Eiffel Tower?",
    "generation": "The Eiffel Tower is in Paris, France.",
    "context": ["The Eiffel Tower is a Paris landmark."],
    "reference": "The Eiffel Tower is in Paris.",
    "geval": {
      "metrics": [
        {
          "name": "location_accuracy",
          "item_fields": ["question", "generation"],
          "criteria": "Answer must state Paris as the location."
        }
      ]
    }
  }
]
```

Save it as `sample.json`.

## 4) Run the CLI

Inspect commands:

```bash
nexagauge --help
nexagauge run --help
nexagauge estimate --help
```

Estimate before execution:

```bash
nexagauge estimate grounding --input sample.json --limit 10
```

Run a branch:

```bash
nexagauge run grounding --input sample.json --limit 10
```

Run full evaluation and write report files:

```bash
nexagauge run eval --input sample.json --output-dir ./report --limit 10
```

## 5) Target Nodes

Available targets:
- `scan`
- `chunk`
- `claims`
- `dedup`
- `geval_steps`
- `relevance`
- `grounding`
- `redteam`
- `geval`
- `reference`
- `eval`
- `report`

Typical paths:
- `grounding`: `scan -> chunk -> claims -> dedup -> grounding`
- `relevance`: `scan -> chunk -> claims -> dedup -> relevance`
- `geval`: `scan -> geval_steps -> geval`
- `eval`: full branch execution + aggregate evaluation

## 6) Common Flags

- data: `--input`, `--adapter`, `--split`, `--start`, `--end`, `--limit`
- model routing: `--model`, `--llm-model`, `--llm-fallback`
- cache: `--force`, `--no-cache`, `--cache-dir`
- execution: `--max-workers`, `--max-in-flight`, `--continue-on-error`
- output: `--output-dir` (run only)
- debug: `--debug` (node logs on; progress bar off)

## 7) Hugging Face Example

```bash
nexagauge estimate relevance \
  --input hf://openai/gsm8k \
  --adapter huggingface \
  --hf-config main \
  --split train \
  --limit 20
```

## 8) Troubleshooting

| Error | Resolution |
|---|---|
| `Unknown node '...'` | Use one of the canonical node names listed above. |
| `Could not resolve dataset source` | Use an existing local path or `hf://<dataset-id>`. |
| `datasets package is required for hf:// adapters` | Install `nexa-gauge[huggingface]`. |
| Authentication errors | Set provider key (for example `OPENAI_API_KEY`). |
