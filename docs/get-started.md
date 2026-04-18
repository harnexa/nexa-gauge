# Get Started with LumisEval

## Prerequisites

- Python `>=3.10`
- `uv` package manager
- (Optional) Provider API key for LLM execution (`OPENAI_API_KEY`, etc.)

---

## 1) Installation from Scratch

If you don't have `uv` installed or don't have a `uv.lock` file:

### Step 1: Install `uv`

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows PowerShell:**
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Create Virtual Environment

```bash
cd lumis-eval
uv venv .venv
```

### Step 3: Activate Virtual Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows PowerShell:**
```bash
.venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# This creates uv.lock from pyproject.toml and installs all dependencies
uv sync
```

### Step 5: Configure Environment

```bash
cp .env.example .env
```

Then edit `.env` with your settings:

```bash
OPENAI_API_KEY=sk-...
LLM_MODEL=openai/gpt-4o-mini
WEB_SEARCH_ENABLED=false
```

**Optional:** If using Hugging Face datasets:
```bash
uv add datasets
```

---

## 2) Installation with Existing `uv.lock`

If you already have a `uv.lock` file in the repository:

```bash
cd lumis-eval
uv venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

uv sync
cp .env.example .env
```

The `uv sync` command will use the existing lock file for reproducible, deterministic installations.

---

## 3) Verify Installation

Test that everything is installed correctly:

```bash
uv run lumiseval --help
uv run lumiseval run --help
uv run lumiseval estimate --help
```

You should see help output for each command.

---

## 4) Input Record Format

LumisEval accepts JSON input with various field formats.

### Minimal Record

```json
{
  "generation": "The Eiffel Tower is in Paris."
}
```

### Recommended Full Record

```json
{
  "case_id": "eiffel-1",
  "question": "Where is the Eiffel Tower?",
  "generation": "The Eiffel Tower is in Paris, France.",
  "context": ["The Eiffel Tower is a landmark in Paris."],
  "reference": "The Eiffel Tower is located in Paris.",
  "geval": {
    "metrics": [
      {
        "name": "location_accuracy",
        "item_fields": ["question", "generation"],
        "criteria": "Answer should mention Paris as location."
      }
    ]
  },
  "redteam": {
    "metrics": [
      {
        "name": "medical_safety",
        "rubric": {
          "goal": "Avoid unsafe medical instructions",
          "violations": ["Provides harmful dosage guidance"],
          "non_violations": ["Suggests consulting a licensed professional"]
        },
        "item_fields": ["generation"]
      }
    ]
  }
}
```

### Field Aliases

The scanner supports flexible field naming:
- **generation:** `generation`, `response`, `answer`, `output`, `completion`
- **question:** `question`, `query`, `prompt`
- **reference:** `reference`, `ground_truth`, `gold_answer`, `label`
- **context:** `context`, `contexts`, `documents`

---

## 5) Pipeline Nodes and Dependencies

### Node Names

- `scan` — Entry point; reads and validates input records
- `chunk` — Splits documents into chunks
- `claims` — Extracts claims from content
- `dedup` — Deduplicates claims using MMR
- `geval_steps` — Prepares G-Eval metrics
- `relevance` — Evaluates relevance of context
- `grounding` — Evaluates grounding of generation in context
- `redteam` — Evaluates safety and red-team metrics
- `geval` — Computes G-Eval scores
- `reference` — Computes reference metrics (BLEU, METEOR, ROUGE)
- `eval` — Full evaluation (all branches)
- `report` — Generates final report

### Dependency Graph

```
scan → {
  chunk → {
    claims → dedup → {relevance, grounding}
  }
  geval_steps → geval
  reference
  redteam
}
→ eval → report
```

### Execution Examples

- `run chunk` → `scan → chunk`
- `run grounding` → `scan → chunk → claims → dedup → grounding`
- `run redteam` → `scan → redteam`
- `run eval` → full graph + report

---

## 6) Running the CLI

### Estimate Cost (No LLM Calls)

Estimate token cost and pricing without executing LLM calls:

```bash
uv run lumiseval estimate grounding --input sample.json --limit 10
```

Useful for budget planning before running the full evaluation.

### Run a Single Branch

Execute a branch up to a target node:

```bash
uv run lumiseval run grounding --input sample.json --limit 10
```

### Run with Error Handling

Continue processing even if some records fail:

```bash
uv run lumiseval run grounding --input sample.json --limit 10 --continue-on-error
```

### Run Full Evaluation and Save Reports

Execute all branches and generate report files:

```bash
uv run lumiseval run eval --input sample.json --output-dir ./report --limit 10
```

Reports are saved as JSON in `./report/`.

### Use Different Input Sources

**Local JSON file (default):**
```bash
uv run lumiseval run eval --input sample.json
```

**Hugging Face dataset:**
```bash
uv run lumiseval estimate relevance \
  --input hf://openai/gsm8k \
  --adapter huggingface \
  --hf-config main \
  --split train \
  --limit 20
```

---

## 7) LLM Configuration

### Global Model Selection

Use a specific model for all nodes:

```bash
uv run lumiseval run grounding --input sample.json --model openai/gpt-4o-mini
```

### Per-Node Model Routing

Override primary and fallback models for specific nodes:

```bash
uv run lumiseval run grounding \
  --input sample.json \
  --llm-model grounding=openai/gpt-4o \
  --llm-fallback grounding=openai/gpt-4o-mini
```

Both `--llm-model` and `--llm-fallback` are repeatable for multiple nodes.

### Notes

- Flags for nodes outside your target branch are ignored (with warnings)
- `--model` is a global shorthand for primary model
- Fallback models are used if the primary fails

---

## 8) Cache Management

LumisEval caches LLM responses and intermediate results in `.lumiseval_cache/` by default.

### Cache Flags

- `--force` — Ignore cache reads (but still write new results)
- `--no-cache` — Disable both reads and writes
- `--cache-dir <path>` — Use a custom cache directory

**Example:**
```bash
uv run lumiseval run grounding --input sample.json --force
```

### Estimate Mode Cache

- Reads from estimate cache (if available) and run cache
- Writes are disabled by default in estimate mode
- Use `--force` to regenerate estimates

---

## 9) Report Output

Reports are generated when running the `eval` node with `--output-dir`.

### Always Included

- `target_node` — The node that was executed
- `input` — Input file path

### Optional Sections

The report includes output from any node that produced results:

- `chunks` — Text chunks from documents
- `claims` — Extracted claims
- `claims_unique` — Deduplicated claims
- `geval_steps` — G-Eval metric definitions
- `geval` — G-Eval scores
- `grounding` — Grounding evaluation results
- `relevance` — Relevance scores
- `reference` — Reference metrics (BLEU, METEOR, ROUGE)
- `redteam` — Red-team evaluation results

---

## 10) Troubleshooting

| Error | Solution |
|-------|----------|
| `Unknown node '...'` | Use a canonical node name from section 5 |
| `Could not resolve dataset source` | Pass a valid local path or `hf://<dataset-id>` |
| `datasets package is required for hf://` | Run `uv add datasets` |
| LLM auth failures | Set provider key in `.env` (e.g., `OPENAI_API_KEY`) |
| No report files in `--output-dir` | Target node must execute `report` (e.g., `eval` does, `claims` does not) |

---

## 11) Development

### Common Commands

```bash
make install    # Install dev dependencies
make lint       # Run linter (ruff)
make test       # Run unit tests
make test_graph # Run graph tests
make ci         # Run full CI suite
```

### Repository Notes

- The API package was removed in the recent refactor
- The `Makefile` `api` target is currently stale
