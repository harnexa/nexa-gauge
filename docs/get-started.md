# Development Get Started (nexa-gauge)

This guide is for local development in this repository, not just CLI usage from PyPI.

## 1) Prerequisites

- Python 3.10+
- `uv` installed: <https://docs.astral.sh/uv/>
- Optional: `make`

## 2) Bootstrap the workspace

From repo root:

```bash
cd /Volumes/Raid1CrucialHD/sardhendu/workspace/harnexa-dev/nexa-gauge
make install
```

`make install` runs `setup.sh`, which creates `.venv` (if missing) and runs `uv sync`.

If you prefer manual setup:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --all-packages --group dev
```

## 3) Configure environment

```bash
cp .env.example .env
```

Set at least one provider key in `.env` before LLM-backed runs, for example:

```env
OPENAI_API_KEY=<your-key>
LLM_MODEL=gpt-4o-mini
```

## 4) Verify your dev install

```bash
source .venv/bin/activate
which nexagauge
nexagauge --help
```

Expected:
- `which nexagauge` points to `.venv/bin/nexagauge`
- help output shows `run`, `estimate`, and `delete`

## 5) Run tests and checks

```bash
make lint
make test
```

Useful targets:
- `make test-pkg PKG=packages/nexagauge-core`
- `make test_graph`
- `make ci`

## 6) Run a local smoke test

```bash
nexagauge estimate grounding --input sample.json --limit 1
nexagauge run eval --input sample.json --limit 1 --output-dir ./report
```

## 7) Development workflow notes

- CLI entrypoint `nexagauge` is defined in:
  - root `pyproject.toml` (`[project.scripts] nexagauge = "ng_cli.main:main"`)
  - `apps/nexagauge-apps/pyproject.toml` (same script)
- `nexagauge-core` and `nexagauge-graph` packages do not expose the `nexagauge` command on their own.
- You can always run commands without activating venv via `uv run ...` if needed.

## 8) Troubleshooting

### `nexagauge: command not found` after `source .venv/bin/activate`

Most common cause: stale or relocated virtualenv where `.venv/bin/activate` points to an old path.

Check:

```bash
rg -n "^VIRTUAL_ENV='" .venv/bin/activate
```

If it does not match your current repo path, recreate the venv:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv sync --all-packages --group dev
which nexagauge
```

### CLI exists but import errors occur

Run a clean sync from repo root:

```bash
uv sync --all-packages --group dev
```

### `OPENAI_API_KEY`/provider auth errors

Set credentials in `.env` (or shell env), then rerun.
