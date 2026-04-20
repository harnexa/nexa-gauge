# Contributing to NexaGauge

## Development Setup

```bash
git clone git@github.com:Sardhendu/nexa-gauge.git
cd nexa-gauge
uv sync
cp .env.example .env
```

## Run Quality Checks

```bash
make lint
make test
make ci
```

## Commit and PR Guidelines

- Keep changes scoped and reviewable.
- Add or update tests for behavioral changes.
- Update docs when interfaces, commands, or report schemas change.
- Avoid unrelated formatting-only edits in feature PRs.

## Local Test Targets

```bash
uv run pytest -s packages/nexagauge-graph/test_ng_graph
uv run pytest -s apps/nexagauge-apps/ng_cli/test_llm_cli_interface.py
uv run pytest -s apps/nexagauge-apps/adapters/test_local_adapter_streaming.py
```

## Reporting Bugs / Requesting Features

- Bugs: use GitHub Issues with repro steps and expected vs actual behavior.
- Features: include use case, user impact, and proposed API/CLI shape.

## Code of Conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
