"""End-to-end CLI test for --transform-file / --transform.

# uv run pytest apps/nexagauge-apps/test_apps/test_ng_cli/test_transforms_cli.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from adapters import create_dataset_adapter as _real_create_dataset_adapter
from ng_cli.main import app
from ng_core.extensions.transforms import _clear_registry_for_tests
from ng_graph.runner import CachedNodeRunner as _real_runner
from typer.testing import CliRunner

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "hotpot_transform.py"


@pytest.fixture(autouse=True)
def _isolate_registry():
    _clear_registry_for_tests()
    yield
    _clear_registry_for_tests()


@pytest.fixture(autouse=True)
def _restore_real_cli_symbols():
    """Defend against test isolation leaks from sibling tests: ``main.run()``
    rebinds ``create_dataset_adapter`` and ``CachedNodeRunner`` onto
    ``ng_cli.run`` and never undoes it, so a prior test that patches the
    main-module symbol leaks its fake into every subsequent run.
    """
    import ng_cli.run as run_module

    run_module.create_dataset_adapter = _real_create_dataset_adapter
    run_module.CachedNodeRunner = _real_runner
    yield
    run_module.create_dataset_adapter = _real_create_dataset_adapter
    run_module.CachedNodeRunner = _real_runner


def _hotpot_shaped_record() -> dict:
    return {
        "id": "hp-1",
        "question": "Which continent has the Atacama Desert?",
        "answer": "South America",
        "context": {
            "title": ["Atacama Desert", "Chile"],
            "sentences": [
                ["The Atacama is a desert plateau.", "It spans 1,000 km."],
                ["Chile is in South America.", "Its capital is Santiago."],
            ],
        },
    }


def test_transform_flattens_hotpot_context_into_paragraphs(tmp_path: Path) -> None:
    src = tmp_path / "hp.jsonl"
    src.write_text(json.dumps(_hotpot_shaped_record()) + "\n")
    output_dir = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "run",
            "chunk",
            "--input",
            str(src),
            "--extension-file",
            str(FIXTURE),
            "--transform",
            "hotpot_qa",
            "--no-cache",
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.stdout
    report_files = list((output_dir / "case_report").glob("*.json"))
    assert len(report_files) == 1
    report = json.loads(report_files[0].read_text())
    context_text = report["input"]["context"]
    # The transform produced a list of paragraphs; scanner joins on "\n\n".
    assert "Atacama Desert" in context_text
    assert "Chile" in context_text
    assert "South America" in context_text
    # The dict-repr leakage we used to see would include "{'title': [...]}".
    assert "{'title':" not in context_text
    assert report["input"]["case_id"] == "hp-1"


def test_unknown_transform_name_fails_with_helpful_error(tmp_path: Path) -> None:
    src = tmp_path / "hp.jsonl"
    src.write_text(json.dumps(_hotpot_shaped_record()) + "\n")

    result = runner.invoke(
        app,
        [
            "run",
            "chunk",
            "--input",
            str(src),
            "--extension-file",
            str(FIXTURE),
            "--transform",
            "does_not_exist",
            "--no-cache",
        ],
    )

    assert result.exit_code != 0
    # InputParseError message lists registered names.
    assert "does_not_exist" in str(result.exception) or "does_not_exist" in result.stdout
    assert "hotpot_qa" in str(result.exception) or "hotpot_qa" in result.stdout


def test_transform_returning_non_dict_raises_input_parse_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.py"
    bad_file.write_text(
        "from ng_core import register_transform\n"
        "@register_transform('bad')\n"
        "def t(record):\n"
        "    return 'not-a-dict'\n"
    )
    src = tmp_path / "x.jsonl"
    src.write_text(json.dumps({"answer": "hi"}) + "\n")

    result = runner.invoke(
        app,
        [
            "run",
            "chunk",
            "--input",
            str(src),
            "--extension-file",
            str(bad_file),
            "--transform",
            "bad",
            "--no-cache",
        ],
    )

    assert result.exit_code != 0
    msg = str(result.exception) + result.stdout
    assert "expected a dict" in msg
