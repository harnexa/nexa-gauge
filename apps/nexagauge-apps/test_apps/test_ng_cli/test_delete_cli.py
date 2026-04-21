from __future__ import annotations

from pathlib import Path

import pytest
from ng_cli.delete import _human_bytes
from ng_cli.main import app
from typer.testing import CliRunner

runner = CliRunner()


def _populate_cache(root: Path) -> tuple[int, int]:
    kv = root / "kv"
    kv.mkdir(parents=True)
    (kv / "a.json").write_bytes(b"x" * 1024)
    (kv / "b.json").write_bytes(b"y" * 2048)
    nested = kv / "sub"
    nested.mkdir()
    (nested / "c.json").write_bytes(b"z" * 512)
    return 3, 1024 + 2048 + 512


def test_human_bytes_formats_each_unit() -> None:
    assert _human_bytes(0) == "0 B"
    assert _human_bytes(512) == "512 B"
    assert _human_bytes(1536) == "1.5 KB"
    assert _human_bytes(5 * 1024 * 1024) == "5.0 MB"


def test_delete_cache_reports_missing_dir(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist"
    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(target)])
    assert result.exit_code == 0
    assert "No cache found" in result.stdout


def test_delete_cache_dry_run_preserves_files(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    expected_files, expected_bytes = _populate_cache(root)

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(root), "--dry-run"])

    assert result.exit_code == 0
    assert "Dry run" in result.stdout
    assert f"{expected_files:,}" in result.stdout
    assert root.exists()
    assert len(list(root.rglob("*.json"))) == expected_files
    del expected_bytes


def test_delete_cache_with_yes_flag_deletes_everything(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    expected_files, _ = _populate_cache(root)

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(root), "--yes"])

    assert result.exit_code == 0
    assert "Freed" in result.stdout
    assert f"{expected_files:,} files" in result.stdout
    assert not root.exists()


def test_delete_cache_prompt_abort_keeps_files(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    expected_files, _ = _populate_cache(root)

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(root)], input="n\n")

    assert result.exit_code == 1
    assert "Aborted" in result.stdout
    assert root.exists()
    assert len(list(root.rglob("*.json"))) == expected_files


def test_delete_cache_prompt_confirm_deletes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    _populate_cache(root)

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(root)], input="y\n")

    assert result.exit_code == 0
    assert "Freed" in result.stdout
    assert not root.exists()


def test_delete_cache_empty_dir_reports_no_op(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(root), "--yes"])

    assert result.exit_code == 0
    assert "already empty" in result.stdout
    assert root.exists()


def test_delete_cache_refuses_non_directory(tmp_path: Path) -> None:
    target = tmp_path / "a-file"
    target.write_text("not a directory")

    result = runner.invoke(app, ["delete", "cache", "--cache-dir", str(target), "--yes"])

    assert result.exit_code == 1
    assert "not a directory" in result.stdout


def test_delete_cache_honors_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "env-cache"
    _populate_cache(root)
    monkeypatch.setenv("NEXAGAUGE_CACHE_DIR", str(root))

    result = runner.invoke(app, ["delete", "cache", "--yes"])

    assert result.exit_code == 0
    assert str(root) in result.stdout.replace("\n", "")
    assert not root.exists()
