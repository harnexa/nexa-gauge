#!/usr/bin/env python3
# Run:
# uv run python packages/lumos-graph/test_lumos_graph/run_sample_graph_example.py

from __future__ import annotations

import builtins
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lumoseval_core.types import MetricCategory, MetricResult  # noqa: F401 – used by fake nodes

ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PATH = ROOT / "sample.json"


METRIC_SECTIONS = ("grounding", "relevance", "redteam", "geval", "reference")


def _print_report(report: dict[str, Any], title: str) -> None:
    print(f"\n{title}")
    print("=" * len(title))
    if not isinstance(report, dict):
        print("No report found.")
        return
    for section in METRIC_SECTIONS:
        if section not in report:
            continue
        metrics = report[section].get("metrics", [])
        cost = report[section].get("cost", {})
        print(f"\n  [{section}]  metrics={len(metrics)}  cost={cost.get('cost', 0)}")
        for idx, result in enumerate(metrics, 1):
            print(f"    {idx}. result={result}")


def _try_real_cli_eval() -> bool:
    out_dir = ROOT / ".tmp_eval_example_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv",
        "run",
        "lumos",
        "run",
        "eval",
        "--input",
        str(SAMPLE_PATH),
        "--output-dir",
        str(out_dir),
        "--no-cache",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("Real CLI run failed in current environment.")
        print(proc.stderr.strip() or proc.stdout.strip())
        return False

    report_files = sorted(out_dir.glob("*.json"))
    if not report_files:
        print("CLI run succeeded but no report JSON files were written.")
        return False

    first_report = json.loads(report_files[0].read_text(encoding="utf-8"))
    if not isinstance(first_report, dict):
        first_report = {}
    _print_report(first_report, "Metrics From Real CLI Run (first case)")
    print(f"\nReport files written: {len(report_files)} at {out_dir}")
    return True


def _run_mocked_graph_eval() -> None:
    # graph.py currently uses `Path` in annotations at import time.
    builtins.Path = Path
    try:
        graph = importlib.import_module("lumos_graph.graph")
    finally:
        if hasattr(builtins, "Path"):
            delattr(builtins, "Path")

    def fake_scan(state: dict[str, Any]) -> dict[str, Any]:
        return {
            "inputs": {
                "generation": state.get("generation"),
                "question": state.get("question"),
                "reference": state.get("reference"),
                "context": state.get("context"),
            }
        }

    def fake_chunk(_state: dict[str, Any]) -> dict[str, Any]:
        return {"generation_chunk": {"chunks": [{"text": "chunk-1"}]}}

    def fake_claims(_state: dict[str, Any]) -> dict[str, Any]:
        return {"generation_claims": {"claims": [{"id": "c1"}], "cost": []}}

    def fake_dedup(_state: dict[str, Any]) -> dict[str, Any]:
        return {"generation_dedup_claims": {"claims": [{"id": "c1"}], "cost": []}}

    def fake_geval_steps(_state: dict[str, Any]) -> dict[str, Any]:
        return {"geval_steps_by_signature": {}}

    def fake_relevance(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "relevance_metrics": [
                MetricResult(name="answer_relevancy", category=MetricCategory.ANSWER, score=0.88)
            ]
        }

    def fake_grounding(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "grounding_metrics": [
                MetricResult(name="grounding", category=MetricCategory.ANSWER, score=0.92)
            ]
        }

    def fake_redteam(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "redteam_metrics": [
                MetricResult(
                    name="vulnerability_prompt_injection",
                    category=MetricCategory.ANSWER,
                    score=0.25,
                )
            ]
        }

    def fake_geval(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "geval_metrics": [
                MetricResult(name="geval_coherence", category=MetricCategory.ANSWER, score=0.81)
            ]
        }

    def fake_reference(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "reference_metrics": [
                MetricResult(name="rouge_l", category=MetricCategory.RETRIEVAL, score=0.73)
            ]
        }

    def fake_eval(_state: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph.node_metadata_scanner = fake_scan
    graph.node_generation_chunk = fake_chunk
    graph.node_generation_claims = fake_claims
    graph.node_generation_claims_dedup = fake_dedup
    graph.node_geval_steps = fake_geval_steps
    graph.node_relevance = fake_relevance
    graph.node_grounding = fake_grounding
    graph.node_redteam = fake_redteam
    graph.node_geval = fake_geval
    graph.node_reference = fake_reference
    graph.node_eval = fake_eval

    app = graph.build_graph().compile()

    records = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    first = records[0]
    state = {
        "case_id": first.get("case_id", "case-0"),
        "generation": first.get("generation"),
        "question": first.get("question"),
        "reference": first.get("reference"),
        "context": first.get("context"),
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    final_state = app.invoke(state)
    report_out = graph.node_report(final_state)
    report = report_out.get("report", {})
    _print_report(report, "Metrics From Mocked Graph Run (all branches)")


def main() -> None:
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"sample.json not found at: {SAMPLE_PATH}")

    print(f"Using sample input: {SAMPLE_PATH}")

    if _try_real_cli_eval():
        return

    print("\nFalling back to mocked graph execution (no external API calls).")
    _run_mocked_graph_eval()


if __name__ == "__main__":
    main()
