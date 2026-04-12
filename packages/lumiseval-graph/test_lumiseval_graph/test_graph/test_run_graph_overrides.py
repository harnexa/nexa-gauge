# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py::test_normalize_runtime_overrides_parses_and_normalizes
# uv run pytest -s -k "run_graph_overrides" packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_run_graph_overrides.py

from __future__ import annotations

from lumiseval_core.types import (
    Claim,
    ClaimArtifacts,
    Chunk,
    ChunkArtifacts,
    CostEstimate,
    Geval,
    GevalMetricInput,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
    Redteam,
    RedteamMetricInput,
    RedteamRubric,
)
from lumiseval_graph.llm import normalize_runtime_overrides


def test_normalize_runtime_overrides_parses_and_normalizes() -> None:
    """normalize_runtime_overrides canonicalises keys and values."""
    overrides = normalize_runtime_overrides({
        "models": {"relevance": "openai/gpt-4.1"},
        "fallback_models": {"relevance": "openai/gpt-4o-mini"},
        "temperatures": {"relevance": 0.2},
    })

    assert overrides["models"]["relevance"] == "openai/gpt-4.1"
    assert overrides["fallback_models"]["relevance"] == "openai/gpt-4o-mini"
    assert overrides["temperatures"]["relevance"] == 0.2


def test_graph_forwards_llm_overrides_to_nodes(graph_module, monkeypatch) -> None:
    """llm_overrides are forwarded to all LLM-routing nodes in node-level execution."""
    captured: list[tuple[str, object]] = []

    def _fake_get_judge_model(node_name: str, default: str, llm_overrides=None) -> str:
        captured.append((node_name, llm_overrides))
        return f"resolved-{node_name}"

    class _FakeClaimExtractorNode:
        def __init__(self, model: str, llm_overrides=None):
            self.model = model
            self.llm_overrides = llm_overrides

        def run(self, _chunks):
            return ClaimArtifacts(
                claims=[Claim(item=Item(text="a claim", tokens=2), source_chunk_index=0)],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    class _FakeGroundingNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            self.judge_model = judge_model
            self.llm_overrides = llm_overrides

        def run(self, claims, context, enable_grounding=True):
            return graph_module.GroundingMetrics(
                metrics=[MetricResult(name="grounding", category=MetricCategory.ANSWER, score=1.0)],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    class _FakeRelevanceNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            self.judge_model = judge_model
            self.llm_overrides = llm_overrides

        def run(self, claims, question):
            return graph_module.RelevanceMetrics(
                metrics=[MetricResult(name="relevance", category=MetricCategory.ANSWER, score=1.0)],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    class _FakeRedteamNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            self.judge_model = judge_model
            self.llm_overrides = llm_overrides

        def run(self, generation, question, reference, context, redteam):
            return graph_module.RedteamMetrics(
                metrics=[MetricResult(name="redteam", category=MetricCategory.ANSWER, score=1.0)],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    class _FakeGevalStepsNode:
        def __init__(self, judge_model: str, llm_overrides=None):
            self.judge_model = judge_model
            self.llm_overrides = llm_overrides

        def run(self, metrics):
            return graph_module.GevalStepsArtifacts(
                resolved_steps=[],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    class _FakeGevalNode:
        def __init__(self, judge_model: str):
            self.judge_model = judge_model

        def run(self, resolved_artifacts, generation, question, reference, context):
            return graph_module.GevalMetrics(
                metrics=[MetricResult(name="geval", category=MetricCategory.ANSWER, score=1.0)],
                cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
            )

    monkeypatch.setattr(graph_module, "get_judge_model", _fake_get_judge_model)
    monkeypatch.setattr(graph_module.claim_extractor, "ClaimExtractorNode", _FakeClaimExtractorNode)
    monkeypatch.setattr(graph_module, "GroundingNode", _FakeGroundingNode)
    monkeypatch.setattr(graph_module, "RelevanceNode", _FakeRelevanceNode)
    monkeypatch.setattr(graph_module, "RedteamNode", _FakeRedteamNode)
    monkeypatch.setattr(graph_module, "GevalStepsNode", _FakeGevalStepsNode)
    monkeypatch.setattr(graph_module, "GevalNode", _FakeGevalNode)

    llm_overrides = {"models": {"claims": "openai/gpt-4o-mini"}}
    inputs = Inputs(
        case_id="t1",
        generation=Item(text="The sky is blue.", tokens=4),
        question=Item(text="What color is the sky?", tokens=6),
        reference=Item(text="Sky appears blue due to Rayleigh scattering.", tokens=8),
        context=Item(text="Atmospheric scattering context.", tokens=4),
        geval=Geval(
            metrics=[
                GevalMetricInput(
                    name="factuality",
                    item_fields=["generation", "reference"],
                    criteria=Item(text="Must be factually correct.", tokens=5),
                    evaluation_steps=[Item(text="Check factual alignment", tokens=3)],
                )
            ]
        ),
        redteam=Redteam(
            metrics=[
                RedteamMetricInput(
                    name="toxicity",
                    rubric=RedteamRubric(goal="No toxicity", violations=["abusive language"]),
                    item_fields=["generation"],
                )
            ]
        ),
    )

    chunk = Chunk(
        index=0,
        item=Item(text="The sky is blue.", tokens=4),
        char_start=0,
        char_end=16,
        sha256="abc",
    )
    base_state = {
        "inputs": inputs,
        "llm_overrides": llm_overrides,
        "generation_chunk": ChunkArtifacts(
            chunks=[chunk],
            cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
        ),
    }

    claims_out = graph_module.node_generation_claims(base_state)["generation_claims"]
    dedup_out = graph_module.node_generation_claims_dedup(
        {**base_state, "generation_claims": claims_out}
    )["generation_dedup_claims"]
    graph_module.node_grounding({**base_state, "generation_dedup_claims": dedup_out})
    graph_module.node_relevance({**base_state, "generation_dedup_claims": dedup_out})
    graph_module.node_redteam(base_state)
    steps_out = graph_module.node_geval_steps(base_state)["geval_steps"]
    graph_module.node_geval({**base_state, "geval_steps": steps_out})

    expected_nodes = ["claims", "grounding", "relevance", "redteam", "geval_steps", "geval"]
    got_nodes = [node for node, _ in captured]
    assert got_nodes == expected_nodes
    assert all(ov == llm_overrides for _, ov in captured)
