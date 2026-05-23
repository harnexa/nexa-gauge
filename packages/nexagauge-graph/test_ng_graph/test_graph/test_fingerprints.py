from unittest.mock import patch

from ng_graph.llm import host_model
from ng_graph.runner.fingerprints import _compute_case_fingerprint, _node_route_fingerprint


def test_compute_case_fingerprint_handles_scalar_context() -> None:
    """A scalar ``context`` (int) must not crash the fingerprint path.

    Regression: previously `for c in (context or [])` raised TypeError when
    `context` was a non-iterable scalar. Scanner normalizes it now via the
    shared scan() path.
    """
    case = {
        "case_id": "scalar-ctx",
        "output": "g",
        "input": "q",
        "context": 42,
    }
    fp = _compute_case_fingerprint(case)
    assert isinstance(fp, str) and len(fp) > 0


def test_compute_case_fingerprint_stable_across_alias_spellings() -> None:
    """Same logical content under different alias keys hashes identically."""
    a = {
        "case_id": "c1",
        "output": "the answer",
        "input": "the input",
        "reference": "the truth",
        "context": "some context",
    }
    b = {
        "case_id": "c1",
        "answer": "the answer",
        "query": "the input",
        "gold_answer": "the truth",
        "documents": "some context",
    }
    assert _compute_case_fingerprint(a) == _compute_case_fingerprint(b)


def test_node_route_fingerprint_changes_with_api_base() -> None:
    base_state = {
        "llm_overrides": {
            "models": {"grounding": "openai/local-model"},
            "api_bases": {"grounding": "http://localhost:8080/v1"},
        }
    }
    other_state = {
        "llm_overrides": {
            "models": {"grounding": "openai/local-model"},
            "api_bases": {"grounding": "http://localhost:8081/v1"},
        }
    }

    fp_a = _node_route_fingerprint("grounding", state=base_state, execution_mode="run")
    fp_b = _node_route_fingerprint("grounding", state=other_state, execution_mode="run")

    assert fp_a != fp_b


def test_node_route_fingerprint_disambiguates_self_hosted_model_swap() -> None:
    """Same local URL serving different models must produce different fingerprints.

    Regression: previously the route fingerprint only saw the static
    HOST_MODEL_ROUTE sentinel plus the URL, so swapping the model loaded
    behind localhost:8080 yielded a cache collision.
    """
    state = {
        "llm_overrides": {
            "models": {"grounding": "openai/local-model"},
            "api_bases": {"grounding": "http://localhost:8080/v1"},
        }
    }
    host_model.reset_cache()
    with patch.object(host_model, "_probe", return_value="llama-3.1-8b"):
        fp_llama = _node_route_fingerprint("grounding", state=state, execution_mode="run")
    host_model.reset_cache()
    with patch.object(host_model, "_probe", return_value="mistral-7b"):
        fp_mistral = _node_route_fingerprint("grounding", state=state, execution_mode="run")
    host_model.reset_cache()

    assert fp_llama != fp_mistral


def test_node_route_fingerprint_stable_when_probe_fails() -> None:
    """If /v1/models is unreachable, fingerprint must remain deterministic."""
    state = {
        "llm_overrides": {
            "models": {"grounding": "openai/local-model"},
            "api_bases": {"grounding": "http://localhost:8080/v1"},
        }
    }
    host_model.reset_cache()
    with patch.object(host_model, "_probe", return_value=None):
        fp_a = _node_route_fingerprint("grounding", state=state, execution_mode="run")
    host_model.reset_cache()
    with patch.object(host_model, "_probe", return_value=None):
        fp_b = _node_route_fingerprint("grounding", state=state, execution_mode="run")
    host_model.reset_cache()

    assert fp_a == fp_b
