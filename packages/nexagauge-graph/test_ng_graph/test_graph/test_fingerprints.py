from ng_graph.runner.fingerprints import _node_route_fingerprint


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
