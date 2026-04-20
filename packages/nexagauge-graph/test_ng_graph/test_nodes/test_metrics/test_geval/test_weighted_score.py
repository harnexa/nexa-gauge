import math

from ng_graph.nodes.metrics.geval.weighted_score import (
    LOG_PROB_FLOOR,
    calculate_weighted_summed_score,
)


def test_weighted_equals_expected_value() -> None:
    tokens = [
        {
            "token": "7",
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": "6", "logprob": math.log(0.3)},
                {"token": "7", "logprob": math.log(0.5)},
                {"token": "8", "logprob": math.log(0.2)},
            ],
        }
    ]
    # (6·0.3 + 7·0.5 + 8·0.2) / (0.3 + 0.5 + 0.2) = 6.9
    assert abs(calculate_weighted_summed_score(7, tokens) - 6.9) < 1e-9


def test_score_token_missing_returns_raw() -> None:
    tokens = [
        {"token": "foo", "logprob": 0.0, "top_logprobs": []},
    ]
    assert calculate_weighted_summed_score(7, tokens) == 7.0


def test_empty_content_returns_raw() -> None:
    assert calculate_weighted_summed_score(7, []) == 7.0


def test_all_below_floor_returns_raw() -> None:
    below = LOG_PROB_FLOOR - 1.0
    tokens = [
        {
            "token": "5",
            "logprob": math.log(0.9),
            "top_logprobs": [
                {"token": "4", "logprob": below},
                {"token": "5", "logprob": below},
                {"token": "6", "logprob": below},
            ],
        }
    ]
    assert calculate_weighted_summed_score(5, tokens) == 5.0


def test_non_decimal_tokens_filtered() -> None:
    tokens = [
        {
            "token": "7",
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": "7", "logprob": math.log(0.5)},
                {"token": ".", "logprob": math.log(0.4)},
                {"token": "/10", "logprob": math.log(0.3)},
                {"token": "seven", "logprob": math.log(0.2)},
            ],
        }
    ]
    # Only "7" survives → weighted collapses to 7.0.
    assert calculate_weighted_summed_score(7, tokens) == 7.0


def test_out_of_range_tokens_filtered() -> None:
    tokens = [
        {
            "token": "7",
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": "7", "logprob": math.log(0.5)},
                {"token": "11", "logprob": math.log(0.4)},
                {"token": "100", "logprob": math.log(0.3)},
            ],
        }
    ]
    assert calculate_weighted_summed_score(7, tokens) == 7.0


def test_whitespace_token_normalization() -> None:
    tokens = [
        {
            "token": " 7",  # provider returns with leading space
            "logprob": math.log(0.5),
            "top_logprobs": [
                {"token": " 7", "logprob": math.log(0.5)},
                {"token": "6", "logprob": math.log(0.3)},
            ],
        }
    ]
    # (7·0.5 + 6·0.3) / 0.8 ≈ 6.625
    expected = (7 * 0.5 + 6 * 0.3) / 0.8
    assert abs(calculate_weighted_summed_score(7, tokens) - expected) < 1e-9


def test_custom_score_range() -> None:
    tokens = [
        {
            "token": "3",
            "logprob": math.log(0.7),
            "top_logprobs": [
                {"token": "2", "logprob": math.log(0.3)},
                {"token": "3", "logprob": math.log(0.7)},
                {"token": "7", "logprob": math.log(0.2)},  # out of range [1,5]
            ],
        }
    ]
    expected = (2 * 0.3 + 3 * 0.7) / (0.3 + 0.7)
    assert (
        abs(calculate_weighted_summed_score(3, tokens, score_min=1, score_max=5) - expected) < 1e-9
    )
