from __future__ import annotations

from adapters.local_file import LocalFileDatasetAdapter


def test_jsonl_limit_does_not_parse_rows_beyond_window(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"case_id":"a","generation":"hello"}',
                "{not-json",
            ]
        )
    )

    adapter = LocalFileDatasetAdapter(path)
    cases = list(adapter.iter_cases(limit=1))

    assert len(cases) == 1
    assert cases[0]["case_id"] == "a"
    assert cases[0]["generation"] == "hello"


def test_csv_streaming_respects_limit(tmp_path) -> None:
    path = tmp_path / "cases.csv"
    path.write_text(
        "\n".join(
            [
                "case_id,generation,question",
                "a,hello,q1",
                "b,world,q2",
                "c,last,q3",
            ]
        )
    )

    adapter = LocalFileDatasetAdapter(path)
    cases = list(adapter.iter_cases(limit=2))

    assert [c["case_id"] for c in cases] == ["a", "b"]
    assert [c["generation"] for c in cases] == ["hello", "world"]
