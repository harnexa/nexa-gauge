"""Example transform that reshapes hotpot_qa-style records into nexa-gauge inputs.

Also used by ``test_transforms_cli.py`` as the canonical UDF integration fixture.
"""

from ng_core import register_transform


@register_transform("hotpot_qa")
def hotpot_qa(record: dict) -> dict:
    """Flatten hotpot_qa's nested context into a list of paragraph strings."""
    ctx = record.get("context") or {}
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    paragraphs = [f"{title}\n{' '.join(sents)}" for title, sents in zip(titles, sentences)]
    output_text = record.get("output") or record.get("answer", "")
    return {
        "case_id": record.get("id"),
        "input": record.get("input", ""),
        "output": output_text,
        "context": paragraphs,
        "reference": output_text,
    }
