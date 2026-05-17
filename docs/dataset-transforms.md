# Dataset Transforms

Sometimes your Logs or a HuggingFace dataset doesn't fit nexa-gauge's expected input shape with simple column-renaming. The canonical example is `hotpotqa/hotpot_qa`, whose `context` is `{title: list[str], sentences: list[list[str]]}` — there is no flat column to alias into `context: str | list[str]`.

**Transforms** let you supply a small Python function that reshapes one raw record into the dict shape nexa-gauge expects, with one CLI flag.

## When to use what

| Mismatch | Use |
|---|---|
| Same data, different column name (`text` → `generation`) | `--field generation=text` |
| Structural reshape (nested dict, multiple source columns → one field, etc.) | `--extension-file ... --transform ...` |
| Both | Both — transform runs first, then aliases resolve column names on the result. |

## Write a transform

Create a Python file anywhere (no packaging required):

```python
# my_transforms.py
from ng_core import register_transform


@register_transform("hotpot_qa")
def hotpot_qa(record: dict) -> dict:
    ctx = record.get("context") or {}
    paragraphs = [
        f"{title}\n{' '.join(sents)}"
        for title, sents in zip(ctx.get("title", []), ctx.get("sentences", []))
    ]
    return {
        "case_id":    record.get("id"),
        "question":   record["question"],
        "generation": record["answer"],
        "context":    paragraphs,
        "reference":  record["answer"],
    }
```

The contract:

- **Input:** one raw record dict (whatever the adapter yields).
- **Output:** a dict with any subset of `case_id`, `question`, `generation`, `context`, `reference`. Keys not listed are ignored.
- **Pure and threadsafe.** No I/O, no shared mutable state.
- **Errors** raise as `InputParseError` with the record index, so they show up in the CLI error path uniformly.

`geval` and `redteam` are nexa-gauge metric configs (not dataset data) — don't try to reshape them via transforms.

## Run it

```bash
nexagauge run eval \
  --input hf://hotpotqa/hotpot_qa \
  --hf-config distractor \
  --extension-file ./my_transforms.py \
  --transform hotpot_qa \
  --host-model-url http://127.0.0.1:8080/v1 \
  --llm-concurrency 4
```

Both `run` and `estimate` accept the flags.

## Compose with `--field`

Transforms reshape; `--field` renames. They compose:

```bash
nexagauge run eval \
  --input hf://my-team/dataset \
  --extension-file ./my_transforms.py \
  --transform team_dataset \
  --field question=user_question
```

The transform runs first, then `--field` aliases resolve column names on the transformed dict.

## Errors

| Situation | Result |
|---|---|
| `--transform` set, name not registered | CLI exits with `InputParseError` listing registered names. |
| `--extension-file` does not exist | `InputParseError: Extension file not found` |
| Transform raises on a record | `InputParseError(record_index=N)` — halts the run. |
| Transform returns non-dict | `InputParseError: Transform '<name>' returned <type>, expected a dict.` |

## Future shape

The same `@register_transform` decorator is the dispatch surface a future Python API or REST endpoint will reuse. Writing a transform today gives you all three call paths for free.
