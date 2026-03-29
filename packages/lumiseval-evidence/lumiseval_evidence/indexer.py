"""
LanceDB Indexer — vectorizes documents and upserts them into a local LanceDB collection.

Used by:
  - The ingest pre-processing step (user-provided files)
  - The Evidence Router (indexes web-retrieved documents on the fly)

TODO: Implement MCP LanceDB connection support.
"""

from pathlib import Path

import lancedb
from lumiseval_core.config import config
from sentence_transformers import SentenceTransformer

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def index_texts(
    texts: list[str],
    doc_ids: list[str],
    db_path: str | None = None,
    table_name: str = "documents",
) -> None:
    """Embed ``texts`` and upsert into LanceDB.

    Args:
        texts: Raw text passages to index.
        doc_ids: Unique identifiers for each passage (must match length of texts).
        db_path: Path to the LanceDB directory. Defaults to config.LANCEDB_PATH.
        table_name: LanceDB table name to write into.
    """
    if not texts:
        return

    db_path = db_path or config.LANCEDB_PATH
    model = _get_model()
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    records = [
        {"id": doc_id, "text": text, "vector": emb}
        for doc_id, text, emb in zip(doc_ids, texts, embeddings)
    ]

    db = lancedb.connect(db_path)
    if table_name in db.table_names():
        table = db.open_table(table_name)
        table.add(records)
    else:
        db.create_table(table_name, records)


def index_file(file_path: str | Path, db_path: str | None = None) -> int:
    """Read a plain text file, split into paragraphs, and index into LanceDB.

    Returns the number of passages indexed.
    """
    path = Path(file_path)
    text = path.read_text()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    doc_ids = [f"{path.stem}:{i}" for i in range(len(paragraphs))]
    index_texts(paragraphs, doc_ids, db_path=db_path)
    return len(paragraphs)
