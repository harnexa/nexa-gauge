"""Shared parallel orchestration helpers for metric fan-out."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    items: Sequence[T],
    worker: Callable[[T], R],
    max_workers: int | None = None,
) -> list[R]:
    """Run ``worker`` across ``items`` in parallel while preserving input order."""
    if not items:
        return []

    workers = len(items) if max_workers is None else int(max_workers)
    workers = max(1, min(len(items), workers))
    if workers <= 1:
        return [worker(item) for item in items]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(worker, items))
