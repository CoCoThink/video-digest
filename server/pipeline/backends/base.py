"""Shared ASR backend contract and errors."""
from __future__ import annotations

from typing import Callable, Protocol


ProgressCallback = Callable[[int], None]


class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> dict: ...


class AsrUnstableError(RuntimeError):
    """The decoder remained invalid after reaching the minimum chunk size."""


class AsrOutputError(RuntimeError):
    """The backend returned text/timestamps that violate the common contract."""
