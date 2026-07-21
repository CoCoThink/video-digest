"""Shared ASR backend contract and errors."""
from __future__ import annotations

import math
import re
from typing import Callable, Protocol
import unicodedata
import wave


ProgressCallback = Callable[[int], None]


class AsrBackend(Protocol):
    name: str
    model_id: str

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        context: str = "",
        progress_cb: ProgressCallback | None = None,
    ) -> dict: ...


class AsrUnstableError(RuntimeError):
    """The decoder remained invalid after reaching the minimum chunk size."""


class AsrOutputError(RuntimeError):
    """The backend returned text/timestamps that violate the common contract."""


def normalized_chars(value: str) -> str:
    """Return the semantic characters shared by ASR text and aligner tokens."""
    return "".join(
        ch for ch in value
        if ch == "'" or unicodedata.category(ch)[:1] in {"L", "N"}
    )


def context_terms(value: str) -> list[str]:
    """Parse a free-form comma/semicolon/newline separated domain vocabulary."""
    terms = []
    seen = set()
    for part in re.split(r"[,，;；\n]+", value):
        term = part.strip()
        if term and term not in seen:
            terms.append(term)
            seen.add(term)
    return terms


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as reader:
        return reader.getnframes() / reader.getframerate()


def validate_asr_output(
    data: dict,
    audio_duration: float,
    *,
    backend_name: str,
) -> None:
    """Validate the text/items contract before layout can create malformed SRT."""
    if not isinstance(data, dict):
        raise AsrOutputError(f"{backend_name} returned a non-object result")
    text = data.get("text")
    items = data.get("items")
    if not isinstance(text, str) or not isinstance(items, list):
        raise AsrOutputError(f"{backend_name} returned malformed text/items")

    source = normalized_chars(text)
    target_parts: list[str] = []
    previous_end = 0.0
    for item in items:
        if not isinstance(item, dict) or not {"t", "s", "e"}.issubset(item):
            raise AsrOutputError(f"{backend_name} returned a malformed timestamp item")
        target_parts.append(normalized_chars(str(item["t"])))
        try:
            start, end = float(item["s"]), float(item["e"])
        except (TypeError, ValueError) as exc:
            raise AsrOutputError(
                f"{backend_name} returned non-numeric timestamps"
            ) from exc
        if (not math.isfinite(start) or not math.isfinite(end)
                or start < -0.001 or end < start
                or end > audio_duration + 0.25
                or start < previous_end - 0.05):
            raise AsrOutputError(
                f"{backend_name} returned invalid absolute timestamps"
            )
        previous_end = max(previous_end, end)

    target = "".join(target_parts)
    if source != target:
        mismatch = next(
            (index for index, (left, right) in enumerate(zip(source, target))
             if left != right),
            min(len(source), len(target)),
        )
        raise AsrOutputError(
            f"{backend_name} text/items mismatch at normalized character {mismatch} "
            f"(text={len(source)}, items={len(target)})"
        )
