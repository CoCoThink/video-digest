"""Experimental Fun-ASR-Nano backend with native character timestamps."""
from __future__ import annotations

from pathlib import Path
import tempfile

from .base import (
    AsrOutputError,
    context_terms,
    normalized_chars,
    validate_asr_output,
    wav_duration,
)
from .wav_chunks import iter_wav_chunks


def _join_text(parts: list[str]) -> str:
    text = ""
    for value in parts:
        part = value.strip()
        if not part:
            continue
        if (text and text[-1].isascii() and text[-1].isalnum()
                and part[0].isascii() and part[0].isalnum()):
            text += " "
        text += part
    return text


def _adapt_result(result: dict, offset: float, chunk_duration: float) -> tuple[str, list[dict]]:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise AsrOutputError("Fun-ASR-Nano returned a malformed result")
    text = result["text"].strip()
    timestamps = result.get("timestamps")
    if normalized_chars(text) and not isinstance(timestamps, list):
        raise AsrOutputError("Fun-ASR-Nano returned text without timestamps")

    items = []
    for entry in timestamps or []:
        if not isinstance(entry, dict):
            raise AsrOutputError("Fun-ASR-Nano returned a malformed timestamp")
        token = str(entry.get("token", ""))
        # Nano aligns punctuation as well; layout assigns punctuation from full text.
        if not normalized_chars(token):
            continue
        try:
            start = float(entry["start_time"])
            end = float(entry["end_time"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AsrOutputError(
                "Fun-ASR-Nano returned a malformed timestamp"
            ) from exc
        if start < -0.001 or end > chunk_duration + 0.25:
            raise AsrOutputError("Fun-ASR-Nano returned an out-of-range timestamp")
        items.append({
            "t": token,
            "s": round(offset + start, 3),
            "e": round(offset + end, 3),
        })
    return text, items


class FunAsrBackend:
    name = "funasr"

    def __init__(self, cfg: dict):
        self.model_id = str(
            cfg.get("funasr_model_path") or "FunAudioLLM/Fun-ASR-Nano-2512"
        )
        self._chunk_sec = float(cfg.get("chunk_sec", 30))
        self._boundary_search_sec = float(
            cfg.get("funasr_boundary_search_sec", 5.0)
        )
        if self._chunk_sec <= 0:
            raise ValueError("VD_CHUNK_SEC must be positive")
        if not 0 <= self._boundary_search_sec < self._chunk_sec:
            raise ValueError(
                "VD_FUNASR_BOUNDARY_SEARCH_SEC must be in [0, VD_CHUNK_SEC)"
            )

        try:
            import torch
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "FunASR is unavailable; install requirements-funasr.txt"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Fun-ASR-Nano backend requires CUDA")

        self._model = AutoModel(
            model=self.model_id,
            device="cuda:0",
            disable_update=True,
            disable_pbar=True,
            disable_log=True,
        )

    def transcribe(
        self, audio_path: str, *, language=None, context="", progress_cb=None,
    ) -> dict:
        import torch

        source = Path(audio_path)
        duration = wav_duration(str(source))
        hotwords = context_terms(context)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_once() -> dict:
            text_parts: list[str] = []
            items: list[dict] = []
            with tempfile.TemporaryDirectory(prefix="vd-funasr-") as value:
                temp_dir = Path(value)
                for path, offset, chunk_duration in iter_wav_chunks(
                    source,
                    temp_dir,
                    self._chunk_sec,
                    boundary_search_sec=self._boundary_search_sec,
                ):
                    kwargs = {"input": str(path), "batch_size": 1}
                    if hotwords:
                        kwargs["hotwords"] = hotwords
                    if language is not None and str(language).strip():
                        kwargs["language"] = str(language).strip()
                    results = self._model.generate(**kwargs)
                    if not isinstance(results, list) or len(results) != 1:
                        raise AsrOutputError(
                            "Fun-ASR-Nano returned an unexpected result count"
                        )
                    chunk_text, chunk_items = _adapt_result(
                        results[0], offset, chunk_duration,
                    )
                    text_parts.append(chunk_text)
                    items.extend(chunk_items)
                    if progress_cb:
                        try:
                            progress_cb(max(
                                1, min(99, int((offset + chunk_duration) / duration * 99))
                            ))
                        except Exception:
                            pass

            data = {
                "text": _join_text(text_parts),
                "language": str(language).strip()
                if language is not None and str(language).strip() else "Unknown",
                "items": items,
                "backend": self.name,
                "model_id": self.model_id,
                "warnings": [],
                "punctuation": True,
            }
            validate_asr_output(
                data, duration, backend_name="Fun-ASR-Nano",
            )
            return data

        try:
            data = run_once()
        except Exception as exc:
            message = str(exc).lower()
            is_oom = ("OutOfMemory" in type(exc).__name__
                      or "out of memory" in message or "cuda driver error" in message)
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            data = run_once()
        torch.cuda.synchronize()
        data["_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        return data
