"""FireRedASR2-AED backend for exact 16 kHz PCM chunks."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .base import AsrUnstableError, validate_asr_output, wav_duration
from .wav_chunks import iter_wav_chunks

PINNED_SOURCE_REVISION = "4e7d9aaf4482a47cec1724807026b9b151926eb5"
REQUIRED_MODEL_FILES = ("model.pth.tar", "dict.txt", "cmvn.ark", "train_bpe1000.model")


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


def _iter_wav_chunks(
    source: Path,
    directory: Path,
    chunk_sec: float,
    *,
    boundary_search_sec: float = 0.0,
):
    """Compatibility wrapper used by tests and the FireRed retry path."""
    return iter_wav_chunks(
        source, directory, chunk_sec,
        boundary_search_sec=boundary_search_sec,
    )


def _split_retry_chunk(source: Path, directory: Path, chunk_sec: float):
    """Materialize only the two children needed by a recursive retry."""
    return _iter_wav_chunks(source, directory, chunk_sec)


def _invalid_timestamps(timestamps, duration: float) -> str | None:
    previous_end = 0.0
    for entry in timestamps:
        if not isinstance(entry, (list, tuple)) or len(entry) < 3:
            return "malformed_timestamp"
        try:
            start, end = float(entry[1]), float(entry[2])
        except (TypeError, ValueError):
            return "malformed_timestamp"
        if not math.isfinite(start) or not math.isfinite(end):
            return "non_finite_timestamp"
        if start < -0.001 or end < start or end > duration + 0.25:
            return "timestamp_out_of_range"
        if start < previous_end - 0.05:
            return "timestamp_overlap"
        previous_end = max(previous_end, end)
    return None


def _suspicious_result(result: dict, duration: float) -> str | None:
    """Detect decoder runaway, invalid alignment and uniform fallback timing."""
    text = result.get("text", "")
    timestamps = result.get("timestamp") or []
    if text and not timestamps:
        return "text_without_timestamps"
    if len(timestamps) > max(100, duration * 12):
        return "implausible_token_rate"
    invalid = _invalid_timestamps(timestamps, duration)
    if invalid:
        return invalid
    if len(timestamps) >= 100:
        durations = [round(float(end) - float(start), 4) for _, start, end in timestamps]
        most_common = max(durations.count(value) for value in set(durations))
        if most_common / len(durations) >= 0.95:
            return "uniform_fallback_timestamps"
    return None


def _transcribe_with_retry(model, chunk_path: Path, offset: float, duration: float,
                           retry_dir: Path, min_chunk_sec: float):
    started = time.perf_counter()
    result = model.transcribe([chunk_path.stem], [str(chunk_path)])[0]
    elapsed = time.perf_counter() - started
    reason = _suspicious_result(result, duration)
    if reason is None:
        return [(offset, duration, result, elapsed)], []
    if duration <= min_chunk_sec + 0.001:
        raise AsrUnstableError(
            f"FireRed output remained unstable at {offset:.3f}s "
            f"({reason}, chunk={duration:.3f}s)"
        )

    current_dir = retry_dir / (
        f"{round(offset * 1000):012d}_{round(duration * 1000):08d}"
    )
    current_dir.mkdir(parents=True, exist_ok=True)
    leaves = []
    events = [{
        "offset_sec": round(offset, 3),
        "duration_sec": round(duration, 3),
        "reason": reason,
        "discarded_text_chars": len(result.get("text", "")),
        "discarded_tokens": len(result.get("timestamp") or []),
    }]
    try:
        for subpath, suboffset, subduration in _split_retry_chunk(
                chunk_path, current_dir, duration / 2):
            subleaves, subevents = _transcribe_with_retry(
                model, subpath, offset + suboffset, subduration,
                retry_dir, min_chunk_sec,
            )
            leaves.extend(subleaves)
            events.extend(subevents)
    finally:
        try:
            current_dir.rmdir()
        except OSError:
            pass
    return leaves, events


def _validate_contract(data: dict, audio_duration: float) -> None:
    validate_asr_output(data, audio_duration, backend_name="FireRedASR2")


def _add_source_dir(source_dir: Path) -> None:
    source_dir = source_dir.expanduser().resolve()
    package_parent = source_dir / "fireredasr2s"
    candidate = package_parent if (package_parent / "fireredasr2").is_dir() else source_dir
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _load_api(source_dir: Path):
    _add_source_dir(source_dir)
    try:
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
    except ImportError:
        try:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        except ImportError as exc:
            raise RuntimeError(
                "FireRedASR2S source is unavailable; configure VD_FIRERED_SOURCE_DIR"
            ) from exc

    # Upstream may feed BF16 log probabilities to forced_align, which only accepts
    # FP16/FP32/FP64. Keep model compute in BF16 and cast at the narrow API boundary.
    import torchaudio

    original = torchaudio.functional.forced_align
    if not getattr(original, "_fireredasr2_bf16_compat", False):
        def forced_align_compat(log_probs, *args, **kwargs):
            if str(log_probs.dtype) == "torch.bfloat16":
                log_probs = log_probs.float()
            return original(log_probs, *args, **kwargs)

        forced_align_compat._fireredasr2_bf16_compat = True
        torchaudio.functional.forced_align = forced_align_compat
    return FireRedAsr2, FireRedAsr2Config


class FireRedBackend:
    name = "firered"
    model_id = "FireRedASR2-AED"

    def __init__(self, cfg: dict):
        import torch

        model_path = cfg.get("firered_model_path")
        if not model_path:
            raise ValueError("VD_FIRERED_MODEL_PATH is required for the firered backend")
        model_path = Path(model_path).expanduser()
        if not model_path.is_dir():
            raise ValueError(f"FireRed model directory does not exist: {model_path}")
        missing = [name for name in REQUIRED_MODEL_FILES if not (model_path / name).is_file()]
        if missing:
            raise ValueError(f"FireRed model directory is incomplete: {', '.join(missing)}")
        source_value = cfg.get("firered_source_dir")
        if not source_value:
            raise ValueError(
                "VD_FIRERED_SOURCE_DIR is required for a reproducible FireRed backend"
            )
        source_dir = Path(source_value).expanduser()
        if not source_dir.is_dir():
            raise ValueError(f"FireRed source directory does not exist: {source_dir}")
        revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            text=True, capture_output=True,
        )
        if revision.returncode != 0:
            raise ValueError("VD_FIRERED_SOURCE_DIR must be a pinned git checkout")
        actual = revision.stdout.strip()
        if actual != PINNED_SOURCE_REVISION:
            raise ValueError(
                f"unsupported FireRedASR2S revision {actual}; "
                f"expected {PINNED_SOURCE_REVISION}"
            )
        dirty = subprocess.run(
            ["git", "-C", str(source_dir), "status", "--porcelain"],
            text=True, capture_output=True,
        )
        if dirty.returncode != 0 or dirty.stdout.strip():
            raise ValueError("VD_FIRERED_SOURCE_DIR must be a clean pinned checkout")
        if not torch.cuda.is_available():
            raise RuntimeError("FireRed backend requires CUDA")

        FireRedAsr2, FireRedAsr2Config = _load_api(source_dir)
        config = FireRedAsr2Config(
            use_gpu=True,
            use_half=True,
            beam_size=int(cfg.get("firered_beam_size", 1)),
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        self._model = FireRedAsr2.from_pretrained("aed", str(model_path), config)
        self._punctuation = None
        if cfg.get("firered_punc_enabled", False):
            punc_path = cfg.get("firered_punc_model_path")
            if not punc_path:
                raise ValueError(
                    "VD_FIRERED_PUNC_MODEL_PATH is required when FireRedPunc is enabled"
                )
            from server.pipeline.punctuation import FireRedPunctuation
            self._punctuation = FireRedPunctuation(
                punc_path, device=str(cfg.get("firered_punc_device", "cpu")),
            )
            self.model_id = "FireRedASR2-AED+FireRedPunc"
        self._chunk_sec = float(cfg.get("chunk_sec", 30))
        self._min_chunk_sec = float(cfg.get("firered_min_chunk_sec", 7.5))
        self._boundary_search_sec = float(
            cfg.get("firered_boundary_search_sec", 5.0)
        )
        if self._min_chunk_sec <= 0 or self._min_chunk_sec > self._chunk_sec:
            raise ValueError("VD_FIRERED_MIN_CHUNK_SEC must be in (0, VD_CHUNK_SEC]")
        if not 0 <= self._boundary_search_sec < self._chunk_sec:
            raise ValueError(
                "VD_FIRERED_BOUNDARY_SEARCH_SEC must be in [0, VD_CHUNK_SEC)"
            )

    @staticmethod
    def _duration(path: Path) -> float:
        return wav_duration(str(path))

    def transcribe(
        self, audio_path: str, *, language=None, context="", progress_cb=None,
    ) -> dict:
        import torch

        if language is not None and str(language).strip():
            raise ValueError(
                "FireRedASR2-AED does not support forced language; "
                "leave VD_ASR_LANGUAGE empty"
            )
        source = Path(audio_path)
        duration = self._duration(source)
        chunk_total = max(1, math.ceil(duration / self._chunk_sec))
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        def run_once():
            text_parts: list[str] = []
            items: list[dict] = []
            retry_events: list[dict] = []
            punc_warnings: list[dict] = []
            punc_elapsed = 0.0
            with tempfile.TemporaryDirectory(prefix="vd-firered-") as value:
                temp_dir = Path(value)
                retry_dir = temp_dir / "retries"
                with torch.no_grad():
                    for index, (path, offset, chunk_duration) in enumerate(
                            _iter_wav_chunks(
                                source, temp_dir, self._chunk_sec,
                                boundary_search_sec=self._boundary_search_sec,
                            ), 1):
                        leaves, events = _transcribe_with_retry(
                            self._model, path, offset, chunk_duration,
                            retry_dir, self._min_chunk_sec,
                        )
                        retry_events.extend(events)
                        leaf_text = []
                        for leaf_offset, _, result, _ in leaves:
                            leaf_text.append(result.get("text", ""))
                            for token, start, end in result.get("timestamp") or []:
                                items.append({
                                    "t": token,
                                    "s": round(leaf_offset + float(start), 3),
                                    "e": round(leaf_offset + float(end), 3),
                                })
                        chunk_text = _join_text(leaf_text)
                        if self._punctuation is not None and chunk_text:
                            punc_started = time.perf_counter()
                            try:
                                chunk_text = self._punctuation.process(chunk_text)
                            except Exception as exc:
                                punc_warnings.append({
                                    "code": "PUNC_TEXT_MISMATCH"
                                    if type(exc).__name__ == "PunctuationProjectionError"
                                    else "PUNC_FAILED",
                                    "offset_sec": round(offset, 3),
                                    "error": f"{type(exc).__name__}: {exc}",
                                })
                            finally:
                                punc_elapsed += time.perf_counter() - punc_started
                        text_parts.append(chunk_text)
                        if progress_cb:
                            try:
                                progress_cb(max(1, min(99, int(index / chunk_total * 100))))
                            except Exception:
                                pass
            data = {
                "text": _join_text(text_parts),
                "language": "Unknown",
                "items": items,
                "backend": self.name,
                "model_id": self.model_id,
                "warnings": [
                    {"code": "ASR_CHUNK_RETRIED", **event} for event in retry_events
                ] + punc_warnings + ([{
                    "code": "ASR_CONTEXT_UNSUPPORTED",
                    "message": "FireRedASR2-AED does not accept context terms",
                }] if context.strip() else []),
                "punctuation": self._punctuation is not None,
                "_punc_sec": round(punc_elapsed, 3),
            }
            _validate_contract(data, duration)
            if progress_cb:
                try:
                    progress_cb(99)
                except Exception:
                    pass
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
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            data["_peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
        return data
