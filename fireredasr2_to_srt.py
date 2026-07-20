#!/usr/bin/env python3
"""Run FireRedASR2-AED on a local audio/video file and build project-style subtitles.

The upstream AED model accepts short utterances, so this script converts the input
to 16 kHz mono PCM and transcribes exact-duration chunks with one resident model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import wave


DEFAULT_MODEL_DIR = os.environ.get("FIREREDASR2_MODEL_DIR")
DEFAULT_SOURCE_DIR = os.environ.get("FIREREDASR2_SOURCE_DIR")


def _run_ffmpeg(source: Path, target: Path) -> None:
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn",
        "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le", str(target),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required but was not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg could not decode {source}") from exc


def _split_wav(source: Path, directory: Path, chunk_sec: float):
    """Split PCM WAV without resampling; return (path, offset_s, duration_s)."""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec must be positive")

    chunks = []
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        if params.nchannels != 1 or params.sampwidth != 2 or params.framerate != 16000:
            raise ValueError("expected 16 kHz, 16-bit, mono PCM WAV")
        frames_per_chunk = max(1, round(chunk_sec * params.framerate))
        start_frame = 0
        index = 0
        while start_frame < params.nframes:
            frame_count = min(frames_per_chunk, params.nframes - start_frame)
            audio = reader.readframes(frame_count)
            path = directory / f"chunk_{index:04d}.wav"
            with wave.open(str(path), "wb") as writer:
                writer.setparams(params)
                writer.setnframes(frame_count)
                writer.writeframes(audio)
            chunks.append((
                path,
                start_frame / params.framerate,
                frame_count / params.framerate,
            ))
            start_frame += frame_count
            index += 1
    return chunks


def _join_text(parts: list[str]) -> str:
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if (text and text[-1].isascii() and text[-1].isalnum()
                and part[0].isascii() and part[0].isalnum()):
            text += " "
        text += part
    return text


def _suspicious_result(result: dict, duration: float) -> str | None:
    """Detect decoder runaway and upstream's uniform-timestamp fallback."""
    text = result.get("text", "")
    timestamps = result.get("timestamp") or []
    if text and not timestamps:
        return "text_without_timestamps"
    if len(timestamps) > max(100, duration * 12):
        return "implausible_token_rate"
    if len(timestamps) >= 100:
        durations = [round(float(end) - float(start), 4) for _, start, end in timestamps]
        most_common = max(durations.count(value) for value in set(durations))
        if most_common / len(durations) >= 0.95:
            return "uniform_fallback_timestamps"
    return None


def _transcribe_with_retry(model, chunk_path: Path, offset: float, duration: float,
                           retry_dir: Path, min_chunk_sec: float = 7.5):
    """Return valid leaf results, recursively halving suspicious ASR output."""
    started = time.perf_counter()
    result = model.transcribe([chunk_path.stem], [str(chunk_path)])[0]
    elapsed = time.perf_counter() - started
    reason = _suspicious_result(result, duration)
    if reason is None:
        return [(offset, duration, result, elapsed)], []
    if duration <= min_chunk_sec + 0.001:
        raise RuntimeError(
            f"suspicious ASR output at {offset:.3f}s ({reason}) even with "
            f"{duration:.3f}s chunk"
        )

    current_retry_dir = retry_dir / (
        f"{round(offset * 1000):012d}_{round(duration * 1000):08d}"
    )
    current_retry_dir.mkdir(parents=True, exist_ok=True)
    subchunks = _split_wav(chunk_path, current_retry_dir, duration / 2)
    leaves = []
    events = [{
        "offset_sec": round(offset, 3),
        "duration_sec": round(duration, 3),
        "reason": reason,
        "discarded_text_chars": len(result.get("text", "")),
        "discarded_tokens": len(result.get("timestamp") or []),
    }]
    for subpath, suboffset, subduration in subchunks:
        subleaves, subevents = _transcribe_with_retry(
            model, subpath, offset + suboffset, subduration,
            retry_dir, min_chunk_sec=min_chunk_sec,
        )
        leaves.extend(subleaves)
        events.extend(subevents)
    return leaves, events


def _add_source_dir(source_dir: Path | None) -> None:
    if source_dir is None:
        return
    source_dir = source_dir.resolve()
    package_parent = source_dir / "fireredasr2s"
    candidate = package_parent if (package_parent / "fireredasr2").is_dir() else source_dir
    sys.path.insert(0, str(candidate))


def _load_backend(source_dir: Path | None):
    _add_source_dir(source_dir)
    try:
        from fireredasr2.asr import FireRedAsr2, FireRedAsr2Config
    except ImportError:
        try:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        except ImportError as exc:
            raise RuntimeError(
                "FireRedASR2S source is unavailable; pass --source-dir or set "
                "FIREREDASR2_SOURCE_DIR"
            ) from exc

    # torchaudio forced_align supports FP16/FP32/FP64, but not BF16. Upstream
    # selects BF16 on Ampere and otherwise silently falls back to uniform timing.
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


def _source_revision(source_dir: Path | None) -> str | None:
    if source_dir is None:
        return None
    result = subprocess.run(
        ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
        text=True, capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Audio or video input")
    parser.add_argument("--out", required=True, type=Path, help="Output SRT path")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR,
                        help="FireRedASR2S checkout (or set FIREREDASR2_SOURCE_DIR)")
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--cpu", action="store_true", help="Run in FP32 on CPU")
    args = parser.parse_args(argv)
    if args.model_dir is None:
        parser.error("--model-dir or FIREREDASR2_MODEL_DIR is required")
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if not args.model_dir.is_dir():
        parser.error(f"model directory does not exist: {args.model_dir}")
    if args.source_dir is not None and not args.source_dir.is_dir():
        parser.error(f"source directory does not exist: {args.source_dir}")
    if args.beam_size < 1:
        parser.error("--beam-size must be at least 1")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    FireRedAsr2, FireRedAsr2Config = _load_backend(args.source_dir)

    import torch
    from server.pipeline import layout, subtitle

    use_gpu = not args.cpu
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu for a slow CPU-only run")

    device = torch.cuda.get_device_name(0) if use_gpu else "cpu"
    print(f"[info] device={device} chunk={args.chunk_sec:g}s beam={args.beam_size}", flush=True)

    with tempfile.TemporaryDirectory(prefix="fireredasr2-") as tmp:
        tmpdir = Path(tmp)
        wav_path = tmpdir / "audio.wav"
        preprocess_started = time.perf_counter()
        _run_ffmpeg(args.input, wav_path)
        chunks = _split_wav(wav_path, tmpdir, args.chunk_sec)
        preprocess_sec = time.perf_counter() - preprocess_started
        total_duration = sum(duration for _, _, duration in chunks)
        print(f"[info] prepared {len(chunks)} chunks, {total_duration:.3f}s audio", flush=True)

        config = FireRedAsr2Config(
            use_gpu=use_gpu,
            use_half=use_gpu,
            beam_size=args.beam_size,
            nbest=1,
            decode_max_len=0,
            softmax_smoothing=1.25,
            aed_length_penalty=0.6,
            eos_penalty=1.0,
            return_timestamp=True,
        )
        load_started = time.perf_counter()
        model = FireRedAsr2.from_pretrained("aed", str(args.model_dir), config)
        load_sec = time.perf_counter() - load_started
        if use_gpu:
            model_gpu_gb = torch.cuda.memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
        else:
            model_gpu_gb = 0.0

        text_parts = []
        items = []
        chunk_metrics = []
        retry_events = []
        retry_dir = tmpdir / "retries"
        inference_started = time.perf_counter()
        for index, (chunk_path, offset, duration) in enumerate(chunks, 1):
            chunk_started = time.perf_counter()
            leaves, events = _transcribe_with_retry(
                model, chunk_path, offset, duration, retry_dir,
            )
            elapsed = time.perf_counter() - chunk_started
            retry_events.extend(events)
            chunk_text_parts = []
            chunk_token_count = 0
            confidences = []
            for leaf_offset, _, result, _ in leaves:
                chunk_text = result.get("text", "")
                timestamps = result.get("timestamp") or []
                chunk_text_parts.append(chunk_text)
                confidences.append(result.get("confidence"))
                chunk_token_count += len(timestamps)
                for token, start, end in timestamps:
                    items.append({
                        "t": token,
                        "s": round(leaf_offset + float(start), 3),
                        "e": round(leaf_offset + float(end), 3),
                    })
            chunk_text = _join_text(chunk_text_parts)
            text_parts.append(chunk_text)
            chunk_metrics.append({
                "index": index,
                "offset_sec": round(offset, 3),
                "duration_sec": round(duration, 3),
                "inference_sec": round(elapsed, 3),
                "rtf": round(elapsed / duration, 4),
                "confidence": confidences[0] if len(confidences) == 1 else confidences,
                "text_chars": len(chunk_text),
                "tokens": chunk_token_count,
                "retry_count": len(events),
            })
            if events:
                print(
                    f"[warn] chunk {index}/{len(chunks)} retried: "
                    + ", ".join(event["reason"] for event in events),
                    flush=True,
                )
            print(
                f"[info] chunk {index}/{len(chunks)}: {duration:.1f}s -> "
                f"{len(chunk_text)} chars in {elapsed:.2f}s",
                flush=True,
            )
        inference_sec = time.perf_counter() - inference_started

    text = _join_text(text_parts)
    data = {
        "text": text,
        "items": items,
        "model": "FireRedASR2-AED",
        "source_revision": _source_revision(args.source_dir),
        "chunk_sec": args.chunk_sec,
        "beam_size": args.beam_size,
    }
    cues, layout_stats = layout.build_cues(data, {})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_srt(cues, args.out, line_width=18)
    txt_path = args.out.with_suffix(".txt")
    items_path = args.out.with_suffix(".items.json")
    metrics_path = args.out.with_suffix(".metrics.json")
    subtitle.write_txt(text, txt_path)
    _write_json(items_path, data)

    metrics = {
        "model": "FireRedASR2-AED",
        "device": device,
        "duration_sec": round(total_duration, 3),
        "chunks": len(chunks),
        "text_chars": len(text),
        "tokens": len(items),
        "cues": len(cues),
        "preprocess_sec": round(preprocess_sec, 3),
        "model_load_sec": round(load_sec, 3),
        "inference_sec": round(inference_sec, 3),
        "inference_rtf": round(inference_sec / total_duration, 4),
        "total_sec": round(time.perf_counter() - started, 3),
        "model_gpu_gb": round(model_gpu_gb, 3),
        "peak_inference_gpu_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 3) if use_gpu else 0.0
        ),
        "retry_events": retry_events,
        "layout": layout_stats,
        "chunk_metrics": chunk_metrics,
    }
    _write_json(metrics_path, metrics)
    print(f"[info] wrote {len(cues)} cues -> {args.out}", flush=True)
    print(f"[info] inference RTF={metrics['inference_rtf']}, "
          f"peak GPU={metrics['peak_inference_gpu_gb']:.3f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
