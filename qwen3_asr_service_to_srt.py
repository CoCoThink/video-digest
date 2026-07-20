#!/usr/bin/env python3
"""Run the project's Qwen3-ASR pipeline on a local media file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
import wave


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--asr", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--chunk-sec", type=int, default=30)
    parser.add_argument("--no-int8", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"input does not exist: {args.input}")
    if args.chunk_sec <= 0:
        parser.error("--chunk-sec must be positive")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    from server.config import Settings
    from server.pipeline import asr, layout, preprocess, subtitle

    cfg = Settings().model_dump()
    cfg.update({
        "asr_model_path": args.asr,
        "aligner_model_path": args.aligner,
        "asr_language": args.language or None,
        "chunk_sec": args.chunk_sec,
        "load_8bit": not args.no_int8,
    })

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qwen3-asr-") as tmp:
        wav_path = Path(tmp) / "audio.wav"
        phase = time.perf_counter()
        preprocess.to_wav_16k(args.input, wav_path)
        preprocess_sec = time.perf_counter() - phase
        with wave.open(str(wav_path), "rb") as reader:
            audio_duration_sec = reader.getnframes() / reader.getframerate()

        phase = time.perf_counter()
        model = asr.load_model(cfg)
        load_sec = time.perf_counter() - phase

        last_progress = -5

        def report_progress(percent: int) -> None:
            nonlocal last_progress
            if percent >= last_progress + 5:
                last_progress = percent
                print(f"[info] transcribing {percent}%", flush=True)

        phase = time.perf_counter()
        data = asr.transcribe(
            model, str(wav_path), language=cfg["asr_language"],
            progress_cb=report_progress,
        )
        transcribe_sec = time.perf_counter() - phase

    peak_gpu_gb = data.pop("_peak_gpu_gb", 0.0)
    cues, layout_stats = layout.build_cues(data, cfg)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_srt(cues, args.out, cfg["layout_line_width"])
    subtitle.write_txt(data["text"], args.out.with_suffix(".txt"))
    _write_json(args.out.with_suffix(".items.json"), data)

    last_token_sec = float(data["items"][-1]["e"]) if data["items"] else 0.0
    metrics = {
        "model": "Qwen3-ASR-1.7B-int8" if cfg["load_8bit"] else "Qwen3-ASR-1.7B",
        "language": data["language"],
        "chunk_sec": cfg["chunk_sec"],
        "duration_sec": round(audio_duration_sec, 3),
        "duration_to_last_token_sec": round(last_token_sec, 3),
        "text_chars": len(data["text"]),
        "tokens": len(data["items"]),
        "cues": len(cues),
        "preprocess_sec": round(preprocess_sec, 3),
        "model_load_sec": round(load_sec, 3),
        "transcribe_sec": round(transcribe_sec, 3),
        "transcribe_rtf": round(transcribe_sec / audio_duration_sec, 4),
        "total_sec": round(time.perf_counter() - started, 3),
        "peak_gpu_gb": peak_gpu_gb,
        "layout": layout_stats,
    }
    _write_json(args.out.with_suffix(".metrics.json"), metrics)
    print(f"[info] wrote {len(cues)} cues -> {args.out}", flush=True)
    print("METRICS " + json.dumps(metrics, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
