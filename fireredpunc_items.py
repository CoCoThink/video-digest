#!/usr/bin/env python3
"""Apply FireRedPunc to an existing FireRed items.json and rebuild subtitles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output SRT path")
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--chunk-sec", type=float, default=30.0)
    return parser.parse_args(argv)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from server.pipeline.backends.firered import _add_source_dir, _join_text
    _add_source_dir(args.source_dir)
    from server.pipeline import layout, subtitle
    from server.pipeline.punctuation import (
        FireRedPunctuation,
        PunctuationProjectionError,
        partition_text_by_time,
    )

    data = json.loads(args.items.read_text(encoding="utf-8"))
    parts = partition_text_by_time(data["text"], data["items"], args.chunk_sec)
    load_started = time.perf_counter()
    punctuation = FireRedPunctuation(args.model_dir, device="cpu")
    load_sec = time.perf_counter() - load_started

    output_parts = []
    warnings = []
    infer_started = time.perf_counter()
    for offset, raw in parts:
        try:
            output_parts.append(punctuation.process(raw))
        except PunctuationProjectionError as exc:
            output_parts.append(raw)
            warnings.append({
                "code": "PUNC_TEXT_MISMATCH",
                "offset_sec": round(offset, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
        except Exception as exc:
            output_parts.append(raw)
            warnings.append({
                "code": "PUNC_FAILED",
                "offset_sec": round(offset, 3),
                "error": f"{type(exc).__name__}: {exc}",
            })
    infer_sec = time.perf_counter() - infer_started

    result = dict(data)
    result.update({
        "text": _join_text(output_parts),
        "backend": "firered",
        "model_id": "FireRedASR2-AED+FireRedPunc",
        "punctuation": True,
        "warnings": list(data.get("warnings") or []) + warnings,
    })
    cues, layout_info = layout.build_cues(result, {})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    subtitle.write_srt(cues, args.out, line_width=18)
    subtitle.write_txt(result["text"], args.out.with_suffix(".txt"))
    _write_json(args.out.with_suffix(".items.json"), result)
    punctuation_counts = {
        mark: result["text"].count(mark) for mark in "，。？！,.?!：；"
    }
    duration = max((float(item["e"]) for item in data["items"]), default=0.0)
    metrics = {
        "model": result["model_id"],
        "duration_sec": round(duration, 3),
        "chunks": len(parts),
        "text_chars_before": len(data["text"]),
        "text_chars_after": len(result["text"]),
        "punctuation_counts": punctuation_counts,
        "punc_model_load_sec": round(load_sec, 3),
        "punc_inference_sec": round(infer_sec, 3),
        "punc_rtf": round(infer_sec / duration, 5) if duration else None,
        "projection_warnings": warnings,
        "cues": len(cues),
        "layout": layout_info,
    }
    _write_json(args.out.with_suffix(".metrics.json"), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
