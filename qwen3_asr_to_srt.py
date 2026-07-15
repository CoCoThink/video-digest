#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-ASR (GPU) -> SRT subtitle extractor.

Loads Qwen3-ASR + Qwen3-ForcedAligner on the GPU (transformers backend),
transcribes an audio/video-derived wav with word/char timestamps, and writes
an SRT file. Tuned for small VRAM (RTX 3050, 4 GB): defaults to the 0.6B models.
"""
import argparse
import os
import sys

# Reduce CUDA fragmentation / OOM on small GPUs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


def fmt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# Sentence-ending / clause punctuation used to break cues.
BREAK_STRONG = set("。！？!?…")
BREAK_SOFT = set("，,、；;：:")


def items_to_cues(items, max_chars=22, max_dur=6.0, max_gap=0.8):
    """Group per-token aligned items into readable subtitle cues."""
    cues = []
    cur_text = ""
    cur_start = None
    cur_end = None
    cur_chars = 0

    def flush():
        nonlocal cur_text, cur_start, cur_end, cur_chars
        t = cur_text.strip()
        if t and cur_start is not None:
            cues.append((cur_start, cur_end, t))
        cur_text = ""
        cur_start = None
        cur_end = None
        cur_chars = 0

    prev_end = None
    for it in items:
        tok = it.text
        st = float(it.start_time)
        et = float(it.end_time)
        if cur_start is None:
            cur_start = st
        # Break on a large silence gap before appending this token.
        if prev_end is not None and (st - prev_end) > max_gap and cur_chars > 0:
            flush()
            cur_start = st
        cur_text += tok
        cur_end = et
        # Count visible (non-space) chars.
        cur_chars += len(tok.strip())
        prev_end = et

        last = tok.strip()[-1:] if tok.strip() else ""
        strong = last in BREAK_STRONG
        soft = last in BREAK_SOFT
        long_enough = (cur_end - cur_start) >= max_dur
        too_many = cur_chars >= max_chars
        if strong or (soft and cur_chars >= max_chars // 2) or long_enough or too_many:
            flush()
    flush()
    return cues


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (st, et, text) in enumerate(cues, 1):
            if et <= st:
                et = st + 0.4
            f.write(f"{i}\n{fmt_ts(st)} --> {fmt_ts(et)}\n{text}\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--asr", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    ap.add_argument("--language", default=None, help="e.g. Chinese; default auto-detect")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--chunk-sec", type=int, default=45,
                    help="Max audio chunk length. On a 4GB GPU, int8-1.7B peaks flat at "
                         "3.02GB up to ~45s (activations tiny); 45s is ~3%% faster than 30s "
                         "with identical memory/text. 90s starts climbing (3.26GB); the "
                         "library default is 180 which OOMs. bf16 full weights need ~30.")
    args = ap.parse_args()

    # Shrink the chunking window so the audio encoder's conv activations fit small VRAM.
    # The library reads these module globals at transcribe() time.
    import qwen_asr.inference.qwen3_asr as _qa
    _qa.MAX_FORCE_ALIGN_INPUT_SECONDS = args.chunk_sec
    _qa.MAX_ASR_INPUT_SECONDS = args.chunk_sec

    from qwen_asr import Qwen3ASRModel

    print(f"[info] torch {torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[info] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"[info] loading ASR={args.asr} + aligner={args.aligner} ...", flush=True)
    model = Qwen3ASRModel.from_pretrained(
        args.asr,
        forced_aligner=args.aligner,
        max_inference_batch_size=args.batch,
        max_new_tokens=512,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    print("[info] models loaded, transcribing (this chunks long audio automatically) ...", flush=True)

    results = model.transcribe(
        audio=args.audio,
        language=args.language,
        return_time_stamps=True,
    )
    r = results[0]
    print(f"[info] detected language: {r.language}", flush=True)
    print(f"[info] full text length: {len(r.text)} chars", flush=True)

    ts = r.time_stamps
    items = getattr(ts, "items", None) if ts is not None else None
    if not items:
        print("[warn] no timestamps returned; writing single-cue SRT from full text.", flush=True)
        write_srt([(0.0, 5.0, r.text.strip())], args.out)
    else:
        print(f"[info] {len(items)} aligned tokens; first: "
              f"{items[0].text!r} [{items[0].start_time:.2f}-{items[0].end_time:.2f}]", flush=True)
        cues = items_to_cues(items)
        write_srt(cues, args.out)
        print(f"[info] wrote {len(cues)} cues -> {args.out}", flush=True)

    # Also dump the raw full transcript next to the SRT.
    txt_path = os.path.splitext(args.out)[0] + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(r.text.strip() + "\n")
    print(f"[info] wrote transcript -> {txt_path}", flush=True)

    if torch.cuda.is_available():
        print(f"[info] peak GPU mem: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
