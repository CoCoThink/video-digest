#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run one Qwen3-ASR model end-to-end and record metrics for comparison.

Keeps the ASR model on the GPU and the forced aligner on CPU so the 1.7B model
fits in 4 GB VRAM. Instruments per-phase wall time (ASR decode vs alignment),
peak GPU memory, and writes an SRT + transcript + a metrics JSON.
"""
import argparse
import json
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch

from qwen3_asr_to_srt import items_to_cues, write_srt  # reuse cue/SRT logic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--tag", required=True, help="label, e.g. 0.6b / 1.7b")
    ap.add_argument("--asr", required=True)
    ap.add_argument("--aligner", default="Qwen/Qwen3-ForcedAligner-0.6B")
    ap.add_argument("--aligner-device", default="cpu")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--context", default="", help="Domain vocabulary / hotwords to bias recognition.")
    ap.add_argument("--chunk-sec", type=int, default=45)
    ap.add_argument("--load-8bit", action="store_true",
                    help="Load ASR weights in int8 (bitsandbytes) to fit big models in small VRAM.")
    ap.add_argument("--gpu-mem-mib", type=int, default=0,
                    help="If >0, use accelerate device_map=auto with this GPU cap (MiB) and "
                         "offload overflow layers to CPU RAM. Full precision, fits big models.")
    args = ap.parse_args()

    import qwen_asr.inference.qwen3_asr as _qa
    _qa.MAX_FORCE_ALIGN_INPUT_SECONDS = args.chunk_sec
    _qa.MAX_ASR_INPUT_SECONDS = args.chunk_sec
    from qwen_asr import Qwen3ASRModel

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    aligner_kwargs = {"device_map": args.aligner_device}
    if args.aligner_device == "cpu":
        aligner_kwargs["dtype"] = torch.float32
    else:
        aligner_kwargs["dtype"] = torch.bfloat16
    asr_kwargs = dict(
        forced_aligner=args.aligner,
        forced_aligner_kwargs=aligner_kwargs,
        max_inference_batch_size=1,
        max_new_tokens=512,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    if args.load_8bit:
        from transformers import BitsAndBytesConfig
        asr_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        # device_map={"":0} (not "cuda:0") avoids the bnb SCB dispatch bug and keeps
        # the whole quantized model on GPU 0 (no CPU offload).
        asr_kwargs["device_map"] = {"": 0}
        asr_kwargs.pop("dtype", None)
    if args.gpu_mem_mib > 0:
        # Split the model across GPU + CPU RAM (accelerate), full precision.
        asr_kwargs["device_map"] = "auto"
        asr_kwargs["max_memory"] = {0: f"{args.gpu_mem_mib}MiB", "cpu": "24GiB"}
    model = Qwen3ASRModel.from_pretrained(args.asr, **asr_kwargs)
    t_load = time.perf_counter() - t0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_after_load = torch.cuda.max_memory_allocated() / 1e9

    # Instrument phase timings by wrapping the two internal hot methods.
    timings = {"asr": 0.0, "align": 0.0}

    orig_infer = model._infer_asr
    def timed_infer(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        s = time.perf_counter()
        out = orig_infer(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings["asr"] += time.perf_counter() - s
        return out
    model._infer_asr = timed_infer

    orig_align = model.forced_aligner.align
    def timed_align(*a, **k):
        s = time.perf_counter()
        out = orig_align(*a, **k)
        timings["align"] += time.perf_counter() - s
        return out
    model.forced_aligner.align = timed_align

    t1 = time.perf_counter()
    results = model.transcribe(audio=args.audio, context=args.context,
                               language=args.language, return_time_stamps=True)
    t_transcribe = time.perf_counter() - t1

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_gpu = 0.0

    r = results[0]
    items = getattr(r.time_stamps, "items", None) if r.time_stamps is not None else None
    srt_path = f"Transformer其实是个简单的架构.qwen3-asr.{args.tag}.srt"
    txt_path = f"Transformer其实是个简单的架构.qwen3-asr.{args.tag}.txt"
    n_cues = 0
    if items:
        cues = items_to_cues(items)
        n_cues = len(cues)
        write_srt(cues, srt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(r.text.strip() + "\n")
    # Dump raw per-token alignment items + full punctuated text for offline re-segmentation.
    if items:
        import json as _json
        dump = {"text": r.text, "language": r.language,
                "items": [{"t": it.text, "s": float(it.start_time), "e": float(it.end_time)}
                          for it in items]}
        with open(f"items.{args.tag}.json", "w", encoding="utf-8") as f:
            _json.dump(dump, f, ensure_ascii=False)

    metrics = {
        "tag": args.tag,
        "asr_model": args.asr,
        "aligner_device": args.aligner_device,
        "chunk_sec": args.chunk_sec,
        "language": r.language,
        "text_chars": len(r.text),
        "n_tokens": len(items) if items else 0,
        "n_cues": n_cues,
        "load_sec": round(t_load, 2),
        "asr_decode_sec": round(timings["asr"], 2),
        "align_sec": round(timings["align"], 2),
        "transcribe_total_sec": round(t_transcribe, 2),
        "peak_gpu_gb": round(peak_gpu, 3),
        "gpu_after_load_gb": round(mem_after_load, 3) if torch.cuda.is_available() else 0,
    }
    with open(f"metrics.{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("METRICS " + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
