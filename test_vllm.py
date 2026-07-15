#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit-test the vLLM backend with bitsandbytes int8 for Qwen3-ASR-1.7B on 4 GB."""
import os, sys, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"


def main():
    import torch
    import qwen_asr.inference.qwen3_asr as _qa
    _qa.MAX_FORCE_ALIGN_INPUT_SECONDS = 30
    _qa.MAX_ASR_INPUT_SECONDS = 30
    from qwen_asr import Qwen3ASRModel

    gmu = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
    print(f"[vllm] init gpu_mem_util={gmu} ...", flush=True)
    t0 = time.perf_counter()
    try:
        model = Qwen3ASRModel.LLM(
            model="Qwen/Qwen3-ASR-1.7B",
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs={"device_map": "cpu", "dtype": torch.float32},
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            gpu_memory_utilization=gmu,
            enforce_eager=True,
            max_model_len=2048,
            max_num_seqs=1,
            dtype="bfloat16",
        )
    except Exception as e:
        print(f"[vllm] INIT FAILED: {type(e).__name__}: {str(e)[:500]}", flush=True)
        raise SystemExit(1)
    print(f"[vllm] init OK in {time.perf_counter()-t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    res = model.transcribe(audio="clip40.wav", language="Chinese", return_time_stamps=False)
    print(f"[vllm] transcribe 40s clip in {time.perf_counter()-t1:.1f}s", flush=True)
    print(f"[vllm] text[:80]={res[0].text[:80]}", flush=True)


if __name__ == "__main__":
    main()
