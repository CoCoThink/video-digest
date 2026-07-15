#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Try several bitsandbytes loading strategies for Qwen3-ASR-1.7B; report which works."""
import os, sys, time
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
import qwen_asr.inference.qwen3_asr as _qa
_qa.MAX_FORCE_ALIGN_INPUT_SECONDS = 20
_qa.MAX_ASR_INPUT_SECONDS = 20
from qwen_asr import Qwen3ASRModel
from transformers import BitsAndBytesConfig

MODE = sys.argv[1] if len(sys.argv) > 1 else "int8_devmap0"

common = dict(max_inference_batch_size=1, max_new_tokens=256)

if MODE == "int8_devmap0":
    qc = BitsAndBytesConfig(load_in_8bit=True)
    kw = dict(quantization_config=qc, device_map={"": 0})
elif MODE == "int8_auto":
    qc = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["audio_tower"])
    kw = dict(quantization_config=qc, device_map="auto")
elif MODE == "nf4":
    qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            llm_int8_skip_modules=["audio_tower"])
    kw = dict(quantization_config=qc, device_map={"": 0})
else:
    raise SystemExit("unknown mode")

print(f"[{MODE}] loading ...", flush=True)
t0 = time.perf_counter()
try:
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs={"device_map": "cpu", "dtype": torch.float32},
        **common, **kw,
    )
except Exception as e:
    print(f"[{MODE}] LOAD FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
    raise SystemExit(1)
print(f"[{MODE}] loaded in {time.perf_counter()-t0:.1f}s, peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

res = model.transcribe(audio="clip40.wav", language="Chinese", return_time_stamps=False)
print(f"[{MODE}] OK text[:80]={res[0].text[:80]}", flush=True)
print(f"[{MODE}] peak_after_infer={torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
