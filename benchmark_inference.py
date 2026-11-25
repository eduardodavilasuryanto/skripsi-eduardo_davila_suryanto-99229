"""Benchmark inference D5 dengan batch yang sama, tanpa mengubah hasil primer.

Masukan: konfigurasi benchmark dan pelatihan, checkpoint lokal, data test.
Keluaran: JSON per model dengan pengulangan, RTF, metrik dan memori CUDA.
Jalankan setiap model dalam proses terpisah dari folder repositori:
    python -u benchmark_inference.py whisper_d5 2>&1 | Tee-Object -FilePath logs/benchmark_inference_whisper_d5.log
    python -u benchmark_inference.py xlsr_d5 2>&1 | Tee-Object -FilePath logs/benchmark_inference_xlsr_d5.log
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from analysis_common import RESULTS_DIR, SPLIT_DIR, sha256_of, write_json
from asr_common import corpus_metrics, normalize_text, read_split
from log_environment import set_seed
from run_inference import apply_numerics, load_config, load_model, numerics_state
from run_inference import prevent_background_downloads, transcribe

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configs" / "inference_benchmark.json"
TEST = SPLIT_DIR / "test.tsv"
MIB = 1024 ** 2


def gpu_snapshot() -> dict[str, Any]:
    """Potret satu GPU sebelum/sesudah uji; bukan puncak seluruh perangkat."""
    fields = ("name", "driver_version", "memory.total", "memory.used",
              "utilization.gpu", "temperature.gpu", "pstate")
    out = subprocess.check_output([
        "nvidia-smi", "-i", "0", "--query-gpu=" + ",".join(fields),
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15).strip().split(",")
    return dict(zip(fields, (v.strip() for v in out), strict=True))


def main() -> None:
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=protocol["models"])
    args = parser.parse_args()
    trial_path = ROOT / "configs" / (args.run + ".yaml")
    config, source_sha = load_config(trial_path)
    previous = json.loads((RESULTS_DIR / f"{args.run}_test_metrics.json").read_text(encoding="utf-8"))
    if source_sha != previous["config_sha256"]:
        raise ValueError("Training configuration differs from primary inference report")
    config["per_device_eval_batch_size"] = protocol["batch_size"]
    raw = sorted(read_split(TEST), key=lambda x: x.utt_id)
    if len(raw) != 3100 or len({x.utt_id for x in raw}) != len(raw):
        raise ValueError("Unexpected test split")
    chosen = random.Random(protocol["seed"]).sample(
        raw, protocol["test_sample_size"] + protocol["warmup_size"],
    )
    sample = chosen[:protocol["test_sample_size"]]
    warmup = chosen[protocol["test_sample_size"]:]
    duration = math.fsum(x.duration for x in sample)
    start = datetime.now(timezone.utc).isoformat()
    print(f"Started {start}; {args.run}; protocol={CONFIG}; SHA-256={sha256_of(CONFIG)}")
    print(f"Training configuration SHA-256={source_sha}; seed={protocol['seed']}; protocol={protocol}")
    print(f"Measured audio: {len(sample)} utterances; {duration:.3f} s")
    start_gpu = gpu_snapshot()
    print(f"GPU before CUDA: {start_gpu}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    prevent_background_downloads()
    os.environ["HF_HUB_OFFLINE"] = "1"
    apply_numerics(config)
    set_seed(protocol["seed"])
    torch.cuda.synchronize()
    load_start = time.perf_counter()
    model, processor = load_model(config)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_start
    baseline = torch.cuda.memory_allocated() / MIB
    transcribe(config, model, processor, warmup)
    torch.cuda.synchronize()
    reps = []
    references = [x.text for x in sample]
    for i in range(protocol["repetitions"]):
        # Pertahankan model dan cache pemanasan; reset hanya pencatatan puncak.
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        hyp = transcribe(config, model, processor, sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        rep = {
            "iteration": i + 1, "elapsed_seconds": elapsed, "rtf": elapsed / duration,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / MIB,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
            "accuracy": corpus_metrics(references, [normalize_text(hyp[x.utt_id]) for x in sample]),
        }
        reps.append(rep)
        print(f"Rep {i + 1}: {rep}")
    finish_gpu = gpu_snapshot()
    summary = {
        "median_seconds": statistics.median(r["elapsed_seconds"] for r in reps),
        "median_rtf": statistics.median(r["rtf"] for r in reps),
        "max_peak_allocated_mib": max(r["peak_allocated_mib"] for r in reps),
        "max_peak_reserved_mib": max(r["peak_reserved_mib"] for r in reps),
        "median_wer": statistics.median(r["accuracy"]["wer"] for r in reps),
        "median_cer": statistics.median(r["accuracy"]["cer"] for r in reps),
    }
    write_json(RESULTS_DIR / f"benchmark_inference_{args.run}.json", {
        "run": args.run, "started_at": start,
        "finished_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "protocol_sha256": sha256_of(CONFIG), "training_config_sha256": source_sha,
        "test_sha256": sha256_of(TEST), "sample_ids": [x.utt_id for x in sample],
        "warmup_ids": [x.utt_id for x in warmup], "audio_seconds": duration,
        "model_load_seconds": load_seconds, "allocated_after_load_mib": baseline,
        "numerics": numerics_state(), "precision": config["precision"],
        "gpu_before": start_gpu, "gpu_after": finish_gpu,
        "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
        "summary": summary, "repetitions": reps,
    })
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
