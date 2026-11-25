"""Inference satu model terlatih dan penyimpanan hipotesis per utterance.

Tujuan
------
Menghasilkan berkas hipotesis per utterance yang menjadi prasyarat seluruh
analisis Milestone 3 (`bab-3.md` Subbab 3.2.2 dan 3.6; Bagian I.8 korpus
konteks), beserta WER dan CER tingkat corpus. Model yang dimuat adalah model
dengan WER validation terendah yang disimpan `train_asr.py`.

Decoding mengikuti Subbab 3.6.1 dan Bab 6 poin 7 dokumen perencanaan: Whisper
memakai `generate` *greedy* dengan bahasa `jw` dan tugas *transcribe*; XLS-R
memakai *greedy CTC decoding* tanpa model bahasa eksternal. Referensi dan
hipotesis dinormalisasi dengan `asr_common.normalize_text` yang sama seperti
saat pelatihan. Presisi, TF32, dan determinisme mengikuti berkas konfigurasi
yang sama dengan pelatihannya.

Masukan
-------
- `--config` : berkas YAML yang dipakai melatih model.
- `--split`  : `test` (bawaan) atau `validation`.
- `outputs/models/<run_name>/` dan berkas pembagian pada `outputs/splits/`.

Keluaran
--------
- `outputs/hypotheses/<run_name>_<split>.tsv` : kolom `id`,
  `reference_normalized`, `hypothesis_normalized`, `hypothesis_raw`,
  berurutan menurut `id`. Kolom mentah disimpan agar uji ketegaran diakritik
  tidak memerlukan inference ulang.
- `outputs/results/<run_name>_<split>_metrics.json` : WER, CER, jumlah
  utterance, dan lingkungan tiga lapis.

Perintah
--------
    python -u run_inference.py --config configs/whisper_d1.yaml 2>&1 |
        Tee-Object -FilePath logs/whisper_d1_inference.log
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_common import SAMPLING_RATE, Utterance, corpus_metrics, load_audio, normalize_text, read_split
from log_environment import collect_environment, set_seed
from train_asr import (
    MODELS_DIR,
    SCRIPT_DIR,
    apply_numerics,
    load_config,
    numerics_state,
    prevent_background_downloads,
)

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SPLIT_DIR = SCRIPT_DIR / "outputs" / "splits"
HYPOTHESES_DIR = SCRIPT_DIR / "outputs" / "hypotheses"
RESULTS_DIR = SCRIPT_DIR / "outputs" / "results"
SPLITS = ("test", "validation")
HYPOTHESIS_HEADER = ["id", "reference_normalized", "hypothesis_normalized", "hypothesis_raw"]


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def transcribe(
    config: dict[str, Any],
    model: Any,
    processor: Any,
    utterances: Sequence[Utterance],
) -> dict[str, str]:
    """Mentranskripsi seluruh utterance; nilai kembalian dipetakan menurut `id`.

    Utterance diurutkan menurut durasi agar padding dalam satu *batch* minimal.
    Urutan ini tidak memengaruhi hasil karena setiap utterance didekode sendiri.
    """
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    ordered = sorted(utterances, key=lambda utt: (utt.duration, utt.utt_id))
    size = config["per_device_eval_batch_size"]
    hypotheses: dict[str, str] = {}
    for start in range(0, len(ordered), size):
        chunk = ordered[start:start + size]
        audio = [load_audio(utt.utt_id) for utt in chunk]
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
            if config["model_family"] == "whisper":
                features = processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE, return_tensors="pt")
                generated = model.generate(
                    features.input_features.to("cuda"),
                    language="javanese",
                    task="transcribe",
                    max_new_tokens=config["generation_max_new_tokens"],
                    num_beams=1,
                    do_sample=False,
                )
                texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            else:
                inputs = processor.feature_extractor(
                    audio, sampling_rate=SAMPLING_RATE, padding=True, return_tensors="pt"
                )
                logits = model(
                    input_values=inputs.input_values.to("cuda"),
                    attention_mask=inputs.attention_mask.to("cuda"),
                ).logits
                texts = processor.tokenizer.batch_decode(logits.argmax(dim=-1).cpu())
        hypotheses.update({utt.utt_id: text for utt, text in zip(chunk, texts, strict=True)})
        if (start // size) % 20 == 0:
            print(f"[inference] {min(start + size, len(ordered))}/{len(ordered)} utterance", flush=True)
    return hypotheses


def load_model(config: dict[str, Any]) -> tuple[Any, Any]:
    """Memuat model terbaik dan processor yang disimpan `train_asr.py`."""
    model_dir = MODELS_DIR / config["run_name"]
    if config["model_family"] == "whisper":
        processor = WhisperProcessor.from_pretrained(model_dir)
        model = WhisperForConditionalGeneration.from_pretrained(model_dir)
    else:
        processor = Wav2Vec2Processor.from_pretrained(model_dir)
        model = Wav2Vec2ForCTC.from_pretrained(model_dir)
    return model.to("cuda").eval(), processor


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Inference dan hipotesis per utterance.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, default="test")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    config, config_sha256 = load_config(args.config)
    prevent_background_downloads()
    apply_numerics(config)
    set_seed(config["seed"])

    print("=" * 78, flush=True)
    print(f"Inference {config['run_name']} pada {args.split}", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Konfigurasi        : {args.config} (SHA-256 {config_sha256[:16]})", flush=True)
    print(f"Model              : {MODELS_DIR / config['run_name']}", flush=True)
    print(f"Random seed        : {config['seed']}", flush=True)
    print(f"Presisi            : {config['precision']}, batch {config['per_device_eval_batch_size']}", flush=True)
    print("=" * 78, flush=True)

    if not torch.cuda.is_available():
        print("GAGAL: CUDA tidak tersedia.", file=sys.stderr)
        return 1

    utterances = read_split(SPLIT_DIR / f"{args.split}.tsv")
    model, processor = load_model(config)
    raw = transcribe(config, model, processor, utterances)

    rows = []
    for utt in sorted(utterances, key=lambda item: item.utt_id):
        rows.append([utt.utt_id, utt.text, normalize_text(raw[utt.utt_id]), raw[utt.utt_id]])
    metrics = corpus_metrics([row[1] for row in rows], [row[2] for row in rows])

    HYPOTHESES_DIR.mkdir(parents=True, exist_ok=True)
    hypotheses_path = HYPOTHESES_DIR / f"{config['run_name']}_{args.split}.tsv"
    with hypotheses_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HYPOTHESIS_HEADER)
        writer.writerows(rows)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "run_name": config["run_name"],
        "split": args.split,
        "hypotheses_path": str(hypotheses_path),
        "metrics": metrics,
        "numerics_state": numerics_state(),
        "environment": collect_environment(config["seed"], started),
        "elapsed_seconds": elapsed,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"{config['run_name']}_{args.split}_metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print("=" * 78, flush=True)
    for key, value in metrics.items():
        print(f"{key:28s}: {value}", flush=True)
    print(f"Hipotesis ditulis  : {hypotheses_path}", flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
