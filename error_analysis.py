"""Bahan analisis kesalahan kualitatif pada konfigurasi data terkecil dan terbesar.

Tujuan
------
Menyiapkan bahan keluaran 6 Subbab 3.6.4 skripsi, yaitu analisis kesalahan
kualitatif pada D1 dan D5 kedua model. Skrip ini hanya menghitung; tafsir
kualitatifnya dikerjakan dengan membaca keluaran ini, sesuai konvensi bahwa
angka yang masuk naskah berasal dari skrip.

Isi analisis per konfigurasi, seluruhnya pada teks ternormalisasi:
- komposisi galat kata: substitusi, penghapusan, penyisipan, dan benar;
- pasangan substitusi, kata terhapus, dan kata tersisip yang paling sering;
- sebaran galat per utterance: porsi utterance tanpa galat, median laju galat
  per utterance, porsi dengan galat sebanyak atau melebihi panjang referensi,
  porsi hipotesis kosong, dan porsi hipotesis lebih dari dua kali panjang
  referensi sebagai penanda pengulangan;
- utterance dengan galat terbanyak.

Sampel acak utterance dengan seed 42 diambil satu kali dan dipakai bagi
keempat konfigurasi, sehingga keluaran kedua model pada kedua tingkat dapat
dibandingkan pada utterance yang sama.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi (dimuat dan
  diperiksa seluruhnya; yang dianalisis D1 dan D5).
- `outputs/results/<run>_test_metrics.json`, hanya untuk pemeriksaan kesamaan.

Keluaran
--------
- `outputs/results/error_analysis_report.json` : ringkasan per konfigurasi.
- `outputs/results/error_analysis_samples.tsv` : sampel acak berdampingan.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u error_analysis.py 2>&1 |
        Tee-Object -FilePath logs/error_analysis.log
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import jiwer
import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    MODELS,
    RESULTS_DIR,
    SUBSETS_REPORT,
    find_config,
    load_configs,
    sha256_of,
    write_json,
)
from asr_common import WER_TRANSFORM
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "error_analysis_report.json"
SAMPLES_NAME = "error_analysis_samples.tsv"

# Konfigurasi data terkecil dan terbesar (Subbab 3.6.4 keluaran 6).
ANALYSIS_LEVELS = ("d1", "d5")

TOP_N = 20
N_WORST = 10
N_RANDOM_SAMPLES = 25

# Hipotesis lebih dari sekian kali panjang referensi ditandai sebagai pengulangan.
LONG_HYPOTHESIS_RATIO = 2.0


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def error_profile(references: Sequence[str], hypotheses: Sequence[str],
                  errors: np.ndarray, lengths: np.ndarray, ids: Sequence[str]) -> dict[str, Any]:
    """Komposisi galat, pasangan tersering, dan sebaran galat per utterance."""
    output = jiwer.process_words(list(references), list(hypotheses),
                                 reference_transform=WER_TRANSFORM, hypothesis_transform=WER_TRANSFORM)
    substitutions: Counter[str] = Counter()
    deletions: Counter[str] = Counter()
    insertions: Counter[str] = Counter()
    for reference, hypothesis, chunks in zip(output.references, output.hypotheses, output.alignments, strict=True):
        for chunk in chunks:
            ref_words = reference[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_words = hypothesis[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if chunk.type == "substitute":
                substitutions.update(f"{r} -> {h}" for r, h in zip(ref_words, hyp_words, strict=True))
            elif chunk.type == "delete":
                deletions.update(ref_words)
            elif chunk.type == "insert":
                insertions.update(hyp_words)

    total_errors = output.substitutions + output.deletions + output.insertions
    hyp_lengths = np.array([len(tokens) for tokens in output.hypotheses])
    utterance_rates = errors / lengths
    worst = sorted(range(len(ids)), key=lambda index: (-int(errors[index]), ids[index]))[:N_WORST]
    return {
        "counts": {
            "reference_words": int(lengths.sum()),
            "hits": output.hits,
            "substitutions": output.substitutions,
            "deletions": output.deletions,
            "insertions": output.insertions,
        },
        "share_of_errors": {
            "substitutions": output.substitutions / total_errors,
            "deletions": output.deletions / total_errors,
            "insertions": output.insertions / total_errors,
        },
        "wer": output.wer,
        "utterances": {
            "share_without_error": float(np.mean(errors == 0)),
            "median_utterance_error_rate": float(np.median(utterance_rates)),
            "share_errors_at_least_reference_length": float(np.mean(errors >= lengths)),
            "share_empty_hypothesis": float(np.mean(hyp_lengths == 0)),
            "share_hypothesis_longer_than_ratio": float(np.mean(hyp_lengths > LONG_HYPOTHESIS_RATIO * lengths)),
            "long_hypothesis_ratio": LONG_HYPOTHESIS_RATIO,
        },
        "top_substitutions": substitutions.most_common(TOP_N),
        "top_deletions": deletions.most_common(TOP_N),
        "top_insertions": insertions.most_common(TOP_N),
        "worst_utterances": [
            {"id": ids[i], "errors": int(errors[i]), "reference_words": int(lengths[i]),
             "reference": references[i], "hypothesis": hypotheses[i]}
            for i in worst
        ],
    }


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Bahan analisis kesalahan kualitatif D1 dan D5.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Folder keluaran; bawaan outputs/results.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    set_seed(ANALYSIS_SEED)

    print("=" * 78, flush=True)
    print("Bahan analisis kesalahan kualitatif D1 dan D5", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED} (sampel acak {N_RANDOM_SAMPLES} utterance)", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    selected = [configs[find_config(configs, model, level)] for level in ANALYSIS_LEVELS for model in MODELS]
    ids = selected[0].ids
    references = selected[0].references

    profiles: dict[str, Any] = {}
    for config in selected:
        profile = error_profile(references, config.hypotheses, config.words.errors, config.words.lengths, ids)
        profiles[config.run_name] = {"model": config.model, "level": config.level, "hours": config.hours, **profile}
        share = profile["share_of_errors"]
        utt = profile["utterances"]
        print(f"{config.run_name:18s} WER {profile['wer']:.4f}  S {share['substitutions']:.1%} "
              f"D {share['deletions']:.1%} I {share['insertions']:.1%}  tanpa galat {utt['share_without_error']:.1%}  "
              f"kosong {utt['share_empty_hypothesis']:.1%}  panjang>2x {utt['share_hypothesis_longer_than_ratio']:.1%}",
              flush=True)

    generator = np.random.default_rng(ANALYSIS_SEED)
    sample_indices = sorted(generator.choice(len(ids), size=N_RANDOM_SAMPLES, replace=False).tolist())
    sample_path = args.results_dir / SAMPLES_NAME
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "reference"]
                        + [c.run_name for c in selected] + [f"{c.run_name}_errors" for c in selected])
        for index in sample_indices:
            writer.writerow([ids[index], references[index]]
                            + [c.hypotheses[index] for c in selected]
                            + [int(c.words.errors[index]) for c in selected])

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "levels": ANALYSIS_LEVELS,
        "text": "referensi dan hipotesis ternormalisasi (Subbab 3.3.4)",
        "top_n": TOP_N,
        "n_random_samples": N_RANDOM_SAMPLES,
        "random_sample_ids": [ids[index] for index in sample_indices],
        "configs": profiles,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Sampel ditulis ke  : {sample_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
