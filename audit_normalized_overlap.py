"""Audit tumpang tindih transkrip antarbagian setelah normalisasi.

Tujuan
------
Pembagian *text-disjoint* (`build_text_disjoint_split.py`) mengelompokkan
utterance menurut transkrip mentah, sebelum normalisasi Subbab 3.3.4. Dua
transkrip mentah yang berbeda dapat menjadi sama setelah huruf kecil,
penghapusan tanda baca, atau pelepasan diakritik, sehingga sebuah kalimat
yang dinilai pada data test mungkin tetap muncul pada data training dalam
bentuk teks yang dinilai. Skrip ini mengukur tumpang tindih itu.

Prosedur
--------
Tiga kunci dibandingkan:
1. `mentah` : transkrip seperti pada berkas pembagian, kunci pembagian.
2. `ringan` : NFC, *casefold*, dan perapatan spasi, yaitu normalisasi
   pemeriksaan ketegaran pada `audit_transcript_leakage.py`.
3. `akhir`  : `asr_common.normalize_text`, yaitu teks yang dinilai WER.

Untuk setiap pasangan bagian, dihitung jumlah dan porsi utterance pada bagian
pertama yang kuncinya muncul pada bagian kedua, serta jumlah kunci bersama.
Pasangan: test terhadap *training pool*, validation, dan D1 sampai D5;
validation terhadap *training pool*. Untuk kunci akhir, setiap tabrakan
dicatat beserta bentuk mentahnya pada kedua bagian. Bila ada utterance test
yang bertabrakan dengan data training suatu tingkat, WER dan CER setiap
konfigurasi dihitung ulang tanpa utterance itu sebagai sensitivitas.

Masukan
-------
- `outputs/splits/test.tsv`, `validation.tsv`, `train_pool.tsv`.
- `outputs/subsets/d1.tsv` sampai `d5.tsv`.
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi, untuk sensitivitas.

Keluaran
--------
- `outputs/results/audit_normalized_overlap_report.json`
- `outputs/results/audit_normalized_overlap_collisions.csv` : tabrakan kunci
  akhir antara data test atau validation dan *training pool*.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u audit_normalized_overlap.py 2>&1 |
        Tee-Object -FilePath logs/audit_normalized_overlap.log
"""

from __future__ import annotations

import argparse
import collections
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from analysis_common import (
    ANALYSIS_SEED,
    LEVELS,
    RESULTS_DIR,
    SCRIPT_DIR,
    SPLIT_DIR,
    load_configs,
    sha256_of,
    utterance_counts,
    write_csv,
    write_json,
)
from asr_common import Utterance, normalize_text, read_split
from audit_transcript_leakage import normalise_text as light_normalise
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "audit_normalized_overlap_report.json"
COLLISION_TABLE_NAME = "audit_normalized_overlap_collisions.csv"
SUBSET_DIR = SCRIPT_DIR / "outputs" / "subsets"

KEYS: dict[str, Callable[[str], str]] = {
    "mentah": lambda text: text,
    "ringan": light_normalise,
    "akhir": normalize_text,
}


# --------------------------------------------------------------------------
# Perhitungan
# --------------------------------------------------------------------------


def overlap(first: Sequence[Utterance], second: Sequence[Utterance], key: Callable[[str], str]) -> dict[str, Any]:
    """Porsi utterance bagian pertama yang kuncinya muncul pada bagian kedua."""
    second_keys = {key(utt.transcript) for utt in second}
    hits = [utt for utt in first if key(utt.transcript) in second_keys]
    shared = {key(utt.transcript) for utt in hits}
    return {
        "n_utterances": len(first),
        "n_overlapping": len(hits),
        "pct_overlapping": 100 * len(hits) / len(first),
        "n_shared_keys": len(shared),
        "overlapping_ids": sorted(utt.utt_id for utt in hits),
    }


def collisions(first: Sequence[Utterance], second: Sequence[Utterance], name: str) -> list[list[Any]]:
    """Tabrakan kunci akhir beserta bentuk mentah pada kedua bagian."""
    second_raw: dict[str, set[str]] = collections.defaultdict(set)
    second_count: collections.Counter[str] = collections.Counter()
    for utt in second:
        second_raw[normalize_text(utt.transcript)].add(utt.transcript)
        second_count[normalize_text(utt.transcript)] += 1
    rows = []
    for utt in first:
        key = normalize_text(utt.transcript)
        if key in second_raw:
            rows.append([name, utt.utt_id, utt.transcript, " || ".join(sorted(second_raw[key])),
                         second_count[key], key])
    return rows


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Audit tumpang tindih transkrip setelah normalisasi.")
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
    print("Audit tumpang tindih transkrip setelah normalisasi (Revisi 01 poin 13)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print("=" * 78, flush=True)

    parts: dict[str, list[Utterance]] = {
        name: read_split(SPLIT_DIR / f"{name}.tsv") for name in ("test", "validation", "train_pool")
    }
    for level in LEVELS:
        parts[level] = read_split(SUBSET_DIR / f"{level}.tsv")
    pairs = [("test", "train_pool"), ("test", "validation"), ("validation", "train_pool"),
             *[("test", level) for level in LEVELS]]

    results: dict[str, dict[str, Any]] = {}
    for key_name, key in KEYS.items():
        results[key_name] = {}
        for first, second in pairs:
            item = overlap(parts[first], parts[second], key)
            results[key_name][f"{first}_in_{second}"] = item
            print(f"{key_name:7s} {first:10s} dalam {second:10s}: {item['n_overlapping']:5d} utterance "
                  f"({item['pct_overlapping']:.2f}%), {item['n_shared_keys']} kunci bersama", flush=True)

    unique_keys = {
        name: {key_name: len({key(utt.transcript) for utt in part}) for key_name, key in KEYS.items()}
        for name, part in parts.items() if name in ("test", "validation", "train_pool")
    }

    rows = (collisions(parts["test"], parts["train_pool"], "test_dalam_train_pool")
            + collisions(parts["validation"], parts["train_pool"], "validation_dalam_train_pool")
            + collisions(parts["test"], parts["validation"], "test_dalam_validation"))

    sensitivity: dict[str, Any] = {}
    affected = {level: set(results["akhir"][f"test_in_{level}"]["overlapping_ids"]) for level in LEVELS}
    if any(affected.values()):
        configs = load_configs()
        for config in configs:
            keep = [i for i, utt in enumerate(config.ids) if utt not in affected[config.level]]
            refs = [config.references[i] for i in keep]
            hyps = [config.hypotheses[i] for i in keep]
            words = utterance_counts(refs, hyps, "word")
            chars = utterance_counts(refs, hyps, "char")
            sensitivity[config.run_name] = {
                "n_excluded": len(config.ids) - len(keep),
                "wer_all": config.words.rate, "wer_without": words.rate,
                "cer_all": config.chars.rate, "cer_without": chars.rate,
            }
            print(f"  {config.run_name:18s} tanpa {len(config.ids) - len(keep):3d}: "
                  f"WER {config.words.rate:.4f} -> {words.rate:.4f}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    inputs = {f"{name}.tsv": sha256_of(SPLIT_DIR / f"{name}.tsv") for name in ("test", "validation", "train_pool")}
    inputs.update({f"{level}.tsv": sha256_of(SUBSET_DIR / f"{level}.tsv") for level in LEVELS})
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "keys": {
            "mentah": "transkrip berkas pembagian, kunci pengelompokan build_text_disjoint_split.py",
            "ringan": "audit_transcript_leakage.normalise_text: NFC, casefold, perapatan spasi",
            "akhir": "asr_common.normalize_text: teks yang dinilai WER (Subbab 3.3.4)",
        },
        "unique_keys": unique_keys,
        "overlap": results,
        "n_collisions_final_key": len(rows),
        "sensitivity_without_test_overlap": sensitivity,
        "input_sha256": inputs,
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / COLLISION_TABLE_NAME,
              ["pasangan", "id", "transkrip_mentah", "bentuk_mentah_pada_bagian_lain",
               "jumlah_utterance_bagian_lain", "kunci_akhir"], rows)

    print("=" * 78, flush=True)
    print(f"Tabrakan kunci akhir: {len(rows)}", flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
