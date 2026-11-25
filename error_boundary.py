"""Galat batas kata kedua model pada kelima tingkat, terutama D3 dan D4 (Revisi 01 poin 33).

Tujuan
------
`bab-4.md` Subbab 4.6.1 menduga bahwa sisa keunggulan WER Whisper pada D3 dan
D4, ketika CER kedua model tidak berbeda, terletak pada penentuan batas kata,
padahal analisis kesalahan hanya mencakup D1 dan D5. Skrip ini mengukur galat
batas kata pada kelima tingkat.

Prosedur
--------
1. **Spasi pada penjajaran karakter.** Dari penjajaran `jiwer` dengan
   transformasi CER penelitian, yang mempertahankan spasi, dihitung spasi
   referensi yang hilang (penggabungan kata) dan spasi hipotesis yang tidak
   ada pada referensi (pemisahan kata), per 1.000 batas kata referensi.
2. **Utterance yang hanya salah batas kata.** Utterance yang referensi dan
   hipotesisnya sama setelah seluruh spasi dihapus, tetapi berbeda sebelum
   spasi dihapus. Seluruh galat kata pada utterance itu berasal dari batas
   kata.
3. **WER tanpa galat batas kata murni.** Galat kata utterance pada langkah 2
   dinolkan, lalu WER tingkat corpus dihitung ulang. Ini batas bawah pengaruh
   galat batas kata, karena galat batas kata pada utterance yang juga
   mengandung galat lain tidak dihapus.
4. Selisih Whisper dikurangi XLS-R pada WER biasa, WER pada langkah 3, dan laju
   pada langkah 1 diberi selang persentil 95% *paired bootstrap* tingkat
   utterance dengan *resample* yang sama dengan hasil utama (seed 42, 10.000
   iterasi).

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.

Keluaran
--------
- `outputs/results/error_boundary_report.json`
- `outputs/results/error_boundary.csv`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u error_boundary.py 2>&1 | Tee-Object -FilePath logs/error_boundary.log
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import jiwer
import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    bootstrap_ratios,
    find_config,
    load_configs,
    percentile_interval,
    write_csv,
    write_json,
)
from asr_common import CER_TRANSFORM
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "error_boundary_report.json"
TABLE_NAME = "error_boundary.csv"


# --------------------------------------------------------------------------
# Perhitungan
# --------------------------------------------------------------------------


def space_errors(references: Sequence[str], hypotheses: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spasi referensi yang hilang, spasi hipotesis berlebih, dan jumlah batas kata referensi per utterance."""
    output = jiwer.process_characters(list(references), list(hypotheses),
                                      reference_transform=CER_TRANSFORM, hypothesis_transform=CER_TRANSFORM)
    merges, splits, boundaries = [], [], []
    for ref, hyp, chunks in zip(output.references, output.hypotheses, output.alignments, strict=True):
        merged = split = 0
        for chunk in chunks:
            ref_part = ref[chunk.ref_start_idx:chunk.ref_end_idx]
            hyp_part = hyp[chunk.hyp_start_idx:chunk.hyp_end_idx]
            if chunk.type == "delete":
                merged += ref_part.count(" ")
            elif chunk.type == "insert":
                split += hyp_part.count(" ")
            elif chunk.type == "substitute":
                merged += sum(1 for r, h in zip(ref_part, hyp_part) if r == " " and h != " ")
                split += sum(1 for r, h in zip(ref_part, hyp_part) if h == " " and r != " ")
        merges.append(merged)
        splits.append(split)
        boundaries.append(ref.count(" "))
    return np.array(merges), np.array(splits), np.array(boundaries)


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Galat batas kata.")
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
    print("Galat batas kata (Revisi 01 poin 33)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    per_config: dict[str, Any] = {}
    columns: dict[str, dict[str, np.ndarray]] = {}
    for config in configs:
        merges, splits, boundaries = space_errors(config.references, config.hypotheses)
        boundary_only = np.array([r != h and r.replace(" ", "") == h.replace(" ", "")
                                  for r, h in zip(config.references, config.hypotheses, strict=True)])
        neutral_errors = np.where(boundary_only, 0, config.words.errors)
        total_errors = int(config.words.errors.sum())
        per_config[config.run_name] = {
            "merges_per_1000_boundaries": float(1000 * merges.sum() / boundaries.sum()),
            "splits_per_1000_boundaries": float(1000 * splits.sum() / boundaries.sum()),
            "n_boundary_only_utterances": int(boundary_only.sum()),
            "word_errors_in_boundary_only_utterances": int(config.words.errors[boundary_only].sum()),
            "share_of_word_errors_pct": float(100 * config.words.errors[boundary_only].sum() / total_errors),
            "wer": config.words.rate,
            "wer_without_boundary_only": float(neutral_errors.sum() / config.words.lengths.sum()),
        }
        columns[config.run_name] = {"neutral": neutral_errors, "merges": merges, "splits": splits,
                                    "boundaries": boundaries}
        item = per_config[config.run_name]
        print(f"{config.run_name:18s} gabung {item['merges_per_1000_boundaries']:6.1f}  pisah "
              f"{item['splits_per_1000_boundaries']:5.1f} per 1000 batas; hanya-batas {item['n_boundary_only_utterances']:3d} "
              f"utt ({item['share_of_word_errors_pct']:.1f}% galat kata); WER {item['wer']:.4f} -> "
              f"{item['wer_without_boundary_only']:.4f}", flush=True)

    numerators, denominators, labels = [], [], []
    for config in configs:
        col = columns[config.run_name]
        numerators += [config.words.errors, col["neutral"], col["merges"] + col["splits"]]
        denominators += [config.words.lengths, config.words.lengths, col["boundaries"]]
        labels += [(config.run_name, "wer"), (config.run_name, "wer_neutral"), (config.run_name, "boundary_rate")]
    boot = bootstrap_ratios(np.column_stack(numerators), np.column_stack(denominators), N_BOOTSTRAP, ANALYSIS_SEED)

    differences: dict[str, Any] = {}
    rows: list[list[Any]] = []
    print("-" * 78, flush=True)
    for level in LEVELS:
        w = configs[find_config(configs, "whisper", level)].run_name
        x = configs[find_config(configs, "xlsr", level)].run_name
        entry = {}
        for measure in ("wer", "wer_neutral", "boundary_rate"):
            i, j = labels.index((w, measure)), labels.index((x, measure))
            point = float(np.sum(numerators[i]) / np.sum(denominators[i]) - np.sum(numerators[j]) / np.sum(denominators[j]))
            entry[measure] = {"diff": point, "ci": list(percentile_interval(boot[:, i] - boot[:, j]))}
        differences[level] = entry
        rows.append([level, *[v for m in ("wer", "wer_neutral", "boundary_rate")
                               for v in (entry[m]["diff"], *entry[m]["ci"])]])
        print(f"{level}: selisih WER {entry['wer']['diff']:+.4f}  tanpa hanya-batas {entry['wer_neutral']['diff']:+.4f} "
              f"[{entry['wer_neutral']['ci'][0]:+.4f}, {entry['wer_neutral']['ci'][1]:+.4f}]  laju galat batas "
              f"{entry['boundary_rate']['diff']:+.4f} [{entry['boundary_rate']['ci'][0]:+.4f}, "
              f"{entry['boundary_rate']['ci'][1]:+.4f}]", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "definitions": {
            "merges": "spasi referensi yang terhapus atau tersubstitusi bukan spasi pada penjajaran karakter",
            "splits": "spasi hipotesis yang tersisip atau menggantikan bukan spasi pada penjajaran karakter",
            "boundary_only": "referensi dan hipotesis sama setelah spasi dihapus, berbeda sebelumnya",
            "wer_without_boundary_only": "galat kata utterance hanya-batas dinolkan; batas bawah pengaruh",
            "boundary_rate": "(penggabungan + pemisahan) per batas kata referensi",
        },
        "per_config": per_config,
        "differences_whisper_minus_xlsr": differences,
        "input_sha256": {f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / TABLE_NAME,
              ["level", "wer_diff", "wer_diff_ci_low", "wer_diff_ci_high", "wer_neutral_diff", "wer_neutral_ci_low",
               "wer_neutral_ci_high", "boundary_rate_diff", "boundary_rate_ci_low", "boundary_rate_ci_high"], rows)
    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
