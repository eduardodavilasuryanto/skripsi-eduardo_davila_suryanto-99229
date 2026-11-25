"""WER dan CER berselang kepercayaan serta uji H1 per tingkat data.

Tujuan
------
Menghasilkan keluaran 1 `bab-3.md` Subbab 3.6.4, yaitu WER dan CER setiap
konfigurasi beserta selang kepercayaan 95%, dan pengujian H1 pada Tabel 3.15:
paired bootstrap tingkat utterance pada setiap tingkat data dengan koreksi
Holm-Bonferroni atas lima pengujian. Lebar selang juga dilaporkan sebagai
bahan pemeriksaan kecukupan kuota data test (Utang nomor 18).

Prosedur
--------
- Resample: 10.000 penarikan berpengembalian atas 3.100 utterance data test
  dengan seed 42, sama bagi seluruh konfigurasi (`analysis_common`).
- Selang: persentil 2,5 dan 97,5.
- H1: selisih WER Whisper dikurangi XLS-R dihitung pada resample yang sama.
  Nilai p bootstrap dua sisi dikoreksi Holm atas lima tingkat; perbedaan
  dinyatakan signifikan bila nilai p terkoreksi di bawah 0,05. Selang 95% dan
  selang pada taraf Holm masing-masing tingkat dilaporkan bersama.
- CER dan selisih CER dilaporkan secara deskriptif; putusan H1 memakai WER.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/results/<run>_test_metrics.json`, hanya untuk pemeriksaan kesamaan.
- `outputs/build_nested_subsets_report.json` : jam data training terukur.

Keluaran
--------
- `outputs/results/bootstrap_metrics_report.json` : laporan lengkap.
- `outputs/results/bootstrap_metrics_configs.csv` : tabel WER dan CER per konfigurasi.
- `outputs/results/bootstrap_metrics_h1.csv` : tabel uji H1 per tingkat.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u bootstrap_metrics.py 2>&1 |
        Tee-Object -FilePath logs/bootstrap_metrics.log
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    ALPHA,
    ANALYSIS_SEED,
    CONFIDENCE,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
    find_config,
    holm_adjust,
    holm_levels,
    load_configs,
    percentile_interval,
    sha256_of,
    two_sided_p,
    write_csv,
    write_json,
)
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "bootstrap_metrics_report.json"
CONFIGS_TABLE_NAME = "bootstrap_metrics_configs.csv"
H1_TABLE_NAME = "bootstrap_metrics_h1.csv"

# Arah selisih H1: model pertama dikurangi model kedua.
H1_MINUEND = "whisper"
H1_SUBTRAHEND = "xlsr"


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def config_summaries(configs: Sequence[Any], wer_boot: np.ndarray, cer_boot: np.ndarray) -> list[dict[str, Any]]:
    """WER dan CER setiap konfigurasi beserta selang dan setengah lebarnya."""
    summaries = []
    for index, config in enumerate(configs):
        wer_low, wer_high = percentile_interval(wer_boot[:, index])
        cer_low, cer_high = percentile_interval(cer_boot[:, index])
        summaries.append({
            "model": config.model,
            "level": config.level,
            "run_name": config.run_name,
            "hours": config.hours,
            "n_utterances": len(config.ids),
            "n_reference_words": int(config.words.lengths.sum()),
            "n_word_errors": int(config.words.errors.sum()),
            "wer": config.words.rate,
            "wer_ci": [wer_low, wer_high],
            "wer_ci_half_width": (wer_high - wer_low) / 2,
            "cer": config.chars.rate,
            "cer_ci": [cer_low, cer_high],
            "cer_ci_half_width": (cer_high - cer_low) / 2,
            "hypotheses_sha256": config.hypotheses_sha256,
        })
    return summaries


def h1_tests(configs: Sequence[Any], wer_boot: np.ndarray, cer_boot: np.ndarray) -> list[dict[str, Any]]:
    """Paired bootstrap selisih WER per tingkat dengan koreksi Holm."""
    rows = []
    for level in LEVELS:
        first = find_config(configs, H1_MINUEND, level)
        second = find_config(configs, H1_SUBTRAHEND, level)
        wer_diff = wer_boot[:, first] - wer_boot[:, second]
        cer_diff = cer_boot[:, first] - cer_boot[:, second]
        rows.append({
            "level": level,
            "hours": configs[first].hours,
            "difference": f"{H1_MINUEND} - {H1_SUBTRAHEND}",
            "wer_diff": configs[first].words.rate - configs[second].words.rate,
            "wer_diff_ci": list(percentile_interval(wer_diff)),
            "wer_diff_p_two_sided": two_sided_p(wer_diff),
            "cer_diff": configs[first].chars.rate - configs[second].chars.rate,
            "cer_diff_ci": list(percentile_interval(cer_diff)),
            "_wer_diff_samples": wer_diff,
        })

    p_values = [row["wer_diff_p_two_sided"] for row in rows]
    for row, adjusted, level in zip(rows, holm_adjust(p_values), holm_levels(p_values), strict=True):
        samples = row.pop("_wer_diff_samples")
        row["wer_diff_p_holm"] = adjusted
        row["holm_ci_level"] = level
        row["wer_diff_ci_holm_level"] = list(percentile_interval(samples, level))
        row["wer_diff_ci_half_width"] = (row["wer_diff_ci"][1] - row["wer_diff_ci"][0]) / 2
        row["significant"] = adjusted < ALPHA
        if row["significant"]:
            row["better_model"] = H1_SUBTRAHEND if row["wer_diff"] > 0 else H1_MINUEND
        else:
            row["better_model"] = None
    return rows


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="WER dan CER berselang serta uji H1.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Jumlah iterasi; selain 10.000 hanya untuk uji coba.")
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
    print("WER dan CER berselang kepercayaan serta uji H1", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print(f"Selang             : persentil {CONFIDENCE:.0%}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    print(f"Konfigurasi dimuat : {len(configs)}, masing-masing {len(configs[0].ids)} utterance; "
          f"WER dan CER hasil cacah sama dengan laporan inference", flush=True)

    numerators = np.column_stack([c.words.errors for c in configs] + [c.chars.errors for c in configs])
    denominators = np.column_stack([c.words.lengths for c in configs] + [c.chars.lengths for c in configs])
    samples = bootstrap_ratios(numerators, denominators, args.n_bootstrap, ANALYSIS_SEED)
    wer_boot, cer_boot = samples[:, :len(configs)], samples[:, len(configs):]

    summaries = config_summaries(configs, wer_boot, cer_boot)
    tests = h1_tests(configs, wer_boot, cer_boot)

    print("-" * 78, flush=True)
    for row in summaries:
        print(f"{row['run_name']:18s} {row['hours']:7.3f} jam  "
              f"WER {row['wer']:.4f} [{row['wer_ci'][0]:.4f}, {row['wer_ci'][1]:.4f}]  "
              f"CER {row['cer']:.4f} [{row['cer_ci'][0]:.4f}, {row['cer_ci'][1]:.4f}]", flush=True)
    print("-" * 78, flush=True)
    print(f"H1: selisih WER {H1_MINUEND} - {H1_SUBTRAHEND}, Holm atas {len(tests)} tingkat", flush=True)
    for row in tests:
        verdict = f"signifikan, lebih baik {row['better_model']}" if row["significant"] else "tidak signifikan"
        print(f"  {row['level']}: {row['wer_diff']:+.4f} [{row['wer_diff_ci'][0]:+.4f}, {row['wer_diff_ci'][1]:+.4f}]  "
              f"p {row['wer_diff_p_two_sided']:.5f}  p Holm {row['wer_diff_p_holm']:.5f}  -> {verdict}", flush=True)
    widths = [row["wer_ci_half_width"] for row in summaries]
    print(f"Setengah lebar selang WER: {min(widths):.4f} sampai {max(widths):.4f}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "confidence": CONFIDENCE,
        "interval_method": "persentil",
        "resampling_unit": "utterance data test, berpengembalian, sama bagi seluruh konfigurasi",
        "aggregation": "tingkat corpus: jumlah galat dibagi jumlah satuan referensi",
        "h1": {
            "difference": f"{H1_MINUEND} - {H1_SUBTRAHEND}",
            "correction": "Holm-Bonferroni atas nilai p bootstrap dua sisi",
            "alpha": ALPHA,
            "n_tests": len(tests),
            "tests": tests,
        },
        "configs": summaries,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / CONFIGS_TABLE_NAME,
              ["model", "level", "run_name", "hours", "wer", "wer_ci_low", "wer_ci_high",
               "cer", "cer_ci_low", "cer_ci_high"],
              [[r["model"], r["level"], r["run_name"], r["hours"], r["wer"], *r["wer_ci"],
                r["cer"], *r["cer_ci"]] for r in summaries])
    write_csv(args.results_dir / H1_TABLE_NAME,
              ["level", "hours", "wer_diff", "wer_diff_ci_low", "wer_diff_ci_high",
               "p_two_sided", "p_holm", "holm_ci_level", "holm_ci_low", "holm_ci_high",
               "significant", "better_model", "cer_diff", "cer_diff_ci_low", "cer_diff_ci_high"],
              [[r["level"], r["hours"], r["wer_diff"], *r["wer_diff_ci"], r["wer_diff_p_two_sided"],
                r["wer_diff_p_holm"], r["holm_ci_level"], *r["wer_diff_ci_holm_level"],
                r["significant"], r["better_model"], r["cer_diff"], *r["cer_diff_ci"]] for r in tests])

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
