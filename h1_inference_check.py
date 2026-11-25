"""Pemeriksaan prosedur inferensi H1: Holm, selang simultan, dan uji permutasi berpasangan (Revisi 01 poin 23).

Tujuan
------
H1 diputuskan dari nilai *p* *bootstrap* dua sisi yang dikoreksi Holm, dan
`bab-4.md` Tabel 4.3 juga melaporkan selang persentil pada taraf urutan Holm.
Selang bertaraf Holm itu tidak dijamin memiliki cakupan simultan, sehingga
skrip ini memisahkan inferensi marginal dari inferensi simultan dan menguji
putusan H1 dengan prosedur berpasangan lain yang terkalibrasi.

Prosedur
--------
1. **Holm.** Nilai *p* terkoreksi dari `analysis_common.holm_adjust`
   dicocokkan dengan `statsmodels.stats.multitest.multipletests(method="holm")`.
2. **Selang simultan.** Selang persentil Bonferroni, yaitu taraf 1 - 0,05/5 =
   99% bagi kelima tingkat, dihitung dari *resample* yang sama dengan hasil
   utama (seed 42, 10.000 iterasi). Selang ini menjamin cakupan bersama
   sekurang-kurangnya 95% bagi kelima selisih, berbeda dari selang 95%
   marginal dan selang bertaraf Holm.
3. **Uji permutasi berpasangan.** *Approximate randomization* (Riezler dan
   Maxwell 2005; Koehn 2004 sebagai pembanding *bootstrap*): pada setiap
   permutasi, galat kata kedua model pada setiap utterance ditukar dengan
   peluang 0,5, lalu selisih WER tingkat corpus dihitung ulang. Nilai *p* dua
   sisi = (1 + #{|d*| >= |d|}) / (R + 1) dengan R = 10.000 dan seed 42, lalu
   dikoreksi Holm. Di bawah hipotesis nol bahwa kedua sistem dapat dipertukarkan
   pada setiap utterance, uji ini eksak sampai galat Monte Carlo, sehingga
   menjadi validasi terkalibrasi bagi putusan *bootstrap*. Hal yang sama
   dijalankan bagi CER.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/results/bootstrap_metrics_report.json` : hasil H1 terbit.

Keluaran
--------
- `outputs/results/h1_inference_check_report.json`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u h1_inference_check.py 2>&1 | Tee-Object -FilePath logs/h1_inference_check.log
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from statsmodels.stats.multitest import multipletests

from analysis_common import (
    ALPHA,
    ANALYSIS_SEED,
    BOOTSTRAP_BLOCK,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    bootstrap_ratios,
    find_config,
    holm_adjust,
    load_configs,
    percentile_interval,
    sha256_of,
    write_json,
)
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "h1_inference_check_report.json"
PUBLISHED = RESULTS_DIR / "bootstrap_metrics_report.json"
N_PERMUTATIONS = 10_000
MATCH_TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Uji permutasi
# --------------------------------------------------------------------------


def paired_permutation_p(first: np.ndarray, second: np.ndarray, lengths: np.ndarray,
                         n_permutations: int, seed: int) -> tuple[float, float]:
    """Selisih rasio teramati dan nilai p dua sisi approximate randomization."""
    total = lengths.sum()
    observed = (first.sum() - second.sum()) / total
    delta = (first - second).astype(np.float64)
    generator = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_permutations, BOOTSTRAP_BLOCK):
        size = min(BOOTSTRAP_BLOCK, n_permutations - start)
        signs = generator.choice(np.array([-1.0, 1.0]), size=(size, delta.size))
        permuted = (signs @ delta) / total
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
    return float(observed), (1 + extreme) / (n_permutations + 1)


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Pemeriksaan prosedur inferensi H1.")
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
    print("Pemeriksaan prosedur inferensi H1 (Revisi 01 poin 23)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi            : bootstrap {N_BOOTSTRAP}, permutasi {N_PERMUTATIONS}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    with PUBLISHED.open("r", encoding="utf-8") as handle:
        published = json.load(handle)["h1"]["tests"]

    p_published = [row["wer_diff_p_two_sided"] for row in published]
    holm_ours = holm_adjust(p_published)
    holm_statsmodels = list(multipletests(p_published, alpha=ALPHA, method="holm")[1])
    holm_match = bool(np.allclose(holm_ours, holm_statsmodels, atol=MATCH_TOLERANCE)
                      and np.allclose(holm_ours, [row["wer_diff_p_holm"] for row in published], atol=MATCH_TOLERANCE))
    print(f"Holm analysis_common = statsmodels = laporan terbit: {holm_match}", flush=True)

    wer_boot = bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                                np.column_stack([c.words.lengths for c in configs]), N_BOOTSTRAP, ANALYSIS_SEED)
    cer_boot = bootstrap_ratios(np.column_stack([c.chars.errors for c in configs]),
                                np.column_stack([c.chars.lengths for c in configs]), N_BOOTSTRAP, ANALYSIS_SEED)
    bonferroni_level = 1 - ALPHA / len(LEVELS)

    rows: list[dict[str, Any]] = []
    for index, level in enumerate(LEVELS):
        w = configs[find_config(configs, "whisper", level)]
        x = configs[find_config(configs, "xlsr", level)]
        first, second = find_config(configs, "whisper", level), find_config(configs, "xlsr", level)
        wer_diff = wer_boot[:, first] - wer_boot[:, second]
        cer_diff = cer_boot[:, first] - cer_boot[:, second]
        marginal = list(percentile_interval(wer_diff))
        if not np.allclose(marginal, published[index]["wer_diff_ci"], atol=MATCH_TOLERANCE):
            print(f"GAGAL: selang marginal {level} tidak sama dengan laporan terbit", flush=True)
            return 1
        observed, p_perm = paired_permutation_p(w.words.errors, x.words.errors, w.words.lengths,
                                                N_PERMUTATIONS, ANALYSIS_SEED)
        observed_cer, p_perm_cer = paired_permutation_p(w.chars.errors, x.chars.errors, w.chars.lengths,
                                                        N_PERMUTATIONS, ANALYSIS_SEED)
        rows.append({
            "level": level,
            "wer_diff": observed,
            "ci_marginal_95": marginal,
            "ci_holm_level": published[index]["wer_diff_ci_holm_level"],
            "holm_level": published[index]["holm_ci_level"],
            "ci_bonferroni_simultaneous": list(percentile_interval(wer_diff, bonferroni_level)),
            "p_bootstrap_holm": published[index]["wer_diff_p_holm"],
            "p_permutation": p_perm,
            "cer_diff": observed_cer,
            "cer_ci_bonferroni_simultaneous": list(percentile_interval(cer_diff, bonferroni_level)),
            "p_permutation_cer": p_perm_cer,
        })

    for name, key in (("wer", "p_permutation"), ("cer", "p_permutation_cer")):
        adjusted = holm_adjust([row[key] for row in rows])
        for row, value in zip(rows, adjusted, strict=True):
            row[f"{key}_holm"] = value
    simultaneous_excludes_zero = all(r["ci_bonferroni_simultaneous"][1] < 0 or r["ci_bonferroni_simultaneous"][0] > 0
                                     for r in rows)
    for r in rows:
        print(f"{r['level']}: selisih {r['wer_diff']:+.4f}  marginal [{r['ci_marginal_95'][0]:+.4f}, "
              f"{r['ci_marginal_95'][1]:+.4f}]  Bonferroni 99% [{r['ci_bonferroni_simultaneous'][0]:+.4f}, "
              f"{r['ci_bonferroni_simultaneous'][1]:+.4f}]  p permutasi Holm {r['p_permutation_holm']:.4f}  "
              f"| CER p permutasi Holm {r['p_permutation_cer_holm']:.4f} Bonferroni "
              f"[{r['cer_ci_bonferroni_simultaneous'][0]:+.4f}, {r['cer_ci_bonferroni_simultaneous'][1]:+.4f}]",
              flush=True)
    print(f"Kelima selang simultan WER tidak memuat nol: {simultaneous_excludes_zero}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
        "holm_matches_statsmodels_and_published": holm_match,
        "bonferroni_level": bonferroni_level,
        "interval_notes": {
            "marginal_95": "cakupan 95% bagi setiap tingkat secara terpisah",
            "holm_level": "taraf mengikuti urutan Holm; tidak dijamin bercakupan simultan",
            "bonferroni_simultaneous": "cakupan bersama sekurang-kurangnya 95% bagi kelima tingkat",
        },
        "simultaneous_wer_intervals_exclude_zero": simultaneous_excludes_zero,
        "levels": rows,
        "input_sha256": {**{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
                         PUBLISHED.name: sha256_of(PUBLISHED)},
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0 if holm_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
