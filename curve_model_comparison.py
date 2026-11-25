"""Perbandingan bentuk fungsi kurva dengan AICc, BIC, dependensi titik, dan prediksi titik yang ditahan.

Tujuan
------
Putusan bentuk kurva pada hasil utama memakai AIC atas sepuluh titik WER
dengan ambang 2 (`fit_scaling_curve.py`). Sepuluh titik dengan empat sampai
enam parameter ditambah variansi galat adalah sampel yang sangat kecil, dan
kesepuluh WER saling berkorelasi karena diukur pada data test yang sama dan
dilatih pada subset bersarang. Skrip ini menambahkan tiga pemeriksaan sebagai
sensitivitas, tanpa mengubah aturan putusan utama.

Prosedur
--------
1. **Kriteria informasi.** Bagi ketiga bentuk (`E0` bersama, `E0` terpisah,
   *power law* murni), dihitung AIC = n ln(RSS/n) + 2K, AICc = AIC +
   2K(K + 1)/(n - K - 1), dan BIC = n ln(RSS/n) + K ln n, dengan n = 10 dan K =
   jumlah parameter ditambah satu bagi variansi galat. Ketiganya bersandar pada
   galat normal yang saling bebas dengan variansi sama, asumsi yang diperiksa
   pada langkah 2.
2. **Dependensi antartitik.** Korelasi kesepuluh WER antarresample dihitung
   dari *bootstrap* tingkat utterance yang sama dengan hasil utama (seed 42,
   10.000 iterasi). Korelasi ini hanya menangkap dependensi yang berasal dari
   data test bersama; dependensi lewat subset training bersarang tidak dapat
   diukur tanpa ulangan pelatihan.
3. **Prediksi titik yang ditahan.** (a) *Leave-one-out*: setiap titik ditahan,
   bentuk fungsi dicocokkan pada sembilan titik sisanya, lalu titik itu
   diprediksi. (b) D5 ditahan: kedua titik D5 ditahan bersama, kurva
   dicocokkan pada D1 sampai D4 kedua model, lalu D5 diprediksi; ini satu-
   satunya uji prediksi ke luar rentang pencocokan, dan karena itu juga
   mengukur risiko ekstrapolasi. Pencocokan memakai pilihan, kendala, dan
   titik awal yang sama dengan `fit_scaling_curve.py`; nilai titik yang ditahan
   tidak dipakai sebagai titik awal.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/build_nested_subsets_report.json` : jam data training terukur.
- `outputs/results/scaling_curve_report.json` : pencocokan titik penuh terbit.

Keluaran
--------
- `outputs/results/curve_model_comparison_report.json`
- `outputs/results/curve_model_comparison_holdout.csv` : galat prediksi setiap
  titik yang ditahan per bentuk fungsi.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u curve_model_comparison.py 2>&1 | Tee-Object -FilePath logs/curve_model_comparison.log
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from analysis_common import (
    ANALYSIS_SEED,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
    load_configs,
    sha256_of,
    write_csv,
    write_json,
)
from fit_scaling_curve import LEAST_SQUARES_OPTIONS, NESTED, PRIMARY_DEFAULT, SPECS, ModelSpec, predict, start_points
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "curve_model_comparison_report.json"
HOLDOUT_TABLE_NAME = "curve_model_comparison_holdout.csv"
PUBLISHED_CURVE = RESULTS_DIR / "scaling_curve_report.json"
MATCH_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# Pencocokan bertopeng
# --------------------------------------------------------------------------


def fit_masked(spec: ModelSpec, observed: np.ndarray, hours: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    """Pencocokan terbaik dari titik awal baku hanya atas titik yang tidak ditahan.

    Titik yang ditahan diganti nilai titik bertetangga dari model yang sama
    sebelum titik awal dibentuk, sehingga nilainya tidak ikut menentukan titik
    awal maupun pencocokan.
    """
    proxy = observed.copy()
    n_levels = len(LEVELS)
    for index in np.flatnonzero(~mask):
        model_start = (index // n_levels) * n_levels
        candidates = [i for i in range(model_start, model_start + n_levels) if mask[i]]
        proxy[index] = observed[min(candidates, key=lambda i: abs(i - index))]
    upper = np.full(len(spec.lower), np.inf)
    best: tuple[np.ndarray, float] | None = None
    for start in start_points(spec, proxy):
        result = least_squares(lambda p: (predict(spec, p, hours) - observed)[mask], start,
                               bounds=(np.asarray(spec.lower), upper), **LEAST_SQUARES_OPTIONS)
        rss = float(np.sum(result.fun ** 2))
        if result.success and math.isfinite(rss) and (best is None or rss < best[1]):
            best = (result.x, rss)
    if best is None:
        raise RuntimeError(f"Pencocokan {spec.name} gagal")
    return best


def criteria(rss: float, n: int, n_params: int) -> dict[str, float]:
    """AIC, AICc, dan BIC least squares dengan K = jumlah parameter + 1."""
    k = n_params + 1
    base = n * math.log(rss / n)
    aic = base + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else float("inf")
    return {"K": k, "aic": aic, "aicc": aicc, "bic": base + k * math.log(n), "n_minus_k_minus_1": n - k - 1}


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="AICc, BIC, dependensi titik, dan prediksi titik yang ditahan.")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                        help="Jumlah iterasi bagi korelasi; selain 10.000 hanya untuk uji coba.")
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
    print("Perbandingan bentuk fungsi: AICc, BIC, dependensi, prediksi tertahan (Revisi 01 poin 20)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    names = [c.run_name for c in configs]
    hours = np.array([c.hours for c in configs[:len(LEVELS)]])
    observed = np.array([c.words.rate for c in configs])
    n = len(observed)
    full_mask = np.ones(n, dtype=bool)
    with PUBLISHED_CURVE.open("r", encoding="utf-8") as handle:
        published = json.load(handle)["full_sample"]

    information: dict[str, Any] = {}
    for name, spec in SPECS.items():
        params, rss = fit_masked(spec, observed, hours, full_mask)
        if abs(rss - published[name]["rss"]) > MATCH_TOLERANCE:
            print(f"GAGAL: RSS {name} {rss} berbeda dengan laporan terbit {published[name]['rss']}", flush=True)
            return 1
        information[name] = {"n_parameters": len(spec.parameters), "rss": rss,
                             "parameters": dict(zip(spec.parameters, map(float, params), strict=True)),
                             "lower_bounds": dict(zip(spec.parameters, map(float, spec.lower), strict=True)),
                             **criteria(rss, n, len(spec.parameters))}
    for name in SPECS:
        for key in ("aic", "aicc", "bic"):
            information[name][f"delta_{key}_vs_shared"] = information[name][key] - information[PRIMARY_DEFAULT][key]
        item = information[name]
        print(f"{name:15s} K={item['K']}  AIC {item['aic']:8.3f} ({item['delta_aic_vs_shared']:+.3f})  "
              f"AICc {item['aicc']:8.3f} ({item['delta_aicc_vs_shared']:+.3f})  "
              f"BIC {item['bic']:8.3f} ({item['delta_bic_vs_shared']:+.3f})", flush=True)

    boot = bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                            np.column_stack([c.words.lengths for c in configs]), args.n_bootstrap, ANALYSIS_SEED)
    corr = np.corrcoef(boot, rowvar=False)
    n_levels = len(LEVELS)
    same_model = [corr[i, j] for i in range(n) for j in range(i + 1, n) if i // n_levels == j // n_levels]
    same_level = [corr[i, i + n_levels] for i in range(n_levels)]
    dependence = {
        "correlation_matrix": {names[i]: dict(zip(names, map(float, corr[i]), strict=True)) for i in range(n)},
        "same_model_across_levels": {"min": float(min(same_model)), "max": float(max(same_model)),
                                     "median": float(np.median(same_model))},
        "same_level_across_models": dict(zip(LEVELS, map(float, same_level), strict=True)),
        "scope": "hanya dependensi lewat data test bersama; dependensi lewat subset training bersarang tidak terukur",
    }
    print(f"Korelasi bootstrap antartingkat dalam satu model: {min(same_model):.2f} sampai {max(same_model):.2f}; "
          f"antarmodel pada tingkat sama: {min(same_level):.2f} sampai {max(same_level):.2f}", flush=True)

    rows: list[list[Any]] = []
    holdout: dict[str, Any] = {}
    for name, spec in SPECS.items():
        loo_errors = []
        for index in range(n):
            mask = full_mask.copy()
            mask[index] = False
            params, _ = fit_masked(spec, observed, hours, mask)
            predicted = float(predict(spec, params, hours)[index])
            error = predicted - observed[index]
            loo_errors.append(error)
            rows.append([name, "leave_one_out", names[index], observed[index], predicted, error])
        mask = full_mask.copy()
        d5 = [n_levels - 1, 2 * n_levels - 1]
        mask[d5] = False
        params, _ = fit_masked(spec, observed, hours, mask)
        prediction = predict(spec, params, hours)
        d5_errors = {names[i]: float(prediction[i] - observed[i]) for i in d5}
        for i in d5:
            rows.append([name, "d5_ditahan", names[i], observed[i], float(prediction[i]), d5_errors[names[i]]])
        loo = np.array(loo_errors)
        interior = np.array([loo[i] for i in range(n) if 0 < i % n_levels < n_levels - 1])
        holdout[name] = {
            "leave_one_out": {"rmse": float(np.sqrt(np.mean(loo ** 2))), "max_abs": float(np.max(np.abs(loo))),
                              "interior_rmse": float(np.sqrt(np.mean(interior ** 2))),
                              "interior_definition": "D2 sampai D4, prediksi di dalam rentang pencocokan",
                              "errors": dict(zip(names, map(float, loo), strict=True))},
            "d5_held_out": {"errors": d5_errors,
                            "rmse": float(np.sqrt(np.mean(np.square(list(d5_errors.values()))))),
                            "curve_fitted_on_d1_to_d4": dict(zip(
                                ("E0_whisper", "a_whisper", "c_whisper", "E0_xlsr", "a_xlsr", "c_xlsr"),
                                map(float, spec.unpack(params)), strict=True))},
        }
        print(f"{name:15s} LOO RMSE {100 * holdout[name]['leave_one_out']['rmse']:.2f} poin, maks "
              f"{100 * holdout[name]['leave_one_out']['max_abs']:.2f}, titik dalam "
              f"{100 * holdout[name]['leave_one_out']['interior_rmse']:.2f}; D5 ditahan: "
              + ", ".join(f"{k} {100 * v:+.2f}" for k, v in d5_errors.items()), flush=True)

    # Sensitivitas terhadap titik yang belum konvergen: E0 bersama tanpa titik tertentu,
    # dengan bootstrap pencocokan ulang yang berangkat dari estimasi titik penuh skenario itu.
    shared = SPECS[PRIMARY_DEFAULT]
    scenarios = {
        "tanpa_xlsr_d5": ["xlsr_d5"],
        "tanpa_xlsr_batas_langkah": ["xlsr_d1_trial", "xlsr_d3", "xlsr_d4"],
    }
    exclusion: dict[str, Any] = {}
    print("-" * 78, flush=True)
    for scenario, excluded in scenarios.items():
        mask = np.array([name not in excluded for name in names])
        params, rss = fit_masked(shared, observed, hours, mask)
        upper = np.full(len(shared.lower), np.inf)
        draws = np.full((boot.shape[0], 3), np.nan)
        for row, values in enumerate(boot):
            result = least_squares(lambda p: (predict(shared, p, hours) - values)[mask], params,
                                   bounds=(np.asarray(shared.lower), upper), **LEAST_SQUARES_OPTIONS)
            if result.success:
                e0, _, c_w, _, _, c_x = shared.unpack(result.x)
                draws[row] = (e0, c_w, c_x)
        curve = dict(zip(("E0_whisper", "a_whisper", "c_whisper", "E0_xlsr", "a_xlsr", "c_xlsr"),
                         map(float, shared.unpack(params)), strict=True))
        diff = draws[:, 1] - draws[:, 2]
        exclusion[scenario] = {
            "excluded": excluded, "rss": rss, "curve": curve,
            "E0_ci": list(map(float, np.nanpercentile(draws[:, 0], [2.5, 97.5]))),
            "c_whisper_ci": list(map(float, np.nanpercentile(draws[:, 1], [2.5, 97.5]))),
            "c_xlsr_ci": list(map(float, np.nanpercentile(draws[:, 2], [2.5, 97.5]))),
            "c_diff": curve["c_whisper"] - curve["c_xlsr"],
            "c_diff_ci": list(map(float, np.nanpercentile(diff, [2.5, 97.5]))),
            "n_bootstrap_used": int(np.sum(~np.isnan(draws[:, 0]))),
        }
        item = exclusion[scenario]
        print(f"{scenario:26s} E0 {item['curve']['E0_whisper']:.4f} [{item['E0_ci'][0]:.4f}, {item['E0_ci'][1]:.4f}]  "
              f"c_w {item['curve']['c_whisper']:.4f}  c_x {item['curve']['c_xlsr']:.4f}  selisih {item['c_diff']:+.4f} "
              f"[{item['c_diff_ci'][0]:+.4f}, {item['c_diff_ci'][1]:+.4f}]  bootstrap {item['n_bootstrap_used']}",
              flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "point_exclusion_sensitivity": {
            "purpose": "kurva E0 bersama dengan dan tanpa titik yang belum konvergen (Revisi 01 poin 17)",
            "scenarios": exclusion,
        },
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "n_points": n,
        "fitting": "least_squares, opsi, kendala, dan titik awal sama dengan fit_scaling_curve.py; c tanpa batas",
        "assumptions": ("AIC, AICc, dan BIC mengandaikan galat normal saling bebas dengan variansi sama; "
                        "korelasi bootstrap menunjukkan titik tidak saling bebas"),
        "matches_published_rss": True,
        "information_criteria": information,
        "dependence": dependence,
        "holdout": holdout,
        "nested_name": NESTED,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
            PUBLISHED_CURVE.name: sha256_of(PUBLISHED_CURVE),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / HOLDOUT_TABLE_NAME,
              ["model_spec", "scheme", "held_out", "observed_wer", "predicted_wer", "error"], rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
