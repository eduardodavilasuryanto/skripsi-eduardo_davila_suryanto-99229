"""Pencocokan kurva penskalaan WER = E0 + a * D^(-c) dan pengujian H2 serta H3.

Tujuan
------
Menghasilkan keluaran 3 sampai 5 Subbab 3.6.4 skripsi dan pengujian H2 dan
H3 pada Tabel 3.15, mengikuti enam langkah Subbab 3.6.2.

Prosedur
--------
1. Nonlinear least squares pada skala asli atas sepuluh titik WER, tanpa
   pembobotan, dengan `scipy.optimize.least_squares` metode trf.
2. Kendala E0 >= 0 dan a > 0; eksponen c tidak dibatasi agar H2 benar-benar
   diuji. Pencocokan titik penuh memakai beberapa titik awal dan memilih
   jumlah kuadrat residual terkecil.
3. Model utama: E0 bersama bagi kedua model (lima parameter). Pembanding:
   E0 terpisah per model (enam parameter) dan power law murni E0 = 0 (empat
   parameter).
4. AIC = n ln(RSS / n) + 2K dengan n = 10 dan K = jumlah parameter + 1.
   E0 bersama dipertahankan kecuali AIC model E0 terpisah lebih rendah lebih
   dari 2; aturan ini ditetapkan penulis sebelum pencocokan dijalankan.
5. Bootstrap: 10.000 resample tingkat utterance (seed 42, sama dengan
   `bootstrap_metrics.py`); kesepuluh WER dihitung ulang dan kurva dicocokkan
   ulang dari estimasi titik penuh sebagai titik awal. Selang persentil 95%.
   Bootstrap dijalankan bagi model E0 bersama dan E0 terpisah; power law murni
   hanya dicocokkan pada titik penuh sebagai pembanding.
6. H2: selang 95% c setiap model seluruhnya di atas nol. H3: selang 95%
   c_whisper - c_xlsr, dihitung per iterasi, tidak memuat nol. Keduanya
   memakai model utama hasil langkah 4.

D adalah jam data training terukur dari laporan `build_nested_subsets.py`,
yaitu 1,000; 3,000; 10,001; 30,001; dan 66,134, atas keputusan penulis.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/results/<run>_test_metrics.json`, hanya untuk pemeriksaan kesamaan.
- `outputs/build_nested_subsets_report.json` : jam data training terukur.

Keluaran
--------
- `outputs/results/scaling_curve_report.json` : estimasi, selang, AIC,
  residual, dan putusan H2 dan H3.
- `outputs/results/scaling_curve_parameters.csv` : tabel parameter.
- `outputs/results/scaling_curve_bootstrap.csv` : parameter setiap iterasi.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u fit_scaling_curve.py 2>&1 |
        Tee-Object -FilePath logs/fit_scaling_curve.log
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.optimize import least_squares

from analysis_common import (
    ANALYSIS_SEED,
    CONFIDENCE,
    LEVELS,
    MODELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
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

REPORT_NAME = "scaling_curve_report.json"
PARAMETERS_TABLE_NAME = "scaling_curve_parameters.csv"
BOOTSTRAP_TABLE_NAME = "scaling_curve_bootstrap.csv"

# Batas bawah a sebagai wujud kendala a > 0.
A_MIN = 1e-9

# E0 dianggap berada pada batas nol bila tidak lebih dari nilai ini.
E0_AT_BOUND = 1e-8

# Selisih AIC yang diperlukan agar model E0 terpisah menggantikan E0 bersama.
AIC_THRESHOLD = 2.0

# Titik awal pencocokan titik penuh: E0 sebagai pecahan WER terkecil, dan c.
START_E0_FRACTIONS = (0.0, 0.5, 0.9)
START_C_VALUES = (0.25, 0.5, 1.0)

LEAST_SQUARES_OPTIONS = {"method": "trf", "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12, "max_nfev": 20_000}

# Nama keenam kuantitas kurva dalam bentuk seragam bagi ketiga model.
CURVE_FIELDS = ("E0_whisper", "a_whisper", "c_whisper", "E0_xlsr", "a_xlsr", "c_xlsr")


# --------------------------------------------------------------------------
# Spesifikasi model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Satu bentuk pencocokan gabungan atas sepuluh titik."""

    name: str
    description: str
    parameters: tuple[str, ...]
    lower: tuple[float, ...]
    unpack: Callable[[np.ndarray], tuple[float, float, float, float, float, float]]


SPECS: dict[str, ModelSpec] = {
    "shared_e0": ModelSpec(
        "shared_e0", "E0 bersama bagi kedua model (model utama, Subbab 8.0c)",
        ("E0", "a_whisper", "c_whisper", "a_xlsr", "c_xlsr"),
        (0.0, A_MIN, -np.inf, A_MIN, -np.inf),
        lambda p: (p[0], p[1], p[2], p[0], p[3], p[4]),
    ),
    "separate_e0": ModelSpec(
        "separate_e0", "E0 terpisah per model (pembanding AIC)",
        ("E0_whisper", "a_whisper", "c_whisper", "E0_xlsr", "a_xlsr", "c_xlsr"),
        (0.0, A_MIN, -np.inf, 0.0, A_MIN, -np.inf),
        lambda p: (p[0], p[1], p[2], p[3], p[4], p[5]),
    ),
    "pure_power_law": ModelSpec(
        "pure_power_law", "Power law murni, kasus bersarang E0 = 0",
        ("a_whisper", "c_whisper", "a_xlsr", "c_xlsr"),
        (A_MIN, -np.inf, A_MIN, -np.inf),
        lambda p: (0.0, p[0], p[1], 0.0, p[2], p[3]),
    ),
}

PRIMARY_DEFAULT = "shared_e0"
ALTERNATIVE = "separate_e0"
NESTED = "pure_power_law"


@dataclass(frozen=True)
class FitResult:
    """Hasil satu pencocokan."""

    params: np.ndarray
    rss: float
    success: bool
    message: str


# --------------------------------------------------------------------------
# Pencocokan
# --------------------------------------------------------------------------


def predict(spec: ModelSpec, params: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """WER prediksi bagi Whisper D1 sampai D5 lalu XLS-R D1 sampai D5."""
    e0_w, a_w, c_w, e0_x, a_x, c_x = spec.unpack(params)
    return np.concatenate([e0_w + a_w * hours ** (-c_w), e0_x + a_x * hours ** (-c_x)])


def start_points(spec: ModelSpec, observed: np.ndarray) -> list[np.ndarray]:
    """Titik awal dari data: a dimulai dari WER D1 dikurangi E0 awal."""
    smallest = float(observed.min())
    wer_d1 = {model: float(observed[index * len(LEVELS)]) for index, model in enumerate(MODELS)}
    points = []
    fractions = (0.0,) if spec.name == NESTED else START_E0_FRACTIONS
    for fraction in fractions:
        e0 = fraction * smallest
        for c in START_C_VALUES:
            a = {model: max(wer_d1[model] - e0, 10 * A_MIN) for model in MODELS}
            if spec.name == "shared_e0":
                points.append(np.array([e0, a["whisper"], c, a["xlsr"], c]))
            elif spec.name == "separate_e0":
                points.append(np.array([e0, a["whisper"], c, e0, a["xlsr"], c]))
            else:
                points.append(np.array([a["whisper"], c, a["xlsr"], c]))
    return points


def fit_once(spec: ModelSpec, observed: np.ndarray, hours: np.ndarray, start: np.ndarray) -> FitResult:
    """Satu pencocokan least squares berkendala dari satu titik awal."""
    upper = np.full(len(spec.lower), np.inf)
    result = least_squares(
        lambda p: predict(spec, p, hours) - observed,
        start, bounds=(np.asarray(spec.lower), upper), **LEAST_SQUARES_OPTIONS,
    )
    rss = float(np.sum(result.fun ** 2))
    return FitResult(result.x, rss, bool(result.success) and math.isfinite(rss), result.message)


def fit_best(spec: ModelSpec, observed: np.ndarray, hours: np.ndarray,
             starts: Sequence[np.ndarray]) -> FitResult | None:
    """Pencocokan dari beberapa titik awal; yang berhasil dengan RSS terkecil dipilih."""
    results = [fit_once(spec, observed, hours, start) for start in starts]
    successful = [result for result in results if result.success]
    return min(successful, key=lambda result: result.rss) if successful else None


def aic(rss: float, n_points: int, n_params: int) -> float:
    """AIC least squares dengan galat normal; K mencakup variansi galat."""
    return n_points * math.log(rss / n_points) + 2 * (n_params + 1)


def bootstrap_fits(spec: ModelSpec, wer_boot: np.ndarray, hours: np.ndarray,
                   full: FitResult) -> tuple[np.ndarray, int, int]:
    """Pencocokan ulang setiap resample; baris gagal bernilai NaN.

    Titik awalnya estimasi titik penuh. Bila gagal, beberapa titik awal
    dicoba; bila tetap gagal, iterasi itu dicatat gagal dan dikeluarkan dari
    selang.
    """
    draws = np.full((len(wer_boot), len(CURVE_FIELDS)), np.nan)
    retried = failed = 0
    for index, observed in enumerate(wer_boot):
        result = fit_once(spec, observed, hours, full.params)
        if not result.success:
            retried += 1
            result = fit_best(spec, observed, hours, start_points(spec, observed))
        if result is None:
            failed += 1
            continue
        draws[index] = spec.unpack(result.params)
    return draws, retried, failed


# --------------------------------------------------------------------------
# Ringkasan
# --------------------------------------------------------------------------


def interval_summary(samples: np.ndarray, estimate: float) -> dict[str, Any]:
    """Estimasi, selang persentil, dan median sebuah kuantitas."""
    low, high = percentile_interval(samples)
    return {"estimate": estimate, "ci": [low, high], "bootstrap_median": float(np.nanmedian(samples))}


def full_fit_summary(spec: ModelSpec, fit: FitResult, observed: np.ndarray, hours: np.ndarray,
                     configs: Sequence[Any]) -> dict[str, Any]:
    """Parameter, RSS, AIC, dan residual satu model pada titik penuh."""
    fitted = predict(spec, fit.params, hours)
    return {
        "description": spec.description,
        "parameters": dict(zip(spec.parameters, map(float, fit.params), strict=True)),
        "curve": dict(zip(CURVE_FIELDS, map(float, spec.unpack(fit.params)), strict=True)),
        "n_parameters": len(spec.parameters),
        "rss": fit.rss,
        "aic": aic(fit.rss, len(observed), len(spec.parameters)),
        "message": fit.message,
        "points": [
            {"run_name": c.run_name, "model": c.model, "level": c.level, "hours": c.hours,
             "observed_wer": float(o), "fitted_wer": float(f), "residual": float(o - f)}
            for c, o, f in zip(configs, observed, fitted, strict=True)
        ],
    }


def bootstrap_summary(spec: ModelSpec, fit: FitResult, draws: np.ndarray,
                      retried: int, failed: int) -> dict[str, Any]:
    """Selang setiap kuantitas kurva dan selisih eksponen dari hasil bootstrap."""
    curve = dict(zip(CURVE_FIELDS, map(float, spec.unpack(fit.params)), strict=True))
    fields = CURVE_FIELDS if spec.name == ALTERNATIVE else ("E0_whisper", "a_whisper", "c_whisper",
                                                            "a_xlsr", "c_xlsr")
    summary = {
        field if not (spec.name == "shared_e0" and field == "E0_whisper") else "E0":
            interval_summary(draws[:, CURVE_FIELDS.index(field)], curve[field])
        for field in fields
    }
    c_diff = draws[:, CURVE_FIELDS.index("c_whisper")] - draws[:, CURVE_FIELDS.index("c_xlsr")]
    summary["c_whisper_minus_c_xlsr"] = {
        **interval_summary(c_diff, curve["c_whisper"] - curve["c_xlsr"]),
        "p_two_sided": two_sided_p(c_diff),
    }
    e0_columns = [CURVE_FIELDS.index("E0_whisper"), CURVE_FIELDS.index("E0_xlsr")]
    summary["share_E0_at_zero_bound"] = {
        name: float(np.nanmean(draws[:, column] <= E0_AT_BOUND))
        for name, column in zip(("whisper", "xlsr"), e0_columns, strict=True)
    }
    summary["n_retried"] = retried
    summary["n_failed"] = failed
    summary["n_used"] = int(np.sum(~np.isnan(draws[:, 0])))
    return summary


def decisions(spec_name: str, boot: dict[str, Any]) -> dict[str, Any]:
    """Putusan H2 dan H3 menurut kriteria Tabel 3.15."""
    h2 = {model: boot[f"c_{model}"]["ci"][0] > 0 for model in MODELS}
    diff_low, diff_high = boot["c_whisper_minus_c_xlsr"]["ci"]
    e0_key = "E0" if spec_name == "shared_e0" else None
    e0_contains_zero = (boot[e0_key]["ci"][0] <= E0_AT_BOUND) if e0_key else {
        model: boot[f"E0_{model}"]["ci"][0] <= E0_AT_BOUND for model in MODELS
    }
    return {
        "model": spec_name,
        "h2_c_ci_entirely_above_zero": h2,
        "h2_supported_for_both_models": all(h2.values()),
        "h3_c_difference_ci_excludes_zero": diff_low > 0 or diff_high < 0,
        "h3_larger_exponent": ("whisper" if diff_low > 0 else "xlsr" if diff_high < 0 else None),
        "e0_ci_reaches_zero": e0_contains_zero,
    }


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Pencocokan kurva penskalaan dan uji H2 serta H3.")
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
    print("Pencocokan kurva penskalaan WER = E0 + a * D^(-c)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print(f"Ambang AIC         : E0 terpisah dipakai bila AIC-nya lebih rendah > {AIC_THRESHOLD}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    hours = np.array([c.hours for c in configs[:len(LEVELS)]])
    observed = np.array([c.words.rate for c in configs])
    print(f"D (jam terukur)    : {', '.join(f'{h:.3f}' for h in hours)}", flush=True)
    for config, value in zip(configs, observed, strict=True):
        print(f"  {config.run_name:18s} WER {value:.4f}", flush=True)

    fits: dict[str, FitResult] = {}
    for name, spec in SPECS.items():
        best = fit_best(spec, observed, hours, start_points(spec, observed))
        if best is None:
            print(f"GAGAL: pencocokan {name} pada titik penuh tidak konvergen.", flush=True)
            return 1
        fits[name] = best
    full = {name: full_fit_summary(SPECS[name], fit, observed, hours, configs) for name, fit in fits.items()}

    delta_aic = full[ALTERNATIVE]["aic"] - full[PRIMARY_DEFAULT]["aic"]
    primary = ALTERNATIVE if delta_aic < -AIC_THRESHOLD else PRIMARY_DEFAULT

    print("-" * 78, flush=True)
    for name, summary in full.items():
        values = ", ".join(f"{k} {v:.4f}" for k, v in summary["parameters"].items())
        print(f"{name:15s} RSS {summary['rss']:.6f}  AIC {summary['aic']:8.3f}  {values}", flush=True)
    print(f"AIC E0 terpisah - E0 bersama = {delta_aic:+.3f} -> model utama {primary}", flush=True)

    wer_boot = bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                                np.column_stack([c.words.lengths for c in configs]),
                                args.n_bootstrap, ANALYSIS_SEED)
    boot: dict[str, dict[str, Any]] = {}
    draws_by_spec: dict[str, np.ndarray] = {}
    for name in (PRIMARY_DEFAULT, ALTERNATIVE):
        draws, retried, failed = bootstrap_fits(SPECS[name], wer_boot, hours, fits[name])
        draws_by_spec[name] = draws
        boot[name] = bootstrap_summary(SPECS[name], fits[name], draws, retried, failed)
        print(f"Bootstrap {name:12s}: {boot[name]['n_used']} dipakai, {retried} dicoba ulang, {failed} gagal",
              flush=True)

    verdict = decisions(primary, boot[primary])
    primary_boot = boot[primary]
    print("-" * 78, flush=True)
    for key in [k for k in primary_boot if isinstance(primary_boot[k], dict) and "ci" in primary_boot[k]]:
        item = primary_boot[key]
        print(f"  {key:24s} {item['estimate']:+.4f} [{item['ci'][0]:+.4f}, {item['ci'][1]:+.4f}]", flush=True)
    print(f"H2 (selang c di atas nol): {verdict['h2_c_ci_entirely_above_zero']}", flush=True)
    print(f"H3 (selang c_whisper - c_xlsr tanpa nol): {verdict['h3_c_difference_ci_excludes_zero']}, "
          f"eksponen lebih besar: {verdict['h3_larger_exponent']}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "confidence": CONFIDENCE,
        "interval_method": "persentil",
        "hours_source": f"{SUBSETS_REPORT.name}, jam terukur",
        "hours": dict(zip(LEVELS, map(float, hours), strict=True)),
        "observed_wer": {c.run_name: float(v) for c, v in zip(configs, observed, strict=True)},
        "fitting": {
            "method": "scipy.optimize.least_squares, skala asli, tanpa pembobotan",
            "options": LEAST_SQUARES_OPTIONS,
            "constraints": "E0 >= 0, a >= " + str(A_MIN) + ", c tanpa batas",
            "starts_e0_fraction_of_min_wer": START_E0_FRACTIONS,
            "starts_c": START_C_VALUES,
        },
        "aic": {
            "formula": "n ln(RSS/n) + 2K, K = jumlah parameter + 1",
            "values": {name: summary["aic"] for name, summary in full.items()},
            "separate_minus_shared": delta_aic,
            "threshold": AIC_THRESHOLD,
            "primary_model": primary,
            "nested_minus_shared": full[NESTED]["aic"] - full[PRIMARY_DEFAULT]["aic"],
        },
        "full_sample": full,
        "bootstrap": boot,
        "decisions": verdict,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)

    parameter_rows = []
    for name, summary in boot.items():
        for key, item in summary.items():
            if isinstance(item, dict) and "ci" in item:
                parameter_rows.append([name, key, item["estimate"], *item["ci"], item["bootstrap_median"]])
    for key, value in full[NESTED]["parameters"].items():
        parameter_rows.append([NESTED, key, value, "", "", ""])
    write_csv(args.results_dir / PARAMETERS_TABLE_NAME,
              ["model_spec", "quantity", "estimate", "ci_low", "ci_high", "bootstrap_median"], parameter_rows)
    write_csv(args.results_dir / BOOTSTRAP_TABLE_NAME, ["model_spec", "iteration", *CURVE_FIELDS],
              [[name, index, *row] for name, draws in draws_by_spec.items() for index, row in enumerate(draws)])

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
