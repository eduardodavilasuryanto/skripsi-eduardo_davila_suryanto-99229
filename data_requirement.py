"""Kebutuhan jam data training untuk mencapai WER sasaran di dalam rentang terukur (Revisi 01 poin 3).

Tujuan
------
*Data efficiency* pada penelitian ini didefinisikan sebagai eksponen `c`, yaitu
laju peluruhan komponen galat yang dapat direduksi. Ukuran itu tidak sama
dengan jumlah data yang dibutuhkan untuk mencapai WER tertentu. Skrip ini
menghitung ukuran praktis yang terpisah itu, yaitu jam data D* yang dibutuhkan
setiap model untuk mencapai WER sasaran, beserta ketidakpastiannya.

Prosedur
--------
1. WER sasaran dipilih di dalam rentang yang dicapai kedua model, yaitu 40%,
   30%, 25%, dan 20%. Sasaran di luar rentang itu tidak dihitung agar kurva
   tidak diekstrapolasi.
2. D* dihitung dengan tiga cara agar kepekaannya terhadap bentuk fungsi
   terlihat: (a) kurva `E0` bersama, D* = (a / (sasaran - E0))^(1/c);
   (b) *power law* murni, D* = (a / sasaran)^(1/c); (c) interpolasi tanpa
   model, yaitu interpolasi linear WER terhadap log jam di antara dua tingkat
   berurutan yang mengapit sasaran, memakai perpotongan pertama.
3. Ketidakpastian berasal dari *bootstrap* tingkat utterance yang sama dengan
   hasil utama (seed 42, 10.000 iterasi): untuk (a) dan (b) kurva dicocokkan
   ulang dengan fungsi `fit_scaling_curve.py`; untuk (c) interpolasi diulang
   pada WER setiap resample. D* di luar rentang 1 sampai 66,134 jam dicatat
   sebagai di luar rentang dan tidak ikut selang. Selang persentil 95% juga
   dihitung bagi rasio D* XLS-R terhadap D* Whisper pada iterasi yang sama.
4. Hasil `E0` bersama pada data penuh dicocokkan dengan
   `scaling_curve_bootstrap.csv` yang terbit sebelum dipakai.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/build_nested_subsets_report.json` : jam terukur.
- `outputs/results/scaling_curve_bootstrap.csv` : pemeriksaan kesamaan.

Keluaran
--------
- `outputs/results/data_requirement_report.json`
- `outputs/results/data_requirement.csv`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u data_requirement.py 2>&1 | Tee-Object -FilePath logs/data_requirement.log
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
    load_configs,
    percentile_interval,
    sha256_of,
    write_csv,
    write_json,
)
from fit_scaling_curve import CURVE_FIELDS, NESTED, PRIMARY_DEFAULT, SPECS, bootstrap_fits, fit_best, start_points
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "data_requirement_report.json"
TABLE_NAME = "data_requirement.csv"
PUBLISHED_DRAWS = RESULTS_DIR / "scaling_curve_bootstrap.csv"
TARGETS = (0.40, 0.30, 0.25, 0.20)
MODELS = ("whisper", "xlsr")
MATCH_TOLERANCE = 1e-9


# --------------------------------------------------------------------------
# Perhitungan
# --------------------------------------------------------------------------


def from_curve(draws: np.ndarray, model: str, target: float) -> np.ndarray:
    """D* dari parameter kurva; NaN bila sasaran tidak tercapai oleh kurva."""
    e0 = draws[:, CURVE_FIELDS.index(f"E0_{model}")]
    a = draws[:, CURVE_FIELDS.index(f"a_{model}")]
    c = draws[:, CURVE_FIELDS.index(f"c_{model}")]
    with np.errstate(divide="ignore", invalid="ignore"):
        hours = (a / (target - e0)) ** (1 / c)
    hours[(target <= e0) | ~np.isfinite(hours)] = np.nan
    return hours


def from_interpolation(wer: np.ndarray, hours: np.ndarray, target: float) -> np.ndarray:
    """D* dari interpolasi linear WER terhadap log jam; perpotongan pertama, NaN bila tidak ada."""
    log_hours = np.log(hours)
    result = np.full(wer.shape[0], np.nan)
    for row in range(wer.shape[0]):
        for i in range(len(hours) - 1):
            w0, w1 = wer[row, i], wer[row, i + 1]
            if (w0 - target) * (w1 - target) <= 0 and w0 != w1:
                fraction = (w0 - target) / (w0 - w1)
                result[row] = float(np.exp(log_hours[i] + fraction * (log_hours[i + 1] - log_hours[i])))
                break
    return result


def summarise(point: float, samples: np.ndarray, low: float, high: float) -> dict[str, Any]:
    """Estimasi titik, selang atas iterasi di dalam rentang, dan porsi di luar rentang."""
    inside = samples[(samples >= low) & (samples <= high)]
    outside = float(1 - inside.size / samples.size)
    ci = list(percentile_interval(inside)) if inside.size else [float("nan"), float("nan")]
    return {"estimate": point, "estimate_in_range": bool(low <= point <= high) if np.isfinite(point) else False,
            "ci": ci, "share_out_of_range_or_unreached": outside}


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Kebutuhan jam data bagi WER sasaran.")
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
    print("Kebutuhan jam data bagi WER sasaran (Revisi 01 poin 3)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    hours = np.array([c.hours for c in configs[:len(LEVELS)]])
    low, high = float(hours.min()), float(hours.max())
    observed = np.array([c.words.rate for c in configs])
    wer_boot = bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                                np.column_stack([c.words.lengths for c in configs]), args.n_bootstrap, ANALYSIS_SEED)

    point_draws: dict[str, np.ndarray] = {}
    boot_draws: dict[str, np.ndarray] = {}
    for name in (PRIMARY_DEFAULT, NESTED):
        spec = SPECS[name]
        fit = fit_best(spec, observed, hours, start_points(spec, observed))
        point_draws[name] = np.array([spec.unpack(fit.params)])
        boot_draws[name], _, _ = bootstrap_fits(spec, wer_boot, hours, fit)

    matches = None
    if args.n_bootstrap == N_BOOTSTRAP:
        with PUBLISHED_DRAWS.open("r", encoding="utf-8", newline="") as handle:
            published = np.array([[float(row[f]) for f in CURVE_FIELDS] for row in csv.DictReader(handle)
                                  if row["model_spec"] == PRIMARY_DEFAULT])
        matches = bool(np.allclose(published, boot_draws[PRIMARY_DEFAULT], atol=MATCH_TOLERANCE, equal_nan=True))
        print(f"Bootstrap E0 bersama cocok dengan scaling_curve_bootstrap.csv: {matches}", flush=True)
        if not matches:
            return 1

    results: dict[str, Any] = {}
    rows: list[list[Any]] = []
    for method in ("shared_e0", "pure_power_law", "interpolation"):
        results[method] = {}
        for target in TARGETS:
            entry: dict[str, Any] = {}
            per_model: dict[str, np.ndarray] = {}
            for index, model in enumerate(MODELS):
                if method == "interpolation":
                    columns = slice(index * len(LEVELS), (index + 1) * len(LEVELS))
                    point = float(from_interpolation(observed[None, columns], hours, target)[0])
                    samples = from_interpolation(wer_boot[:, columns], hours, target)
                else:
                    point = float(from_curve(point_draws[method], model, target)[0])
                    samples = from_curve(boot_draws[method], model, target)
                per_model[model] = samples
                entry[model] = summarise(point, samples, low, high)
            ratio_samples = per_model["xlsr"] / per_model["whisper"]
            valid = np.all([(per_model[m] >= low) & (per_model[m] <= high) for m in MODELS], axis=0)
            ratio_point = entry["xlsr"]["estimate"] / entry["whisper"]["estimate"]
            entry["ratio_xlsr_to_whisper"] = {
                "estimate": ratio_point,
                "ci": list(percentile_interval(ratio_samples[valid])) if valid.any() else [float("nan")] * 2,
                "n_iterations_both_in_range": int(valid.sum()),
            }
            results[method][f"{int(round(100 * target))}"] = entry
            rows.append([method, target, entry["whisper"]["estimate"], *entry["whisper"]["ci"],
                         entry["xlsr"]["estimate"], *entry["xlsr"]["ci"], ratio_point, *entry["ratio_xlsr_to_whisper"]["ci"]])
            print(f"{method:15s} WER {100 * target:4.0f}%: Whisper {entry['whisper']['estimate']:6.2f} jam "
                  f"[{entry['whisper']['ci'][0]:.2f}; {entry['whisper']['ci'][1]:.2f}]  XLS-R "
                  f"{entry['xlsr']['estimate']:6.2f} [{entry['xlsr']['ci'][0]:.2f}; {entry['xlsr']['ci'][1]:.2f}]  "
                  f"rasio {ratio_point:.2f} [{entry['ratio_xlsr_to_whisper']['ci'][0]:.2f}; "
                  f"{entry['ratio_xlsr_to_whisper']['ci'][1]:.2f}]", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "targets": TARGETS,
        "hours_range": [low, high],
        "methods": {
            "shared_e0": "D* = (a / (sasaran - E0))^(1/c), kurva E0 bersama",
            "pure_power_law": "D* = (a / sasaran)^(1/c), power law murni",
            "interpolation": "interpolasi linear WER terhadap log jam, perpotongan pertama",
        },
        "matches_published_bootstrap": matches,
        "results": results,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
            PUBLISHED_DRAWS.name: sha256_of(PUBLISHED_DRAWS),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / TABLE_NAME,
              ["method", "target_wer", "whisper_hours", "whisper_ci_low", "whisper_ci_high", "xlsr_hours",
               "xlsr_ci_low", "xlsr_ci_high", "ratio", "ratio_ci_low", "ratio_ci_high"], rows)
    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
