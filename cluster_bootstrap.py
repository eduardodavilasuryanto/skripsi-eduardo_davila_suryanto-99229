"""Sensitivitas inferensi terhadap *bootstrap* berkelompok penutur dan prompt (Revisi 01 poin 22).

Tujuan
------
*Bootstrap* hasil utama menarik ulang utterance satu per satu, sehingga
menganggap utterance cukup saling bebas. Pada data test 3.100 utterance,
satu penutur dapat menyumbang beberapa utterance dan satu prompt dapat
dibacakan beberapa kali, sehingga galat di dalam satu penutur atau satu
prompt mungkin berkorelasi. Skrip ini mengulang inferensi utama dengan
*bootstrap* yang menarik ulang kelompok utuh.

Prosedur
--------
Dua definisi kelompok: `penutur` (`speaker_id`) dan `prompt` (transkrip
mentah, yaitu kunci pengelompokan pembagian *text-disjoint*). Pada setiap
iterasi, sebanyak K kelompok ditarik berpengembalian dari K kelompok yang ada
sebagai cacah multinomial, dan setiap utterance memperoleh bobot sebesar cacah
kelompoknya. WER dan CER tingkat corpus dihitung sebagai jumlah galat berbobot
dibagi jumlah panjang referensi berbobot, sama seperti `analysis_common`.
Resample dibangkitkan dengan seed 42 dalam blok 500 dan dipakai bersama oleh
kesepuluh konfigurasi, sehingga pasangan *checkpoint* yang sama dipertahankan
pada setiap resample dan selisih H1 tetap berpasangan.

Hasil resample itu diteruskan tanpa perubahan ke fungsi hasil utama:
`bootstrap_metrics.config_summaries` dan `h1_tests` (selang persentil 95%,
nilai *p* dua sisi, Holm) serta pencocokan ulang kurva `fit_scaling_curve`
bagi model `E0` bersama dan `E0` terpisah. Lebar selang dibandingkan dengan
laporan *bootstrap* tingkat utterance yang sudah terbit.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/splits/test.tsv` : penutur dan transkrip mentah.
- `outputs/results/bootstrap_metrics_report.json` dan `scaling_curve_report.json`.

Keluaran
--------
- `outputs/results/cluster_bootstrap_report.json`
- `outputs/results/cluster_bootstrap_configs.csv`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u cluster_bootstrap.py 2>&1 | Tee-Object -FilePath logs/cluster_bootstrap.log
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    BOOTSTRAP_BLOCK,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SPLIT_DIR,
    SUBSETS_REPORT,
    load_configs,
    sha256_of,
    write_csv,
    write_json,
)
from asr_common import read_split
from bootstrap_metrics import config_summaries, h1_tests
from fit_scaling_curve import (
    ALTERNATIVE,
    PRIMARY_DEFAULT,
    SPECS,
    bootstrap_fits,
    bootstrap_summary,
    decisions,
    fit_best,
    start_points,
)
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "cluster_bootstrap_report.json"
CONFIGS_TABLE_NAME = "cluster_bootstrap_configs.csv"
PUBLISHED_BOOTSTRAP = RESULTS_DIR / "bootstrap_metrics_report.json"
PUBLISHED_CURVE = RESULTS_DIR / "scaling_curve_report.json"
CURVE_KEYS = ("E0", "a_whisper", "c_whisper", "a_xlsr", "c_xlsr", "c_whisper_minus_c_xlsr")


# --------------------------------------------------------------------------
# Bootstrap berkelompok
# --------------------------------------------------------------------------


def cluster_index(labels: Sequence[str]) -> tuple[np.ndarray, int]:
    """Indeks kelompok setiap utterance, dengan kelompok diurutkan menurut label."""
    names = sorted(set(labels))
    position = {name: index for index, name in enumerate(names)}
    return np.array([position[label] for label in labels]), len(names)


def cluster_bootstrap_ratios(numerators: np.ndarray, denominators: np.ndarray, clusters: np.ndarray,
                             n_clusters: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Rasio jumlah berbobot setiap kolom, dengan bobot utterance dari cacah kelompoknya."""
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    generator = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, numerators.shape[1]), dtype=np.float64)
    for start in range(0, n_bootstrap, BOOTSTRAP_BLOCK):
        size = min(BOOTSTRAP_BLOCK, n_bootstrap - start)
        counts = generator.multinomial(n_clusters, probabilities, size=size)
        weights = counts[:, clusters]
        output[start:start + size] = (weights @ numerators) / (weights @ denominators)
    return output


def analyse(configs: Sequence[Any], clusters: np.ndarray, n_clusters: int, n_bootstrap: int) -> dict[str, Any]:
    """Inferensi utama dengan resample berkelompok."""
    wer_boot = cluster_bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                                        np.column_stack([c.words.lengths for c in configs]),
                                        clusters, n_clusters, n_bootstrap, ANALYSIS_SEED)
    cer_boot = cluster_bootstrap_ratios(np.column_stack([c.chars.errors for c in configs]),
                                        np.column_stack([c.chars.lengths for c in configs]),
                                        clusters, n_clusters, n_bootstrap, ANALYSIS_SEED)
    hours = np.array([c.hours for c in configs[:len(LEVELS)]])
    observed = np.array([c.words.rate for c in configs])
    boot = {}
    for name in (PRIMARY_DEFAULT, ALTERNATIVE):
        fit = fit_best(SPECS[name], observed, hours, start_points(SPECS[name], observed))
        if fit is None:
            raise RuntimeError(f"Pencocokan {name} tidak konvergen")
        draws, retried, failed = bootstrap_fits(SPECS[name], wer_boot, hours, fit)
        boot[name] = bootstrap_summary(SPECS[name], fit, draws, retried, failed)
    return {
        "n_clusters": n_clusters,
        "configs": config_summaries(configs, wer_boot, cer_boot),
        "h1": h1_tests(configs, wer_boot, cer_boot),
        "bootstrap": boot,
        "decisions": decisions(PRIMARY_DEFAULT, boot[PRIMARY_DEFAULT]),
    }


def width(interval: Sequence[float]) -> float:
    """Lebar selang."""
    return float(interval[1] - interval[0])


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Bootstrap berkelompok penutur dan prompt.")
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
    print("Bootstrap berkelompok penutur dan prompt (Revisi 01 poin 22)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    test = {utt.utt_id: utt for utt in read_split(SPLIT_DIR / "test.tsv")}
    labels = {
        "penutur": [test[utt].speaker_id for utt in configs[0].ids],
        "prompt": [test[utt].transcript for utt in configs[0].ids],
    }
    with PUBLISHED_BOOTSTRAP.open("r", encoding="utf-8") as handle:
        published = json.load(handle)
    with PUBLISHED_CURVE.open("r", encoding="utf-8") as handle:
        curve = json.load(handle)["bootstrap"][PRIMARY_DEFAULT]

    structure = {}
    for name, values in labels.items():
        sizes = collections.Counter(values)
        structure[name] = {
            "n_clusters": len(sizes),
            "max_utterances_per_cluster": max(sizes.values()),
            "mean_utterances_per_cluster": len(values) / len(sizes),
            "n_clusters_with_one_utterance": sum(1 for size in sizes.values() if size == 1),
        }
        print(f"{name:8s}: {structure[name]['n_clusters']} kelompok, rerata "
              f"{structure[name]['mean_utterances_per_cluster']:.2f}, maks "
              f"{structure[name]['max_utterances_per_cluster']} utterance", flush=True)

    results: dict[str, Any] = {}
    rows: list[list[Any]] = []
    for name, values in labels.items():
        clusters, n_clusters = cluster_index(values)
        result = analyse(configs, clusters, n_clusters, args.n_bootstrap)
        comparisons = []
        for mine, theirs in zip(result["configs"], published["configs"], strict=True):
            ratio = width(mine["wer_ci"]) / width(theirs["wer_ci"])
            comparisons.append({"run_name": mine["run_name"], "wer_ci": mine["wer_ci"],
                                "wer_ci_utterance": theirs["wer_ci"], "width_ratio": ratio})
            rows.append([name, mine["run_name"], mine["wer"], *mine["wer_ci"], *theirs["wer_ci"], ratio,
                         *mine["cer_ci"]])
        h1_comparisons = []
        for mine, theirs in zip(result["h1"], published["h1"]["tests"], strict=True):
            h1_comparisons.append({"level": mine["level"], "wer_diff_ci": mine["wer_diff_ci"],
                                   "wer_diff_ci_utterance": theirs["wer_diff_ci"],
                                   "width_ratio": width(mine["wer_diff_ci"]) / width(theirs["wer_diff_ci"]),
                                   "p_holm": mine["wer_diff_p_holm"],
                                   "wer_diff_ci_holm_level": mine["wer_diff_ci_holm_level"],
                                   "cer_diff_ci": mine["cer_diff_ci"],
                                   "cer_diff_ci_utterance": theirs["cer_diff_ci"]})
        curve_comparisons = {key: {"ci": result["bootstrap"][PRIMARY_DEFAULT][key]["ci"],
                                   "ci_utterance": curve[key]["ci"],
                                   "width_ratio": width(result["bootstrap"][PRIMARY_DEFAULT][key]["ci"])
                                   / width(curve[key]["ci"])}
                             for key in CURVE_KEYS}
        result["comparison_with_utterance_bootstrap"] = {
            "configs": comparisons, "h1": h1_comparisons, "curve_shared_e0": curve_comparisons,
        }
        results[name] = result

        print("-" * 78, flush=True)
        ratios = [item["width_ratio"] for item in comparisons]
        print(f"[{name}] lebar selang WER terhadap tingkat utterance: {min(ratios):.2f} sampai {max(ratios):.2f} kali",
              flush=True)
        for item in h1_comparisons:
            print(f"[{name}] H1 {item['level']}: [{item['wer_diff_ci'][0]:+.4f}, {item['wer_diff_ci'][1]:+.4f}] "
                  f"(utterance [{item['wer_diff_ci_utterance'][0]:+.4f}, {item['wer_diff_ci_utterance'][1]:+.4f}]) "
                  f"p_holm {item['p_holm']:.4f}; CER [{item['cer_diff_ci'][0]:+.4f}, {item['cer_diff_ci'][1]:+.4f}]",
                  flush=True)
        for key, item in curve_comparisons.items():
            print(f"[{name}] {key:24s} [{item['ci'][0]:+.4f}, {item['ci'][1]:+.4f}] "
                  f"(utterance [{item['ci_utterance'][0]:+.4f}, {item['ci_utterance'][1]:+.4f}])", flush=True)
        print(f"[{name}] putusan: {result['decisions']}", flush=True)

    with PUBLISHED_CURVE.open("r", encoding="utf-8") as handle:
        points = json.load(handle)["full_sample"][PRIMARY_DEFAULT]["points"]
    utterance_half = {item["run_name"]: item["wer_ci_half_width"] for item in published["configs"]}
    residuals = []
    print("-" * 78, flush=True)
    for point in points:
        entry = {"run_name": point["run_name"], "residual": point["residual"],
                 "half_width_utterance": utterance_half[point["run_name"]]}
        for name in labels:
            half = next(c["wer_ci_half_width"] for c in results[name]["configs"] if c["run_name"] == point["run_name"])
            entry[f"half_width_{name}"] = half
        for unit in ("utterance", *labels):
            entry[f"exceeds_twice_half_width_{unit}"] = abs(point["residual"]) > 2 * entry[f"half_width_{unit}"]
        residuals.append(entry)
        print(f"residual {point['run_name']:18s} {point['residual']:+.4f}  2x setengah lebar: utterance "
              f"{2 * entry['half_width_utterance']:.4f}, penutur {2 * entry['half_width_penutur']:.4f}, "
              f"prompt {2 * entry['half_width_prompt']:.4f}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "residual_vs_interval": {
            "definition": "residual kurva E0 bersama titik penuh terhadap dua kali setengah lebar selang WER 95% titik",
            "points": residuals,
        },
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "resampling": "cacah multinomial atas kelompok, bobot utterance = cacah kelompoknya, blok 500",
        "cluster_definitions": {"penutur": "speaker_id pada test.tsv",
                                "prompt": "transkrip mentah, kunci pengelompokan pembagian text-disjoint"},
        "n_test_utterances": len(configs[0].ids),
        "cluster_structure": structure,
        "results": results,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            "test.tsv": sha256_of(SPLIT_DIR / "test.tsv"),
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
            PUBLISHED_BOOTSTRAP.name: sha256_of(PUBLISHED_BOOTSTRAP),
            PUBLISHED_CURVE.name: sha256_of(PUBLISHED_CURVE),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / CONFIGS_TABLE_NAME,
              ["cluster", "run_name", "wer", "wer_ci_low", "wer_ci_high", "wer_ci_low_utterance",
               "wer_ci_high_utterance", "width_ratio", "cer_ci_low", "cer_ci_high"], rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
