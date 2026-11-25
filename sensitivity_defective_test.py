"""Sensitivitas hasil terhadap dua utterance data test yang cacat (Revisi 01 poin 32).

Tujuan
------
Dua utterance data test tidak sesuai dengan audionya (`DraftResearch.md`
Lampiran B.17, Keputusan Metodologis I.3 poin 17): `3d653a789b` rekamannya
terpotong sesudah tiga kata, dan `174723df08` memuat ucapan di luar
transkripnya. Keduanya tetap pada data test sebagai hasil utama. Skrip ini
menghitung ulang WER, CER, uji H1, dan kurva penskalaan tanpa keduanya, agar
pengaruhnya terhadap simpulan terukur.

Prosedur
--------
Seluruh langkah sama dengan `bootstrap_metrics.py` dan `fit_scaling_curve.py`,
dengan fungsi yang diimpor dari keduanya tanpa perubahan: *bootstrap*
multinomial tingkat utterance, seed 42, 10.000 iterasi, selang persentil 95%,
Holm atas nilai *p* dua sisi, pencocokan `E0` bersama, `E0` terpisah, dan
*power law* murni dengan aturan AIC berambang 2. Perhitungan dijalankan dua
kali, pada 3.100 utterance dan pada 3.098 utterance. Hasil pada 3.100
utterance dicocokkan dengan laporan `bootstrap_metrics_report.json` dan
`scaling_curve_report.json`, sehingga perbandingan bertumpu pada jalur yang
terbukti sama dengan hasil utama. Karena jumlah utterance berbeda, resample
kedua perhitungan tidak identik.

Utterance training bertranskrip pesan galat (`05e59eab6d`, Lampiran B.3) tidak
dapat diukur tanpa melatih ulang D5, yang tidak dikerjakan atas keputusan
pengguna (I.3 poin 20(a)); laporan menyatakannya terbuka.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/results/bootstrap_metrics_report.json` dan `scaling_curve_report.json`.
- `outputs/build_nested_subsets_report.json` : jam data training terukur.

Keluaran
--------
- `outputs/results/sensitivity_defective_test_report.json`
- `outputs/results/sensitivity_defective_test_configs.csv`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u sensitivity_defective_test.py 2>&1 |
        Tee-Object -FilePath logs/sensitivity_defective_test.log
"""

from __future__ import annotations

import argparse
import dataclasses
import json
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
    ConfigData,
    ErrorCounts,
    bootstrap_ratios,
    load_configs,
    sha256_of,
    write_csv,
    write_json,
)
from bootstrap_metrics import config_summaries, h1_tests
from fit_scaling_curve import (
    ALTERNATIVE,
    AIC_THRESHOLD,
    NESTED,
    PRIMARY_DEFAULT,
    SPECS,
    bootstrap_fits,
    bootstrap_summary,
    decisions,
    fit_best,
    full_fit_summary,
    start_points,
)
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "sensitivity_defective_test_report.json"
CONFIGS_TABLE_NAME = "sensitivity_defective_test_configs.csv"

# Lampiran B.17: rekaman terpotong dan rekaman berucapan tambahan.
EXCLUDED_IDS: tuple[str, ...] = ("3d653a789b", "174723df08")
TRAINING_DEFECT_ID = "05e59eab6d"

PUBLISHED_BOOTSTRAP = RESULTS_DIR / "bootstrap_metrics_report.json"
PUBLISHED_CURVE = RESULTS_DIR / "scaling_curve_report.json"

# Toleransi kecocokan dengan laporan terbit.
MATCH_TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def without(config: ConfigData, excluded: set[str]) -> ConfigData:
    """Konfigurasi yang sama tanpa utterance terpilih."""
    keep = np.array([utt not in excluded for utt in config.ids])

    def pick(values: Sequence[str]) -> tuple[str, ...]:
        return tuple(value for value, flag in zip(values, keep, strict=True) if flag)

    return dataclasses.replace(
        config,
        ids=pick(config.ids), references=pick(config.references),
        hypotheses=pick(config.hypotheses), hypotheses_raw=pick(config.hypotheses_raw),
        words=ErrorCounts(config.words.errors[keep], config.words.lengths[keep]),
        chars=ErrorCounts(config.chars.errors[keep], config.chars.lengths[keep]),
    )


def analyse(configs: Sequence[ConfigData], n_bootstrap: int) -> dict[str, Any]:
    """WER, CER, H1, dan kurva penskalaan dengan jalur hasil utama."""
    wer_boot = bootstrap_ratios(np.column_stack([c.words.errors for c in configs]),
                                np.column_stack([c.words.lengths for c in configs]), n_bootstrap, ANALYSIS_SEED)
    cer_boot = bootstrap_ratios(np.column_stack([c.chars.errors for c in configs]),
                                np.column_stack([c.chars.lengths for c in configs]), n_bootstrap, ANALYSIS_SEED)
    hours = np.array([c.hours for c in configs[:len(LEVELS)]])
    observed = np.array([c.words.rate for c in configs])

    fits = {}
    for name, spec in SPECS.items():
        best = fit_best(spec, observed, hours, start_points(spec, observed))
        if best is None:
            raise RuntimeError(f"Pencocokan {name} tidak konvergen")
        fits[name] = best
    full = {name: full_fit_summary(SPECS[name], fit, observed, hours, configs) for name, fit in fits.items()}
    delta = full[ALTERNATIVE]["aic"] - full[PRIMARY_DEFAULT]["aic"]
    primary = ALTERNATIVE if delta < -AIC_THRESHOLD else PRIMARY_DEFAULT
    boot = {}
    for name in (PRIMARY_DEFAULT, ALTERNATIVE):
        draws, retried, failed = bootstrap_fits(SPECS[name], wer_boot, hours, fits[name])
        boot[name] = bootstrap_summary(SPECS[name], fits[name], draws, retried, failed)
    return {
        "n_utterances": len(configs[0].ids),
        "configs": config_summaries(configs, wer_boot, cer_boot),
        "h1": h1_tests(configs, wer_boot, cer_boot),
        "aic": {
            "separate_minus_shared": delta,
            "nested_minus_shared": full[NESTED]["aic"] - full[PRIMARY_DEFAULT]["aic"],
            "primary_model": primary,
        },
        "full_sample": full,
        "bootstrap": boot,
        "decisions": decisions(primary, boot[primary]),
    }


def check_against_published(result: dict[str, Any]) -> list[str]:
    """Selisih hasil 3.100 utterance terhadap laporan terbit; kosong bila cocok."""
    problems = []
    with PUBLISHED_BOOTSTRAP.open("r", encoding="utf-8") as handle:
        published = json.load(handle)
    with PUBLISHED_CURVE.open("r", encoding="utf-8") as handle:
        curve = json.load(handle)
    pairs = []
    for mine, theirs in zip(result["configs"], published["configs"], strict=True):
        pairs += [(f"{mine['run_name']} wer", mine["wer"], theirs["wer"]),
                  (f"{mine['run_name']} wer_ci", mine["wer_ci"], theirs["wer_ci"])]
    for mine, theirs in zip(result["h1"], published["h1"]["tests"], strict=True):
        pairs += [(f"h1 {mine['level']} p_holm", mine["wer_diff_p_holm"], theirs["wer_diff_p_holm"]),
                  (f"h1 {mine['level']} ci", mine["wer_diff_ci"], theirs["wer_diff_ci"])]
    pairs.append(("aic separate-shared", result["aic"]["separate_minus_shared"], curve["aic"]["separate_minus_shared"]))
    for key in ("E0", "c_whisper", "c_xlsr", "c_whisper_minus_c_xlsr"):
        pairs.append((f"curve {key}", result["bootstrap"][PRIMARY_DEFAULT][key]["ci"],
                      curve["bootstrap"][PRIMARY_DEFAULT][key]["ci"]))
    for label, mine, theirs in pairs:
        if not np.allclose(np.asarray(mine, dtype=float), np.asarray(theirs, dtype=float),
                           rtol=0, atol=MATCH_TOLERANCE):
            problems.append(f"{label}: {mine} != {theirs}")
    return problems


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Sensitivitas terhadap dua utterance data test cacat.")
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
    print("Sensitivitas terhadap dua utterance data test cacat (Revisi 01 poin 32)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print(f"Dikeluarkan        : {', '.join(EXCLUDED_IDS)}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    missing = set(EXCLUDED_IDS) - set(configs[0].ids)
    if missing:
        print(f"GAGAL: utterance tidak ada pada data test: {sorted(missing)}", flush=True)
        return 1

    per_utterance: dict[str, dict[str, Any]] = {}
    for utt in EXCLUDED_IDS:
        per_utterance[utt] = {}
        for config in configs:
            index = config.ids.index(utt)
            per_utterance[utt][config.run_name] = {
                "reference_words": int(config.words.lengths[index]),
                "word_errors": int(config.words.errors[index]),
                "char_errors": int(config.chars.errors[index]),
                "hypothesis_words": len(config.hypotheses[index].split()),
            }
        print(f"{utt}: galat kata " + ", ".join(f"{name.split('_')[0][0]}{name.split('_')[1]} {item['word_errors']}"
                                                 for name, item in per_utterance[utt].items()), flush=True)

    full = analyse(configs, args.n_bootstrap)
    problems = check_against_published(full) if args.n_bootstrap == N_BOOTSTRAP else ["dilewati, uji coba"]
    print(f"Cocok dengan laporan terbit: {'ya' if not problems else problems}", flush=True)
    if problems and args.n_bootstrap == N_BOOTSTRAP:
        return 1
    reduced = analyse([without(c, set(EXCLUDED_IDS)) for c in configs], args.n_bootstrap)

    rows = []
    print("-" * 78, flush=True)
    for a, b in zip(full["configs"], reduced["configs"], strict=True):
        rows.append([a["run_name"], a["wer"], b["wer"], b["wer"] - a["wer"], a["cer"], b["cer"], b["cer"] - a["cer"],
                     *b["wer_ci"]])
        print(f"{a['run_name']:18s} WER {a['wer']:.4f} -> {b['wer']:.4f} ({b['wer'] - a['wer']:+.4f})  "
              f"CER {a['cer']:.4f} -> {b['cer']:.4f}", flush=True)
    for a, b in zip(full["h1"], reduced["h1"], strict=True):
        print(f"H1 {a['level']}: {a['wer_diff']:+.4f} -> {b['wer_diff']:+.4f} [{b['wer_diff_ci'][0]:+.4f}, "
              f"{b['wer_diff_ci'][1]:+.4f}] p_holm {b['wer_diff_p_holm']:.4f}; CER {a['cer_diff']:+.4f} -> "
              f"{b['cer_diff']:+.4f} [{b['cer_diff_ci'][0]:+.4f}, {b['cer_diff_ci'][1]:+.4f}]", flush=True)
    for key in ("E0", "c_whisper", "c_xlsr", "c_whisper_minus_c_xlsr"):
        a, b = full["bootstrap"][PRIMARY_DEFAULT][key], reduced["bootstrap"][PRIMARY_DEFAULT][key]
        print(f"{key:24s} {a['estimate']:+.4f} -> {b['estimate']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]",
              flush=True)
    print(f"AIC terpisah-bersama {full['aic']['separate_minus_shared']:+.3f} -> "
          f"{reduced['aic']['separate_minus_shared']:+.3f}; murni-bersama "
          f"{full['aic']['nested_minus_shared']:+.3f} -> {reduced['aic']['nested_minus_shared']:+.3f}", flush=True)
    print(f"Putusan penuh   : {full['decisions']}", flush=True)
    print(f"Putusan dikurangi: {reduced['decisions']}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "excluded_test_ids": EXCLUDED_IDS,
        "per_utterance_errors": per_utterance,
        "matches_published_reports": not problems,
        "training_defect": {
            "id": TRAINING_DEFECT_ID,
            "status": "tidak diukur; memerlukan pelatihan ulang D5, tidak dikerjakan (I.3 poin 20(a))",
        },
        "full": full,
        "without_defective": reduced,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
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
              ["run_name", "wer_full", "wer_without", "wer_change", "cer_full", "cer_without", "cer_change",
               "wer_without_ci_low", "wer_without_ci_high"], rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
