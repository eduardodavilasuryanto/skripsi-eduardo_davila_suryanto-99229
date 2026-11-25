"""Analisis penutur dikenal dan belum dikenal per tingkat data (Revisi 01 poin 12).

Tujuan
------
Subset bersarang menahan komposisi data agar tidak berganti antartingkat,
tetapi tidak menahan cakupan penutur: porsi utterance data test yang
penuturnya juga muncul pada data training naik dari 63,94% pada D1 menjadi
99,90% pada D5 (`build_nested_subsets_report.json`). Skrip ini memisahkan
pengaruh itu dari jumlah data dengan dua cara.

Prosedur
--------
1. **Dikenal dan belum dikenal per tingkat.** Pada setiap tingkat, utterance
   data test dibagi menurut apakah penuturnya muncul pada subset training
   tingkat itu. WER dan CER setiap model dihitung pada kedua kelompok, dengan
   selang persentil 95% dari *bootstrap* tingkat utterance di dalam kelompok,
   beserta selisih berpasangan Whisper dikurangi XLS-R. Porsi kelompok dikenal
   dicocokkan dengan `test_speakers_in_level_pct` laporan subset.
2. **Irisan tetap.** Utterance data test yang penuturnya sudah muncul pada D1.
   Karena subset bersarang, penutur ini dikenal pada kelima tingkat, sehingga
   keakraban penutur tetap di sepanjang grid. Pada irisan ini seluruh analisis
   utama dijalankan ulang dengan fungsi hasil utama: WER dan CER berselang, H1
   dengan Holm, kurva `E0` bersama, `E0` terpisah, dan *power law* murni, AIC,
   H2, dan H3 (seed 42, 10.000 iterasi).

Irisan tetap tidak meniadakan perubahan lain antartingkat, yaitu isi data
tambahan dan lama pelatihan, dan jumlah utterance per penutur pada data
training tetap bertambah. Yang ditahan hanya status dikenal atau belum.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/splits/test.tsv` dan `outputs/subsets/d1.tsv` sampai `d5.tsv`.
- `outputs/build_nested_subsets_report.json`.

Keluaran
--------
- `outputs/results/analysis_speaker_coverage_report.json`
- `outputs/results/analysis_speaker_coverage_groups.csv` : WER dan CER per
  tingkat, model, dan kelompok.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u analysis_speaker_coverage.py 2>&1 |
        Tee-Object -FilePath logs/analysis_speaker_coverage.log
"""

from __future__ import annotations

import argparse
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
    SCRIPT_DIR,
    SPLIT_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
    find_config,
    load_configs,
    percentile_interval,
    sha256_of,
    write_csv,
    write_json,
)
from asr_common import read_split
from fit_scaling_curve import PRIMARY_DEFAULT
from log_environment import collect_environment, set_seed
from sensitivity_defective_test import analyse, without

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "analysis_speaker_coverage_report.json"
GROUPS_TABLE_NAME = "analysis_speaker_coverage_groups.csv"
SUBSET_DIR = SCRIPT_DIR / "outputs" / "subsets"

# Porsi pada laporan subset ditulis dua desimal.
PCT_TOLERANCE = 0.005

# Kelompok dengan utterance lebih sedikit dari ini ditandai tidak layak ditafsirkan.
MIN_GROUP_UTTERANCES = 100


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def group_metrics(config: Any, mask: np.ndarray, n_bootstrap: int) -> dict[str, Any]:
    """WER dan CER satu konfigurasi pada satu kelompok utterance."""
    words_e, words_l = config.words.errors[mask], config.words.lengths[mask]
    chars_e, chars_l = config.chars.errors[mask], config.chars.lengths[mask]
    boot = bootstrap_ratios(np.column_stack([words_e, chars_e]), np.column_stack([words_l, chars_l]),
                            n_bootstrap, ANALYSIS_SEED)
    return {
        "n_utterances": int(mask.sum()),
        "wer": float(words_e.sum() / words_l.sum()),
        "wer_ci": list(percentile_interval(boot[:, 0])),
        "cer": float(chars_e.sum() / chars_l.sum()),
        "cer_ci": list(percentile_interval(boot[:, 1])),
    }


def paired_difference(first: Any, second: Any, mask: np.ndarray, n_bootstrap: int) -> dict[str, Any]:
    """Selisih WER berpasangan dua konfigurasi pada satu kelompok."""
    boot = bootstrap_ratios(np.column_stack([first.words.errors[mask], second.words.errors[mask]]),
                            np.column_stack([first.words.lengths[mask], second.words.lengths[mask]]),
                            n_bootstrap, ANALYSIS_SEED)
    estimate = (first.words.errors[mask].sum() / first.words.lengths[mask].sum()
                - second.words.errors[mask].sum() / second.words.lengths[mask].sum())
    return {"wer_diff": float(estimate), "wer_diff_ci": list(percentile_interval(boot[:, 0] - boot[:, 1]))}


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Analisis penutur dikenal dan belum dikenal per tingkat.")
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
    print("Penutur dikenal dan belum dikenal per tingkat (Revisi 01 poin 12)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    speaker_of = {utt.utt_id: utt.speaker_id for utt in read_split(SPLIT_DIR / "test.tsv")}
    test_speakers = np.array([speaker_of[utt] for utt in configs[0].ids])
    with SUBSETS_REPORT.open("r", encoding="utf-8") as handle:
        subsets_report = json.load(handle)["levels"]

    seen_masks: dict[str, np.ndarray] = {}
    checks: dict[str, Any] = {}
    for level in LEVELS:
        speakers = {utt.speaker_id for utt in read_split(SUBSET_DIR / f"{level}.tsv")}
        seen_masks[level] = np.array([speaker in speakers for speaker in test_speakers])
        pct = 100 * seen_masks[level].mean()
        reported = subsets_report[level]["test_speakers_in_level_pct"]
        checks[level] = {"seen_pct": pct, "reported_pct": reported, "match": abs(pct - reported) <= PCT_TOLERANCE}
    if not all(item["match"] for item in checks.values()):
        print(f"GAGAL: porsi dikenal tidak cocok dengan laporan subset: {checks}", flush=True)
        return 1
    for level in LEVELS[1:]:
        previous = LEVELS[LEVELS.index(level) - 1]
        if np.any(seen_masks[previous] & ~seen_masks[level]):
            print("GAGAL: penutur dikenal pada tingkat lebih kecil tidak dikenal pada tingkat lebih besar", flush=True)
            return 1
    print("Porsi dikenal cocok dengan laporan subset, dan bersarang.", flush=True)

    groups: dict[str, Any] = {}
    rows: list[list[Any]] = []
    print("-" * 78, flush=True)
    for level in LEVELS:
        groups[level] = {}
        for group_name, mask in (("dikenal", seen_masks[level]), ("belum_dikenal", ~seen_masks[level])):
            entry: dict[str, Any] = {"n_utterances": int(mask.sum()),
                                     "n_speakers": int(len(set(test_speakers[mask]))),
                                     "interpretable": bool(mask.sum() >= MIN_GROUP_UTTERANCES)}
            if mask.sum() > 0:
                for model in ("whisper", "xlsr"):
                    config = configs[find_config(configs, model, level)]
                    entry[model] = group_metrics(config, mask, args.n_bootstrap)
                    rows.append([level, group_name, model, entry["n_utterances"], entry["n_speakers"],
                                 entry[model]["wer"], *entry[model]["wer_ci"], entry[model]["cer"],
                                 *entry[model]["cer_ci"]])
                entry["h1"] = paired_difference(configs[find_config(configs, "whisper", level)],
                                                configs[find_config(configs, "xlsr", level)], mask, args.n_bootstrap)
            groups[level][group_name] = entry
            if mask.sum() > 0:
                print(f"{level} {group_name:13s} n={entry['n_utterances']:5d}  WER whisper {entry['whisper']['wer']:.4f}"
                      f"  xlsr {entry['xlsr']['wer']:.4f}  selisih {entry['h1']['wer_diff']:+.4f} "
                      f"[{entry['h1']['wer_diff_ci'][0]:+.4f}, {entry['h1']['wer_diff_ci'][1]:+.4f}]", flush=True)

    slice_ids = {utt for utt, flag in zip(configs[0].ids, seen_masks["d1"], strict=True) if not flag}
    slice_configs = [without(config, slice_ids) for config in configs]
    print("-" * 78, flush=True)
    print(f"Irisan tetap (penutur dikenal sejak D1): {len(slice_configs[0].ids)} utterance, "
          f"{len(set(test_speakers[seen_masks['d1']]))} penutur", flush=True)
    fixed = analyse(slice_configs, args.n_bootstrap)
    full = analyse(configs, args.n_bootstrap)
    for a, b in zip(full["configs"], fixed["configs"], strict=True):
        print(f"  {a['run_name']:18s} WER penuh {a['wer']:.4f}  irisan {b['wer']:.4f}", flush=True)
    for b in fixed["h1"]:
        print(f"  H1 {b['level']}: {b['wer_diff']:+.4f} [{b['wer_diff_ci'][0]:+.4f}, {b['wer_diff_ci'][1]:+.4f}] "
              f"p_holm {b['wer_diff_p_holm']:.4f}", flush=True)
    for key in ("E0", "c_whisper", "c_xlsr", "c_whisper_minus_c_xlsr"):
        a, b = full["bootstrap"][PRIMARY_DEFAULT][key], fixed["bootstrap"][PRIMARY_DEFAULT][key]
        print(f"  {key:24s} penuh {a['estimate']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}]  irisan "
              f"{b['estimate']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]", flush=True)
    print(f"  AIC terpisah-bersama {fixed['aic']['separate_minus_shared']:+.3f}, murni-bersama "
          f"{fixed['aic']['nested_minus_shared']:+.3f}", flush=True)
    print(f"  Putusan irisan: {fixed['decisions']}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "seen_definition": "penutur utterance data test muncul pada subset training tingkat itu",
        "checks_against_subset_report": checks,
        "min_group_utterances_for_interpretation": MIN_GROUP_UTTERANCES,
        "groups_per_level": groups,
        "fixed_slice": {
            "definition": "utterance data test yang penuturnya sudah muncul pada D1, sehingga dikenal pada kelima tingkat",
            "n_utterances": len(slice_configs[0].ids),
            "n_speakers": int(len(set(test_speakers[seen_masks["d1"]]))),
            "n_reference_words": int(slice_configs[0].words.lengths.sum()),
            "analysis": fixed,
        },
        "full_test_reference": {"configs": full["configs"], "h1": full["h1"], "aic": full["aic"],
                                "bootstrap": full["bootstrap"], "decisions": full["decisions"]},
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            "test.tsv": sha256_of(SPLIT_DIR / "test.tsv"),
            **{f"{level}.tsv": sha256_of(SUBSET_DIR / f"{level}.tsv") for level in LEVELS},
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / GROUPS_TABLE_NAME,
              ["level", "group", "model", "n_utterances", "n_speakers", "wer", "wer_ci_low", "wer_ci_high",
               "cer", "cer_ci_low", "cer_ci_high"], rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
