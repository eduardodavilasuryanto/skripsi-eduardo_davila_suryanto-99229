"""Uji sensitivitas normalisasi: WER dan CER pada teks yang mempertahankan diakritik.

Tujuan
------
Menghasilkan keluaran 7 Subbab 3.6.4 skripsi, yaitu uji sensitivitas atas
keputusan pelepasan diakritik (Subbab 3.3.4),
untuk menunjukkan besaran pengaruh keputusan normalisasi itu terhadap WER dan
CER setiap konfigurasi.

Normalisasi pembanding
----------------------
Langkah 1, 2, 4, dan 6 preprocessing teks dijalankan tanpa langkah 3:
normalisasi NFC, huruf kecil, whitespace dan garis miring menjadi spasi,
penghapusan seluruh karakter selain huruf dan tanda diakritik (kategori
Unicode L dan M), lalu perapatan spasi. Huruf beraksen dan huruf di luar 26
huruf Latin karena itu dipertahankan. Fungsi ini sengaja ditulis di skrip ini
dan bukan di `asr_common.py`, yang dikunci sejak titik pembekuan.

Referensi diambil dari transkrip mentah `outputs/splits/test.tsv` dan
hipotesis dari kolom `hypothesis_raw`, yaitu keluaran dekoder sebelum
normalisasi, lalu keduanya dinormalisasi dengan fungsi pembanding. Model
dilatih pada label tanpa diakritik, sehingga hasil uji ini menunjukkan
seberapa jauh angka utama bergantung pada keputusan normalisasi, bukan
kemampuan model mengenali diakritik.

Prosedur bootstrap sama dengan `bootstrap_metrics.py` (seed 42, resample yang
sama); selisih terhadap hasil utama dihitung pada resample yang sama.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/splits/test.tsv` : transkrip referensi mentah.
- `outputs/results/<run>_test_metrics.json`, hanya untuk pemeriksaan kesamaan.

Keluaran
--------
- `outputs/results/robustness_diacritics_report.json` : laporan lengkap.
- `outputs/results/robustness_diacritics.csv` : tabel per konfigurasi.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u robustness_diacritics.py 2>&1 |
        Tee-Object -FilePath logs/robustness_diacritics.log
"""

from __future__ import annotations

import argparse
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    CONFIDENCE,
    MODELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SPLIT_DIR,
    SUBSETS_REPORT,
    bootstrap_ratios,
    load_configs,
    percentile_interval,
    sha256_of,
    utterance_counts,
    write_csv,
    write_json,
)
from asr_common import WORD_SEPARATORS, read_split
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "robustness_diacritics_report.json"
TABLE_NAME = "robustness_diacritics.csv"
TEST_SPLIT = SPLIT_DIR / "test.tsv"

# Kategori Unicode yang dipertahankan: huruf (L) dan tanda diakritik (M).
KEPT_CATEGORY_PREFIXES = ("L", "M")


# --------------------------------------------------------------------------
# Normalisasi pembanding
# --------------------------------------------------------------------------


def normalize_keep_diacritics(text: str) -> str:
    """Langkah 1, 2, 4, dan 6 Subbab 3.3.4 tanpa pelepasan diakritik."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if char.isspace() or char in WORD_SEPARATORS else char for char in text)
    text = "".join(
        char for char in text
        if char == " " or unicodedata.category(char).startswith(KEPT_CATEGORY_PREFIXES)
    )
    return " ".join(text.split())


def non_ascii_letters(texts: Sequence[str]) -> Counter[str]:
    """Cacah karakter non-ASCII yang tersisa pada teks hasil normalisasi pembanding."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(char for char in text if ord(char) > 127)
    return counter


def char_name(char: str) -> str:
    """Nama Unicode sebuah karakter, agar laporan dapat dibaca tanpa font khusus."""
    return f"U+{ord(char):04X} {unicodedata.name(char, '?')}"


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Uji ketegaran normalisasi diakritik.")
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
    print("Uji ketegaran: teks yang mempertahankan diakritik", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED}", flush=True)
    print(f"Iterasi bootstrap  : {args.n_bootstrap}", flush=True)
    print("=" * 78, flush=True)

    configs = load_configs()
    raw_references = {utt.utt_id: utt.transcript for utt in read_split(TEST_SPLIT)}
    if set(raw_references) != set(configs[0].ids):
        print("GAGAL: id test.tsv berbeda dengan id berkas hipotesis.", flush=True)
        return 1
    references = [normalize_keep_diacritics(raw_references[utt_id]) for utt_id in configs[0].ids]
    if any(not reference for reference in references):
        print("GAGAL: referensi kosong setelah normalisasi pembanding.", flush=True)
        return 1
    changed = sum(1 for kept, main_ref in zip(references, configs[0].references, strict=True) if kept != main_ref)
    reference_chars = non_ascii_letters(references)
    print(f"Referensi berubah  : {changed} dari {len(references)} utterance berbeda dengan normalisasi utama",
          flush=True)

    robust_words, robust_chars, hypothesis_chars = [], [], {}
    for config in configs:
        hypotheses = [normalize_keep_diacritics(text) for text in config.hypotheses_raw]
        robust_words.append(utterance_counts(references, hypotheses, "word"))
        robust_chars.append(utterance_counts(references, hypotheses, "char"))
        hypothesis_chars[config.run_name] = non_ascii_letters(hypotheses)

    k = len(configs)
    numerators = np.column_stack([c.errors for c in robust_words] + [c.errors for c in robust_chars]
                                 + [c.words.errors for c in configs] + [c.chars.errors for c in configs])
    denominators = np.column_stack([c.lengths for c in robust_words] + [c.lengths for c in robust_chars]
                                   + [c.words.lengths for c in configs] + [c.chars.lengths for c in configs])
    samples = bootstrap_ratios(numerators, denominators, args.n_bootstrap, ANALYSIS_SEED)
    robust_wer, robust_cer = samples[:, :k], samples[:, k:2 * k]
    main_wer, main_cer = samples[:, 2 * k:3 * k], samples[:, 3 * k:]

    rows: list[dict[str, Any]] = []
    print("-" * 78, flush=True)
    for index, config in enumerate(configs):
        row = {
            "model": config.model,
            "level": config.level,
            "run_name": config.run_name,
            "hours": config.hours,
            "wer_main": config.words.rate,
            "wer_kept": robust_words[index].rate,
            "wer_kept_ci": list(percentile_interval(robust_wer[:, index])),
            "wer_kept_minus_main": robust_words[index].rate - config.words.rate,
            "wer_kept_minus_main_ci": list(percentile_interval(robust_wer[:, index] - main_wer[:, index])),
            "cer_main": config.chars.rate,
            "cer_kept": robust_chars[index].rate,
            "cer_kept_ci": list(percentile_interval(robust_cer[:, index])),
            "cer_kept_minus_main": robust_chars[index].rate - config.chars.rate,
            "cer_kept_minus_main_ci": list(percentile_interval(robust_cer[:, index] - main_cer[:, index])),
            "hypothesis_non_ascii_letters": {char_name(ch): n for ch, n in hypothesis_chars[config.run_name].most_common()},
        }
        rows.append(row)
        print(f"{row['run_name']:18s} WER {row['wer_main']:.4f} -> {row['wer_kept']:.4f} "
              f"({row['wer_kept_minus_main']:+.4f})  CER {row['cer_main']:.4f} -> {row['cer_kept']:.4f} "
              f"({row['cer_kept_minus_main']:+.4f})", flush=True)
    for model in MODELS:
        letters = sum(sum(hypothesis_chars[c.run_name].values()) for c in configs if c.model == model)
        print(f"Huruf non-ASCII pada hipotesis {model}: {letters}", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": args.n_bootstrap,
        "confidence": CONFIDENCE,
        "interval_method": "persentil",
        "normalization": "langkah 1, 2, 4, dan 6 Subbab 3.3.4 tanpa langkah 3; huruf dan tanda diakritik "
                         "(kategori Unicode L dan M) dipertahankan",
        "reference_source": f"{TEST_SPLIT.name}, kolom transcript",
        "hypothesis_source": "kolom hypothesis_raw berkas hipotesis",
        "n_references_changed": changed,
        "reference_non_ascii_letters": {char_name(ch): n for ch, n in reference_chars.most_common()},
        "configs": rows,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            TEST_SPLIT.name: sha256_of(TEST_SPLIT),
            SUBSETS_REPORT.name: sha256_of(SUBSETS_REPORT),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / TABLE_NAME,
              ["model", "level", "run_name", "hours", "wer_main", "wer_kept", "wer_kept_ci_low",
               "wer_kept_ci_high", "wer_kept_minus_main", "wer_diff_ci_low", "wer_diff_ci_high",
               "cer_main", "cer_kept", "cer_kept_ci_low", "cer_kept_ci_high", "cer_kept_minus_main",
               "cer_diff_ci_low", "cer_diff_ci_high"],
              [[r["model"], r["level"], r["run_name"], r["hours"], r["wer_main"], r["wer_kept"], *r["wer_kept_ci"],
                r["wer_kept_minus_main"], *r["wer_kept_minus_main_ci"], r["cer_main"], r["cer_kept"],
                *r["cer_kept_ci"], r["cer_kept_minus_main"], *r["cer_kept_minus_main_ci"]] for r in rows])

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
