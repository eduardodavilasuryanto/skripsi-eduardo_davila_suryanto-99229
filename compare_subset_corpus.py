"""Bukti mekanisme partisi arsip dan perbandingan subset kerja dengan corpus penuh (Revisi 01 poin 11).

Tujuan
------
Naskah menyatakan pengambilan empat arsip pertama SLR35 setara dengan sampel
acak seragam 25% karena partisi antararsip berbasis *hash* identifier
utterance. Skrip ini memeriksa klaim itu dari dua arah: mekanisme partisinya,
dan kesamaan sebaran metadata subset kerja dengan corpus penuh.

Prosedur
--------
1. **Mekanisme.** Untuk setiap berkas FLAC pada keempat arsip di disk,
   diperiksa apakah karakter heksadesimal pertama identifier utterance sama
   dengan nomor arsipnya. Sebaran karakter pertama dan kedua identifier pada
   metadata corpus penuh diuji terhadap sebaran seragam 16 kategori dengan uji
   khi-kuadrat, dan kebebasan karakter pertama dari penutur diperiksa lewat
   porsi penutur yang utterance-nya tersebar ke lebih dari satu arsip.
2. **Sebaran metadata.** Subset kerja (karakter pertama 0 sampai 3) dibandingkan
   dengan corpus penuh pada panjang transkrip dalam karakter dan kata, jumlah
   utterance per penutur, dan jumlah pembacaan setiap kalimat pada corpus penuh.
   Ukuran perbedaannya statistik Kolmogorov-Smirnov dua sampel antara subset
   kerja dan dua belas arsip lainnya, beserta selisih rerata dan median. Dengan
   puluhan ribu utterance, uji ini peka terhadap perbedaan yang sangat kecil,
   sehingga yang ditafsirkan adalah besar statistiknya, bukan nilai *p* semata.
   Durasi tidak dibandingkan, karena berkas audio corpus penuh tidak ada di disk.

Masukan
-------
- `_03_dataset/asr_javanese_0/utt_spk_text.tsv` : metadata corpus penuh.
- `_03_dataset/asr_javanese_{0..3}/data/**.flac` : keanggotaan arsip.

Keluaran
--------
- `outputs/compare_subset_corpus_report.json`
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u compare_subset_corpus.py 2>&1 | Tee-Object -FilePath logs/compare_subset_corpus.log
"""

from __future__ import annotations

import argparse
import collections
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import stats

from audit_transcript_leakage import DEFAULT_DATASET_ROOT, DEFAULT_METADATA, WORKING_ARCHIVES, load_metadata
from analysis_common import sha256_of, write_json
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_PATH = Path(__file__).resolve().parent / "outputs" / "compare_subset_corpus_report.json"
HEX = "0123456789abcdef"
SUBSET_PREFIXES = frozenset("0123")
SEED = 42


# --------------------------------------------------------------------------
# Analisis
# --------------------------------------------------------------------------


def chi_square_uniform(values: Sequence[str]) -> dict[str, Any]:
    """Uji khi-kuadrat sebaran 16 karakter heksadesimal terhadap sebaran seragam."""
    counts = collections.Counter(values)
    observed = np.array([counts.get(char, 0) for char in HEX])
    result = stats.chisquare(observed)
    return {"counts": dict(zip(HEX, map(int, observed), strict=True)), "chi2": float(result.statistic),
            "df": 15, "p_value": float(result.pvalue),
            "max_share_pct": float(100 * observed.max() / observed.sum()),
            "min_share_pct": float(100 * observed.min() / observed.sum())}


def compare(name: str, subset: np.ndarray, rest: np.ndarray, full: np.ndarray) -> dict[str, Any]:
    """Ringkasan dan uji KS satu besaran antara subset kerja dan arsip lainnya."""
    ks = stats.ks_2samp(subset, rest)
    summary = {
        "subset_mean": float(subset.mean()), "full_mean": float(full.mean()), "rest_mean": float(rest.mean()),
        "subset_median": float(np.median(subset)), "full_median": float(np.median(full)),
        "relative_mean_difference_pct": float(100 * (subset.mean() - full.mean()) / full.mean()),
        "ks_statistic_subset_vs_rest": float(ks.statistic), "ks_p_value": float(ks.pvalue),
        "n_subset": int(subset.size), "n_rest": int(rest.size),
    }
    print(f"{name:34s} rerata subset {summary['subset_mean']:8.3f}  penuh {summary['full_mean']:8.3f}  "
          f"selisih {summary['relative_mean_difference_pct']:+6.2f}%  KS {summary['ks_statistic_subset_vs_rest']:.4f} "
          f"(p {summary['ks_p_value']:.3g})", flush=True)
    return summary


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Mekanisme partisi arsip dan perbandingan subset kerja.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="Berkas laporan JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    set_seed(SEED)

    print("=" * 78, flush=True)
    print("Mekanisme partisi arsip dan perbandingan subset kerja (Revisi 01 poin 11)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print("=" * 78, flush=True)

    rows = load_metadata(DEFAULT_METADATA)
    ids = [row.utt_id for row in rows]
    in_subset = np.array([utt[0] in SUBSET_PREFIXES for utt in ids])

    membership: dict[str, Any] = {}
    for archive in WORKING_ARCHIVES:
        files = list((DEFAULT_DATASET_ROOT / archive / "data").rglob("*.flac"))
        expected = archive[-1]
        mismatched = [f.stem for f in files if f.stem[0] != expected]
        membership[archive] = {"n_flac": len(files), "first_char_expected": expected, "n_mismatched": len(mismatched)}
        print(f"{archive}: {len(files)} FLAC, {len(mismatched)} berawalan selain '{expected}'", flush=True)
    n_metadata_subset = int(in_subset.sum())
    n_flac = sum(item["n_flac"] for item in membership.values())
    print(f"Metadata berawalan 0-3: {n_metadata_subset}; FLAC pada keempat arsip: {n_flac}", flush=True)

    first = chi_square_uniform([utt[0] for utt in ids])
    second = chi_square_uniform([utt[1] for utt in ids])
    print(f"Karakter pertama : khi2 {first['chi2']:.2f} (p {first['p_value']:.3f}), porsi "
          f"{first['min_share_pct']:.2f}% sampai {first['max_share_pct']:.2f}%", flush=True)
    print(f"Karakter kedua   : khi2 {second['chi2']:.2f} (p {second['p_value']:.3f})", flush=True)

    by_speaker: dict[str, set[str]] = collections.defaultdict(set)
    per_speaker = collections.Counter(row.speaker_id for row in rows)
    for row in rows:
        by_speaker[row.speaker_id].add(row.utt_id[0])
    multi = [s for s, n in per_speaker.items() if n >= 2]
    spread = sum(1 for s in multi if len(by_speaker[s]) > 1)
    speakers_subset = {row.speaker_id for row, flag in zip(rows, in_subset, strict=True) if flag}
    speaker_info = {
        "n_speakers_full": len(per_speaker),
        "n_speakers_in_subset": len(speakers_subset),
        "pct_speakers_in_subset": 100 * len(speakers_subset) / len(per_speaker),
        "n_speakers_with_two_or_more_utterances": len(multi),
        "pct_of_those_spread_over_archives": 100 * spread / len(multi),
    }
    print(f"Penutur: {len(speakers_subset)} dari {len(per_speaker)} muncul pada subset; "
          f"{speaker_info['pct_of_those_spread_over_archives']:.2f}% penutur ber-2+ utterance tersebar ke "
          f"lebih dari satu arsip", flush=True)

    reads = collections.Counter(row.text for row in rows)
    chars = np.array([len(row.text) for row in rows])
    words = np.array([len(row.text.split()) for row in rows])
    readings = np.array([reads[row.text] for row in rows])
    speaker_size = np.array([per_speaker[row.speaker_id] for row in rows])
    print("-" * 78, flush=True)
    comparisons = {
        "transcript_characters": compare("Panjang transkrip (karakter)", chars[in_subset], chars[~in_subset], chars),
        "transcript_words": compare("Panjang transkrip (kata)", words[in_subset], words[~in_subset], words),
        "speaker_total_utterances": compare("Utterance per penutur (corpus)", speaker_size[in_subset],
                                            speaker_size[~in_subset], speaker_size),
        "sentence_readings_in_full_corpus": compare("Pembacaan kalimat pada corpus penuh", readings[in_subset],
                                                    readings[~in_subset], readings),
    }

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "n_metadata_rows": len(rows),
        "n_subset_rows": n_metadata_subset,
        "subset_share_pct": 100 * n_metadata_subset / len(rows),
        "archive_membership": membership,
        "n_flac_working_archives": n_flac,
        "id_first_char_uniformity": first,
        "id_second_char_uniformity": second,
        "speakers": speaker_info,
        "comparisons": comparisons,
        "notes": ("durasi tidak dibandingkan karena audio corpus penuh tidak ada di disk; KS dibandingkan antara "
                  "subset kerja dan dua belas arsip lainnya"),
        "input_sha256": {DEFAULT_METADATA.name: sha256_of(DEFAULT_METADATA)},
        "environment": collect_environment(SEED, started),
        "elapsed_seconds": elapsed,
    }
    write_json(args.report, report)
    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
