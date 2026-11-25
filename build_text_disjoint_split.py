"""Pembagian data produksi text-disjoint beserta penyaringan subset kerja.

Tujuan
------
Menghasilkan pembagian data yang mengunci kesepuluh konfigurasi penelitian,
yaitu data test, data validation, dan training pool, sesuai `bab-3.md`
Subbab 3.3.3. Algoritma pembagiannya sama dengan strategi text-disjoint pada
`audit_transcript_leakage.py`, dengan dua perbedaan yang disengaja:

1. Kuota dipatok dalam jumlah utterance, yaitu 3.100 test dan 1.900
   validation (Keputusan Metodologis I.3 poin 4), bukan sebagai proporsi.
2. Utterance yang dikeluarkan preprocessing disaring **sebelum** pembagian,
   sehingga populasi yang dibagi sama dengan populasi yang dievaluasi.
   Kriteria pengeluarannya mengikuti `bab-3.md` Subbab 3.3.4: durasi di bawah
   1,5 detik, transkrip memuat angka, atau transkrip memuat karakter U+FFFD.
   Pemeriksaan angka dan U+FFFD dilakukan pada transkrip mentah. Langkah
   normalisasi 1 sampai 4 (NFC, huruf kecil, pelepasan diakritik, penghapusan
   tanda baca) tidak menambah maupun menghapus angka atau U+FFFD, sehingga
   hasilnya sama dengan memeriksa sesudah langkah tersebut.

Skrip juga menghitung statistik durasi, format audio, dan ukuran berkas subset
kerja dari header berkas FLAC, sebelum dan sesudah penyaringan, karena
statistik tersebut belum pernah memiliki skrip penghasil (Utang nomor 24).

Masukan
-------
- `utt_spk_text.tsv` : metadata corpus, dibaca dengan `load_metadata` dari
  `audit_transcript_leakage.py` agar cara pembacaannya identik.
- Akar folder dataset : keanggotaan subset kerja ditetapkan dari berkas FLAC
  yang ada di disk, dan durasi dibaca dari header berkas tersebut.

Keluaran
--------
- `outputs/splits/test.tsv`, `outputs/splits/validation.tsv`,
  `outputs/splits/train_pool.tsv` : kolom `id`, `speaker_id`,
  `duration_seconds`, `transcript`, berurutan menurut `id`.
- `outputs/splits/excluded.tsv` : utterance yang disaring beserta alasannya.
- `outputs/build_text_disjoint_split_report.json` : laporan lengkap.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u build_text_disjoint_split.py 2>&1 |
        Tee-Object -FilePath logs/build_text_disjoint_split.log
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import soundfile

from audit_transcript_leakage import (
    DEFAULT_DATASET_ROOT,
    DEFAULT_METADATA,
    WORKING_ARCHIVES,
    Utterance,
    load_metadata,
)

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = SCRIPT_DIR / "outputs" / "splits"
DEFAULT_REPORT = SCRIPT_DIR / "outputs" / "build_text_disjoint_split_report.json"

# Seed pembagian produksi, ditetapkan pengguna pada Sesi 15 sebelum hasilnya
# dilihat. Tidak boleh diganti demi ukuran hasil yang lebih rapi.
SPLIT_SEED = 42

# Kuota rancangan dalam utterance, turunan dari ± 5 jam dan ± 3 jam.
TEST_QUOTA = 3_100
VALIDATION_QUOTA = 1_900

MIN_DURATION_SECONDS = 1.5
DIGIT_PATTERN = re.compile(r"\d")
REPLACEMENT_CHAR = "�"

SECONDS_PER_HOUR = 3_600


# --------------------------------------------------------------------------
# Struktur data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """Satu utterance subset kerja beserta durasinya."""

    utt: Utterance
    duration: float


# --------------------------------------------------------------------------
# Pemuatan
# --------------------------------------------------------------------------


def scan_flac(dataset_root: Path) -> tuple[dict[str, float], dict[str, object]]:
    """Membaca durasi dan format setiap berkas FLAC subset kerja dari header-nya.

    Nilai kembalian kedua merangkum format audio dan ukuran berkas, agar
    baris format pada Tabel 3.1 naskah memiliki skrip penghasil.
    """
    durations: dict[str, float] = {}
    formats: collections.Counter[str] = collections.Counter()
    sizes: list[int] = []
    for archive in WORKING_ARCHIVES:
        data_dir = dataset_root / archive / "data"
        if not data_dir.is_dir():
            continue
        for flac in data_dir.rglob("*.flac"):
            info = soundfile.info(str(flac))
            durations[flac.stem] = info.frames / info.samplerate
            formats[f"{info.format} {info.samplerate} Hz {info.channels} kanal {info.subtype}"] += 1
            sizes.append(flac.stat().st_size)
    audio = {
        "format_counts": dict(formats),
        "mean_file_size_kb": round(statistics.fmean(sizes) / 1024, 1) if sizes else 0.0,
    }
    return durations, audio


# --------------------------------------------------------------------------
# Statistik dan penyaringan
# --------------------------------------------------------------------------


def duration_statistics(records: Sequence[Record]) -> dict[str, float | int]:
    """Statistik durasi satu populasi utterance."""
    values = [record.duration for record in records]
    return {
        "n_utterances": len(values),
        "total_hours": round(sum(values) / SECONDS_PER_HOUR, 3),
        "mean_seconds": round(statistics.fmean(values), 3),
        "median_seconds": round(statistics.median(values), 3),
        "min_seconds": round(min(values), 3),
        "max_seconds": round(max(values), 3),
        "n_below_min_duration": sum(1 for value in values if value < MIN_DURATION_SECONDS),
        "n_speakers": len({record.utt.speaker_id for record in records}),
        "n_unique_transcripts": len({record.utt.text for record in records}),
    }


def exclusion_reasons(record: Record) -> list[str]:
    """Alasan pengeluaran satu utterance, kosong bila utterance dipertahankan."""
    reasons: list[str] = []
    if record.duration < MIN_DURATION_SECONDS:
        reasons.append("durasi_di_bawah_1_5_detik")
    if DIGIT_PATTERN.search(record.utt.text):
        reasons.append("memuat_angka")
    if REPLACEMENT_CHAR in record.utt.text:
        reasons.append("memuat_U+FFFD")
    return reasons


# --------------------------------------------------------------------------
# Pembagian
# --------------------------------------------------------------------------


def take_groups_until(
    keys: Sequence[str],
    groups: dict[str, list[Record]],
    quota: int,
) -> tuple[list[Record], list[str]]:
    """Mengambil grup utuh berurutan sampai jumlah utterance mencapai kuota."""
    taken: list[Record] = []
    used: list[str] = []
    for key in keys:
        if len(taken) >= quota:
            break
        taken.extend(groups[key])
        used.append(key)
    return taken, used


def split_text_disjoint(
    records: Sequence[Record],
    seed: int,
) -> tuple[list[Record], list[Record], list[Record], int]:
    """Pembagian text-disjoint sesuai lima langkah `bab-3.md` Subbab 3.3.3.

    Grup diurutkan lebih dulu agar hasilnya tidak bergantung pada urutan baris
    masukan, lalu diacak dengan seed yang ditetapkan.
    """
    groups: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        groups[record.utt.text].append(record)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    test, used_test = take_groups_until(keys, groups, TEST_QUOTA)
    remaining = keys[len(used_test):]
    validation, used_val = take_groups_until(remaining, groups, VALIDATION_QUOTA)
    train = [record for key in remaining[len(used_val):] for record in groups[key]]
    return train, validation, test, len(groups)


def part_summary(part: Sequence[Record], total: int) -> dict[str, float | int]:
    """Ukuran, durasi, dan porsi satu bagian hasil pembagian."""
    hours = sum(record.duration for record in part) / SECONDS_PER_HOUR
    return {
        "n_utterances": len(part),
        "hours": round(hours, 3),
        "mean_duration_seconds": round(hours * SECONDS_PER_HOUR / len(part), 3),
        "pct_of_population": round(100 * len(part) / total, 2),
        "n_speakers": len({record.utt.speaker_id for record in part}),
        "n_unique_transcripts": len({record.utt.text for record in part}),
    }


def pct_shared(part: Sequence[Record], other: Sequence[Record], attr: str) -> float:
    """Persentase utterance `part` yang nilai `attr`-nya muncul di `other`."""
    seen = {getattr(record.utt, attr) for record in other}
    hits = sum(1 for record in part if getattr(record.utt, attr) in seen)
    return round(100 * hits / len(part), 2) if part else 0.0


# --------------------------------------------------------------------------
# Penulisan
# --------------------------------------------------------------------------


def write_part(path: Path, part: Sequence[Record], reasons: dict[str, list[str]] | None = None) -> None:
    """Menulis satu bagian ke TSV, berurutan menurut `id`."""
    header = ["id", "speaker_id", "duration_seconds", "transcript"]
    if reasons is not None:
        header.append("reasons")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for record in sorted(part, key=lambda item: item.utt.utt_id):
            row = [record.utt.utt_id, record.utt.speaker_id,
                   f"{record.duration:.3f}", record.utt.text]
            if reasons is not None:
                row.append(",".join(reasons[record.utt.utt_id]))
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Pembagian data produksi text-disjoint.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    print("=" * 78, flush=True)
    print("Pembagian data produksi text-disjoint", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Metadata           : {args.metadata}", flush=True)
    print(f"Random seed        : {SPLIT_SEED}", flush=True)
    print(f"Kuota              : test {TEST_QUOTA}, validation {VALIDATION_QUOTA}", flush=True)
    print(f"Batas durasi       : {MIN_DURATION_SECONDS} detik", flush=True)
    print("=" * 78, flush=True)

    if not args.metadata.is_file():
        print(f"GAGAL: metadata tidak ditemukan pada {args.metadata}", file=sys.stderr)
        return 1

    rows = load_metadata(args.metadata)
    durations, audio = scan_flac(args.dataset_root)
    records = [Record(row, durations[row.utt_id]) for row in rows if row.utt_id in durations]
    missing_metadata = len(durations) - len(records)
    print(f"Berkas FLAC        : {len(durations)}", flush=True)
    print(f"Format audio       : {audio['format_counts']}", flush=True)
    print(f"Ukuran rata-rata   : {audio['mean_file_size_kb']} KB", flush=True)
    print(f"Subset kerja       : {len(records)} utterance bermetadata", flush=True)
    if missing_metadata:
        print(f"PERINGATAN: {missing_metadata} berkas FLAC tanpa metadata", flush=True)

    subset_stats = duration_statistics(records)

    reasons_by_id: dict[str, list[str]] = {}
    kept: list[Record] = []
    excluded: list[Record] = []
    for record in records:
        reasons = exclusion_reasons(record)
        if reasons:
            reasons_by_id[record.utt.utt_id] = reasons
            excluded.append(record)
        else:
            kept.append(record)
    reason_counts = collections.Counter(
        reason for reasons in reasons_by_id.values() for reason in reasons
    )
    filtered_stats = duration_statistics(kept)

    print(f"Dikeluarkan        : {len(excluded)} utterance "
          f"({', '.join(f'{k} {v}' for k, v in sorted(reason_counts.items()))})", flush=True)
    print(f"Subset tersaring   : {len(kept)} utterance, "
          f"{filtered_stats['total_hours']} jam", flush=True)

    train, validation, test, n_groups = split_text_disjoint(kept, SPLIT_SEED)
    total = len(kept)
    parts = {
        "test": part_summary(test, total),
        "validation": part_summary(validation, total),
        "train_pool": part_summary(train, total),
    }

    checks = {
        "sizes_sum_to_population": len(test) + len(validation) + len(train) == total,
        "test_meets_quota": len(test) >= TEST_QUOTA,
        "validation_meets_quota": len(validation) >= VALIDATION_QUOTA,
        "test_transcripts_in_train_pct": pct_shared(test, train, "text"),
        "test_transcripts_in_validation_pct": pct_shared(test, validation, "text"),
        "validation_transcripts_in_train_pct": pct_shared(validation, train, "text"),
        "test_speakers_in_train_pct": pct_shared(test, train, "speaker_id"),
    }

    for name, summary in parts.items():
        print(f"{name:18s} : {summary['n_utterances']} utterance, "
              f"{summary['hours']} jam, {summary['pct_of_population']}%", flush=True)
    print(f"Grup transkrip     : {n_groups}", flush=True)
    for name, value in checks.items():
        print(f"Periksa {name:34s}: {value}", flush=True)

    if not (checks["sizes_sum_to_population"] and checks["test_meets_quota"]
            and checks["validation_meets_quota"]
            and checks["test_transcripts_in_train_pct"] == 0.0
            and checks["test_transcripts_in_validation_pct"] == 0.0
            and checks["validation_transcripts_in_train_pct"] == 0.0):
        print("GAGAL: pemeriksaan pembagian tidak terpenuhi.", file=sys.stderr)
        return 1

    args.split_dir.mkdir(parents=True, exist_ok=True)
    write_part(args.split_dir / "test.tsv", test)
    write_part(args.split_dir / "validation.tsv", validation)
    write_part(args.split_dir / "train_pool.tsv", train)
    write_part(args.split_dir / "excluded.tsv", excluded, reasons_by_id)

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "metadata_path": str(args.metadata),
        "split_seed": SPLIT_SEED,
        "test_quota": TEST_QUOTA,
        "validation_quota": VALIDATION_QUOTA,
        "min_duration_seconds": MIN_DURATION_SECONDS,
        "working_subset_archives": list(WORKING_ARCHIVES),
        "n_flac_files": len(durations),
        "n_flac_without_metadata": missing_metadata,
        "audio_format": audio,
        "working_subset_duration_statistics": subset_stats,
        "exclusion": {
            "n_excluded": len(excluded),
            "reason_counts": dict(sorted(reason_counts.items())),
            "excluded_ids": sorted(reasons_by_id),
        },
        "filtered_subset_duration_statistics": filtered_stats,
        "n_transcript_groups": n_groups,
        "parts": parts,
        "checks": checks,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 78, flush=True)
    print(f"Pembagian ditulis  : {args.split_dir}", flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
