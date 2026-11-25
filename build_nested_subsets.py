"""Pembentukan subset training bersarang D1 sampai D5 dari training pool.

Tujuan
------
Membentuk kelima tingkat data training sesuai Subbab 3.3.2 skripsi, yaitu
1, 3, 10, dan 30 jam ditambah seluruh training pool, secara bersarang
D1 ⊂ D2 ⊂ D3 ⊂ D4 ⊂ D5. Training pool diurutkan menurut `id`, diacak satu kali
pada tingkat utterance dengan seed yang ditetapkan, lalu setiap tingkat diambil
sebagai awalan terpendek permutasi tersebut yang total durasinya mencapai
target. Karena seluruh tingkat adalah awalan dari permutasi yang sama, sifat
bersarang terjamin oleh konstruksinya dan pemeriksaan sesudahnya hanya
menegaskan hal itu.

Aturan berhenti sama dengan aturan kuota pada `build_text_disjoint_split.py`:
target dipenuhi, dan kelebihannya kurang dari satu utterance. D5 tidak memakai
target, melainkan seluruh training pool.

Durasi dibaca dari kolom `duration_seconds` training pool, yaitu berkas yang
dikunci sebelum subset dibentuk, lalu dijumlahkan dalam milidetik bulat agar keputusan
berhenti tidak dipengaruhi galat pembulatan floating point.

Masukan
-------
- `outputs/splits/train_pool.tsv` : keluaran `build_text_disjoint_split.py`.
- `outputs/splits/test.tsv`, `outputs/splits/validation.tsv` : hanya untuk
  memeriksa bahwa tidak ada transkrip training pool yang muncul di keduanya.

Keluaran
--------
- `outputs/subsets/d1.tsv` sampai `outputs/subsets/d5.tsv` : kolom sama dengan
  berkas pembagian, berurutan menurut `id`.
- `outputs/build_nested_subsets_report.json` : laporan lengkap, termasuk hash
  SHA-256 masukan dan keluaran.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u build_nested_subsets.py 2>&1 |
        Tee-Object -FilePath logs/build_nested_subsets.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_DIR = SCRIPT_DIR / "outputs" / "splits"
DEFAULT_SUBSET_DIR = SCRIPT_DIR / "outputs" / "subsets"
DEFAULT_REPORT = SCRIPT_DIR / "outputs" / "build_nested_subsets_report.json"

# Seed pengacakan training pool, ditetapkan penulis sebelum
# hasilnya dilihat, sama dengan seed pembagian. Tidak boleh diganti demi
# ukuran tingkat yang lebih rapi.
SUBSET_SEED = 42

# Target durasi D1 sampai D4 dalam jam. D5 memakai seluruh training pool.
LEVEL_TARGET_HOURS: dict[str, int] = {"d1": 1, "d2": 3, "d3": 10, "d4": 30}
FULL_POOL_LEVEL = "d5"

TSV_HEADER = ["id", "speaker_id", "duration_seconds", "transcript"]

MS_PER_SECOND = 1_000
SECONDS_PER_HOUR = 3_600
MS_PER_HOUR = MS_PER_SECOND * SECONDS_PER_HOUR


# --------------------------------------------------------------------------
# Struktur data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """Satu baris berkas pembagian, disimpan apa adanya agar ditulis ulang tanpa perubahan."""

    utt_id: str
    speaker_id: str
    duration_text: str
    transcript: str
    duration_ms: int


# --------------------------------------------------------------------------
# Pemuatan
# --------------------------------------------------------------------------


def read_split(path: Path) -> list[Row]:
    """Membaca satu berkas pembagian dengan dialek yang sama seperti penulisnya.

    Transkrip yang memuat tab atau tanda kutip ditulis berkutip oleh
    `csv.writer`, sehingga pembacaannya wajib melalui `csv.reader` dan bukan
    pemisahan string biasa.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header != TSV_HEADER:
            raise ValueError(f"Header {path.name} tidak dikenal: {header}")
        return [
            Row(utt_id, speaker_id, duration, transcript,
                round(float(duration) * MS_PER_SECOND))
            for utt_id, speaker_id, duration, transcript in reader
        ]


def sha256_of(path: Path) -> str:
    """Hash SHA-256 isi berkas, untuk mengikat laporan pada berkas yang tepat."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Pembentukan tingkat
# --------------------------------------------------------------------------


def build_levels(pool: Sequence[Row], seed: int) -> dict[str, list[Row]]:
    """Mengacak training pool satu kali lalu mengambil awalan bagi setiap tingkat.

    Pool diurutkan lebih dulu agar hasilnya tidak bergantung pada urutan baris
    masukan. Setiap daftar yang dikembalikan masih berurutan menurut
    permutasi, sehingga elemen terakhirnya adalah utterance yang menutup target.
    """
    order = sorted(pool, key=lambda row: row.utt_id)
    random.Random(seed).shuffle(order)

    levels: dict[str, list[Row]] = {}
    cumulative_ms = 0
    cut = 0
    for name, hours in LEVEL_TARGET_HOURS.items():
        target_ms = hours * MS_PER_HOUR
        while cumulative_ms < target_ms:
            cumulative_ms += order[cut].duration_ms
            cut += 1
        levels[name] = order[:cut]
    levels[FULL_POOL_LEVEL] = order
    return levels


def level_summary(
    rows: Sequence[Row],
    target_hours: int | None,
    test: Sequence[Row],
) -> dict[str, float | int | None]:
    """Ukuran, durasi, dan keragaman satu tingkat.

    Cakupan penutur test dicatat per tingkat karena tingkat yang lebih kecil
    memuat lebih sedikit penutur, sehingga derajat speaker-dependent evaluasi
    berbeda antartingkat meskipun data test-nya identik.
    """
    total_ms = sum(row.duration_ms for row in rows)
    overshoot = None
    if target_hours is not None:
        overshoot = round((total_ms - target_hours * MS_PER_HOUR) / MS_PER_SECOND, 3)
    speakers = {row.speaker_id for row in rows}
    return {
        "target_hours": target_hours,
        "n_utterances": len(rows),
        "hours": round(total_ms / MS_PER_HOUR, 3),
        "overshoot_seconds": overshoot,
        "mean_duration_seconds": round(total_ms / MS_PER_SECOND / len(rows), 3),
        "n_speakers": len(speakers),
        "n_unique_transcripts": len({row.transcript for row in rows}),
        "test_speakers_in_level_pct": round(
            100 * sum(1 for row in test if row.speaker_id in speakers) / len(test), 2
        ),
    }


def stops_at_first_crossing(rows: Sequence[Row], target_hours: int) -> bool:
    """Benar bila target tercapai dan tidak tercapai tanpa utterance terakhir."""
    total_ms = sum(row.duration_ms for row in rows)
    target_ms = target_hours * MS_PER_HOUR
    return total_ms >= target_ms and total_ms - rows[-1].duration_ms < target_ms


# --------------------------------------------------------------------------
# Penulisan
# --------------------------------------------------------------------------


def write_level(path: Path, rows: Sequence[Row]) -> None:
    """Menulis satu tingkat ke TSV, berurutan menurut `id`."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_HEADER)
        for row in sorted(rows, key=lambda item: item.utt_id):
            writer.writerow([row.utt_id, row.speaker_id, row.duration_text, row.transcript])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Pembentukan subset training bersarang D1 sampai D5.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    pool_path = args.split_dir / "train_pool.tsv"
    test_path = args.split_dir / "test.tsv"
    validation_path = args.split_dir / "validation.tsv"

    print("=" * 78, flush=True)
    print("Pembentukan subset training bersarang D1 sampai D5", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Training pool      : {pool_path}", flush=True)
    print(f"Random seed        : {SUBSET_SEED}", flush=True)
    print(f"Target             : "
          f"{', '.join(f'{k} {v} jam' for k, v in LEVEL_TARGET_HOURS.items())}, "
          f"{FULL_POOL_LEVEL} seluruh pool", flush=True)
    print("=" * 78, flush=True)

    for path in (pool_path, test_path, validation_path):
        if not path.is_file():
            print(f"GAGAL: berkas pembagian tidak ditemukan pada {path}", file=sys.stderr)
            return 1

    pool = read_split(pool_path)
    test = read_split(test_path)
    held_out_texts = {row.transcript for row in test}
    held_out_texts |= {row.transcript for row in read_split(validation_path)}
    pool_ms = sum(row.duration_ms for row in pool)
    input_sha256 = {path.name: sha256_of(path) for path in (pool_path, test_path, validation_path)}
    print(f"Isi training pool  : {len(pool)} utterance, {pool_ms / MS_PER_HOUR:.3f} jam", flush=True)
    print(f"SHA-256 pool       : {input_sha256[pool_path.name]}", flush=True)

    if pool_ms < max(LEVEL_TARGET_HOURS.values()) * MS_PER_HOUR:
        print("GAGAL: training pool lebih kecil daripada target terbesar.", file=sys.stderr)
        return 1

    levels = build_levels(pool, SUBSET_SEED)
    summaries = {
        name: level_summary(rows, LEVEL_TARGET_HOURS.get(name), test)
        for name, rows in levels.items()
    }

    names = list(levels)
    id_sets = {name: {row.utt_id for row in rows} for name, rows in levels.items()}
    checks = {
        "pool_ids_unique": len({row.utt_id for row in pool}) == len(pool),
        "levels_nested": all(id_sets[a] < id_sets[b] for a, b in zip(names, names[1:])),
        "targets_met_at_first_crossing": all(
            stops_at_first_crossing(levels[name], hours)
            for name, hours in LEVEL_TARGET_HOURS.items()
        ),
        "full_level_equals_pool": id_sets[FULL_POOL_LEVEL] == {row.utt_id for row in pool},
        "pool_utterances_with_held_out_transcript": sum(
            1 for row in pool if row.transcript in held_out_texts
        ),
    }

    for name, summary in summaries.items():
        target = f"target {summary['target_hours']} jam" if summary["target_hours"] else "seluruh pool"
        overshoot = (f", lebih {summary['overshoot_seconds']} detik"
                     if summary["overshoot_seconds"] is not None else "")
        print(f"{name:18s} : {summary['n_utterances']} utterance, {summary['hours']} jam "
              f"({target}{overshoot}), {summary['n_speakers']} penutur, "
              f"utterance test berpenutur tercakup {summary['test_speakers_in_level_pct']}%",
              flush=True)
    for name, value in checks.items():
        print(f"Periksa {name:41s}: {value}", flush=True)

    if not (checks["pool_ids_unique"] and checks["levels_nested"]
            and checks["targets_met_at_first_crossing"]
            and checks["full_level_equals_pool"]
            and checks["pool_utterances_with_held_out_transcript"] == 0):
        print("GAGAL: pemeriksaan subset bersarang tidak terpenuhi.", file=sys.stderr)
        return 1

    args.subset_dir.mkdir(parents=True, exist_ok=True)
    output_sha256: dict[str, str] = {}
    for name, rows in levels.items():
        path = args.subset_dir / f"{name}.tsv"
        write_level(path, rows)
        output_sha256[path.name] = sha256_of(path)

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "subset_seed": SUBSET_SEED,
        "level_target_hours": LEVEL_TARGET_HOURS,
        "full_pool_level": FULL_POOL_LEVEL,
        "stopping_rule": "awalan terpendek permutasi yang total durasinya mencapai target",
        "input_sha256": input_sha256,
        "pool": {
            "n_utterances": len(pool),
            "hours": round(pool_ms / MS_PER_HOUR, 3),
        },
        "levels": summaries,
        "checks": checks,
        "output_sha256": output_sha256,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 78, flush=True)
    print(f"Subset ditulis     : {args.subset_dir}", flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
