"""Durasi komponen graf penutur dan transkrip pada subset kerja.

Tujuan
------
Menghasilkan kolom durasi Tabel 3.9 `bab-3.md` dan Tabel 8.2.4
`DraftResearch.md`. Sebelum skrip ini ada, kolom tersebut berupa perkiraan,
yaitu jumlah utterance komponen dikalikan durasi rata-rata, sehingga tidak
memenuhi daftar periksa angka (Utang nomor 26). Skrip ini menjumlahkan durasi
setiap utterance di dalam komponennya.

Komponen dihitung ulang dengan definisi yang sama seperti `giant_component`
pada `audit_transcript_leakage.py`: simpul berupa penutur dan transkrip mentah,
sisi menghubungkan penutur dengan setiap transkrip yang dibacanya, dan sebuah
utterance termasuk komponen tempat simpul penuturnya berada. Fungsi audit itu
hanya mengembalikan ukuran komponen, bukan keanggotaan utterance, sehingga
union-find ditulis ulang di sini. Kesamaan definisinya diperiksa dengan
mencocokkan seluruh daftar ukuran komponen terhadap laporan audit, dan skrip
menolak menulis laporan bila ada selisih.

Perhitungannya deterministik dan tidak memakai random seed.

Masukan
-------
- `outputs/splits/test.tsv`, `validation.tsv`, `train_pool.tsv`, dan
  `excluded.tsv` : gabungan keempatnya adalah subset kerja 46.309 utterance,
  dengan durasi yang dibaca `build_text_disjoint_split.py` dari header FLAC.
- `outputs/audit_transcript_leakage_report.json` : ukuran komponen pembanding.

Keluaran
--------
- `outputs/measure_component_durations_report.json` : laporan lengkap.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u measure_component_durations.py 2>&1 |
        Tee-Object -FilePath logs/measure_component_durations.log
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
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
DEFAULT_AUDIT_REPORT = SCRIPT_DIR / "outputs" / "audit_transcript_leakage_report.json"
DEFAULT_REPORT = SCRIPT_DIR / "outputs" / "measure_component_durations_report.json"

# Gabungan keempat berkas ini adalah subset kerja sebelum penyaringan, yaitu
# populasi yang diaudit dan populasi Tabel 3.9.
SPLIT_FILES = ("test.tsv", "validation.tsv", "train_pool.tsv", "excluded.tsv")
AUDIT_POPULATION = "subset_kerja"

MS_PER_SECOND = 1_000
MS_PER_MINUTE = 60_000
MS_PER_HOUR = 3_600_000

Node = tuple[str, str]


# --------------------------------------------------------------------------
# Struktur data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """Satu utterance subset kerja beserta durasinya dalam milidetik bulat."""

    utt_id: str
    speaker_id: str
    transcript: str
    duration_ms: int


# --------------------------------------------------------------------------
# Pemuatan
# --------------------------------------------------------------------------


def read_working_subset(split_dir: Path) -> list[Row]:
    """Menggabungkan keempat berkas pembagian menjadi subset kerja.

    `excluded.tsv` memiliki kolom tambahan `reasons`, sehingga pembacaan
    memakai nama kolom dan bukan posisi.
    """
    rows: list[Row] = []
    for name in SPLIT_FILES:
        with (split_dir / name).open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle, delimiter="\t"):
                rows.append(Row(
                    record["id"],
                    record["speaker_id"],
                    record["transcript"],
                    round(float(record["duration_seconds"]) * MS_PER_SECOND),
                ))
    return rows


def audit_component_sizes(path: Path) -> list[int]:
    """Daftar ukuran komponen subset kerja menurut laporan audit."""
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    for population in report["populations"]:
        if population["population"] == AUDIT_POPULATION:
            return list(population["speaker_transcript_graph"]["component_sizes_desc"])
    raise ValueError(f"Populasi {AUDIT_POPULATION} tidak ada pada {path.name}")


# --------------------------------------------------------------------------
# Komponen graf
# --------------------------------------------------------------------------


def component_roots(rows: Sequence[Row]) -> list[Node]:
    """Akar komponen setiap utterance, dengan definisi `giant_component`."""
    parent: dict[Node, Node] = {}

    def find(node: Node) -> Node:
        root = node
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left: Node, right: Node) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for row in rows:
        speaker_node = ("speaker", row.speaker_id)
        text_node = ("text", row.transcript)
        parent.setdefault(speaker_node, speaker_node)
        parent.setdefault(text_node, text_node)
        union(speaker_node, text_node)

    return [find(("speaker", row.speaker_id)) for row in rows]


def part_summary(
    keys: Sequence[Node],
    sizes: dict[Node, int],
    durations: dict[Node, int],
    total: int,
) -> dict[str, float | int]:
    """Ukuran dan durasi gabungan sekelompok komponen."""
    n_utterances = sum(sizes[key] for key in keys)
    duration_ms = sum(durations[key] for key in keys)
    return {
        "n_components": len(keys),
        "n_utterances": n_utterances,
        "pct_utterances": round(100 * n_utterances / total, 2),
        "hours": round(duration_ms / MS_PER_HOUR, 3),
        "minutes": round(duration_ms / MS_PER_MINUTE, 1),
    }


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Durasi komponen graf penutur dan transkrip.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    print("=" * 78, flush=True)
    print("Durasi komponen graf penutur dan transkrip pada subset kerja", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Berkas pembagian   : {args.split_dir}", flush=True)
    print(f"Laporan audit      : {args.audit_report}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print("=" * 78, flush=True)

    for path in [args.split_dir / name for name in SPLIT_FILES] + [args.audit_report]:
        if not path.is_file():
            print(f"GAGAL: berkas masukan tidak ditemukan pada {path}", file=sys.stderr)
            return 1

    rows = read_working_subset(args.split_dir)
    expected_sizes = audit_component_sizes(args.audit_report)
    roots = component_roots(rows)

    sizes: collections.Counter[Node] = collections.Counter()
    durations: collections.Counter[Node] = collections.Counter()
    for row, root in zip(rows, roots):
        sizes[root] += 1
        durations[root] += row.duration_ms

    # Ukuran sama diurutkan menurut akarnya agar urutan komponen kecil tetap.
    ordered = sorted(sizes, key=lambda key: (-sizes[key], key))
    total = len(rows)
    total_ms = sum(row.duration_ms for row in rows)
    parts = {
        "largest": part_summary(ordered[:1], sizes, durations, total),
        "second": part_summary(ordered[1:2], sizes, durations, total),
        "remaining": part_summary(ordered[2:], sizes, durations, total),
    }
    # Bila satu komponen besar dijadikan data test, seluruh sisa subset kerja
    # adalah batas atas bagi training pool dan validation sekaligus.
    for name, key in (("largest", ordered[0]), ("second", ordered[1])):
        parts[name]["hours_left_if_test"] = round((total_ms - durations[key]) / MS_PER_HOUR, 3)
    measured_sizes = [sizes[key] for key in ordered]

    checks = {
        "utterance_ids_unique": len({row.utt_id for row in rows}) == total,
        "component_sizes_match_audit": measured_sizes == expected_sizes,
    }

    print(f"Subset kerja       : {total} utterance, {total_ms / MS_PER_HOUR:.3f} jam", flush=True)
    print(f"Komponen           : {len(ordered)}", flush=True)
    for name, summary in parts.items():
        left = (f", sisa bila jadi test {summary['hours_left_if_test']} jam"
                if "hours_left_if_test" in summary else "")
        print(f"{name:18s} : {summary['n_components']} komponen, "
              f"{summary['n_utterances']} utterance ({summary['pct_utterances']}%), "
              f"{summary['hours']} jam = {summary['minutes']} menit{left}", flush=True)
    for name, value in checks.items():
        print(f"Periksa {name:28s}: {value}", flush=True)

    if not all(checks.values()):
        print("GAGAL: subset kerja atau komponen tidak sesuai laporan audit.", file=sys.stderr)
        return 1

    elapsed = round(time.perf_counter() - started, 2)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "population": AUDIT_POPULATION,
        "split_files": list(SPLIT_FILES),
        "audit_report": str(args.audit_report),
        "n_utterances": total,
        "total_hours": round(total_ms / MS_PER_HOUR, 3),
        "n_components": len(ordered),
        "parts": parts,
        "checks": checks,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
