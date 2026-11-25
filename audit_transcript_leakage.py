"""Audit kebocoran transkrip pada corpus OpenSLR SLR35 (Bahasa Jawa).

Tujuan
------
Menghasilkan angka yang dikutip pada `DraftResearch.md` Subbab 8.2 dan
`bab-2.md` Subbab 2.2.9, yaitu besaran kebocoran transkrip antara data
training dan data test menurut strategi pembagian data. Sebelum skrip ini
dibuat, angka tersebut tercantum di naskah tanpa berkas yang
menghasilkannya, sehingga tidak memenuhi daftar periksa Bagian 8
`strutkur-script-python.md`.

Skrip menghitung lima kelompok keluaran:

1. Statistik dasar corpus: jumlah utterance, kalimat unik, rata-rata
   pembacaan per kalimat, jumlah kalimat yang dibaca lebih dari sekali,
   dan jumlah speaker unik.
2. Besaran kebocoran transkrip untuk tiga strategi pembagian, yaitu acak
   per-utterance, speaker-disjoint, dan text-disjoint, masing-masing
   diulang pada sejumlah random seed.
3. Sensitivitas kebocoran terhadap rasio data test, karena rasio tersebut
   menentukan besaran yang dilaporkan.
4. Komponen terbesar pada graf keterhubungan speaker dan transkrip.
5. Seluruh butir 1 sampai 4 dihitung dua kali, yaitu pada corpus penuh
   (16 arsip) dan pada subset kerja (4 arsip, Subbab 9.1), karena kedua
   populasi memberi angka yang berbeda jauh.

Masukan
-------
- `utt_spk_text.tsv` : metadata corpus, tiga kolom bertab, yaitu
  `utterance_id`, `speaker_id`, dan transkrip. Berkas ini memuat seluruh
  185.076 utterance corpus penuh meskipun disalin di dalam setiap arsip.
- Akar folder dataset (opsional) : dipakai untuk menetapkan keanggotaan
  subset kerja dari berkas FLAC yang benar-benar ada di disk. Bila tidak
  diberikan, keanggotaan ditetapkan dari prefiks heksadesimal
  `utterance_id` sesuai skema partisi corpus.

Keluaran
--------
- JSON laporan lengkap ke `outputs/audit_transcript_leakage_report.json`.
- Ringkasan terbaca manusia ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u audit_transcript_leakage.py 2>&1 |
        Tee-Object -FilePath logs/audit_transcript_leakage.log

Catatan
-------
Seluruh operasi berkas memakai `encoding="utf-8"` secara eksplisit karena
transkrip Bahasa Jawa memuat karakter beraksen dan Windows tidak memakai
UTF-8 sebagai default.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SKRIPSI_ROOT = Path(r"D:\_01_Bachelor_Information_System\_00-SKRIPSI")
DEFAULT_METADATA = SKRIPSI_ROOT / "_03_dataset" / "asr_javanese_0" / "utt_spk_text.tsv"
DEFAULT_DATASET_ROOT = SKRIPSI_ROOT / "_03_dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "audit_transcript_leakage_report.json"

# Arsip yang diekstrak sebagai subset kerja (Subbab 9.1).
WORKING_ARCHIVES: tuple[str, ...] = (
    "asr_javanese_0",
    "asr_javanese_1",
    "asr_javanese_2",
    "asr_javanese_3",
)

# Partisi corpus dilakukan berdasarkan hash identifier utterance. Arsip ke-k
# memuat utterance yang `utterance_id`-nya berawalan heksadesimal `k`.
WORKING_ID_PREFIXES: frozenset[str] = frozenset("0123")

# Proporsi pembagian data mengikuti Subbab 9.1, yaitu 3.100 utterance test
# dan 1.900 utterance validation dari subset kerja 46.300 utterance.
TEST_FRACTION = 3_100 / 46_300
VAL_FRACTION = 1_900 / 46_300

# Sapuan sensitivitas terhadap rasio data test.
TEST_FRACTION_SWEEP: tuple[float, ...] = (0.05, 0.067, 0.10, 0.15, 0.20, 0.30)

# Random seed dicatat eksplisit sesuai butir 10 konvensi penulisan skrip.
SEEDS: tuple[int, ...] = (1, 7, 13, 42, 101, 2024, 31337, 60221, 99991, 123457)

STRATEGIES: tuple[str, ...] = ("random_utterance", "speaker_disjoint", "text_disjoint")

MIN_COLUMNS = 3


# --------------------------------------------------------------------------
# Struktur data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Utterance:
    """Satu baris metadata corpus."""

    utt_id: str
    speaker_id: str
    text: str


@dataclass
class SplitOutcome:
    """Hasil satu simulasi pembagian data pada satu seed."""

    seed: int
    n_train: int
    n_val: int
    n_test: int
    transcript_leak_pct: float
    speaker_overlap_pct: float


@dataclass
class StrategySummary:
    """Ringkasan satu strategi pembagian atas seluruh seed."""

    strategy: str
    n_seeds: int
    transcript_leak_mean: float
    transcript_leak_sd: float
    transcript_leak_min: float
    transcript_leak_max: float
    speaker_overlap_mean: float
    runs: list[SplitOutcome]


# --------------------------------------------------------------------------
# Pemuatan dan normalisasi
# --------------------------------------------------------------------------


def load_metadata(path: Path) -> list[Utterance]:
    """Membaca `utt_spk_text.tsv` menjadi daftar `Utterance`.

    Kolom ketiga dan seterusnya digabung kembali, karena transkrip yang
    memuat karakter tab akan terpecah oleh pembaca CSV.
    """
    rows: list[Utterance] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for fields in csv.reader(handle, delimiter="\t"):
            if len(fields) < MIN_COLUMNS:
                continue
            text = "\t".join(fields[2:]).strip()
            rows.append(Utterance(fields[0], fields[1], text))
    return rows


def normalise_text(text: str) -> str:
    """Normalisasi ringan untuk pemeriksaan ketegaran hasil.

    Bukan prosedur normalisasi naskah pada Subbab 8.3. Fungsi ini hanya
    menyatukan bentuk Unicode, menyeragamkan huruf besar-kecil, dan
    merapatkan spasi, agar dapat diperiksa apakah besaran kebocoran
    berubah bila kecocokan transkrip dilonggarkan.
    """
    folded = unicodedata.normalize("NFC", text).casefold()
    return " ".join(folded.split())


def resolve_working_subset(
    rows: Sequence[Utterance],
    dataset_root: Path | None,
) -> tuple[list[Utterance], str]:
    """Menetapkan keanggotaan subset kerja 4 arsip.

    Bila `dataset_root` tersedia, keanggotaan diambil dari berkas FLAC yang
    benar-benar ada di disk. Bila tidak, keanggotaan diturunkan dari prefiks
    heksadesimal `utterance_id`. Nilai kembalian kedua menyatakan metode
    yang dipakai, agar tercatat pada laporan.
    """
    if dataset_root is not None and dataset_root.is_dir():
        present: set[str] = set()
        for archive in WORKING_ARCHIVES:
            data_dir = dataset_root / archive / "data"
            if not data_dir.is_dir():
                continue
            present.update(flac.stem for flac in data_dir.rglob("*.flac"))
        if present:
            subset = [row for row in rows if row.utt_id in present]
            return subset, "berkas FLAC pada disk"

    subset = [row for row in rows if row.utt_id[:1] in WORKING_ID_PREFIXES]
    return subset, "prefiks heksadesimal utterance_id"


# --------------------------------------------------------------------------
# Statistik dasar
# --------------------------------------------------------------------------


def base_statistics(rows: Sequence[Utterance]) -> dict[str, float | int]:
    """Menghitung statistik dasar corpus."""
    texts = [row.text for row in rows]
    counts = collections.Counter(texts)
    unique_raw = len(counts)
    unique_norm = len({normalise_text(text) for text in texts})
    repeated = [text for text, n in counts.items() if n > 1]

    cross_speaker = 0
    speakers_per_text: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        speakers_per_text[row.text].add(row.speaker_id)
    for text in repeated:
        if len(speakers_per_text[text]) > 1:
            cross_speaker += 1

    return {
        "n_utterances": len(rows),
        "n_unique_transcripts_raw": unique_raw,
        "n_unique_transcripts_normalised": unique_norm,
        "mean_readings_per_transcript": round(len(rows) / unique_raw, 4),
        "n_transcripts_read_more_than_once": len(repeated),
        "pct_repeated_read_by_different_speakers": round(
            100 * cross_speaker / len(repeated), 2
        )
        if repeated
        else 0.0,
        "n_unique_speakers": len({row.speaker_id for row in rows}),
    }


# --------------------------------------------------------------------------
# Simulasi pembagian data
# --------------------------------------------------------------------------


def _take_groups_until(
    group_order: Iterable[str],
    groups: dict[str, list[Utterance]],
    target: int,
) -> tuple[list[Utterance], set[str]]:
    """Mengambil grup berurutan sampai jumlah utterance mencapai `target`."""
    taken: list[Utterance] = []
    used: set[str] = set()
    for key in group_order:
        if len(taken) >= target:
            break
        taken.extend(groups[key])
        used.add(key)
    return taken, used


def split_random_utterance(
    rows: Sequence[Utterance],
    rng: random.Random,
    test_fraction: float,
    val_fraction: float,
) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
    """Pembagian acak pada tingkat utterance."""
    order = list(rows)
    rng.shuffle(order)
    n_test = int(len(order) * test_fraction)
    n_val = int(len(order) * val_fraction)
    test = order[:n_test]
    val = order[n_test : n_test + n_val]
    train = order[n_test + n_val :]
    return train, val, test


def split_by_group(
    rows: Sequence[Utterance],
    key_of: "collections.abc.Callable[[Utterance], str]",
    rng: random.Random,
    test_fraction: float,
    val_fraction: float,
) -> tuple[list[Utterance], list[Utterance], list[Utterance]]:
    """Pembagian pada tingkat grup, dipakai speaker-disjoint dan text-disjoint.

    Grup diacak lalu diambil berurutan sampai kuota utterance test dan
    validation terpenuhi. Sisanya menjadi data training. Karena pembagian
    dilakukan pada tingkat grup, tidak ada satu pun nilai kunci yang muncul
    di dua bagian sekaligus.
    """
    groups: dict[str, list[Utterance]] = collections.defaultdict(list)
    for row in rows:
        groups[key_of(row)].append(row)

    keys = sorted(groups)
    rng.shuffle(keys)

    test, used_test = _take_groups_until(keys, groups, int(len(rows) * test_fraction))
    remaining = [key for key in keys if key not in used_test]
    val, used_val = _take_groups_until(remaining, groups, int(len(rows) * val_fraction))
    train = [row for key in remaining if key not in used_val for row in groups[key]]
    return train, val, test


def measure_leakage(
    train: Sequence[Utterance],
    test: Sequence[Utterance],
) -> tuple[float, float]:
    """Mengukur kebocoran transkrip dan tumpang tindih speaker.

    Kebocoran transkrip adalah persentase utterance test yang transkripnya
    juga muncul pada data training. Tumpang tindih speaker adalah
    persentase utterance test yang speaker-nya juga muncul pada data
    training.
    """
    if not test:
        return 0.0, 0.0
    train_texts = {row.text for row in train}
    train_speakers = {row.speaker_id for row in train}
    leaked = sum(1 for row in test if row.text in train_texts)
    overlapped = sum(1 for row in test if row.speaker_id in train_speakers)
    return 100 * leaked / len(test), 100 * overlapped / len(test)


def run_strategy(
    rows: Sequence[Utterance],
    strategy: str,
    seeds: Sequence[int],
    test_fraction: float,
    val_fraction: float,
) -> StrategySummary:
    """Menjalankan satu strategi pembagian pada seluruh seed."""
    outcomes: list[SplitOutcome] = []
    for seed in seeds:
        rng = random.Random(seed)
        if strategy == "random_utterance":
            train, val, test = split_random_utterance(rows, rng, test_fraction, val_fraction)
        elif strategy == "speaker_disjoint":
            train, val, test = split_by_group(
                rows, lambda row: row.speaker_id, rng, test_fraction, val_fraction
            )
        elif strategy == "text_disjoint":
            train, val, test = split_by_group(
                rows, lambda row: row.text, rng, test_fraction, val_fraction
            )
        else:
            raise ValueError(f"Strategi tidak dikenal: {strategy}")

        leak, overlap = measure_leakage(train, test)
        outcomes.append(
            SplitOutcome(
                seed=seed,
                n_train=len(train),
                n_val=len(val),
                n_test=len(test),
                transcript_leak_pct=round(leak, 2),
                speaker_overlap_pct=round(overlap, 2),
            )
        )

    leaks = [item.transcript_leak_pct for item in outcomes]
    overlaps = [item.speaker_overlap_pct for item in outcomes]
    return StrategySummary(
        strategy=strategy,
        n_seeds=len(outcomes),
        transcript_leak_mean=round(statistics.fmean(leaks), 2),
        transcript_leak_sd=round(statistics.stdev(leaks), 2) if len(leaks) > 1 else 0.0,
        transcript_leak_min=min(leaks),
        transcript_leak_max=max(leaks),
        speaker_overlap_mean=round(statistics.fmean(overlaps), 2),
        runs=outcomes,
    )


def sweep_test_fraction(
    rows: Sequence[Utterance],
    seeds: Sequence[int],
    fractions: Sequence[float],
) -> list[dict[str, float]]:
    """Memeriksa kepekaan kebocoran terhadap rasio data test."""
    sweep: list[dict[str, float]] = []
    for fraction in fractions:
        entry: dict[str, float] = {"test_fraction": fraction}
        for strategy in ("random_utterance", "speaker_disjoint"):
            summary = run_strategy(rows, strategy, seeds, fraction, VAL_FRACTION)
            entry[f"{strategy}_mean"] = summary.transcript_leak_mean
            entry[f"{strategy}_min"] = summary.transcript_leak_min
            entry[f"{strategy}_max"] = summary.transcript_leak_max
        sweep.append(entry)
    return sweep


# --------------------------------------------------------------------------
# Graf keterhubungan speaker dan transkrip
# --------------------------------------------------------------------------


def giant_component(rows: Sequence[Utterance]) -> dict[str, float | int]:
    """Menghitung komponen terbesar pada graf bipartit speaker dan transkrip.

    Simpul berupa speaker dan transkrip. Sisi menghubungkan speaker dengan
    setiap transkrip yang dibacanya. Komponen terhubung dicari dengan
    union-find. Besaran yang dilaporkan adalah persentase utterance yang
    berada di dalam komponen terbesar, karena komponen tersebut tidak dapat
    dipisah menjadi bagian speaker-disjoint sekaligus text-disjoint.
    """
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(node: tuple[str, str]) -> tuple[str, str]:
        root = node
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        return root

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_left] = root_right

    for row in rows:
        speaker_node = ("speaker", row.speaker_id)
        text_node = ("text", row.text)
        parent.setdefault(speaker_node, speaker_node)
        parent.setdefault(text_node, text_node)
        union(speaker_node, text_node)

    sizes: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in rows:
        sizes[find(("speaker", row.speaker_id))] += 1

    if not rows or not sizes:
        return {"n_components": 0, "component_sizes_desc": []}

    ordered = sorted(sizes.values(), reverse=True)
    total = len(rows)
    largest = ordered[0]
    second = ordered[1] if len(ordered) > 1 else 0
    remainder = ordered[2:]

    # Kuota data test hanya dapat dipenuhi oleh gabungan komponen utuh, karena
    # membelah satu komponen selalu menautkan kembali speaker atau transkrip di
    # kedua sisi. Ukuran komponen yang tersedia karena itu perlu dilaporkan.
    return {
        "n_components": len(ordered),
        "largest_component_utterances": largest,
        "largest_component_pct": round(100 * largest / total, 2),
        "second_component_utterances": second,
        "second_component_pct": round(100 * second / total, 2),
        "n_remaining_components": len(remainder),
        "remaining_components_utterances": sum(remainder),
        "remaining_components_pct": round(100 * sum(remainder) / total, 3),
        "test_quota_utterances": int(total * TEST_FRACTION),
        "component_sizes_desc": ordered,
    }


# --------------------------------------------------------------------------
# Orkestrasi
# --------------------------------------------------------------------------


def analyse_population(
    name: str,
    rows: Sequence[Utterance],
    seeds: Sequence[int],
) -> dict[str, object]:
    """Menjalankan seluruh analisis untuk satu populasi utterance."""
    print(f"\n[{name}] {len(rows)} utterance", flush=True)

    stats = base_statistics(rows)
    print(
        f"[{name}] kalimat unik {stats['n_unique_transcripts_raw']}, "
        f"rata-rata pembacaan {stats['mean_readings_per_transcript']}, "
        f"speaker {stats['n_unique_speakers']}",
        flush=True,
    )

    summaries: dict[str, object] = {}
    for strategy in STRATEGIES:
        summary = run_strategy(rows, strategy, seeds, TEST_FRACTION, VAL_FRACTION)
        summaries[strategy] = asdict(summary)
        print(
            f"[{name}] {strategy:18s} kebocoran "
            f"{summary.transcript_leak_mean:6.2f}% "
            f"(sd {summary.transcript_leak_sd:.2f}, "
            f"rentang {summary.transcript_leak_min:.2f} sampai "
            f"{summary.transcript_leak_max:.2f}), "
            f"speaker tumpang tindih {summary.speaker_overlap_mean:6.2f}%",
            flush=True,
        )

    graph = giant_component(rows)
    print(
        f"[{name}] graf speaker-transkrip: {graph['n_components']} komponen, "
        f"terbesar {graph['largest_component_utterances']} utterance "
        f"({graph['largest_component_pct']}%), "
        f"kedua {graph['second_component_utterances']} "
        f"({graph['second_component_pct']}%), "
        f"sisa {graph['n_remaining_components']} komponen "
        f"berjumlah {graph['remaining_components_utterances']} utterance "
        f"({graph['remaining_components_pct']}%). "
        f"Kuota data test {graph['test_quota_utterances']} utterance",
        flush=True,
    )

    sweep = sweep_test_fraction(rows, seeds, TEST_FRACTION_SWEEP)

    return {
        "population": name,
        "base_statistics": stats,
        "split_strategies": summaries,
        "test_fraction_sweep": sweep,
        "speaker_transcript_graph": graph,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Audit kebocoran transkrip corpus OpenSLR SLR35 Bahasa Jawa."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Path berkas utt_spk_text.tsv.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Akar folder dataset, dipakai menetapkan keanggotaan subset kerja.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path berkas JSON laporan.",
    )
    parser.add_argument(
        "--skip-disk-check",
        action="store_true",
        help="Tetapkan subset kerja dari prefiks utterance_id, tanpa memindai FLAC.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    print("=" * 78, flush=True)
    print("Audit kebocoran transkrip corpus OpenSLR SLR35", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Metadata           : {args.metadata}", flush=True)
    print(f"Random seed        : {', '.join(str(seed) for seed in SEEDS)}", flush=True)
    print(
        f"Rasio pembagian    : test {TEST_FRACTION:.4f}, validation {VAL_FRACTION:.4f}",
        flush=True,
    )
    print("=" * 78, flush=True)

    if not args.metadata.is_file():
        print(f"GAGAL: metadata tidak ditemukan pada {args.metadata}", file=sys.stderr)
        return 1

    rows = load_metadata(args.metadata)
    if not rows:
        print("GAGAL: metadata kosong.", file=sys.stderr)
        return 1

    dataset_root = None if args.skip_disk_check else args.dataset_root
    subset, subset_method = resolve_working_subset(rows, dataset_root)
    print(f"Keanggotaan subset : {subset_method}", flush=True)

    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "metadata_path": str(args.metadata),
        "seeds": list(SEEDS),
        "test_fraction": round(TEST_FRACTION, 6),
        "val_fraction": round(VAL_FRACTION, 6),
        "working_subset_archives": list(WORKING_ARCHIVES),
        "working_subset_membership_method": subset_method,
        "populations": [
            analyse_population("corpus_penuh", rows, SEEDS),
            analyse_population("subset_kerja", subset, SEEDS),
        ],
    }

    elapsed = round(time.perf_counter() - started, 2)
    report["elapsed_seconds"] = elapsed

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78, flush=True)
    print(f"Laporan ditulis ke : {args.output}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
