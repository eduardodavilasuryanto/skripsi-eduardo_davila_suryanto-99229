"""Modul bersama analisis utama: hipotesis, cacah galat, dan bootstrap.

Tujuan
------
Menyediakan satu jalur pemuatan dan penarikan ulang bagi seluruh skrip
analisis utama (Subbab 3.6 skripsi), agar WER yang dihitung ulang
pada setiap iterasi bootstrap teragregasi persis seperti Subbab 3.6.1 dan agar
seluruh skrip memakai himpunan resample yang sama.

WER tingkat corpus adalah jumlah substitusi, penghapusan, dan penyisipan atas
seluruh utterance dibagi jumlah kata referensi, dan CER sama pada tingkat
karakter. Cacah galat dan panjang referensi setiap utterance karena itu cukup
dihitung satu kali dengan `jiwer`; metrik satu resample adalah jumlah
berbobot cacah tersebut, dengan bobot berupa berapa kali setiap utterance
terambil. Kesamaan cara ini dengan `asr_common.corpus_metrics` dan dengan
laporan inference diperiksa setiap kali hipotesis dimuat, sehingga
penyimpangan sekecil apa pun menghentikan analisis.

Resample dibangkitkan sebagai cacah multinomial atas utterance data test,
setara dengan penarikan berpengembalian, dalam blok berukuran tetap dari
generator ber-seed. Skrip mana pun yang memakai seed dan jumlah iterasi yang
sama memperoleh resample yang identik, sehingga selang WER, uji H1, dan
selang parameter kurva bertumpu pada penarikan yang sama.

Hasil D1 kedua model berasal dari run percobaan awal yang memakai nilai grid,
sehingga berkas hipotesisnya bernama `*_d1_trial`.

Modul ini tidak dijalankan sendiri. Teks yang dicetak skrip pemakainya
dibatasi pada ASCII, karena keluaran Python yang disalurkan ke PowerShell
memakai halaman kode cp1252.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jiwer
import numpy as np

from asr_common import CER_TRANSFORM, WER_TRANSFORM, corpus_metrics

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
HYPOTHESES_DIR = SCRIPT_DIR / "outputs" / "hypotheses"
RESULTS_DIR = SCRIPT_DIR / "outputs" / "results"
SPLIT_DIR = SCRIPT_DIR / "outputs" / "splits"
SUBSETS_REPORT = SCRIPT_DIR / "outputs" / "build_nested_subsets_report.json"

MODELS: tuple[str, ...] = ("whisper", "xlsr")
LEVELS: tuple[str, ...] = ("d1", "d2", "d3", "d4", "d5")

# Konfigurasi D1 adalah run percobaan awal yang memakai nilai grid.
RUN_NAME_OVERRIDES: dict[tuple[str, str], str] = {
    ("whisper", "d1"): "whisper_d1_trial",
    ("xlsr", "d1"): "xlsr_d1_trial",
}

HYPOTHESIS_HEADER = ["id", "reference_normalized", "hypothesis_normalized", "hypothesis_raw"]

# Seed analisis sama dengan seed pembagian, subset, dan pelatihan.
ANALYSIS_SEED = 42
N_BOOTSTRAP = 10_000
CONFIDENCE = 0.95
ALPHA = 1 - CONFIDENCE

# Ukuran blok penarikan; tetap agar urutan resample identik di seluruh skrip.
BOOTSTRAP_BLOCK = 500

# Toleransi kesamaan metrik hasil cacah dengan laporan inference.
METRIC_TOLERANCE = 1e-12


# --------------------------------------------------------------------------
# Struktur data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorCounts:
    """Cacah galat dan panjang referensi setiap utterance pada satu satuan."""

    errors: np.ndarray
    lengths: np.ndarray

    @property
    def rate(self) -> float:
        """Laju galat tingkat corpus."""
        return float(self.errors.sum() / self.lengths.sum())


@dataclass(frozen=True)
class ConfigData:
    """Hipotesis satu konfigurasi beserta cacah galat kata dan karakternya."""

    model: str
    level: str
    run_name: str
    hours: float
    ids: tuple[str, ...]
    references: tuple[str, ...]
    hypotheses: tuple[str, ...]
    hypotheses_raw: tuple[str, ...]
    words: ErrorCounts
    chars: ErrorCounts
    hypotheses_sha256: str


# --------------------------------------------------------------------------
# Pemuatan
# --------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    """Hash SHA-256 isi berkas, untuk mengikat laporan pada berkas yang tepat."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_name(model: str, level: str) -> str:
    """Nama run sebuah konfigurasi grid."""
    return RUN_NAME_OVERRIDES.get((model, level), f"{model}_{level}")


def level_hours(report_path: Path = SUBSETS_REPORT) -> dict[str, float]:
    """Jam data training terukur setiap tingkat dari laporan `build_nested_subsets.py`."""
    with report_path.open("r", encoding="utf-8") as handle:
        levels = json.load(handle)["levels"]
    return {level: float(levels[level]["hours"]) for level in LEVELS}


def read_hypotheses(path: Path) -> list[list[str]]:
    """Membaca berkas hipotesis dengan dialek yang sama seperti `run_inference.py`."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        if header != HYPOTHESIS_HEADER:
            raise ValueError(f"Header {path.name} tidak dikenal: {header}")
        return [row for row in reader]


def utterance_counts(references: Sequence[str], hypotheses: Sequence[str], unit: str) -> ErrorCounts:
    """Cacah galat per utterance dengan transformasi minimal Subbab 3.6.1.

    Satu pemanggilan `jiwer` atas seluruh pasangan menghasilkan penjajaran per
    utterance; galatnya dihitung dari potongan penjajaran, lalu jumlahnya
    dicocokkan dengan total yang dilaporkan `jiwer` sendiri.
    """
    if unit == "word":
        output = jiwer.process_words(list(references), list(hypotheses),
                                     reference_transform=WER_TRANSFORM,
                                     hypothesis_transform=WER_TRANSFORM)
    elif unit == "char":
        output = jiwer.process_characters(list(references), list(hypotheses),
                                          reference_transform=CER_TRANSFORM,
                                          hypothesis_transform=CER_TRANSFORM)
    else:
        raise ValueError(f"Satuan tidak dikenal: {unit}")

    errors, lengths = [], []
    for reference_tokens, chunks in zip(output.references, output.alignments, strict=True):
        count = 0
        for chunk in chunks:
            if chunk.type in ("substitute", "delete"):
                count += chunk.ref_end_idx - chunk.ref_start_idx
            elif chunk.type == "insert":
                count += chunk.hyp_end_idx - chunk.hyp_start_idx
        errors.append(count)
        lengths.append(len(reference_tokens))

    counts = ErrorCounts(np.asarray(errors, dtype=np.int64), np.asarray(lengths, dtype=np.int64))
    expected = output.substitutions + output.deletions + output.insertions
    if int(counts.errors.sum()) != expected:
        raise RuntimeError(f"Cacah galat per utterance ({counts.errors.sum()}) tidak sama dengan total jiwer ({expected})")
    return counts


def load_configs(
    hypotheses_dir: Path = HYPOTHESES_DIR,
    results_dir: Path = RESULTS_DIR,
    subsets_report: Path = SUBSETS_REPORT,
) -> list[ConfigData]:
    """Memuat kesepuluh konfigurasi dan memeriksa keutuhannya.

    Urutannya Whisper D1 sampai D5 lalu XLS-R D1 sampai D5. Pemeriksaan yang
    menghentikan analisis bila gagal: `id` dan referensi identik di seluruh
    berkas, tidak ada referensi kosong, serta WER dan CER hasil cacah sama
    dengan `corpus_metrics` dan dengan laporan inference.
    """
    hours = level_hours(subsets_report)
    configs: list[ConfigData] = []
    for model in MODELS:
        for level in LEVELS:
            name = run_name(model, level)
            path = hypotheses_dir / f"{name}_test.tsv"
            rows = read_hypotheses(path)
            ids = tuple(row[0] for row in rows)
            references = tuple(row[1] for row in rows)
            hypotheses = tuple(row[2] for row in rows)
            if any(not reference for reference in references):
                raise ValueError(f"{path.name} memuat referensi kosong")
            words = utterance_counts(references, hypotheses, "word")
            chars = utterance_counts(references, hypotheses, "char")

            library = corpus_metrics(references, hypotheses)
            with (results_dir / f"{name}_test_metrics.json").open("r", encoding="utf-8") as handle:
                reported = json.load(handle)["metrics"]
            for metric, value in (("wer", words.rate), ("cer", chars.rate)):
                if abs(value - library[metric]) > METRIC_TOLERANCE or abs(value - reported[metric]) > METRIC_TOLERANCE:
                    raise RuntimeError(f"{metric} {name} dari cacah ({value}) berbeda dengan "
                                       f"corpus_metrics ({library[metric]}) atau laporan ({reported[metric]})")

            configs.append(ConfigData(
                model=model, level=level, run_name=name, hours=hours[level],
                ids=ids, references=references, hypotheses=hypotheses,
                hypotheses_raw=tuple(row[3] for row in rows),
                words=words, chars=chars, hypotheses_sha256=sha256_of(path),
            ))

    first = configs[0]
    if list(first.ids) != sorted(first.ids) or len(set(first.ids)) != len(first.ids):
        raise ValueError("id data test tidak terurut atau tidak unik")
    for config in configs[1:]:
        if config.ids != first.ids or config.references != first.references:
            raise ValueError(f"id atau referensi {config.run_name} berbeda dengan {first.run_name}")
    return configs


def find_config(configs: Sequence[ConfigData], model: str, level: str) -> int:
    """Indeks konfigurasi pada daftar hasil `load_configs`."""
    for index, config in enumerate(configs):
        if config.model == model and config.level == level:
            return index
    raise KeyError(f"Konfigurasi {model} {level} tidak ada")


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def bootstrap_ratios(
    numerators: np.ndarray,
    denominators: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = ANALYSIS_SEED,
) -> np.ndarray:
    """Rasio jumlah berbobot setiap kolom pada setiap resample.

    `numerators` dan `denominators` berukuran (utterance, kolom). Setiap baris
    keluaran adalah satu resample yang sama bagi seluruh kolom, sehingga
    selisih antarkolom pada baris yang sama merupakan selisih berpasangan.
    Resample hanya bergantung pada seed, jumlah utterance, dan urutan blok,
    tidak pada isi kolom.
    """
    n_items = numerators.shape[0]
    probabilities = np.full(n_items, 1.0 / n_items)
    generator = np.random.default_rng(seed)
    output = np.empty((n_bootstrap, numerators.shape[1]), dtype=np.float64)
    for start in range(0, n_bootstrap, BOOTSTRAP_BLOCK):
        size = min(BOOTSTRAP_BLOCK, n_bootstrap - start)
        weights = generator.multinomial(n_items, probabilities, size=size)
        output[start:start + size] = (weights @ numerators) / (weights @ denominators)
    return output


def percentile_interval(samples: np.ndarray, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Selang persentil dari sampel bootstrap; nilai NaN diabaikan."""
    tail = 100 * (1 - confidence) / 2
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return float(low), float(high)


def two_sided_p(samples: np.ndarray, null: float = 0.0) -> float:
    """Nilai p bootstrap dua sisi terhadap nilai nol.

    Memakai koreksi (1 + cacah) / (1 + B) agar nilai p tidak pernah nol pada
    jumlah iterasi yang terbatas.
    """
    values = samples[~np.isnan(samples)]
    total = len(values) + 1
    lower = (1 + int(np.sum(values <= null))) / total
    upper = (1 + int(np.sum(values >= null))) / total
    return float(min(1.0, 2 * min(lower, upper)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Nilai p terkoreksi Holm-Bonferroni, dalam urutan masukan."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def holm_levels(p_values: Sequence[float], alpha: float = ALPHA) -> list[float]:
    """Taraf selang setiap pengujian menurut urutan Holm, dalam urutan masukan."""
    m = len(p_values)
    order = sorted(range(m), key=lambda index: p_values[index])
    levels = [0.0] * m
    for rank, index in enumerate(order):
        levels[index] = 1 - alpha / (m - rank)
    return levels


# --------------------------------------------------------------------------
# Penulisan
# --------------------------------------------------------------------------


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Menulis laporan JSON dengan format yang sama seperti skrip lain."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    """Menulis tabel CSV berpemisah koma dengan akhir baris LF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
