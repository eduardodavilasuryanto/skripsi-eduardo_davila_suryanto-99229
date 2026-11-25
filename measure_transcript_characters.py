"""Statistik karakter dan ejaan transkrip corpus penuh SLR35.

Tujuan
------
Menghasilkan ulang, dengan skrip berlog, statistik transkrip yang dikutip
`DraftResearch.md` Subbab 3.5 butir b dan c serta statistik tambahannya, dan
`bab-3.md` Subbab 3.1 dan 3.3.4. Angka-angka itu dihitung pada 10 September
2026 tanpa skrip yang tersimpan (`DraftResearch.md` Lampiran A), sehingga
belum memenuhi Daftar Periksa I.5. Skrip ini lahir pada Sesi 30 ketika
penulisan paper menemukan bahwa angka tersebut tidak ada pada keluaran skrip
mana pun.

Populasinya corpus penuh, yaitu seluruh 185.076 baris metadata
`utt_spk_text.tsv`, sama dengan populasi yang dinyatakan Lampiran A.
Pembacaan metadata memakai `load_metadata` dari `audit_transcript_leakage.py`,
sehingga kolom ketiga dan seterusnya digabung kembali dengan tab dan jumlah
kalimat uniknya sama dengan audit kebocoran.

Beberapa besaran pada naskah tidak menyebut definisinya, misalnya apa yang
dihitung sebagai karakter beraksen atau sebagai bentuk kata. Bagi besaran
semacam itu, skrip melaporkan beberapa definisi sekaligus pada laporan,
sehingga pemilihan definisi tercatat dan tidak dilakukan diam-diam.

Definisi yang dipakai
---------------------
- Karakter berdiakritik: karakter yang dekomposisi NFD-nya memuat tanda
  gabung berkategori Unicode `Mn`.
- Pelepasan diakritik: NFD, buang `Mn`, lalu NFC, sama dengan langkah 3
  normalisasi naskah.
- Kata mentah: potongan transkrip menurut *whitespace*, apa adanya.
- Kata huruf: kata mentah yang di-*casefold* dan hanya mempertahankan
  karakter berkategori `L` atau `M`; kata tanpa huruf dibuang.
- Tanda baca: karakter berkategori Unicode `P`. Tanda baca atau simbol:
  karakter berkategori `P` atau `S`.
- Angka: karakter berkategori Unicode `Nd`.
- Bentuk kata yang menjadi identik bila diakritik dilepas: kata huruf
  berbeda yang hasil pelepasan diakritiknya sama. Laporan memuat jumlah
  kelompoknya, yaitu bentuk tanpa diakritik yang ditulis dalam dua varian
  atau lebih, dan jumlah bentuk tertulis di dalam kelompok itu.

Selain itu, setiap baris mentah metadata dihitung jumlah karakter tabnya,
untuk memeriksa apakah ada transkrip yang memuat tab di dalam isinya.

Skrip hanya membaca, deterministik tanpa random seed, dan tidak memakai GPU.

Masukan
-------
- `../_03_dataset/asr_javanese_0/utt_spk_text.tsv` : metadata corpus penuh.

Keluaran
--------
- `outputs/measure_transcript_characters_report.json` : seluruh statistik,
  daftar karakter langka, perbandingan dengan angka naskah, dan SHA-256
  metadata.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u measure_transcript_characters.py 2>&1 |
        Tee-Object -FilePath logs/measure_transcript_characters.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from audit_transcript_leakage import DEFAULT_METADATA, load_metadata

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_PATH = SCRIPT_DIR / "outputs" / "measure_transcript_characters_report.json"

ACCENT_EXAMPLES = ("é", "è", "ê", "å", "â")
RARE_EXAMPLES = ("ð", "ò", "¥", "—", "Â", "�")
RARE_MAX_COUNT = 4
VARIANT_WORD = "dheweke"
REPLACEMENT_CHARACTER = "�"

# Angka naskah yang dicocokkan, dengan letaknya. Seluruhnya berpopulasi corpus
# penuh menurut DraftResearch.md Lampiran A dan dihitung tanpa skrip.
NASKAH_VALUES = {
    "pct_utterances_with_accent": (26.9, "DraftResearch.md 3.5b; bab-3.md 3.1"),
    "dheweke_variants": (17, "DraftResearch.md 3.5b"),
    "word_form_collision_groups": (2626, "DraftResearch.md 3.5b"),
    "n_distinct_characters": (86, "DraftResearch.md 3.5c; bab-3.md 3.1 dan 3.3.4"),
    "replacement_occurrences": (4, "DraftResearch.md 3.5c"),
    "replacement_utterances": (2, "DraftResearch.md 3.5c; bab-3.md 3.3.4"),
    "pct_utterances_with_uppercase": (99.5, "DraftResearch.md 3.5"),
    "utterances_with_punctuation_or_symbol": (845, "DraftResearch.md 3.5"),
    "utterances_with_digit": (52, "DraftResearch.md 3.5; bab-3.md 3.3.4"),
    "words_mean": (6.9, "DraftResearch.md 3.5"),
    "words_median": (7, "DraftResearch.md 3.5"),
    "words_max": (24, "DraftResearch.md 3.5"),
}
RARE_NASKAH = {"ð": 1, "ò": 1, "¥": 1, "—": 1, "Â": 4, REPLACEMENT_CHARACTER: 4}


# --------------------------------------------------------------------------
# Fungsi teks
# --------------------------------------------------------------------------


def has_diacritic(char: str) -> bool:
    """Apakah dekomposisi NFD karakter memuat tanda gabung Mn."""
    return any(unicodedata.category(part) == "Mn" for part in unicodedata.normalize("NFD", char))


def strip_diacritics(text: str) -> str:
    """Pelepasan diakritik: NFD, buang Mn, lalu NFC."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if unicodedata.category(c) != "Mn"))


def letter_word(token: str) -> str:
    """Kata huruf: casefold, hanya karakter berkategori L atau M."""
    return "".join(c for c in token.casefold() if unicodedata.category(c)[0] in "LM")


def describe(char: str) -> str:
    """Kode titik dan nama Unicode sebuah karakter."""
    return f"U+{ord(char):04X} {unicodedata.name(char, 'TANPA NAMA')}"


def sha256_of(path: Path) -> str:
    """Hash SHA-256 isi berkas."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Pengukuran
# --------------------------------------------------------------------------


def collisions(forms: Iterable[str]) -> dict[str, int]:
    """Bentuk kata yang menjadi identik dengan bentuk lain bila diakritik dilepas."""
    groups: dict[str, set[str]] = defaultdict(set)
    distinct = set(forms)
    for form in distinct:
        groups[strip_diacritics(form)].add(form)
    colliding = [group for group in groups.values() if len(group) > 1]
    return {
        "n_distinct_forms": len(distinct),
        "n_forms_with_diacritic": sum(1 for form in distinct if any(has_diacritic(c) for c in form)),
        "n_forms_colliding": sum(len(group) for group in colliding),
        "n_collision_groups": len(colliding),
        "n_forms_colliding_with_diacritic": sum(
            1 for group in colliding for form in group if any(has_diacritic(c) for c in form)),
    }


def tab_counts(path: Path) -> dict[str, Any]:
    """Jumlah karakter tab pada setiap baris mentah metadata."""
    lines = [line for line in path.read_bytes().split(b"\n") if line.strip()]
    counts = Counter(line.count(b"\t") for line in lines)
    return {
        "n_lines": len(lines),
        "lines_by_tab_count": {str(k): v for k, v in sorted(counts.items())},
        "lines_with_tab_in_transcript": sum(v for k, v in counts.items() if k > 2),
    }


def measure(texts: Sequence[str]) -> dict[str, Any]:
    """Seluruh statistik karakter, ejaan, dan panjang transkrip."""
    n = len(texts)
    characters = Counter(char for text in texts for char in text)
    raw_words = [token for text in texts for token in text.split()]
    letter_words = [word for word in (letter_word(token) for token in raw_words) if word]
    lengths = [len(text.split()) for text in texts]

    rare = {describe(char): count for char, count in sorted(characters.items(), key=lambda item: (item[1], item[0]))
            if count <= RARE_MAX_COUNT}
    accent_any = sum(1 for text in texts if any(has_diacritic(c) for c in text))
    accent_examples = sum(1 for text in texts if any(c in ACCENT_EXAMPLES for c in text.casefold()))
    accent_examples_case = sum(1 for text in texts if any(c in ACCENT_EXAMPLES for c in text))

    def variants(words: Iterable[str], fold: bool) -> list[str]:
        return sorted({w for w in words if strip_diacritics(w.casefold() if fold else w) == VARIANT_WORD})

    return {
        "n_utterances": n,
        "n_unique_transcripts": len(set(texts)),
        "accent": {
            "pct_utterances_any_diacritic": 100 * accent_any / n,
            "pct_utterances_listed_examples_casefold": 100 * accent_examples / n,
            "pct_utterances_listed_examples_case_sensitive": 100 * accent_examples_case / n,
            "listed_examples": list(ACCENT_EXAMPLES),
        },
        "dheweke": {
            "letter_words": variants(letter_words, fold=False),
            "raw_words_casefold": variants(raw_words, fold=True),
            "raw_forms_matching_letter_word": sorted({w for w in raw_words
                                                      if strip_diacritics(letter_word(w)) == VARIANT_WORD}),
        },
        "collisions": {
            "letter_words": collisions(letter_words),
            "raw_words": collisions(raw_words),
        },
        "characters": {
            "n_distinct_all": len(characters),
            "n_distinct_excluding_whitespace": sum(1 for c in characters if not c.isspace()),
            "whitespace_characters": [describe(c) for c in sorted(characters) if c.isspace()],
            "rare_at_most_4": rare,
            "listed_rare_examples": {describe(c): characters.get(c, 0) for c in RARE_EXAMPLES},
        },
        "replacement_character": {
            "occurrences": characters.get(REPLACEMENT_CHARACTER, 0),
            "utterances": sum(1 for text in texts if REPLACEMENT_CHARACTER in text),
        },
        "other": {
            "pct_utterances_with_uppercase": 100 * sum(1 for t in texts if any(c.isupper() for c in t)) / n,
            "utterances_with_punctuation": sum(1 for t in texts
                                               if any(unicodedata.category(c).startswith("P") for c in t)),
            "utterances_with_punctuation_or_symbol": sum(1 for t in texts
                                                         if any(unicodedata.category(c)[0] in "PS" for c in t)),
            "utterances_with_digit": sum(1 for t in texts if any(unicodedata.category(c) == "Nd" for c in t)),
            "words_mean": statistics.fmean(lengths),
            "words_median": statistics.median(lengths),
            "words_max": max(lengths),
        },
    }


def compare(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Angka naskah lawan hasil skrip, dengan definisi utama setiap besaran."""
    measured = {
        "pct_utterances_with_accent": round(stats["accent"]["pct_utterances_any_diacritic"], 1),
        "dheweke_variants": len(stats["dheweke"]["letter_words"]),
        "word_form_collision_groups": stats["collisions"]["letter_words"]["n_collision_groups"],
        "n_distinct_characters": stats["characters"]["n_distinct_all"],
        "replacement_occurrences": stats["replacement_character"]["occurrences"],
        "replacement_utterances": stats["replacement_character"]["utterances"],
        "pct_utterances_with_uppercase": round(stats["other"]["pct_utterances_with_uppercase"], 1),
        "utterances_with_punctuation_or_symbol": stats["other"]["utterances_with_punctuation_or_symbol"],
        "utterances_with_digit": stats["other"]["utterances_with_digit"],
        "words_mean": round(stats["other"]["words_mean"], 1),
        "words_median": stats["other"]["words_median"],
        "words_max": stats["other"]["words_max"],
    }
    result = {key: {"naskah": value, "letak": place, "skrip": measured[key], "sama": measured[key] == value}
              for key, (value, place) in NASKAH_VALUES.items()}
    with_tab = stats["metadata_lines"]["lines_with_tab_in_transcript"]
    result["transcripts_with_tab"] = {
        "naskah": "ada", "letak": "bab-3.md 3.3.1 dan 3.6.1", "skrip": with_tab, "sama": with_tab > 0}
    rare = stats["characters"]["listed_rare_examples"]
    for char, count in RARE_NASKAH.items():
        key = f"rare {describe(char)}"
        result[key] = {"naskah": count, "letak": "DraftResearch.md 3.5c", "skrip": rare[describe(char)],
                       "sama": rare[describe(char)] == count}
    return result


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Statistik karakter dan ejaan transkrip corpus penuh.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Path berkas utt_spk_text.tsv.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help="Berkas laporan; bawaan outputs/measure_transcript_characters_report.json.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print("=" * 78, flush=True)
    print("Statistik karakter dan ejaan transkrip, corpus penuh SLR35", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print(f"Metadata           : {args.metadata}", flush=True)
    print("=" * 78, flush=True)

    texts = [row.text for row in load_metadata(args.metadata)]
    stats = measure(texts)
    stats["metadata_lines"] = tab_counts(args.metadata)
    comparison = compare(stats)
    for key, item in comparison.items():
        status = "sama" if item["sama"] else "BERBEDA"
        line = f"{key:45s} naskah {item['naskah']!s:>8s}  skrip {item['skrip']!s:>8s}  {status}"
        print(line.encode("ascii", "backslashreplace").decode("ascii"), flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": None,
        "population": "corpus_penuh",
        "metadata_path": str(args.metadata),
        "metadata_sha256": sha256_of(args.metadata),
        "statistics": stats,
        "naskah_comparison": comparison,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 78, flush=True)
    print(f"Utterance          : {stats['n_utterances']}", flush=True)
    print(f"Kalimat unik       : {stats['n_unique_transcripts']}", flush=True)
    print(f"Berbeda dari naskah: {sum(not item['sama'] for item in comparison.values())} besaran", flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
