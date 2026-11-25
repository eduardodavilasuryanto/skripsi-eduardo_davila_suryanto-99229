"""Komponen bersama pipeline ASR: normalisasi teks, pemuatan data, dan metrik.

Tujuan
------
Menjadi satu-satunya tempat bagi tiga prosedur yang wajib identik di seluruh
tahap pelatihan, validation, dan inference, yaitu preprocessing teks
(`bab-3.md` Subbab 3.3.4), pembacaan berkas pembagian beserta audionya, dan
perhitungan WER serta CER (Subbab 3.6.1). Modul ini tidak dijalankan sendiri;
ia diimpor oleh `train_asr.py` dan `run_inference.py`.

Kelas dataset, penyandi, dan collator diletakkan di sini, bukan di skrip
pelatihan, karena pekerja `DataLoader` pada Windows memuat ulang kelas lewat
impor modul. Kelas yang didefinisikan pada `__main__` rawan gagal dimuat ulang.

Masukan
-------
- Berkas TSV keluaran `build_text_disjoint_split.py` dan
  `build_nested_subsets.py`, berkolom `id`, `speaker_id`, `duration_seconds`,
  `transcript`.
- Berkas FLAC pada akar dataset `audit_transcript_leakage.DEFAULT_DATASET_ROOT`.
"""

from __future__ import annotations

import csv
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import jiwer
import numpy as np
import soundfile
import torch
from torch.utils.data import Dataset

from audit_transcript_leakage import DEFAULT_DATASET_ROOT

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SAMPLING_RATE = 16_000

# Inventaris akhir preprocessing teks: 26 huruf Latin dan spasi (Subbab 3.3.4).
ALLOWED_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz ")

# Tanda baca yang disebut eksplisit pada `DraftResearch.md` Subbab 8.3 langkah 4.
# Dipakai hanya untuk diagnosis: karakter terhapus di luar daftar ini dicatat ke
# log agar penghapusan simbol yang tidak disebut naskah tidak terjadi diam-diam.
LISTED_PUNCTUATION = frozenset("?!./'~–—\"“”")

# Tanda baca yang memisahkan dua kata tanpa spasi, misalnya "désa/kalurahan".
# Diganti spasi, bukan dihapus, agar tidak lahir token gabungan seperti
# "desakalurahan" (Utang nomor 30, Sesi 18). Apostrof sengaja tidak termasuk
# karena berada di dalam satu kata, misalnya "dike'i".
WORD_SEPARATORS = frozenset("/")

# Rangkaian transformasi `jiwer` yang diwajibkan Subbab 3.6.1: hanya pemecahan
# menjadi token, tanpa perubahan isi teks.
WER_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfWords()])
CER_TRANSFORM = jiwer.Compose([jiwer.ReduceToListOfListOfChars()])

LABEL_PAD = -100


# --------------------------------------------------------------------------
# Preprocessing teks
# --------------------------------------------------------------------------


def _strip_diacritics(text: str) -> str:
    """Langkah 3: dekomposisi NFD lalu penghapusan *combining marks*."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(text: str) -> str:
    """Langkah 1 sampai 4 dan 6 preprocessing teks Subbab 3.3.4.

    Langkah 5, yaitu pengeluaran utterance berangka atau bermuatan U+FFFD,
    bukan transformasi teks dan sudah dijalankan sebelum pembagian data.

    Langkah 4 diwujudkan sebagai penghapusan setiap karakter di luar 26 huruf
    dan spasi, sehingga inventaris akhirnya tepat seperti yang dinyatakan
    naskah. Pengecualiannya `WORD_SEPARATORS`, yaitu garis miring, yang
    diganti spasi karena memisahkan dua kata. Pada subset kerja tersaring,
    karakter yang terhapus berupa tanda baca yang disebut naskah ditambah satu
    huruf `ð`. Seluruh *whitespace*, termasuk tab, lebih dulu dijadikan spasi,
    lalu spasi dirapatkan dan teks dipangkas, karena `jiwer` dipanggil tanpa
    `Strip` maupun `RemoveMultipleSpaces` (Subbab 3.6.1).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = _strip_diacritics(text)
    text = "".join(
        " " if char.isspace() or char in WORD_SEPARATORS else char for char in text
    )
    text = "".join(char for char in text if char in ALLOWED_CHARACTERS)
    return " ".join(text.split())


def unlisted_removed_characters(text: str) -> set[str]:
    """Karakter yang dihapus langkah 4 tetapi tidak disebut sebagai tanda baca."""
    prepared = _strip_diacritics(unicodedata.normalize("NFC", text).lower())
    return {
        char for char in prepared
        if char not in ALLOWED_CHARACTERS and char not in LISTED_PUNCTUATION
        and not char.isspace()
    }


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Utterance:
    """Satu utterance berkas pembagian beserta transkrip ternormalisasinya."""

    utt_id: str
    speaker_id: str
    duration: float
    transcript: str
    text: str


def read_split(path: Path) -> list[Utterance]:
    """Membaca satu berkas TSV pembagian dengan dialek yang sama seperti penulisnya."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            Utterance(
                record["id"],
                record["speaker_id"],
                float(record["duration_seconds"]),
                record["transcript"],
                normalize_text(record["transcript"]),
            )
            for record in csv.DictReader(handle, delimiter="\t")
        ]


def flac_path(utt_id: str, dataset_root: Path = DEFAULT_DATASET_ROOT) -> Path:
    """Lokasi berkas FLAC: arsip ditentukan digit heksadesimal pertama `id`."""
    archive = f"asr_javanese_{int(utt_id[0], 16)}"
    return dataset_root / archive / "data" / utt_id[:2] / f"{utt_id}.flac"


def load_audio(utt_id: str, dataset_root: Path = DEFAULT_DATASET_ROOT) -> np.ndarray:
    """Membaca audio 16 kHz mono tanpa *resampling* (Subbab 3.3.4)."""
    audio, rate = soundfile.read(str(flac_path(utt_id, dataset_root)), dtype="float32")
    if rate != SAMPLING_RATE or audio.ndim != 1:
        raise ValueError(f"{utt_id}: format audio {rate} Hz, {audio.ndim} dimensi")
    return audio


class WhisperEncoder:
    """Mengubah satu utterance menjadi log-mel dan token label Whisper."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, audio: np.ndarray, text: str) -> dict[str, Any]:
        features = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE)
        return {
            "input_features": features.input_features[0],
            "labels": self.processor.tokenizer(text).input_ids,
        }


class CTCEncoder:
    """Mengubah satu utterance menjadi bentuk gelombang ternormalisasi dan label CTC."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, audio: np.ndarray, text: str) -> dict[str, Any]:
        values = self.processor.feature_extractor(audio, sampling_rate=SAMPLING_RATE)
        return {
            "input_values": values.input_values[0],
            "labels": self.processor.tokenizer(text).input_ids,
        }


class SpeechDataset(Dataset):
    """Dataset yang membaca audio saat diakses, agar tidak ada salinan fitur di disk."""

    def __init__(
        self,
        utterances: Sequence[Utterance],
        encode: Callable[[np.ndarray, str], dict[str, Any]],
        dataset_root: Path = DEFAULT_DATASET_ROOT,
    ) -> None:
        self.utterances = list(utterances)
        self.encode = encode
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.utterances)

    def __getitem__(self, index: int) -> dict[str, Any]:
        utt = self.utterances[index]
        return self.encode(load_audio(utt.utt_id, self.dataset_root), utt.text)


class WhisperCollator:
    """Menyatukan log-mel dan memberi label padding −100 bagi Whisper.

    Token awal transkripsi dibuang dari label karena model menambahkannya
    sendiri saat menggeser label menjadi masukan *decoder*.
    """

    def __init__(self, processor: Any, decoder_start_token_id: int) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": item["input_features"]} for item in features],
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features], return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), LABEL_PAD)
        if bool((labels[:, 0] == self.decoder_start_token_id).all()):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class CTCCollator:
    """Menyatukan bentuk gelombang berpanjang beragam dan label CTC."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_values": item["input_values"]} for item in features],
            padding=True,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": item["labels"]} for item in features], padding=True, return_tensors="pt"
        )
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), LABEL_PAD
        )
        return batch


# --------------------------------------------------------------------------
# Metrik
# --------------------------------------------------------------------------


def corpus_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float | int]:
    """WER dan CER tingkat corpus dengan transformasi minimal Subbab 3.6.1.

    Pasangan berreferensi kosong dikeluarkan dan jumlahnya dilaporkan, karena
    `jiwer` mengembalikan cacah galat alih-alih rasio untuk referensi kosong.
    """
    pairs = [(ref, hyp) for ref, hyp in zip(references, hypotheses, strict=True) if ref]
    refs = [ref for ref, _ in pairs]
    hyps = [hyp for _, hyp in pairs]
    return {
        "wer": jiwer.wer(refs, hyps, reference_transform=WER_TRANSFORM,
                         hypothesis_transform=WER_TRANSFORM),
        "cer": jiwer.cer(refs, hyps, reference_transform=CER_TRANSFORM,
                         hypothesis_transform=CER_TRANSFORM),
        "n_utterances": len(pairs),
        "n_excluded_empty_reference": len(references) - len(pairs),
        "n_empty_hypothesis": sum(1 for hyp in hyps if not hyp),
    }
