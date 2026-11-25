"""Fine-tuning satu konfigurasi ASR, Whisper small atau XLS-R 300M.

Tujuan
------
Melatih satu sel matriks konfigurasi (`bab-3.md` Tabel 3.6) dari berkas
konfigurasi YAML, sehingga perbedaan antarkonfigurasi dapat dibuktikan dari
berkas konfigurasinya saja. Aturan yang identik bagi seluruh konfigurasi:

- Presisi, status TF32, dan *flag* determinisme ditetapkan eksplisit dari
  konfigurasi sebelum model dimuat (Tabel 3.12).
- *Batch size* efektif = *batch* per langkah × akumulasi gradien, dan akumulasi
  dihitung dari konfigurasi agar *batch* efektif identik (Subbab 3.4.3).
- Setiap langkah optimizer memuat tepat *batch* efektif. Setiap epoch mengacak
  ulang seluruh utterance training lalu memakai kelipatan *batch* efektif
  pertama; sisanya, kurang dari satu *batch* efektif, disisihkan dan berganti
  antarepoch (`DraftResearch.md` Lampiran B.12). Loss dirata-ratakan atas
  akumulasi gradien, sehingga gradien setiap langkah adalah rerata, bukan
  jumlah, atas *micro-batch*-nya (Lampiran B.14).
- Lama pelatihan ditentukan *early stopping* atas WER validation: evaluasi
  setiap `eval_steps` langkah, berhenti bila WER validation tidak membaik pada
  `early_stopping_patience` evaluasi terakhir, tetapi tidak sebelum
  `early_stopping_min_steps` dan tidak melewati `max_steps`. *Checkpoint*
  dengan WER validation terendah dimuat kembali dan disimpan sebagai model.
  Batas bawah langkah mencegah penghentian pada dataran awal pelatihan CTC,
  ketika WER tertahan 100% sebelum penjajaran terbentuk; batas itu tidak
  memengaruhi pemilihan *checkpoint* terbaik.
- Memori yang tercadang selama training dilepas sebelum dan sesudah setiap
  evaluasi, dan laju training dihitung tanpa waktu evaluasi maupun
  penyimpanan *checkpoint* (`DraftResearch.md` Lampiran B.10 dan B.11).
- Allocator CUDA memakai `expandable_segments`, agar cadangan memori tidak
  terfragmentasi oleh panjang masukan yang berubah-ubah (Lampiran B.13).
- Model prapelatihan dimuat pada revisi yang dikunci konfigurasi (`revision`,
  hash komit 40 karakter) dan hanya dari cache lokal, tanpa menghubungi
  Hugging Face Hub, agar kesepuluh konfigurasi tidak diam-diam memakai bobot
  awal yang berbeda bila repositori model berubah. Hash yang benar-benar
  dimuat dicocokkan dengan `revision` (Utang nomor 31, Sesi 18).

Mode `--probe-vram` tidak melatih model. Mode ini mencoba beberapa *batch* per
langkah pada utterance terpanjang *training pool*, yaitu kasus terburuk bagi
VRAM di seluruh tingkat data, lalu melaporkan *batch* terbesar yang muat tanpa
melampaui VRAM bebas. Pada driver WDDM, alokasi yang melampaui VRAM tidak
selalu gagal tetapi dilimpahkan ke RAM sistem, sehingga batas muat ditentukan
dari VRAM bebas, bukan dari ada tidaknya galat kehabisan memori.

Masukan
-------
- `--config` : berkas YAML, contoh `configs/whisper_d1.yaml`.
- Berkas subset training (`outputs/subsets/`), data validation
  (`outputs/splits/validation.tsv`), dan audio FLAC.
- Model prapelatihan pada revisi `revision` di cache Hugging Face. Di mesin
  yang belum memilikinya, unduh lebih dulu secara eksplisit, misalnya
  `hf download openai/whisper-small --revision <hash>`; skrip ini sengaja
  tidak mengunduh apa pun.

Konfigurasi probe yang dieksekusi sebelum Sesi 18 tidak memuat `revision`
sehingga tidak lolos pemeriksaan versi skrip ini; ketiganya direproduksi
dengan `train_asr.py` pada *commit* `b84332a`.

Keluaran
--------
- `outputs/runs/<run_name>/` : salinan konfigurasi, `run_report.json`
  (konfigurasi, hasil, lingkungan tiga lapis dari `log_environment.py`,
  peringatan determinisme), `trainer_state.json`.
- `outputs/models/<run_name>/` : model dengan WER validation terendah beserta
  *processor*-nya. Folder ini tidak masuk git karena ukurannya.
- Mode probe: `outputs/runs/<run_name>/probe_vram.json`.

Perintah
--------
    python -u train_asr.py --config configs/whisper_d1.yaml 2>&1 |
        Tee-Object -FilePath logs/whisper_d1.log
    python -u train_asr.py --config configs/trial/whisper_probe.yaml --probe-vram 2>&1 |
        Tee-Object -FilePath logs/whisper_probe.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import Sampler
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from asr_common import (
    LABEL_PAD,
    CTCCollator,
    CTCEncoder,
    SpeechDataset,
    WhisperCollator,
    WhisperEncoder,
    corpus_metrics,
    normalize_text,
    read_split,
    unlisted_removed_characters,
)
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "outputs" / "runs"
MODELS_DIR = SCRIPT_DIR / "outputs" / "models"
TRAIN_POOL = SCRIPT_DIR / "outputs" / "splits" / "train_pool.tsv"

MODEL_FAMILIES = ("whisper", "xlsr")
REQUIRED_KEYS = (
    "run_name", "model_family", "pretrained", "revision", "train_subset", "validation_split",
    "seed", "precision", "tf32", "deterministic", "effective_batch_size",
    "per_device_train_batch_size", "per_device_eval_batch_size", "learning_rate",
    "warmup_steps", "lr_scheduler_type", "optim", "max_steps", "eval_steps",
    "early_stopping_patience", "early_stopping_min_steps", "gradient_checkpointing",
    "logging_steps", "dataloader_num_workers", "save_only_model",
    "keep_checkpoints", "model_overrides",
)

# Kosakata CTC tetap bagi seluruh tingkat data: 26 huruf, pembatas kata, dan
# dua token khusus (Subbab 3.3.4). Kosakata yang diturunkan dari data training
# akan berbeda antartingkat bila sebuah huruf tidak muncul pada D1.
CTC_VOCAB = ["<pad>", "<unk>", "|"] + list("abcdefghijklmnopqrstuvwxyz")

# Revisi model prapelatihan wajib berupa hash komit, bukan nama cabang seperti
# `main` yang dapat berpindah (Utang nomor 31).
COMMIT_HASH = re.compile(r"[0-9a-f]{40}")

# Allocator CUDA memperluas satu segmen memori alih-alih memesan blok baru
# untuk setiap ukuran tensor. Tanpa ini, panjang masukan XLS-R yang berubah pada
# setiap micro-batch memfragmentasi cadangan memori sampai melampaui VRAM fisik
# meskipun memori yang benar-benar dipakai tetap (Lampiran B.13).
CUDA_ALLOC_CONF = "expandable_segments:True"

# Ruang aman di bawah VRAM bebas saat probe, dalam MiB.
PROBE_MARGIN_MIB = 256
PROBE_STEPS_PER_SIZE = 2
BYTES_PER_MIB = 1024 ** 2


# --------------------------------------------------------------------------
# Konfigurasi dan keadaan numerik
# --------------------------------------------------------------------------


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    """Membaca dan memeriksa konfigurasi; nilai kembalian kedua adalah SHA-256-nya."""
    raw = path.read_bytes()
    config = yaml.safe_load(raw.decode("utf-8"))
    missing = [key for key in REQUIRED_KEYS if key not in config]
    if missing:
        raise ValueError(f"Kunci konfigurasi tidak ada: {missing}")
    if config["model_family"] not in MODEL_FAMILIES:
        raise ValueError(f"model_family harus salah satu dari {MODEL_FAMILIES}")
    if not COMMIT_HASH.fullmatch(str(config["revision"])):
        raise ValueError("revision harus hash komit 40 karakter heksadesimal, bukan nama cabang")
    if config["model_family"] == "whisper" and "generation_max_new_tokens" not in config:
        raise ValueError("Konfigurasi Whisper wajib memuat generation_max_new_tokens")
    if config["precision"] not in ("bf16", "fp32"):
        raise ValueError("precision harus bf16 atau fp32")
    if config["effective_batch_size"] % config["per_device_train_batch_size"]:
        raise ValueError("effective_batch_size harus habis dibagi per_device_train_batch_size")
    if config["max_steps"] % config["eval_steps"]:
        raise ValueError("max_steps harus kelipatan eval_steps agar evaluasi terakhir terjadi")
    return config, hashlib.sha256(raw).hexdigest()


def gradient_accumulation(config: dict[str, Any]) -> int:
    """Akumulasi gradien yang menjaga *batch* efektif tetap."""
    return config["effective_batch_size"] // config["per_device_train_batch_size"]


def apply_numerics(config: dict[str, Any]) -> None:
    """Menetapkan status TF32, *flag* determinisme, dan allocator CUDA secara eksplisit.

    `CUBLAS_WORKSPACE_CONFIG` wajib ditetapkan sebelum konteks CUDA terbentuk
    agar *matmul* cuBLAS deterministik. `warn_only=True` membuat operasi yang
    tidak memiliki implementasi deterministik, misalnya *backward* CTC loss di
    CUDA, tetap berjalan dengan peringatan yang dicatat ke laporan.
    `PYTORCH_CUDA_ALLOC_CONF` dibaca saat allocator pertama kali dipakai,
    sehingga juga ditetapkan di sini; pengaturan ini mengubah cara memori
    dicadangkan, bukan perhitungan (Lampiran B.13).
    """
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = CUDA_ALLOC_CONF
    if config["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["tf32"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(config["deterministic"])
    torch.use_deterministic_algorithms(bool(config["deterministic"]), warn_only=True)


def prevent_background_downloads() -> None:
    """Mencegah `transformers` mengunduh konversi *safetensors* di latar belakang.

    Bila bobot prapelatihan hanya tersedia sebagai `.bin`, seperti XLS-R 300M,
    `from_pretrained` meluncurkan thread non-daemon yang mengunduh versi
    *safetensors* dari *pull request* konversi, sehingga proses tidak dapat
    keluar sebelum unduhan itu selesai. `use_safetensors=False` tidak
    mencegahnya. Variabel ini dibaca saat `from_pretrained` dipanggil, bukan
    saat impor, sehingga cukup ditetapkan sebelum model dimuat.
    """
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"


def numerics_state() -> dict[str, Any]:
    """Status numerik yang benar-benar berlaku, dibaca ulang dari `torch`."""
    return {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }


class WarningRecorder:
    """Mencatat pesan peringatan unik, terutama peringatan operasi nondeterministik."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self._original = warnings.showwarning

    def install(self) -> None:
        """Memasang pencatat tanpa mematikan pencetakan peringatan ke log."""
        def record(message: Any, category: Any, filename: str, lineno: int,
                   file: Any = None, line: str | None = None) -> None:
            text = f"{category.__name__}: {message}"
            if text not in self.messages:
                self.messages.append(text)
            self._original(message, category, filename, lineno, file, line)
        warnings.showwarning = record

    def determinism_messages(self) -> list[str]:
        """Peringatan yang menyangkut determinisme saja."""
        return [text for text in self.messages if "determinis" in text.lower()]


# --------------------------------------------------------------------------
# Model dan processor
# --------------------------------------------------------------------------


def pinned_source(config: dict[str, Any]) -> dict[str, Any]:
    """Argumen pemuatan yang mengunci revisi dan melarang akses jaringan.

    Tanpa `local_files_only`, `from_pretrained` tetap menghubungi Hub untuk
    berkas opsional meskipun revisinya hash komit; bila jaringan terputus,
    pemuatan tertunda beberapa menit oleh percobaan ulang (Sesi 18).
    """
    return {"revision": config["revision"], "local_files_only": True}


def verify_revision(model: Any, config: dict[str, Any]) -> None:
    """Memastikan bobot yang dimuat berasal dari revisi yang dikunci konfigurasi.

    Processor tidak menyimpan hash revisinya, sehingga yang dapat diperiksa
    hanya model; processor dimuat dengan argumen pengunci yang sama.
    """
    loaded = getattr(model.config, "_commit_hash", None)
    if loaded != config["revision"]:
        raise RuntimeError(f"Revisi model yang dimuat {loaded} berbeda dari konfigurasi {config['revision']}")


def build_whisper(config: dict[str, Any]) -> tuple[Any, Any, Callable, Callable]:
    """Model Whisper, processor, penyandi, dan collator untuk bahasa Jawa (`jw`)."""
    processor = WhisperProcessor.from_pretrained(
        config["pretrained"], language="javanese", task="transcribe", **pinned_source(config)
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        config["pretrained"], **pinned_source(config), **config["model_overrides"]
    )
    verify_revision(model, config)
    model.generation_config.language = "javanese"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    # Batas yang sama dipakai evaluasi validation dan inference test.
    model.generation_config.max_new_tokens = config["generation_max_new_tokens"]
    if config["gradient_checkpointing"]:
        model.config.use_cache = False
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)
    return model, processor, WhisperEncoder(processor), collator


def build_ctc_processor(vocab_dir: Path, config: dict[str, Any]) -> Wav2Vec2Processor:
    """Processor XLS-R dengan kosakata tetap `CTC_VOCAB`."""
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_file = vocab_dir / "vocab.json"
    with vocab_file.open("w", encoding="utf-8") as handle:
        json.dump({token: index for index, token in enumerate(CTC_VOCAB)}, handle, ensure_ascii=False)
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_file), unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|",
        bos_token=None, eos_token=None, do_lower_case=False,
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(config["pretrained"], **pinned_source(config))
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_xlsr(config: dict[str, Any], vocab_dir: Path) -> tuple[Any, Any, Callable, Callable]:
    """Model XLS-R dengan kepala CTC baru, processor, penyandi, dan collator."""
    processor = build_ctc_processor(vocab_dir, config)
    model = Wav2Vec2ForCTC.from_pretrained(
        config["pretrained"],
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        **pinned_source(config),
        **config["model_overrides"],
    )
    verify_revision(model, config)
    model.freeze_feature_encoder()
    return model, processor, CTCEncoder(processor), CTCCollator(processor)


def build(config: dict[str, Any], vocab_dir: Path) -> tuple[Any, Any, Callable, Callable]:
    """Memilih pembangun sesuai `model_family`."""
    if config["model_family"] == "whisper":
        return build_whisper(config)
    return build_xlsr(config, vocab_dir)


# --------------------------------------------------------------------------
# Metrik validation
# --------------------------------------------------------------------------


def make_compute_metrics(config: dict[str, Any], processor: Any) -> Callable[[Any], dict[str, float]]:
    """WER validation dengan normalisasi dan agregasi yang sama seperti evaluasi test."""
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id

    def compute(prediction: Any) -> dict[str, float]:
        pred_ids = np.where(prediction.predictions != LABEL_PAD, prediction.predictions, pad_id)
        label_ids = np.where(prediction.label_ids != LABEL_PAD, prediction.label_ids, pad_id)
        if config["model_family"] == "whisper":
            hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        else:
            hyps = tokenizer.batch_decode(pred_ids)
            refs = tokenizer.batch_decode(label_ids, group_tokens=False)
        metrics = corpus_metrics([normalize_text(r) for r in refs], [normalize_text(h) for h in hyps])
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    return compute


def argmax_logits(logits: torch.Tensor | tuple, labels: torch.Tensor) -> torch.Tensor:
    """Menyimpan hanya indeks argmax CTC agar evaluasi tidak menumpuk logit penuh."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# --------------------------------------------------------------------------
# Callback
# --------------------------------------------------------------------------


class EarlyStoppingWithFloor(TrainerCallback):
    """Berhenti bila WER validation tidak membaik pada `patience` evaluasi terakhir.

    Penghentian tidak pernah terjadi sebelum `min_steps`. Hitungan evaluasi tanpa
    perbaikan tetap berjalan sebelum batas itu, sehingga aturannya dapat
    dinyatakan dalam satu kalimat dan sama bagi seluruh konfigurasi.
    """

    def __init__(self, patience: int, min_steps: int) -> None:
        self.patience = patience
        self.min_steps = min_steps
        self.best: float | None = None
        self.best_step: int | None = None
        self.evals_without_improvement = 0
        self.history: list[dict[str, float | int]] = []

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float], **kwargs: Any) -> None:
        wer = metrics["eval_wer"]
        self.history.append({"step": state.global_step, "wer": wer, "cer": metrics.get("eval_cer")})
        if self.best is None or wer < self.best:
            self.best, self.best_step = wer, state.global_step
            self.evals_without_improvement = 0
        else:
            self.evals_without_improvement += 1
        print(f"[evaluasi] langkah {state.global_step}: WER {wer:.4f}, terbaik {self.best:.4f} "
              f"pada langkah {self.best_step}, tanpa perbaikan {self.evals_without_improvement} kali",
              flush=True)
        if state.global_step >= self.min_steps and self.evals_without_improvement >= self.patience:
            print("[evaluasi] early stopping terpicu", flush=True)
            control.should_training_stop = True


class TrainingMonitor(TrainerCallback):
    """Mengukur laju training dan biaya evaluasi, serta melepas memori di sekitar evaluasi.

    Laju training dihitung dari selang antara dua akhir langkah, kecuali selang
    yang memuat evaluasi atau penyimpanan *checkpoint*. `train_runtime` keluaran
    `Trainer` mencakup keduanya sehingga tidak dipakai sebagai laju training
    (Lampiran B.11). Langkah pertama dicatat terpisah karena memuat waktu
    menyalakan *worker* dataloader.

    Blok memori yang tercadang selama training dilepas dengan `empty_cache`
    sebelum dan sesudah setiap evaluasi, agar evaluasi tidak menambah alokasi
    di atas blok training lalu melimpah ke RAM sistem (Lampiran B.10).
    Pelepasan ini hanya mengembalikan blok yang tidak terpakai ke driver dan
    tidak mengubah perhitungan. VRAM puncak training dicetak agar pelimpahan
    WDDM terlihat. Puncak teralokasi dicatat berdampingan dengan puncak
    tercadang, karena selisih keduanya adalah memori yang ditahan *caching
    allocator* tanpa dipakai tensor, sehingga fragmentasi terlihat (Lampiran
    B.13).
    """

    def __init__(self) -> None:
        self.last_step_end = time.perf_counter()
        self.pause_in_interval = False
        self.first_step_seconds: float | None = None
        self.step_seconds: list[float] = []
        self.logged_intervals = 0
        self.train_peak_mib = 0.0
        self.train_peak_allocated_mib = 0.0
        self.reserved_before_eval_mib = 0.0
        self.evaluations: dict[int, dict[str, float]] = {}
        self.eval_seconds_total = 0.0

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self.last_step_end = time.perf_counter()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        now = time.perf_counter()
        if self.first_step_seconds is None:
            self.first_step_seconds = now - self.last_step_end
        elif not self.pause_in_interval:
            self.step_seconds.append(now - self.last_step_end)
        self.last_step_end = now
        self.pause_in_interval = bool(control.should_evaluate or control.should_save)
        self.train_peak_mib = max(self.train_peak_mib, torch.cuda.max_memory_reserved() / BYTES_PER_MIB)
        self.train_peak_allocated_mib = max(self.train_peak_allocated_mib,
                                            torch.cuda.max_memory_allocated() / BYTES_PER_MIB)
        if control.should_evaluate:
            torch.cuda.empty_cache()
            self.reserved_before_eval_mib = torch.cuda.memory_reserved() / BYTES_PER_MIB
            torch.cuda.reset_peak_memory_stats()

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: dict[str, float], **kwargs: Any) -> None:
        cost = {
            "eval_seconds": round(metrics.get("eval_runtime", 0.0), 1),
            "eval_reserved_start_mib": round(self.reserved_before_eval_mib),
            "eval_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / BYTES_PER_MIB),
            "eval_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / BYTES_PER_MIB),
        }
        self.evaluations[state.global_step] = cost
        self.eval_seconds_total += metrics.get("eval_runtime", 0.0)
        print(f"[evaluasi] langkah {state.global_step}: {cost['eval_seconds']} detik, VRAM tercadang "
              f"awal {cost['eval_reserved_start_mib']} MiB, puncak {cost['eval_peak_reserved_mib']} MiB, "
              f"puncak teralokasi {cost['eval_peak_allocated_mib']} MiB", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if not logs or "loss" not in logs:
            return
        recent = self.step_seconds[self.logged_intervals:]
        self.logged_intervals = len(self.step_seconds)
        if recent:
            rate = f"{sum(recent) / len(recent):.2f} s/langkah"
        elif state.global_step == 1:
            rate = f"{self.first_step_seconds:.2f} s pada langkah pertama"
        else:
            rate = "selang memuat evaluasi atau penyimpanan, tidak dihitung"
        print(f"[progres] langkah {state.global_step}: {rate}, "
              f"VRAM puncak training tercadang {self.train_peak_mib:.0f} MiB, "
              f"teralokasi {self.train_peak_allocated_mib:.0f} MiB", flush=True)

    def summary(self) -> dict[str, Any]:
        """Ringkasan laju training dan biaya evaluasi bagi laporan run."""
        timed = len(self.step_seconds)
        return {
            "seconds_per_step": round(sum(self.step_seconds) / timed, 3) if timed else None,
            "timed_steps": timed,
            "first_step_seconds": round(self.first_step_seconds or 0.0, 1),
            "train_peak_reserved_mib": round(self.train_peak_mib),
            "train_peak_allocated_mib": round(self.train_peak_allocated_mib),
            "evaluation_seconds_total": round(self.eval_seconds_total, 1),
        }


# --------------------------------------------------------------------------
# Probe VRAM
# --------------------------------------------------------------------------


def make_optimizer(config: dict[str, Any], params: Any) -> torch.optim.Optimizer:
    """Optimizer yang sama jenisnya dengan yang dipakai `Trainer`."""
    if config["optim"] == "adamw_torch":
        return torch.optim.AdamW(params, lr=config["learning_rate"])
    if config["optim"] == "adamw_bnb_8bit":
        import bitsandbytes
        return bitsandbytes.optim.AdamW8bit(params, lr=config["learning_rate"])
    raise ValueError(f"optim {config['optim']} tidak didukung probe")


def probe_vram(config: dict[str, Any], run_dir: Path, sizes: Sequence[int], eval_sizes: Sequence[int]) -> dict[str, Any]:
    """Mencoba *batch* per langkah pada utterance terpanjang *training pool*."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    budget_mib = free_bytes / BYTES_PER_MIB - PROBE_MARGIN_MIB
    print(f"VRAM bebas         : {free_bytes / BYTES_PER_MIB:.0f} dari {total_bytes / BYTES_PER_MIB:.0f} MiB, "
          f"anggaran {budget_mib:.0f} MiB", flush=True)

    model, processor, encode, collate = build(config, run_dir)
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to("cuda").train()
    optimizer = make_optimizer(config, [p for p in model.parameters() if p.requires_grad])
    dtype = torch.bfloat16 if config["precision"] == "bf16" else torch.float32
    longest = sorted(read_split(TRAIN_POOL), key=lambda utt: utt.duration, reverse=True)
    dataset = SpeechDataset(longest[: max(list(sizes) + list(eval_sizes))], encode)

    results: dict[str, Any] = {"budget_mib": round(budget_mib), "train": [], "eval": []}
    for size in sorted(sizes):
        batch = {k: v.to("cuda") for k, v in collate([dataset[i] for i in range(size)]).items()}
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(PROBE_STEPS_PER_SIZE):
                with torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                    loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / BYTES_PER_MIB
            entry = {"batch": size, "peak_reserved_mib": round(peak), "fits": peak <= budget_mib}
        except torch.OutOfMemoryError:
            entry = {"batch": size, "peak_reserved_mib": None, "fits": False, "oom": True}
        del batch
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        results["train"].append(entry)
        print(f"Probe train batch {size:3d} : {entry}", flush=True)
        if not entry["fits"]:
            break

    # Evaluasi berjalan saat state optimizer masih menempati VRAM.
    model.eval()
    for size in sorted(eval_sizes):
        batch = collate([dataset[i] for i in range(size)])
        torch.cuda.reset_peak_memory_stats()
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                if config["model_family"] == "whisper":
                    model.generate(batch["input_features"].to("cuda"), max_new_tokens=config["generation_max_new_tokens"])
                else:
                    model(input_values=batch["input_values"].to("cuda"),
                          attention_mask=batch["attention_mask"].to("cuda"))
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / BYTES_PER_MIB
            entry = {"batch": size, "peak_reserved_mib": round(peak), "fits": peak <= budget_mib}
        except torch.OutOfMemoryError:
            entry = {"batch": size, "peak_reserved_mib": None, "fits": False, "oom": True}
        torch.cuda.empty_cache()
        results["eval"].append(entry)
        print(f"Probe eval batch {size:3d}  : {entry}", flush=True)
        if not entry["fits"]:
            break

    fitting = [entry["batch"] for entry in results["train"] if entry["fits"]]
    results["largest_fitting_train_batch"] = max(fitting) if fitting else None
    fitting_eval = [entry["batch"] for entry in results["eval"] if entry["fits"]]
    results["largest_fitting_eval_batch"] = max(fitting_eval) if fitting_eval else None
    results["longest_utterance_seconds"] = longest[0].duration
    return results


# --------------------------------------------------------------------------
# Pelatihan
# --------------------------------------------------------------------------


class FullBatchSampler(Sampler[int]):
    """Urutan acak per epoch yang panjangnya kelipatan *batch* efektif.

    `Trainer` membentuk langkah optimizer terakhir setiap epoch dari sisa
    *micro-batch*, sehingga tanpa sampler ini langkah itu memuat kurang dari
    *batch* efektif (Lampiran B.12). Setiap epoch, seluruh utterance diacak
    ulang dengan generator ber-*seed* `seed + epoch`, lalu hanya kelipatan
    *batch* efektif pertama yang diserahkan. Sisanya, kurang dari satu *batch*
    efektif, disisihkan dan berganti antarepoch karena urutannya berganti.

    Diturunkan dari `Sampler`, bukan `RandomSampler`, karena `accelerate`
    mengganti setiap `RandomSampler` dengan samplernya sendiri. Nomor epoch
    diterima lewat `set_epoch`, yang dipanggil `accelerate` sebelum setiap epoch.
    """

    def __init__(self, n_items: int, multiple: int, seed: int) -> None:
        if n_items < multiple:
            raise ValueError(f"Data training {n_items} utterance lebih sedikit dari batch efektif {multiple}")
        self.n_items = n_items
        self.n_used = n_items // multiple * multiple
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Menetapkan nomor epoch yang menentukan urutan acak berikutnya."""
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_used

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.n_items, generator=generator)[: self.n_used].tolist())


class FullBatchMixin:
    """Memasang `FullBatchSampler` dan merata-ratakan loss atas akumulasi gradien.

    Kedua model menerima argumen kata kunci bebas pada `forward`, sehingga
    `Trainer` menganggap model menormalkan loss dengan `num_items_in_batch`
    dan tidak lagi membagi loss dengan akumulasi gradien. Kedua model tidak
    memakai argumen itu, sehingga tanpa penetapan di bawah, gradien setiap
    langkah menjadi jumlah, bukan rerata, gradien per *micro-batch*
    (Lampiran B.14).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def _get_train_sampler(self, train_dataset: Any = None) -> FullBatchSampler:
        dataset = self.train_dataset if train_dataset is None else train_dataset
        multiple = self.args.train_batch_size * self.args.gradient_accumulation_steps
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return FullBatchSampler(len(dataset), multiple, seed)


class FullBatchTrainer(FullBatchMixin, Trainer):
    """`Trainer` yang setiap langkah optimizernya memakai rerata tepat *batch* efektif."""


class FullBatchSeq2SeqTrainer(FullBatchMixin, Seq2SeqTrainer):
    """`Seq2SeqTrainer` yang setiap langkah optimizernya memakai rerata tepat *batch* efektif."""


def training_arguments(config: dict[str, Any], run_dir: Path) -> TrainingArguments:
    """Argumen `Trainer` yang seluruh nilainya berasal dari konfigurasi."""
    common = dict(
        output_dir=str(run_dir / "checkpoints"),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=gradient_accumulation(config),
        learning_rate=config["learning_rate"],
        warmup_steps=config["warmup_steps"],
        lr_scheduler_type=config["lr_scheduler_type"],
        optim=config["optim"],
        max_steps=config["max_steps"],
        bf16=config["precision"] == "bf16",
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["eval_steps"],
        save_total_limit=2,
        save_only_model=config["save_only_model"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=config["logging_steps"],
        logging_first_step=True,
        seed=config["seed"],
        data_seed=config["seed"],
        full_determinism=False,
        dataloader_num_workers=config["dataloader_num_workers"],
        dataloader_persistent_workers=config["dataloader_num_workers"] > 0,
        remove_unused_columns=False,
        report_to="none",
        disable_tqdm=True,
    )
    if config["model_family"] == "whisper":
        return Seq2SeqTrainingArguments(predict_with_generate=True, **common)
    return TrainingArguments(**common)


def prepare_split(path: Path, label: str) -> list[Any]:
    """Membaca satu berkas pembagian dan mencatat hasil normalisasinya ke log."""
    utterances = read_split(path)
    empty = [utt.utt_id for utt in utterances if not utt.text]
    stray = sorted({char for utt in utterances for char in unlisted_removed_characters(utt.transcript)})
    hours = sum(utt.duration for utt in utterances) / 3600
    print(f"{label:18s} : {len(utterances)} utterance, {hours:.3f} jam, "
          f"referensi kosong {len(empty)}, simbol tak tercantum terhapus {stray}", flush=True)
    return [utt for utt in utterances if utt.text]


def train(config: dict[str, Any], run_dir: Path, recorder: WarningRecorder) -> dict[str, Any]:
    """Menjalankan satu pelatihan lengkap dan mengembalikan ringkasannya."""
    model_dir = MODELS_DIR / config["run_name"]
    train_utts = prepare_split(SCRIPT_DIR / config["train_subset"], "Training")
    val_utts = prepare_split(SCRIPT_DIR / config["validation_split"], "Validation")

    effective = config["effective_batch_size"]
    steps_per_epoch = len(train_utts) // effective
    print(f"Per epoch          : {steps_per_epoch * effective} dari {len(train_utts)} utterance, "
          f"{steps_per_epoch} langkah penuh, {len(train_utts) - steps_per_epoch * effective} "
          f"utterance disisihkan bergantian", flush=True)

    model, processor, encode, collate = build(config, model_dir)
    trainer_cls = FullBatchSeq2SeqTrainer if config["model_family"] == "whisper" else FullBatchTrainer
    early_stopping = EarlyStoppingWithFloor(config["early_stopping_patience"],
                                            config["early_stopping_min_steps"])
    monitor = TrainingMonitor()
    trainer = trainer_cls(
        model=model,
        args=training_arguments(config, run_dir),
        data_collator=collate,
        train_dataset=SpeechDataset(train_utts, encode),
        eval_dataset=SpeechDataset(val_utts, encode),
        processing_class=processor,
        compute_metrics=make_compute_metrics(config, processor),
        preprocess_logits_for_metrics=None if config["model_family"] == "whisper" else argmax_logits,
        callbacks=[early_stopping, monitor],
    )
    print(f"Status numerik     : {numerics_state()}", flush=True)
    output = trainer.train()

    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
    if not config["keep_checkpoints"]:
        shutil.rmtree(run_dir / "checkpoints", ignore_errors=True)

    history = [{**entry, **monitor.evaluations.get(int(entry["step"]), {})} for entry in early_stopping.history]
    return {
        "n_train_utterances": len(train_utts),
        "n_train_utterances_per_epoch": steps_per_epoch * effective,
        "steps_per_epoch": steps_per_epoch,
        "n_validation_utterances": len(val_utts),
        "gradient_accumulation_steps": gradient_accumulation(config),
        "global_step": trainer.state.global_step,
        "epochs_completed": round(trainer.state.global_step / steps_per_epoch, 2),
        "stopped_early": trainer.state.global_step < config["max_steps"],
        "best_validation_wer": early_stopping.best,
        "best_step": early_stopping.best_step,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "validation_history": history,
        "train_runtime_seconds": round(output.metrics.get("train_runtime", 0.0), 1),
        **monitor.summary(),
        "model_dir": str(model_dir),
    }


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Fine-tuning satu konfigurasi ASR.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--probe-vram", action="store_true")
    parser.add_argument("--probe-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--probe-eval-sizes", type=int, nargs="+", default=[8, 16, 32])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    config, config_sha256 = load_config(args.config)
    run_dir = RUNS_DIR / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / args.config.name)

    recorder = WarningRecorder()
    recorder.install()
    prevent_background_downloads()
    apply_numerics(config)
    set_seed(config["seed"])

    print("=" * 78, flush=True)
    print(f"{'Probe VRAM' if args.probe_vram else 'Pelatihan'} {config['run_name']}", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Konfigurasi        : {args.config} (SHA-256 {config_sha256[:16]})", flush=True)
    print(f"Random seed        : {config['seed']}", flush=True)
    print(f"Model              : {config['pretrained']} @ {config['revision']}", flush=True)
    print(f"Presisi / TF32     : {config['precision']} / {config['tf32']}", flush=True)
    print(f"Batch efektif      : {config['effective_batch_size']} = "
          f"{config['per_device_train_batch_size']} x {gradient_accumulation(config)} akumulasi", flush=True)
    print(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), flush=True)
    print("=" * 78, flush=True)

    if not torch.cuda.is_available():
        print("GAGAL: CUDA tidak tersedia.", file=sys.stderr)
        return 1

    report: dict[str, Any] = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "mode": "probe_vram" if args.probe_vram else "train",
        "config_path": str(args.config),
        "config_sha256": config_sha256,
        "config": config,
    }
    if args.probe_vram:
        report["probe"] = probe_vram(config, run_dir, args.probe_sizes, args.probe_eval_sizes)
        report_name = "probe_vram.json"
    else:
        report["result"] = train(config, run_dir, recorder)
        report_name = "run_report.json"

    report["numerics_state_at_end"] = numerics_state()
    report["determinism_warnings"] = recorder.determinism_messages()
    report["environment"] = collect_environment(config["seed"], started)
    report["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    with (run_dir / report_name).open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)

    print("=" * 78, flush=True)
    for key, value in report.get("result", report.get("probe", {})).items():
        if key != "validation_history":
            print(f"{key:28s}: {value}", flush=True)
    print(f"Peringatan determinisme: {len(report['determinism_warnings'])}", flush=True)
    print(f"Laporan ditulis ke : {run_dir / report_name}", flush=True)
    print(f"Durasi eksekusi    : {report['elapsed_seconds']} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
