"""Analisis proses pelaksanaan kesepuluh run grid dari laporan, log, dan pemantau.

Tujuan
------
Menghasilkan angka proses pelaksanaan eksperimen bagi `bab-4.md` Subbab 4.1,
`lampiran.md` Lampiran B (Tabel B.1 sampai B.8 dan kedua gambarnya),
`bab-3.md` Subbab 3.6.6, serta `DraftResearch.md` Lampiran B.15 dan B.20.
Yang dianalisis adalah jalannya pelatihan
dan inference, bukan hasil pada data test: lama setiap run beserta
rinciannya, laju per langkah dan kestabilannya, dinamika loss, norma gradien,
dan WER validation, memori GPU, suhu, daya, dan pembatasan GPU, beban CPU dan
disk, peristiwa charger, durasi inference, keseragaman lingkungan antarrun,
serta run pengembangan sebelum grid.

Skrip ini hanya membaca berkas yang ditulis `train_asr.py`, `run_inference.py`,
dan `monitor_run.ps1`. Skrip tidak memuat model, tidak memakai GPU, dan tidak
mengubah berkas apa pun pada `outputs/runs/` maupun `logs/`. Perhitungannya
deterministik dan tidak memakai random seed.

Definisi
--------
- Jendela laju      : satu baris `[progres]` log training, yaitu rerata detik
                      per langkah sejak baris sebelumnya, tanpa selang yang
                      memuat evaluasi atau penyimpanan checkpoint dan tanpa
                      langkah pertama (`TrainingMonitor` pada `train_asr.py`).
                      Jumlah selang terukur setiap jendela direkonstruksi dari
                      jadwal evaluasi dan dicocokkan dengan `timed_steps`
                      laporan run.
- Laju acuan model  : median seluruh jendela laju kelima run satu model.
- Jendela lambat    : jendela dengan laju di atas 1,25 kali laju acuan model.
                      Jumlah jendela di antara 1,25 dan 1,5 kali acuan
                      dilaporkan sebagai uji kepekaan ambang.
- Periode lambat    : jendela lambat yang berurutan.
- Waktu tambahan    : jumlah (laju jendela lambat - laju acuan) x selang
                      terukurnya, yaitu waktu yang hilang karena periode lambat.
- Evaluasi lambat   : evaluasi validation yang lebih lama daripada 1,5 kali
                      median lama evaluasi kelima run satu model.
- Biaya evaluasi pertama : lama evaluasi pertama dikurangi lama evaluasi kedua
                      pada run yang sama.
- Rincian durasi    : durasi proses = langkah terukur + evaluasi + sisa. Sisa
                      memuat persiapan sebelum `Trainer`, langkah pertama,
                      penyimpanan checkpoint, dan langkah yang selangnya
                      memuat evaluasi atau penyimpanan.
- Norma gradien     : `grad_norm` pada `log_history`, yaitu norma gradien
                      sebelum clipping pada langkah terakhir setiap 25 langkah
                      (`transformers/trainer.py` baris 1862-1865 dan 2664-2670).
                      Clipping aktif bila norma melebihi `max_grad_norm` 1,0,
                      nilai bawaan `TrainingArguments` yang tidak diubah
                      konfigurasi (BAB III Tabel 3.14).
- Lonjakan loss     : loss jendela sesudah warmup yang lebih dari dua kali loss
                      jendela sebelumnya dan naik sekurangnya 0,01. Rasio loss
                      tertinggi sesudah warmup terhadap loss pada akhir warmup
                      dilaporkan bersamanya.
- Batas epoch       : kelipatan `steps_per_epoch` laporan run. Perubahan loss di
                      sekitarnya adalah rerata loss jendela dalam 100 langkah
                      sesudah batas dibagi rerata 100 langkah sebelumnya,
                      dikurangi satu; median atas seluruh batas sesudah warmup
                      dilaporkan bagi run yang satu epochnya sekurangnya 150
                      langkah. Ukuran yang sama di tengah setiap epoch menjadi
                      kontrol bagi turunnya loss sepanjang run.
- Laju normal       : median jendela yang tidak lambat pada satu run; dipakai
                      bagi detik audio yang diproses per detik pelatihan.
- Potret pemantau   : potret berkala `monitor_run.ps1` setiap 30 menit selagi
                      proses training hidup. Potret awal, yang diambil pada
                      langkah pertama, dan potret akhir, sesudah proses keluar,
                      tidak dipakai bagi statistik selama training. Potret
                      lambat adalah potret yang jendela laju terakhirnya lambat.
- Pembatasan GPU    : salah satu dari empat alasan pembatasan clock
                      (`sw_power_cap`, `hw_slowdown`, `hw_thermal_slowdown`,
                      `sw_thermal_slowdown`) bernilai Active pada potret.
- Perkiraan energi  : rerata daya GPU pada potret berkala dikali durasi proses.
                      Hanya daya papan GPU menurut `nvidia-smi`, tanpa CPU,
                      RAM, dan layar, dan dari sampel 30 menit sehingga
                      kasar.
- RTF inference     : durasi eksekusi `run_inference.py`, termasuk memuat model
                      dan audio, dibagi durasi audio data test.

Masukan
-------
- `outputs/runs/<run>/run_report.json`, `trainer_state.json`, dan salinan
  konfigurasi kesepuluh run grid; laporan run atau probe lainnya pada
  `outputs/runs/` sebagai run pengembangan.
- `logs/<run>.log` dan `logs/<run>.monitor.log` kesepuluh run grid.
- `logs/kernel_power_105_20261004-20261006.log`: peristiwa charger 4 sampai 6
  Oktober 2026 yang disalin dari log Windows.
- `outputs/results/<run>_test_metrics.json` kesepuluh run grid.
- `outputs/build_nested_subsets_report.json` dan
  `outputs/build_text_disjoint_split_report.json`: jam dan rerata durasi data.

Keluaran
--------
- `outputs/results/training_process_runs.csv`        : satu baris per run.
- `outputs/results/training_process_windows.csv`     : satu baris per jendela laju.
- `outputs/results/training_process_evaluations.csv` : satu baris per evaluasi.
- `outputs/results/training_process_monitor.csv`     : satu baris per potret.
- `outputs/results/training_process_appendix_b1_durations.csv` sampai
  `..._b8_snapshot_groups.csv`: nilai Tabel B.1 sampai B.8 `lampiran.md`
  dalam satuan naskah (jam, menit, persen) tanpa pembulatan, agar setiap
  konversi satuan dihasilkan skrip ini.
- `outputs/results/training_process_report.json`     : laporan lengkap,
  ringkasan lintas run, definisi, dan SHA-256 masukan.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u analyze_training_process.py 2>&1 |
        Tee-Object -FilePath logs/analyze_training_process.log
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    LEVELS,
    MODELS,
    RESULTS_DIR,
    run_name,
    sha256_of,
    write_csv,
    write_json,
)

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "outputs" / "runs"
LOGS_DIR = SCRIPT_DIR / "logs"
SUBSETS_REPORT = SCRIPT_DIR / "outputs" / "build_nested_subsets_report.json"
SPLIT_REPORT = SCRIPT_DIR / "outputs" / "build_text_disjoint_split_report.json"
KERNEL_POWER_LOG = LOGS_DIR / "kernel_power_105_20261004-20261006.log"

REPORT_NAME = "training_process_report.json"
RUNS_TABLE = "training_process_runs.csv"
WINDOWS_TABLE = "training_process_windows.csv"
EVALUATIONS_TABLE = "training_process_evaluations.csv"
MONITOR_TABLE = "training_process_monitor.csv"
# Nilai Tabel B.1 sampai B.8 `lampiran.md` dalam satuan naskah, tanpa
# pembulatan: `training_process_appendix_b1_durations.csv` dan seterusnya.
APPENDIX_PREFIX = "training_process_appendix_"

# Seluruh run dan stempel waktu pemantau berada pada WIB.
LOCAL_TZ = timezone(timedelta(hours=7))

# Ambang jendela lambat dan uji kepekaannya, kelipatan laju acuan model.
SLOW_FACTOR = 1.25
SLOW_FACTOR_SENSITIVITY = 1.5

# Ambang evaluasi lambat, kelipatan median lama evaluasi satu model. Evaluasi
# pertama setiap run sekitar 1,1 sampai 1,4 kali median karena biaya awal yang
# muncul pada kesepuluh run, sehingga ambang 1,5 memisahkan biaya awal itu dari
# evaluasi yang jatuh pada periode lambat. Biaya awal dilaporkan tersendiri.
SLOW_EVAL_FACTOR = 1.5

# `max_grad_norm` bawaan TrainingArguments transformers 5.17.0; konfigurasi
# penelitian tidak mengubahnya (BAB III Tabel 3.14).
MAX_GRAD_NORM = 1.0

# Lonjakan loss: loss jendela lebih dari dua kali jendela sebelumnya dan naik
# sekurangnya 0,01. Batas mutlak mencegah fluktuasi loss yang sudah mendekati
# nol, misalnya 0,004 menjadi 0,008, terhitung sebagai lonjakan.
LOSS_JUMP_FACTOR = 2.0
LOSS_JUMP_MIN_INCREASE = 0.01

# Rentang langkah akhir untuk mengukur perbaikan WER validation yang tersisa.
LATE_WINDOW_STEPS = 1_000

# Perubahan loss training di sekitar batas epoch: rerata loss jendela dalam
# 100 langkah sesudah batas dibagi rerata 100 langkah sebelumnya. Hanya bagi
# run yang satu epochnya sekurangnya 150 langkah, karena jendela laju
# berukuran 25 langkah.
EPOCH_BOUNDARY_SPAN = 100
EPOCH_BOUNDARY_MIN_STEPS = 150

# Toleransi rekonstruksi waktu langkah terukur terhadap laporan run. Laju
# jendela dicetak dua desimal, sehingga galat pembulatannya paling besar
# 0,005 detik per langkah.
ROUNDING_PER_STEP = 0.005

DETERMINISM_LABELS = {
    "ctc_loss_backward_gpu": "ctc_loss_backward_gpu",
    "Memory Efficient attention": "memory_efficient_attention_backward",
}

GPU_COLUMNS = [
    "gpu_memory_used_mib", "gpu_memory_free_mib", "gpu_utilization_pct", "gpu_temperature_c",
    "gpu_power_w", "gpu_sm_clock_mhz", "gpu_sm_clock_max_mhz", "throttle_mask",
    "throttle_sw_power_cap", "throttle_hw_slowdown", "throttle_hw_thermal", "throttle_sw_thermal",
]
THROTTLE_FLAGS = ["throttle_sw_power_cap", "throttle_hw_slowdown", "throttle_hw_thermal",
                  "throttle_sw_thermal"]

WINDOW_HEADER = [
    "run_name", "model", "level", "step", "previous_step", "timed_intervals", "seconds_per_step",
    "timed_seconds", "slow", "train_peak_reserved_mib", "train_peak_allocated_mib", "loss",
    "grad_norm", "learning_rate",
]
EVALUATION_HEADER = [
    "run_name", "model", "level", "step", "wer", "cer", "eval_loss", "eval_seconds",
    "eval_reserved_start_mib", "eval_peak_reserved_mib", "eval_peak_allocated_mib",
    "best_wer_so_far", "evals_without_improvement", "slow", "in_slow_period",
]
MONITOR_HEADER = [
    "run_name", "model", "level", "timestamp", "kind", "process_alive", "progress_step",
    "progress_seconds_per_step", "slow", *GPU_COLUMNS, "power_source", "battery_pct",
    "cpu_utilization_pct", "cpu_performance_pct", "cpu_time_pct", "disk_busy_pct", "disk_mb_s",
    "python_cpu_pct", "python_processes", "ac_online", "charging_mw", "discharging_mw",
    "battery_capacity_mwh", "charger_events",
]
RUN_HEADER = [
    "model", "level", "run_name", "start", "train_end", "inference_start", "inference_end",
    "wall_seconds", "setup_seconds", "train_runtime_seconds", "timed_steps", "timed_seconds_report",
    "timed_seconds_reconstructed", "first_step_seconds", "evaluation_seconds", "rest_seconds",
    "evaluation_share", "inference_seconds", "inference_rtf", "inference_utterances_per_second",
    "last_step", "best_step", "epochs", "utterances_processed", "audio_hours_processed",
    "median_seconds_per_step", "p05_seconds_per_step", "p95_seconds_per_step", "cv_seconds_per_step",
    "min_seconds_per_step", "max_seconds_per_step", "median_normal_seconds_per_step",
    "audio_seconds_per_second", "n_windows",
    "n_slow_windows", "slow_step_share", "n_slow_periods", "longest_slow_period_steps",
    "slow_excess_seconds", "n_evaluations", "median_eval_seconds", "min_eval_seconds",
    "max_eval_seconds", "n_slow_evaluations", "n_slow_evaluations_in_slow_periods",
    "first_eval_extra_seconds", "validation_rtf",
    "seconds_after_best",
    "share_steps_after_best", "first_eval_step_below_100", "n_improvements",
    "max_evals_without_improvement", "best_validation_wer", "final_validation_wer",
    "wer_gain_last_1000_points", "eval_loss_min_step", "eval_loss_min", "eval_loss_at_best_wer",
    "eval_loss_final", "train_loss_first", "train_loss_at_best", "train_loss_final",
    "lr_fraction_at_best", "n_nonfinite", "n_loss_jumps", "max_loss_ratio_after_warmup",
    "steps_per_epoch", "n_epoch_boundaries", "epoch_boundary_loss_ratio_median",
    "epoch_boundary_loss_ratio_min", "epoch_boundary_loss_ratio_max", "n_epoch_midpoints",
    "epoch_midpoint_loss_ratio_median", "grad_norm_warmup_max",
    "grad_norm_median_after_warmup", "grad_norm_p90_after_warmup", "grad_norm_max_after_warmup",
    "grad_clip_share_after_warmup", "test_wer", "test_minus_validation_points",
    "train_peak_reserved_mib", "train_peak_allocated_mib", "eval_peak_reserved_mib_max",
    "eval_peak_allocated_mib_max", "gpu_headroom_mib", "inference_peak_reserved_mib",
    "n_monitor_snapshots", "gpu_temp_median", "gpu_temp_max", "gpu_power_median",
    "gpu_power_max", "gpu_utilization_median", "gpu_utilization_min", "gpu_sm_clock_median",
    "gpu_memory_free_min", "n_throttled_snapshots", "energy_wh_estimate", "n_charger_events",
    "determinism_warnings",
]


# --------------------------------------------------------------------------
# Pembacaan dan pemeriksaan
# --------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Membaca berkas JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    """Menghentikan skrip bila sebuah pemeriksaan gagal."""
    if not condition:
        raise ValueError(message)


def local_time(text: str) -> datetime:
    """Stempel waktu pemantau tanpa zona menjadi waktu WIB."""
    return datetime.strptime(text, "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)


def iso(moment: datetime) -> str:
    """Waktu ISO 8601 sampai detik."""
    return moment.isoformat(timespec="seconds")


def median(values: Sequence[float]) -> float | None:
    """Median, atau None bila kosong."""
    return float(statistics.median(values)) if values else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """Persentil dengan interpolasi linear, atau None bila kosong."""
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else None


def number(text: str) -> float:
    """Bilangan pertama pada sebuah teks."""
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    require(match is not None, f"tidak ada bilangan pada '{text}'")
    return float(match.group(0))


# --------------------------------------------------------------------------
# Log training: jendela laju
# --------------------------------------------------------------------------

PROGRESS_RATE = re.compile(
    r"^\[progres\] langkah (\d+): ([\d.]+) s/langkah, VRAM puncak training tercadang (\d+) MiB, "
    r"teralokasi (\d+) MiB")
PROGRESS_FIRST = re.compile(r"^\[progres\] langkah 1: ([\d.]+) s pada langkah pertama")
PROGRESS_ANY = re.compile(r"^\[progres\] langkah (\d+):")


def excluded_intervals(previous: int, step: int, eval_steps: int, last_step: int) -> int:
    """Banyak selang tak terukur di antara dua baris progres.

    Selang yang berakhir pada langkah k + 1 tidak diukur bila langkah k memicu
    evaluasi atau penyimpanan, yaitu kelipatan `eval_steps` sebelum langkah
    terakhir (`TrainingMonitor.on_step_end`).
    """
    count = 0
    for end in range(previous + 1, step + 1):
        trigger = end - 1
        if trigger >= eval_steps and trigger % eval_steps == 0 and trigger < last_step:
            count += 1
    return count


def read_windows(log_path: Path, eval_steps: int, last_step: int) -> list[dict[str, Any]]:
    """Jendela laju dari baris `[progres]` log training."""
    windows: list[dict[str, Any]] = []
    previous = 1
    saw_first = False
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if PROGRESS_FIRST.match(line):
            saw_first = True
            continue
        match = PROGRESS_RATE.match(line)
        if match is None:
            require(PROGRESS_ANY.match(line) is None, f"{log_path.name}: baris progres tak dikenal: {line}")
            continue
        step = int(match.group(1))
        timed = (step - previous) - excluded_intervals(previous, step, eval_steps, last_step)
        rate = float(match.group(2))
        windows.append({
            "step": step,
            "previous_step": previous,
            "timed_intervals": timed,
            "seconds_per_step": rate,
            "timed_seconds": rate * timed,
            "train_peak_reserved_mib": int(match.group(3)),
            "train_peak_allocated_mib": int(match.group(4)),
        })
        previous = step
    require(saw_first, f"{log_path.name}: baris langkah pertama tidak ditemukan")
    require(windows and windows[-1]["step"] == last_step,
            f"{log_path.name}: jendela terakhir tidak berakhir pada langkah {last_step}")
    return windows


def slow_periods(windows: Sequence[dict[str, Any]]) -> list[dict[str, int]]:
    """Jendela lambat yang berurutan sebagai satu periode."""
    periods: list[dict[str, int]] = []
    current: dict[str, int] | None = None
    for window in windows:
        if window["slow"]:
            if current is None:
                current = {"first_step": window["previous_step"] + 1, "last_step": window["step"],
                           "n_windows": 0, "timed_intervals": 0}
            current["last_step"] = window["step"]
            current["n_windows"] += 1
            current["timed_intervals"] += window["timed_intervals"]
        elif current is not None:
            periods.append(current)
            current = None
    if current is not None:
        periods.append(current)
    return periods


# --------------------------------------------------------------------------
# Log pemantau
# --------------------------------------------------------------------------

SNAPSHOT_HEADER = re.compile(r"^==== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (.+?) ====\s*$")
FIELD_LINE = re.compile(r"^([A-Za-z][A-Za-z0-9 ]*?)\s+: (.*)$")
CPU_DISK = re.compile(
    r"utilitas CPU % ([\d.]+), kinerja CPU % ([\d.]+), waktu CPU % ([\d.]+), "
    r"disk sibuk % ([\d.]+), disk MB/s ([\d.]+)")
PYTHON_CPU = re.compile(r"python total ([\d.]+)% \((\d+) proses\)")
BATTERY = re.compile(
    r"AC (True|False), mengisi (True|False) (\d+) mW, menguras (True|False) (\d+) mW, "
    r"kapasitas (\d+) mWh")
POWER_SOURCE = re.compile(r"sumber (\w+), baterai (\d+)%")
CHARGER_EVENT = re.compile(r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}) AC=(true|false)")


def parse_gpu(text: str) -> dict[str, Any]:
    """Kolom GPU `nvidia-smi` pada satu potret."""
    parts = [part.strip() for part in text.split(",")]
    require(len(parts) == len(GPU_COLUMNS), f"kolom GPU tidak lengkap: {text}")
    values: dict[str, Any] = {}
    for column, part in zip(GPU_COLUMNS, parts):
        if column == "throttle_mask":
            values[column] = part
        elif column.startswith("throttle_"):
            require(part in ("Active", "Not Active"), f"nilai pembatasan tak dikenal: {part}")
            values[column] = part == "Active"
        else:
            values[column] = number(part)
    return values


def read_monitor(path: Path, year: int) -> list[dict[str, Any]]:
    """Potret pemantau satu run beserta peristiwa charger yang dicatatnya."""
    snapshots: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        header = SNAPSHOT_HEADER.match(line)
        if header:
            current = {"timestamp": local_time(header.group(1)),
                       "kind": header.group(2).split(" ")[0].rstrip(":").lower(), "fields": {}}
            snapshots.append(current)
            continue
        if current is None:
            continue
        field = FIELD_LINE.match(line)
        if field and field.group(1).strip() not in current["fields"]:
            current["fields"][field.group(1).strip()] = field.group(2).strip()

    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        fields = snapshot["fields"]
        row: dict[str, Any] = {column: None for column in MONITOR_HEADER}
        row.update({"timestamp": snapshot["timestamp"], "kind": snapshot["kind"],
                    "process_alive": fields.get("Proses training", "").startswith("hidup")})
        progress = fields.get("Progres terakhir", "")
        step_match = re.search(r"langkah (\d+):", progress)
        rate_match = re.search(r": ([\d.]+) s/langkah", progress)
        row["progress_step"] = int(step_match.group(1)) if step_match else None
        row["progress_seconds_per_step"] = float(rate_match.group(1)) if rate_match else None
        if "GPU" in fields:
            row.update(parse_gpu(fields["GPU"]))
        source = POWER_SOURCE.search(fields.get("Daya", ""))
        if source:
            row["power_source"], row["battery_pct"] = source.group(1), int(source.group(2))
        cpu = CPU_DISK.search(fields.get("CPU dan disk", ""))
        if cpu:
            (row["cpu_utilization_pct"], row["cpu_performance_pct"], row["cpu_time_pct"],
             row["disk_busy_pct"], row["disk_mb_s"]) = (float(value) for value in cpu.groups())
        python = PYTHON_CPU.search(fields.get("Proses CPU", ""))
        if python:
            row["python_cpu_pct"], row["python_processes"] = float(python.group(1)), int(python.group(2))
        battery = BATTERY.search(fields.get("Baterai", ""))
        if battery:
            row["ac_online"] = battery.group(1) == "True"
            row["charging_mw"] = int(battery.group(3))
            row["discharging_mw"] = int(battery.group(5))
            row["battery_capacity_mwh"] = int(battery.group(6))
        events = [(local_time(f"{year}-{stamp}"), state == "true")
                  for stamp, state in CHARGER_EVENT.findall(fields.get("Peristiwa charger", ""))]
        row["charger_events"] = events
        rows.append(row)
    return rows


def read_kernel_power(path: Path) -> list[tuple[datetime, bool]]:
    """Peristiwa charger Kernel-Power 105 yang disalin dari log Windows."""
    events: list[tuple[datetime, bool]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        stamp, state = line.split("\t")[:2]
        events.append((local_time(stamp), state == "AcOnline=true"))
    return events


# --------------------------------------------------------------------------
# Satu run
# --------------------------------------------------------------------------


def analyse_run(model: str, level: str, runs_dir: Path, logs_dir: Path, results_dir: Path,
                level_info: dict[str, Any], test_seconds: float,
                validation_seconds: float) -> dict[str, Any]:
    """Seluruh bahan proses satu run sebelum ambang lintas run diterapkan."""
    name = run_name(model, level)
    folder = runs_dir / name
    report_path = folder / "run_report.json"
    state_path = folder / "trainer_state.json"
    config_path = folder / f"{name}.yaml"
    log_path = logs_dir / f"{name}.log"
    monitor_path = logs_dir / f"{name}.monitor.log"
    metrics_path = results_dir / f"{name}_test_metrics.json"

    report = read_json(report_path)
    state = read_json(state_path)
    metrics = read_json(metrics_path)
    config = report["config"]
    result = report["result"]

    require(report.get("mode") == "train", f"{name}: laporan run bukan mode pelatihan")
    require(config["run_name"] == name, f"{name}: run_name laporan adalah {config['run_name']}")
    require(sha256_of(config_path) == report["config_sha256"], f"{name}: SHA-256 konfigurasi tidak cocok")
    require(metrics["config_sha256"] == report["config_sha256"],
            f"{name}: laporan inference memakai konfigurasi lain")
    require(state["global_step"] == result["global_step"], f"{name}: langkah terakhir berbeda antarberkas")
    require(state["best_global_step"] == result["best_step"], f"{name}: langkah terbaik berbeda antarberkas")

    last_step = int(result["global_step"])
    best_step = int(result["best_step"])
    eval_steps = int(config["eval_steps"])
    warmup = int(config["warmup_steps"])
    peak_lr = float(config["learning_rate"])
    batch = int(config["effective_batch_size"])

    windows = read_windows(log_path, eval_steps, last_step)
    timed_reconstructed = sum(window["timed_intervals"] for window in windows)
    require(timed_reconstructed == result["timed_steps"],
            f"{name}: selang terukur hasil rekonstruksi {timed_reconstructed}, "
            f"laporan run {result['timed_steps']}")
    timed_seconds_report = float(result["seconds_per_step"]) * int(result["timed_steps"])
    timed_seconds_windows = sum(window["timed_seconds"] for window in windows)
    tolerance = ROUNDING_PER_STEP * timed_reconstructed + 0.0005 * int(result["timed_steps"])
    require(abs(timed_seconds_windows - timed_seconds_report) <= tolerance,
            f"{name}: waktu langkah terukur dari jendela {timed_seconds_windows:.1f} detik, "
            f"laporan run {timed_seconds_report:.1f} detik")

    log_entries = state["log_history"]
    train_logs = {entry["step"]: entry for entry in log_entries if "loss" in entry}
    eval_logs = {entry["step"]: entry for entry in log_entries if "eval_wer" in entry}
    for window in windows:
        entry = train_logs.get(window["step"])
        require(entry is not None, f"{name}: log_history tanpa loss pada langkah {window['step']}")
        window.update({"loss": entry["loss"], "grad_norm": entry.get("grad_norm"),
                       "learning_rate": entry.get("learning_rate")})

    evaluations: list[dict[str, Any]] = []
    best_so_far: float | None = None
    without = 0
    for cost in result["validation_history"]:
        step = int(cost["step"])
        entry = eval_logs.get(step)
        require(entry is not None, f"{name}: log_history tanpa evaluasi pada langkah {step}")
        require(abs(entry["eval_wer"] - cost["wer"]) <= 1e-12, f"{name}: WER evaluasi {step} berbeda antarberkas")
        if best_so_far is None or cost["wer"] < best_so_far:
            best_so_far, without = cost["wer"], 0
        else:
            without += 1
        evaluations.append({
            "step": step, "wer": float(cost["wer"]), "cer": float(cost["cer"]),
            "eval_loss": float(entry["eval_loss"]), "eval_seconds": float(cost["eval_seconds"]),
            "eval_reserved_start_mib": int(cost["eval_reserved_start_mib"]),
            "eval_peak_reserved_mib": int(cost["eval_peak_reserved_mib"]),
            "eval_peak_allocated_mib": int(cost["eval_peak_allocated_mib"]),
            "best_wer_so_far": best_so_far, "evals_without_improvement": without,
        })
    require(len(evaluations) == len(eval_logs), f"{name}: jumlah evaluasi berbeda antarberkas")
    require(abs(sum(e["eval_seconds"] for e in evaluations) - result["evaluation_seconds_total"]) <= 0.05 * len(evaluations),
            f"{name}: jumlah lama evaluasi berbeda dengan laporan run")

    start = datetime.fromisoformat(report["generated_at"])
    wall = float(report["elapsed_seconds"])
    inference_start = datetime.fromisoformat(metrics["generated_at"])
    inference_seconds = float(metrics["elapsed_seconds"])
    inference_layer = metrics["environment"]["lapis_3_jejak_eksekusi"]

    snapshots: list[dict[str, Any]] = []
    if monitor_path.exists():
        snapshots = read_monitor(monitor_path, start.year)

    environment = report["environment"]
    hardware = json.loads(json.dumps(environment["lapis_1_perangkat_keras"]))
    hardware.get("ram_sistem", {}).pop("tersedia_saat_pencatatan_gib", None)
    signature = json.dumps({"hardware": hardware,
                            "software": environment["lapis_2_perangkat_lunak_terikat_perangkat_keras"]},
                           sort_keys=True)

    warnings = []
    for text in report.get("determinism_warnings", []):
        label = next((value for key, value in DETERMINISM_LABELS.items() if key in text), "lain")
        warnings.append(label)

    return {
        "model": model, "level": level, "name": name, "config": config, "result": result,
        "state": state, "metrics": metrics, "windows": windows, "evaluations": evaluations,
        "snapshots": snapshots, "start": start, "wall": wall, "inference_start": inference_start,
        "inference_seconds": inference_seconds, "inference_layer": inference_layer,
        "timed_seconds_report": timed_seconds_report, "timed_seconds_windows": timed_seconds_windows,
        "last_step": last_step, "best_step": best_step, "warmup": warmup, "peak_lr": peak_lr,
        "batch": batch, "level_info": level_info, "test_seconds": test_seconds,
        "validation_seconds": validation_seconds, "signature": signature,
        "gpu_total_mib": int(environment["lapis_1_perangkat_keras"]["gpu"]["vram_total_mib"]),
        "numerics": json.dumps(report["numerics_state_at_end"], sort_keys=True),
        "inference_numerics": json.dumps(metrics["numerics_state"], sort_keys=True),
        "warnings": warnings,
        "sources": {
            f"runs/{name}/run_report.json": sha256_of(report_path),
            f"runs/{name}/trainer_state.json": sha256_of(state_path),
            f"logs/{name}.log": sha256_of(log_path),
            **({f"logs/{name}.monitor.log": sha256_of(monitor_path)} if monitor_path.exists() else {}),
            f"results/{name}_test_metrics.json": sha256_of(metrics_path),
        },
    }


def finish_run(run: dict[str, Any], reference_rate: float, reference_eval: float, gpu_total_mib: int,
               charger_events: Sequence[tuple[datetime, bool]]) -> dict[str, Any]:
    """Ringkasan satu run sesudah laju acuan dan lama evaluasi acuan diketahui."""
    windows = run["windows"]
    evaluations = run["evaluations"]
    result = run["result"]
    last_step, best_step, warmup = run["last_step"], run["best_step"], run["warmup"]
    threshold = SLOW_FACTOR * reference_rate

    for window in windows:
        window["slow"] = window["seconds_per_step"] > threshold
    for evaluation in evaluations:
        evaluation["slow"] = evaluation["eval_seconds"] > SLOW_EVAL_FACTOR * reference_eval
    periods = slow_periods(windows)
    # Evaluasi pada langkah s berjalan sesudah langkah s, sehingga jatuh pada
    # periode lambat bila s berada di dalam rentang langkah periode itu.
    for evaluation in evaluations:
        evaluation["in_slow_period"] = any(p["first_step"] <= evaluation["step"] <= p["last_step"]
                                           for p in periods)
    rates = [window["seconds_per_step"] for window in windows]
    slow_windows = [window for window in windows if window["slow"]]
    timed_total = sum(window["timed_intervals"] for window in windows)
    excess = sum((window["seconds_per_step"] - reference_rate) * window["timed_intervals"]
                 for window in slow_windows)

    for snapshot in run["snapshots"]:
        rate = snapshot["progress_seconds_per_step"]
        snapshot["slow"] = None if rate is None else rate > threshold
    training_snapshots = [s for s in run["snapshots"] if s["kind"] == "berkala" and s["process_alive"]]

    def snap(column: str) -> list[float]:
        return [s[column] for s in training_snapshots if s.get(column) is not None]

    train_end = run["start"] + timedelta(seconds=run["wall"])
    inference_end = run["inference_start"] + timedelta(seconds=run["inference_seconds"])
    events_in_run = sorted({event for event in charger_events if run["start"] <= event[0] <= train_end})

    train_entries = [entry for entry in run["state"]["log_history"] if "loss" in entry]
    after_warmup = [entry for entry in train_entries if entry["step"] > warmup]
    grad_after = [float(entry["grad_norm"]) for entry in after_warmup if entry.get("grad_norm") is not None]
    grad_warmup = [float(entry["grad_norm"]) for entry in train_entries
                   if entry["step"] <= warmup and entry.get("grad_norm") is not None]
    nonfinite = sum(1 for entry in train_entries
                    for key in ("loss", "grad_norm") if entry.get(key) is not None
                    and not math.isfinite(float(entry[key])))
    jumps = sum(1 for previous, entry in zip(train_entries, train_entries[1:])
                if entry["step"] > warmup and entry["loss"] > LOSS_JUMP_FACTOR * previous["loss"]
                and entry["loss"] - previous["loss"] >= LOSS_JUMP_MIN_INCREASE)
    loss_by_step = {entry["step"]: float(entry["loss"]) for entry in train_entries}
    require(warmup in loss_by_step, f"{run['name']}: loss tidak tercatat pada akhir warmup")
    loss_ratio_after_warmup = max(float(entry["loss"]) for entry in after_warmup) / loss_by_step[warmup]

    steps_per_epoch = int(result["steps_per_epoch"])

    def loss_change(point: int) -> float | None:
        """Rerata loss 100 langkah sesudah titik dibagi 100 langkah sebelumnya."""
        before = [w["loss"] for w in windows
                  if w["previous_step"] >= point - EPOCH_BOUNDARY_SPAN and w["step"] <= point]
        after = [w["loss"] for w in windows
                 if w["previous_step"] >= point and w["step"] <= point + EPOCH_BOUNDARY_SPAN]
        return statistics.fmean(after) / statistics.fmean(before) if before and after else None

    # Titik kontrol di tengah setiap epoch memisahkan efek batas epoch dari
    # turunnya loss sepanjang run.
    boundary_ratios: list[float] = []
    midpoint_ratios: list[float] = []
    if steps_per_epoch >= EPOCH_BOUNDARY_MIN_STEPS:
        boundary = steps_per_epoch
        while boundary + EPOCH_BOUNDARY_SPAN <= last_step:
            for point, store in ((boundary, boundary_ratios),
                                 (boundary - steps_per_epoch // 2, midpoint_ratios)):
                ratio = loss_change(point) if point > warmup else None
                if ratio is not None:
                    store.append(ratio)
            boundary += steps_per_epoch
    lr_by_step = {entry["step"]: float(entry["learning_rate"]) for entry in train_entries}

    wers = [evaluation["wer"] for evaluation in evaluations]
    best_eval = min(evaluations, key=lambda evaluation: evaluation["wer"])
    require(best_eval["step"] == best_step, f"{run['name']}: langkah terbaik bukan evaluasi WER terendah")
    loss_min = min(evaluations, key=lambda evaluation: evaluation["eval_loss"])
    early = [evaluation["wer"] for evaluation in evaluations if evaluation["step"] <= last_step - LATE_WINDOW_STEPS]
    below = next((evaluation["step"] for evaluation in evaluations if evaluation["wer"] < 1.0), None)
    improvements = sum(1 for evaluation in evaluations if evaluation["evals_without_improvement"] == 0)

    seconds_after_best = (sum(window["timed_seconds"] for window in windows if window["step"] > best_step)
                          + sum(evaluation["eval_seconds"] for evaluation in evaluations
                                if evaluation["step"] > best_step))
    eval_seconds = [evaluation["eval_seconds"] for evaluation in evaluations]
    evaluation_total = float(result["evaluation_seconds_total"])
    train_runtime = float(result["train_runtime_seconds"])
    rest = run["wall"] - run["timed_seconds_report"] - evaluation_total
    mean_duration = float(run["level_info"]["mean_duration_seconds"])
    utterances = last_step * run["batch"]
    median_rate = float(statistics.median(rates))
    normal_rate = float(statistics.median([window["seconds_per_step"] for window in windows
                                           if not window["slow"]]))
    mean_rate = float(statistics.fmean(rates))
    test_wer = float(run["metrics"]["metrics"]["wer"])
    powers = snap("gpu_power_w")
    energy = statistics.fmean(powers) * run["wall"] / 3600 if len(powers) >= 3 else None

    row = {
        "model": run["model"], "level": run["level"], "run_name": run["name"],
        "start": iso(run["start"]), "train_end": iso(train_end),
        "inference_start": iso(run["inference_start"]), "inference_end": iso(inference_end),
        "wall_seconds": run["wall"], "setup_seconds": run["wall"] - train_runtime,
        "train_runtime_seconds": train_runtime, "timed_steps": int(result["timed_steps"]),
        "timed_seconds_report": run["timed_seconds_report"],
        "timed_seconds_reconstructed": run["timed_seconds_windows"],
        "first_step_seconds": float(result["first_step_seconds"]),
        "evaluation_seconds": evaluation_total, "rest_seconds": rest,
        "evaluation_share": evaluation_total / run["wall"],
        "inference_seconds": run["inference_seconds"],
        "inference_rtf": run["inference_seconds"] / run["test_seconds"],
        "inference_utterances_per_second": int(run["metrics"]["metrics"]["n_utterances"]) / run["inference_seconds"],
        "last_step": last_step, "best_step": best_step, "epochs": float(run["state"]["epoch"]),
        "utterances_processed": utterances,
        "audio_hours_processed": utterances * mean_duration / 3600,
        "median_seconds_per_step": median_rate,
        "p05_seconds_per_step": percentile(rates, 5), "p95_seconds_per_step": percentile(rates, 95),
        "cv_seconds_per_step": float(statistics.pstdev(rates)) / mean_rate,
        "min_seconds_per_step": min(rates), "max_seconds_per_step": max(rates),
        "median_normal_seconds_per_step": normal_rate,
        "audio_seconds_per_second": run["batch"] * mean_duration / normal_rate,
        "n_windows": len(windows), "n_slow_windows": len(slow_windows),
        "slow_step_share": sum(w["timed_intervals"] for w in slow_windows) / timed_total,
        "n_slow_periods": len(periods),
        "longest_slow_period_steps": max((p["timed_intervals"] for p in periods), default=0),
        "slow_excess_seconds": excess,
        "n_evaluations": len(evaluations), "median_eval_seconds": median(eval_seconds),
        "min_eval_seconds": min(eval_seconds), "max_eval_seconds": max(eval_seconds),
        "n_slow_evaluations": sum(1 for evaluation in evaluations if evaluation["slow"]),
        "n_slow_evaluations_in_slow_periods": sum(1 for evaluation in evaluations
                                                  if evaluation["slow"] and evaluation["in_slow_period"]),
        "first_eval_extra_seconds": eval_seconds[0] - eval_seconds[1],
        "validation_rtf": median(eval_seconds) / run["validation_seconds"],
        "seconds_after_best": seconds_after_best,
        "share_steps_after_best": (last_step - best_step) / last_step,
        "first_eval_step_below_100": below, "n_improvements": improvements,
        "max_evals_without_improvement": max(evaluation["evals_without_improvement"] for evaluation in evaluations),
        "best_validation_wer": best_eval["wer"], "final_validation_wer": wers[-1],
        "wer_gain_last_1000_points": (min(early) - best_eval["wer"]) * 100 if early else None,
        "eval_loss_min_step": loss_min["step"], "eval_loss_min": loss_min["eval_loss"],
        "eval_loss_at_best_wer": best_eval["eval_loss"], "eval_loss_final": evaluations[-1]["eval_loss"],
        "train_loss_first": float(train_entries[0]["loss"]), "train_loss_at_best": loss_by_step.get(best_step),
        "train_loss_final": float(train_entries[-1]["loss"]),
        "lr_fraction_at_best": lr_by_step[best_step] / run["peak_lr"] if best_step in lr_by_step else None,
        "n_nonfinite": nonfinite, "n_loss_jumps": jumps,
        "max_loss_ratio_after_warmup": loss_ratio_after_warmup,
        "steps_per_epoch": steps_per_epoch,
        "n_epoch_boundaries": len(boundary_ratios),
        "epoch_boundary_loss_ratio_median": median(boundary_ratios),
        "epoch_boundary_loss_ratio_min": min(boundary_ratios, default=None),
        "epoch_boundary_loss_ratio_max": max(boundary_ratios, default=None),
        "n_epoch_midpoints": len(midpoint_ratios),
        "epoch_midpoint_loss_ratio_median": median(midpoint_ratios),
        "grad_norm_warmup_max": max(grad_warmup) if grad_warmup else None,
        "grad_norm_median_after_warmup": median(grad_after),
        "grad_norm_p90_after_warmup": percentile(grad_after, 90),
        "grad_norm_max_after_warmup": max(grad_after) if grad_after else None,
        "grad_clip_share_after_warmup": (sum(1 for value in grad_after if value > MAX_GRAD_NORM) / len(grad_after)
                                         if grad_after else None),
        "test_wer": test_wer, "test_minus_validation_points": (test_wer - best_eval["wer"]) * 100,
        "train_peak_reserved_mib": int(result["train_peak_reserved_mib"]),
        "train_peak_allocated_mib": int(result["train_peak_allocated_mib"]),
        "eval_peak_reserved_mib_max": max(evaluation["eval_peak_reserved_mib"] for evaluation in evaluations),
        "eval_peak_allocated_mib_max": max(evaluation["eval_peak_allocated_mib"] for evaluation in evaluations),
        "gpu_headroom_mib": gpu_total_mib - int(result["train_peak_reserved_mib"]),
        "inference_peak_reserved_mib": float(run["inference_layer"]["vram_puncak_tercadang_mib"]),
        "n_monitor_snapshots": len(training_snapshots),
        "gpu_temp_median": median(snap("gpu_temperature_c")), "gpu_temp_max": max(snap("gpu_temperature_c"), default=None),
        "gpu_power_median": median(powers), "gpu_power_max": max(powers, default=None),
        "gpu_utilization_median": median(snap("gpu_utilization_pct")),
        "gpu_utilization_min": min(snap("gpu_utilization_pct"), default=None),
        "gpu_sm_clock_median": median(snap("gpu_sm_clock_mhz")),
        "gpu_memory_free_min": min(snap("gpu_memory_free_mib"), default=None),
        "n_throttled_snapshots": sum(1 for s in training_snapshots if any(s[flag] for flag in THROTTLE_FLAGS)),
        "energy_wh_estimate": energy,
        "n_charger_events": len(events_in_run),
        "determinism_warnings": "+".join(sorted(set(run["warnings"]))),
    }
    run["row"] = row
    run["periods"] = periods
    run["charger_events_in_run"] = [{"time": iso(moment), "ac_online": online} for moment, online in events_in_run]
    run["training_snapshots"] = training_snapshots
    run["train_end"] = train_end
    run["inference_end"] = inference_end
    return row


# --------------------------------------------------------------------------
# Ringkasan lintas run
# --------------------------------------------------------------------------


def config_differences(configs: Sequence[dict[str, Any]], ignore: Sequence[str]) -> list[str]:
    """Kunci konfigurasi yang nilainya tidak sama pada seluruh konfigurasi."""
    keys = sorted(set().union(*(config.keys() for config in configs)))
    return [key for key in keys if key not in ignore
            and len({json.dumps(config.get(key), sort_keys=True) for config in configs}) > 1]


def snapshot_summary(snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Statistik potret berkala selama training."""
    def values(column: str) -> list[float]:
        return [s[column] for s in snapshots if s.get(column) is not None]

    summary: dict[str, Any] = {"n": len(snapshots)}
    for column in ("gpu_utilization_pct", "gpu_power_w", "gpu_temperature_c", "gpu_sm_clock_mhz",
                   "gpu_memory_free_mib", "cpu_utilization_pct", "cpu_performance_pct",
                   "disk_busy_pct", "disk_mb_s", "python_cpu_pct"):
        data = values(column)
        summary[column] = ({"n": len(data), "min": min(data), "median": median(data), "max": max(data)}
                           if data else {"n": 0})
    summary["n_throttled"] = sum(1 for s in snapshots if any(s[flag] for flag in THROTTLE_FLAGS))
    summary["throttle_flag_counts"] = {flag: sum(1 for s in snapshots if s[flag]) for flag in THROTTLE_FLAGS}
    summary["on_ac"] = sum(1 for s in snapshots if s.get("power_source") == "AC")
    return summary


def development_runs(runs_dir: Path, grid: set[str]) -> list[dict[str, Any]]:
    """Run probe dan uji pada `outputs/runs/` di luar grid."""
    rows: list[dict[str, Any]] = []
    for folder in sorted(path for path in runs_dir.iterdir() if path.is_dir() and path.name not in grid):
        for file_name in ("run_report.json", "probe_vram.json"):
            path = folder / file_name
            if path.exists():
                data = read_json(path)
                rows.append({
                    "run_name": folder.name, "kind": data.get("mode", file_name),
                    "start": data.get("generated_at"), "elapsed_seconds": float(data["elapsed_seconds"]),
                    "global_step": data.get("result", {}).get("global_step"),
                    "sha256": sha256_of(path),
                })
                break
    return rows


def summarise(runs: Sequence[dict[str, Any]], references: dict[str, dict[str, float]],
              development: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Ringkasan lintas run yang dikutip prosa naskah."""
    rows = [run["row"] for run in runs]
    ordered = sorted(runs, key=lambda run: run["start"])
    gaps = []
    for previous, current in zip(ordered, ordered[1:]):
        gaps.append({"after": previous["name"], "before": current["name"],
                     "idle_hours": (current["start"] - previous["inference_end"]).total_seconds() / 3600})

    per_model: dict[str, Any] = {}
    for model in MODELS:
        model_runs = [run for run in runs if run["model"] == model]
        model_rows = [run["row"] for run in model_runs]
        windows = [window for run in model_runs for window in run["windows"]]
        reference = references[model]["seconds_per_step"]
        between = [window for window in windows
                   if SLOW_FACTOR * reference < window["seconds_per_step"] <= SLOW_FACTOR_SENSITIVITY * reference]
        snapshots = [s for run in model_runs for s in run["training_snapshots"]]
        energies = [row["energy_wh_estimate"] for row in model_rows if row["energy_wh_estimate"] is not None]
        per_model[model] = {
            "reference_seconds_per_step": reference,
            "slow_threshold_seconds_per_step": SLOW_FACTOR * reference,
            "reference_eval_seconds": references[model]["eval_seconds"],
            "slow_eval_threshold_seconds": SLOW_EVAL_FACTOR * references[model]["eval_seconds"],
            "wall_hours": sum(row["wall_seconds"] for row in model_rows) / 3600,
            "evaluation_hours": sum(row["evaluation_seconds"] for row in model_rows) / 3600,
            "inference_hours": sum(row["inference_seconds"] for row in model_rows) / 3600,
            "evaluation_share_range": [min(row["evaluation_share"] for row in model_rows),
                                       max(row["evaluation_share"] for row in model_rows)],
            "median_rate_range": [min(row["median_seconds_per_step"] for row in model_rows),
                                  max(row["median_seconds_per_step"] for row in model_rows)],
            "window_rate_range": [min(w["seconds_per_step"] for w in windows),
                                  max(w["seconds_per_step"] for w in windows)],
            "normal_window_rate_range": [min(w["seconds_per_step"] for w in windows if not w["slow"]),
                                         max(w["seconds_per_step"] for w in windows if not w["slow"])],
            "n_windows": len(windows),
            "n_slow_windows": sum(1 for w in windows if w["slow"]),
            "n_windows_between_thresholds": len(between),
            "slow_runs": [row["run_name"] for row in model_rows if row["n_slow_windows"] > 0],
            "slow_excess_hours": sum(row["slow_excess_seconds"] for row in model_rows) / 3600,
            "n_slow_evaluations": sum(row["n_slow_evaluations"] for row in model_rows),
            "n_slow_evaluations_in_slow_periods": sum(row["n_slow_evaluations_in_slow_periods"]
                                                      for row in model_rows),
            "n_evaluations": sum(row["n_evaluations"] for row in model_rows),
            "eval_seconds_range_normal": [
                min(e["eval_seconds"] for run in model_runs for e in run["evaluations"] if not e["slow"]),
                max(e["eval_seconds"] for run in model_runs for e in run["evaluations"] if not e["slow"])],
            "hours_after_best": sum(row["seconds_after_best"] for row in model_rows) / 3600,
            "first_eval_extra_seconds_range": [min(row["first_eval_extra_seconds"] for row in model_rows),
                                               max(row["first_eval_extra_seconds"] for row in model_rows)],
            "grad_clip_share_range": [min(row["grad_clip_share_after_warmup"] for row in model_rows),
                                      max(row["grad_clip_share_after_warmup"] for row in model_rows)],
            "inference_rtf_range": [min(row["inference_rtf"] for row in model_rows),
                                    max(row["inference_rtf"] for row in model_rows)],
            "validation_rtf_range": [min(row["validation_rtf"] for row in model_rows),
                                     max(row["validation_rtf"] for row in model_rows)],
            "audio_seconds_per_second_range": [min(row["audio_seconds_per_second"] for row in model_rows),
                                               max(row["audio_seconds_per_second"] for row in model_rows)],
            "test_minus_validation_points_range": [min(row["test_minus_validation_points"] for row in model_rows),
                                                   max(row["test_minus_validation_points"] for row in model_rows)],
            "train_peak_reserved_mib_range": [min(row["train_peak_reserved_mib"] for row in model_rows),
                                              max(row["train_peak_reserved_mib"] for row in model_rows)],
            "monitor": snapshot_summary(snapshots),
            "monitor_normal": snapshot_summary([s for s in snapshots if s["slow"] is False]),
            "monitor_slow": snapshot_summary([s for s in snapshots if s["slow"] is True]),
            "energy_kwh_estimate": sum(energies) / 1000 if energies else None,
            "energy_runs": len(energies),
        }

    signatures = {run["signature"] for run in runs}
    numerics = {run["numerics"] for run in runs} | {run["inference_numerics"] for run in runs}
    within = {model: config_differences([run["config"] for run in runs if run["model"] == model],
                                        ("run_name", "train_subset")) for model in MODELS}
    between_models = config_differences([run["config"] for run in runs], ("run_name", "train_subset"))
    first = ordered[0]
    last = max(runs, key=lambda run: run["inference_end"])
    return {
        "n_runs": len(rows),
        "grid_wall_hours": sum(row["wall_seconds"] for row in rows) / 3600,
        "grid_inference_hours": sum(row["inference_seconds"] for row in rows) / 3600,
        "grid_evaluation_hours": sum(row["evaluation_seconds"] for row in rows) / 3600,
        "grid_slow_excess_hours": sum(row["slow_excess_seconds"] for row in rows) / 3600,
        "grid_hours_after_best": sum(row["seconds_after_best"] for row in rows) / 3600,
        "calendar_first_start": iso(first["start"]),
        "calendar_last_end": iso(last["inference_end"]),
        "calendar_span_hours": (last["inference_end"] - first["start"]).total_seconds() / 3600,
        "run_order": [run["name"] for run in ordered],
        "gaps": gaps,
        "per_model": per_model,
        "n_nonfinite": sum(row["n_nonfinite"] for row in rows),
        "n_loss_jumps": sum(row["n_loss_jumps"] for row in rows),
        "max_loss_ratio_after_warmup": max(row["max_loss_ratio_after_warmup"] for row in rows),
        "n_throttled_snapshots": sum(row["n_throttled_snapshots"] for row in rows),
        "n_training_snapshots": sum(row["n_monitor_snapshots"] for row in rows),
        "throttled_snapshots": [
            {"run_name": run["name"], "timestamp": iso(s["timestamp"]), "progress_step": s["progress_step"],
             "progress_seconds_per_step": s["progress_seconds_per_step"], "temperature_c": s["gpu_temperature_c"],
             "power_w": s["gpu_power_w"], "sm_clock_mhz": s["gpu_sm_clock_mhz"], "mask": s["throttle_mask"],
             "flags": [flag for flag in THROTTLE_FLAGS if s[flag]]}
            for run in runs for s in run["training_snapshots"] if any(s[flag] for flag in THROTTLE_FLAGS)],
        "charger_events_in_runs": {run["name"]: run["charger_events_in_run"] for run in runs
                                   if run["charger_events_in_run"]},
        "monitor_by_run": {
            run["name"]: {
                "first_snapshot": iso(run["training_snapshots"][0]["timestamp"])
                if run["training_snapshots"] else None,
                "first_snapshot_step": run["training_snapshots"][0]["progress_step"]
                if run["training_snapshots"] else None,
                "normal": snapshot_summary([s for s in run["training_snapshots"] if s["slow"] is False]),
                "slow": snapshot_summary([s for s in run["training_snapshots"] if s["slow"] is True]),
            } for run in runs},
        "slow_evaluations_by_run": {
            run["name"]: {"n": sum(1 for e in run["evaluations"] if e["slow"]),
                          "range_seconds": [min((e["eval_seconds"] for e in run["evaluations"] if e["slow"]),
                                                default=None),
                                            max((e["eval_seconds"] for e in run["evaluations"] if e["slow"]),
                                                default=None)],
                          "normal_range_seconds": [min(e["eval_seconds"] for e in run["evaluations"] if not e["slow"]),
                                                   max(e["eval_seconds"] for e in run["evaluations"] if not e["slow"])],
                          "steps": [e["step"] for e in run["evaluations"] if e["slow"]]}
            for run in runs},
        "environment_signatures_distinct": len(signatures),
        "numerics_states_distinct": len(numerics),
        "config_differences_within_model": within,
        "config_differences_between_models": between_models,
        "determinism_warnings": {run["name"]: run["row"]["determinism_warnings"] for run in runs},
        "development_runs": {
            "n_runs": len(development),
            "elapsed_hours": sum(row["elapsed_seconds"] for row in development) / 3600,
            "runs": list(development),
        },
        "energy_kwh_estimate_all": sum(m["energy_kwh_estimate"] or 0.0 for m in per_model.values()),
    }


# --------------------------------------------------------------------------
# Tabel lampiran dalam satuan naskah
# --------------------------------------------------------------------------


def group_summary(label: str, snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Satu baris Tabel B.8: median dan rentang potret satu kelompok."""
    def stats(column: str) -> tuple[int, float | None, float | None, float | None]:
        data = [s[column] for s in snapshots if s.get(column) is not None]
        return (len(data), median(data), min(data, default=None), max(data, default=None))

    row: dict[str, Any] = {"group": label, "n_snapshots": len(snapshots)}
    for column in ("gpu_utilization_pct", "gpu_power_w", "gpu_temperature_c", "gpu_memory_free_mib",
                   "cpu_utilization_pct", "cpu_performance_pct", "disk_busy_pct", "python_cpu_pct"):
        n, mid, low, high = stats(column)
        row.update({f"{column}_n": n, f"{column}_median": mid, f"{column}_min": low, f"{column}_max": high})
    return row


def appendix_tables(runs: Sequence[dict[str, Any]], summary: dict[str, Any]) -> dict[str, tuple[list[str], list[list[Any]]]]:
    """Nilai Tabel B.1 sampai B.8 `lampiran.md` dalam satuan naskah, tanpa pembulatan."""
    tables: dict[str, tuple[list[str], list[list[Any]]]] = {}
    rows = [run["row"] for run in runs]

    header = ["model", "level", "start_local", "wall_hours", "timed_step_hours", "evaluation_hours",
              "evaluation_pct", "rest_minutes", "inference_seconds"]
    tables["b1_durations"] = (header, [[
        row["model"], row["level"], datetime.fromisoformat(row["start"]).strftime("%Y-%m-%d %H:%M"),
        row["wall_seconds"] / 3600, row["timed_seconds_report"] / 3600, row["evaluation_seconds"] / 3600,
        row["evaluation_share"] * 100, (row["wall_seconds"] - row["timed_seconds_report"]
                                        - row["evaluation_seconds"]) / 60,
        row["inference_seconds"]] for row in rows])

    header = ["model", "level", "median_seconds_per_step", "p05_seconds_per_step", "p95_seconds_per_step",
              "cv_pct", "n_slow_windows", "n_windows", "n_slow_periods", "slow_excess_hours",
              "n_slow_evaluations", "n_evaluations", "median_eval_seconds"]
    tables["b2_rate"] = (header, [[
        row["model"], row["level"], row["median_seconds_per_step"], row["p05_seconds_per_step"],
        row["p95_seconds_per_step"], row["cv_seconds_per_step"] * 100, row["n_slow_windows"],
        row["n_windows"], row["n_slow_periods"], row["slow_excess_seconds"] / 3600,
        row["n_slow_evaluations"], row["n_evaluations"], row["median_eval_seconds"]] for row in rows])

    header = ["model", "level", "train_loss_first", "train_loss_at_best", "train_loss_final",
              "max_loss_ratio_after_warmup", "steps_per_epoch", "n_epoch_boundaries",
              "epoch_boundary_loss_change_pct_median", "epoch_midpoint_loss_change_pct_median",
              "grad_norm_warmup_max",
              "grad_norm_median_after_warmup", "grad_norm_max_after_warmup", "grad_clip_pct_after_warmup"]
    tables["b3_loss_gradient"] = (header, [[
        row["model"], row["level"], row["train_loss_first"], row["train_loss_at_best"],
        row["train_loss_final"], row["max_loss_ratio_after_warmup"], row["steps_per_epoch"],
        row["n_epoch_boundaries"],
        None if row["epoch_boundary_loss_ratio_median"] is None
        else (row["epoch_boundary_loss_ratio_median"] - 1) * 100,
        None if row["epoch_midpoint_loss_ratio_median"] is None
        else (row["epoch_midpoint_loss_ratio_median"] - 1) * 100,
        row["grad_norm_warmup_max"], row["grad_norm_median_after_warmup"],
        row["grad_norm_max_after_warmup"], row["grad_clip_share_after_warmup"] * 100] for row in rows])

    header = ["model", "level", "first_eval_step_below_100", "n_improvements",
              "max_evals_without_improvement", "eval_loss_min_step", "best_step", "lr_pct_at_best",
              "wer_gain_last_1000_points", "share_steps_after_best_pct", "hours_after_best",
              "test_minus_validation_points"]
    tables["b4_validation"] = (header, [[
        row["model"], row["level"], row["first_eval_step_below_100"], row["n_improvements"],
        row["max_evals_without_improvement"], row["eval_loss_min_step"], row["best_step"],
        row["lr_fraction_at_best"] * 100, row["wer_gain_last_1000_points"],
        row["share_steps_after_best"] * 100, row["seconds_after_best"] / 3600,
        row["test_minus_validation_points"]] for row in rows])

    header = ["model", "level", "train_peak_reserved_mib", "train_peak_allocated_mib",
              "eval_peak_reserved_mib_max", "eval_peak_allocated_mib_max", "gpu_headroom_mib",
              "gpu_memory_free_min_mib", "inference_peak_reserved_mib"]
    tables["b5_memory"] = (header, [[
        row["model"], row["level"], row["train_peak_reserved_mib"], row["train_peak_allocated_mib"],
        row["eval_peak_reserved_mib_max"], row["eval_peak_allocated_mib_max"], row["gpu_headroom_mib"],
        row["gpu_memory_free_min"], row["inference_peak_reserved_mib"]] for row in rows])

    header = ["model", "level", "n_snapshots", "gpu_temp_median", "gpu_temp_max", "gpu_power_median",
              "gpu_power_max", "gpu_utilization_median", "gpu_utilization_min", "gpu_sm_clock_median",
              "n_throttled_snapshots", "energy_kwh_estimate", "n_charger_events"]
    tables["b6_thermal_power"] = (header, [[
        row["model"], row["level"], row["n_monitor_snapshots"], row["gpu_temp_median"], row["gpu_temp_max"],
        row["gpu_power_median"], row["gpu_power_max"], row["gpu_utilization_median"],
        row["gpu_utilization_min"], row["gpu_sm_clock_median"], row["n_throttled_snapshots"],
        None if row["energy_wh_estimate"] is None else row["energy_wh_estimate"] / 1000,
        row["n_charger_events"]] for row in rows])

    header = ["run_name", "timestamp_local", "progress_step", "progress_seconds_per_step", "temperature_c",
              "power_w", "sm_clock_mhz", "throttle_mask", "flags"]
    tables["b7_throttled"] = (header, [[
        item["run_name"], datetime.fromisoformat(item["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
        item["progress_step"], item["progress_seconds_per_step"], item["temperature_c"], item["power_w"],
        item["sm_clock_mhz"], item["mask"], "+".join(item["flags"])] for item in summary["throttled_snapshots"]])

    groups = [
        group_summary("whisper_all", [s for run in runs if run["model"] == "whisper"
                                      for s in run["training_snapshots"]]),
        group_summary("xlsr_normal", [s for run in runs if run["model"] == "xlsr"
                                      for s in run["training_snapshots"] if s["slow"] is False]),
        group_summary("xlsr_slow", [s for run in runs if run["model"] == "xlsr"
                                    for s in run["training_snapshots"] if s["slow"] is True]),
    ]
    header = list(groups[0].keys())
    tables["b8_snapshot_groups"] = (header, [[group[column] for column in header] for group in groups])
    return tables


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Analisis proses pelaksanaan kesepuluh run grid.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR, help="Folder laporan run.")
    parser.add_argument("--logs-dir", type=Path, default=LOGS_DIR, help="Folder log training dan pemantau.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Folder keluaran; bawaan outputs/results. Laporan inference dibaca dari "
                             "outputs/results apa pun nilai argumen ini.")
    return parser.parse_args(argv)


def as_cell(value: Any) -> Any:
    """Nilai sel CSV: waktu menjadi ISO, daftar peristiwa menjadi teks."""
    if isinstance(value, datetime):
        return iso(value)
    if isinstance(value, list):
        return "; ".join(f"{iso(moment)} AC={'true' if online else 'false'}" for moment, online in value)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    print("=" * 78, flush=True)
    print("Analisis proses pelaksanaan kesepuluh run grid", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print(f"Ambang lambat      : {SLOW_FACTOR} x laju acuan model (kepekaan {SLOW_FACTOR_SENSITIVITY} x); "
          f"evaluasi {SLOW_EVAL_FACTOR} x median model", flush=True)
    print("=" * 78, flush=True)

    subsets = read_json(SUBSETS_REPORT)
    split = read_json(SPLIT_REPORT)
    test_seconds = float(split["parts"]["test"]["hours"]) * 3600
    validation_seconds = float(split["parts"]["validation"]["hours"]) * 3600
    charger_events = read_kernel_power(KERNEL_POWER_LOG)

    runs = [analyse_run(model, level, args.runs_dir, args.logs_dir, RESULTS_DIR,
                        subsets["levels"][level], test_seconds, validation_seconds)
            for model in MODELS for level in LEVELS]
    for run in runs:
        for snapshot in run["snapshots"]:
            charger_events.extend(snapshot["charger_events"])
    charger_events = sorted(set(charger_events))
    print(f"Run dibaca         : {len(runs)}; selang terukur dan lama evaluasi cocok dengan laporan run",
          flush=True)

    references: dict[str, dict[str, float]] = {}
    for model in MODELS:
        model_runs = [run for run in runs if run["model"] == model]
        references[model] = {
            "seconds_per_step": float(statistics.median(
                [w["seconds_per_step"] for run in model_runs for w in run["windows"]])),
            "eval_seconds": float(statistics.median(
                [e["eval_seconds"] for run in model_runs for e in run["evaluations"]])),
        }
    totals = {run["gpu_total_mib"] for run in runs}
    require(len(totals) == 1, f"VRAM terpasang berbeda antarrun: {sorted(totals)}")
    gpu_total = totals.pop()
    rows =[finish_run(run, references[run["model"]]["seconds_per_step"],
                       references[run["model"]]["eval_seconds"], gpu_total, charger_events) for run in runs]
    development = development_runs(args.runs_dir, {run["name"] for run in runs})
    summary = summarise(runs, references, development)

    print("-" * 78, flush=True)
    print(f"{'run':17s} {'jam':>5s} {'eval%':>5s} {'s/lgk':>5s} {'CV%':>5s} {'lambat':>6s} "
          f"{'+jam':>5s} {'clip%':>5s} {'T med/max':>9s} {'W med':>6s} {'thr':>3s} {'RTF':>6s}", flush=True)
    for row in rows:
        temp = (f"{row['gpu_temp_median']:.0f}/{row['gpu_temp_max']:.0f}"
                if row["gpu_temp_median"] is not None else "-")
        print(f"{row['run_name']:17s} {row['wall_seconds'] / 3600:5.2f} {row['evaluation_share'] * 100:5.1f} "
              f"{row['median_seconds_per_step']:5.2f} {row['cv_seconds_per_step'] * 100:5.1f} "
              f"{row['n_slow_windows']:6d} {row['slow_excess_seconds'] / 3600:5.2f} "
              f"{row['grad_clip_share_after_warmup'] * 100:5.1f} {temp:>9s} "
              f"{row['gpu_power_median'] or 0:6.1f} {row['n_throttled_snapshots']:3d} "
              f"{row['inference_rtf']:6.4f}", flush=True)
    print("-" * 78, flush=True)
    for model in MODELS:
        data = summary["per_model"][model]
        mon = data["monitor"]
        print(f"{model:8s}: acuan {data['reference_seconds_per_step']:.2f} s/lgk, ambang "
              f"{data['slow_threshold_seconds_per_step']:.2f}; jendela lambat {data['n_slow_windows']} dari "
              f"{data['n_windows']} ({data['n_windows_between_thresholds']} di antara kedua ambang); "
              f"waktu tambahan {data['slow_excess_hours']:.2f} jam; evaluasi lambat "
              f"{data['n_slow_evaluations']} dari {data['n_evaluations']} "
              f"({data['n_slow_evaluations_in_slow_periods']} di dalam periode lambat)", flush=True)
        print(f"          proses {data['wall_hours']:.2f} jam, evaluasi {data['evaluation_hours']:.2f} jam, "
              f"inference {data['inference_hours'] * 60:.1f} menit; suhu {mon['gpu_temperature_c'].get('min')}"
              f"-{mon['gpu_temperature_c'].get('max')} C, daya {mon['gpu_power_w'].get('min')}"
              f"-{mon['gpu_power_w'].get('max')} W, potret terbatas {mon['n_throttled']} dari {mon['n']}",
              flush=True)
    print(f"Grid               : {summary['grid_wall_hours']:.2f} jam proses training, "
          f"{summary['grid_inference_hours'] * 60:.1f} menit inference, rentang kalender "
          f"{summary['calendar_span_hours']:.1f} jam", flush=True)
    print(f"Run pengembangan   : {summary['development_runs']['n_runs']} run, "
          f"{summary['development_runs']['elapsed_hours']:.2f} jam", flush=True)
    print(f"Lingkungan         : {summary['environment_signatures_distinct']} tanda tangan, "
          f"{summary['numerics_states_distinct']} keadaan numerik; beda konfigurasi dalam model "
          f"{summary['config_differences_within_model']}", flush=True)
    print(f"Loss               : {summary['n_nonfinite']} nilai tak hingga atau NaN, "
          f"{summary['n_loss_jumps']} lonjakan sesudah warmup; loss tertinggi sesudah warmup "
          f"{summary['max_loss_ratio_after_warmup']:.2f} x loss akhir warmup", flush=True)
    print(f"Energi GPU (kasar) : {summary['energy_kwh_estimate_all']:.2f} kWh", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    sources: dict[str, str] = {}
    for run in runs:
        sources.update(run["sources"])
    sources.update({
        "build_nested_subsets_report.json": sha256_of(SUBSETS_REPORT),
        "build_text_disjoint_split_report.json": sha256_of(SPLIT_REPORT),
        f"logs/{KERNEL_POWER_LOG.name}": sha256_of(KERNEL_POWER_LOG),
    })
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": None,
        "thresholds": {
            "slow_factor": SLOW_FACTOR, "slow_factor_sensitivity": SLOW_FACTOR_SENSITIVITY,
            "slow_eval_factor": SLOW_EVAL_FACTOR, "max_grad_norm": MAX_GRAD_NORM,
            "loss_jump_factor": LOSS_JUMP_FACTOR, "loss_jump_min_increase": LOSS_JUMP_MIN_INCREASE,
            "late_window_steps": LATE_WINDOW_STEPS,
        },
        "references": references,
        "gpu_total_mib": gpu_total,
        "test_audio_seconds": test_seconds,
        "validation_audio_seconds": validation_seconds,
        "summary": summary,
        "runs": rows,
        "slow_periods": {run["name"]: run["periods"] for run in runs},
        "input_sha256": sources,
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / RUNS_TABLE, RUN_HEADER, [[row[column] for column in RUN_HEADER] for row in rows])
    write_csv(args.results_dir / WINDOWS_TABLE, WINDOW_HEADER,
              [[run["name"], run["model"], run["level"], *[window[column] for column in WINDOW_HEADER[3:]]]
               for run in runs for window in run["windows"]])
    write_csv(args.results_dir / EVALUATIONS_TABLE, EVALUATION_HEADER,
              [[run["name"], run["model"], run["level"], *[evaluation[column] for column in EVALUATION_HEADER[3:]]]
               for run in runs for evaluation in run["evaluations"]])
    write_csv(args.results_dir / MONITOR_TABLE, MONITOR_HEADER,
              [[run["name"], run["model"], run["level"],
                *[as_cell(snapshot.get(column)) for column in MONITOR_HEADER[3:]]]
               for run in runs for snapshot in run["snapshots"]])
    for key, (header, table_rows) in appendix_tables(runs, summary).items():
        write_csv(args.results_dir / f"{APPENDIX_PREFIX}{key}.csv", header, table_rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
