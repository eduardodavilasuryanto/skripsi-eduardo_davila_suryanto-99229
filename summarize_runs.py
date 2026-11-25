"""Rekap status pelatihan kesepuluh konfigurasi grid dari laporan setiap run.

Tujuan
------
Menghasilkan Tabel 4.1 `bab-4.md`, yaitu status pelatihan setiap konfigurasi,
sesuai komitmen `DraftResearch.md` Lampiran B.16 bahwa status penghentian
setiap run dilaporkan bersama hasilnya. Rekap yang sama memuat angka prosa
`bab-4.md` Subbab 4.1 dan puncak VRAM pelatihan pada `bab-3.md` Subbab 3.5.1.
Sebelum skrip ini ada, Tabel 4.1 disusun oleh skrip bantu sesi di luar
repositori.

Skrip ini hanya membaca laporan yang ditulis `train_asr.py` selama pelatihan.
Skrip tidak memuat model, tidak memakai GPU, dan tidak mengubah berkas apa pun
pada `outputs/runs/`. Perhitungannya deterministik dan tidak memakai random
seed.

Definisi kolom
--------------
- Langkah terakhir       : `result.global_step` laporan run.
- Epoch                  : `epoch` pada `trainer_state.json`.
- Penghentian            : `early_stopping` bila `result.stopped_early`,
                           selain itu `max_steps`, yaitu batas 10.000 langkah.
- Langkah terbaik        : `result.best_step`, yaitu checkpoint dengan WER
                           validation terendah yang disimpan sebagai model
                           dan dipakai `run_inference.py`.
- WER validation terbaik : `result.best_validation_wer` atas 1.900 utterance.
- Learning rate akhir    : nilai `learning_rate` terakhir pada `log_history`
                           `trainer_state.json`, yang tercatat pada langkah
                           terakhir, dibagi learning rate puncak konfigurasi.
- Durasi                 : `result.train_runtime_seconds`, mencakup evaluasi
                           validation dan penyimpanan checkpoint selama
                           pelatihan (Lampiran B.11).
- Detik per langkah      : `result.seconds_per_step`, tanpa evaluasi dan
                           penyimpanan.
- Puncak VRAM            : `result.train_peak_reserved_mib` dan
                           `result.train_peak_allocated_mib`.

Nilai yang tercatat pada kedua berkas dicocokkan lebih dulu, dan salinan
konfigurasi pada folder run dicocokkan dengan SHA-256 yang tercatat pada
laporan run. Ketidakcocokan apa pun menghentikan skrip sebelum keluaran ditulis.

Masukan
-------
- `outputs/runs/<run>/run_report.json` kesepuluh konfigurasi.
- `outputs/runs/<run>/trainer_state.json` kesepuluh konfigurasi.
- `outputs/runs/<run>/<run>.yaml` : salinan konfigurasi, hanya untuk pemeriksaan.

Keluaran
--------
- `outputs/results/run_status.csv` : satu baris per konfigurasi.
- `outputs/results/run_status_report.json` : laporan lengkap, ringkasan, dan
  SHA-256 masukan.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u summarize_runs.py 2>&1 |
        Tee-Object -FilePath logs/summarize_runs.log
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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

RUNS_DIR = Path(__file__).resolve().parent / "outputs" / "runs"

REPORT_NAME = "run_status_report.json"
TABLE_NAME = "run_status.csv"

STOP_EARLY = "early_stopping"
STOP_MAX_STEPS = "max_steps"

# Toleransi kesamaan nilai pecahan yang tercatat di dua berkas.
FLOAT_TOLERANCE = 1e-12

TABLE_HEADER = [
    "model", "level", "run_name", "last_step", "epoch", "stop_reason", "best_step",
    "best_validation_wer", "peak_learning_rate", "final_learning_rate",
    "final_lr_fraction_of_peak", "train_runtime_hours", "seconds_per_step",
    "train_peak_reserved_mib", "train_peak_allocated_mib", "n_train_utterances",
    "steps_per_epoch",
]


# --------------------------------------------------------------------------
# Pembacaan dan pemeriksaan
# --------------------------------------------------------------------------


def read_json(path: Path) -> dict[str, Any]:
    """Membaca berkas JSON."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    """Menghentikan skrip bila sebuah pemeriksaan kesamaan gagal."""
    if not condition:
        raise ValueError(message)


def summarize_run(model: str, level: str, runs_dir: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """Status pelatihan satu konfigurasi beserta SHA-256 berkas sumbernya."""
    name = run_name(model, level)
    folder = runs_dir / name
    report_path = folder / "run_report.json"
    state_path = folder / "trainer_state.json"
    config_path = folder / f"{name}.yaml"

    report = read_json(report_path)
    state = read_json(state_path)
    config = report["config"]
    result = report["result"]

    require(report.get("mode") == "train", f"{name}: laporan run bukan mode pelatihan")
    require(config["run_name"] == name, f"{name}: run_name pada laporan adalah {config['run_name']}")
    require(sha256_of(config_path) == report["config_sha256"],
            f"{name}: salinan konfigurasi tidak cocok dengan SHA-256 laporan run")

    require(state["global_step"] == result["global_step"], f"{name}: langkah terakhir berbeda antarberkas")
    require(state["best_global_step"] == result["best_step"], f"{name}: langkah terbaik berbeda antarberkas")
    require(abs(state["best_metric"] - result["best_validation_wer"]) <= FLOAT_TOLERANCE,
            f"{name}: WER validation terbaik berbeda antarberkas")
    require(state["max_steps"] == config["max_steps"], f"{name}: max_steps berbeda antarberkas")

    best_eval = min(result["validation_history"], key=lambda entry: entry["wer"])
    require(best_eval["step"] == result["best_step"],
            f"{name}: langkah terbaik tidak sama dengan evaluasi WER terendah")

    stopped_early = bool(result["stopped_early"])
    if stopped_early:
        require(result["global_step"] < config["max_steps"], f"{name}: berhenti dini pada atau sesudah max_steps")
    else:
        require(result["global_step"] == config["max_steps"], f"{name}: tidak berhenti dini tetapi belum mencapai max_steps")

    logged = [entry for entry in state["log_history"] if "learning_rate" in entry]
    require(bool(logged), f"{name}: log_history tidak memuat learning_rate")
    require(logged[-1]["step"] == result["global_step"],
            f"{name}: learning rate terakhir tercatat pada langkah {logged[-1]['step']}, bukan langkah terakhir")
    peak_lr = float(config["learning_rate"])
    final_lr = float(logged[-1]["learning_rate"])

    row = {
        "model": model,
        "level": level,
        "run_name": name,
        "last_step": int(result["global_step"]),
        "epoch": float(state["epoch"]),
        "stop_reason": STOP_EARLY if stopped_early else STOP_MAX_STEPS,
        "best_step": int(result["best_step"]),
        "best_validation_wer": float(result["best_validation_wer"]),
        "peak_learning_rate": peak_lr,
        "final_learning_rate": final_lr,
        "final_lr_fraction_of_peak": final_lr / peak_lr,
        "train_runtime_hours": float(result["train_runtime_seconds"]) / 3600,
        "seconds_per_step": float(result["seconds_per_step"]),
        "train_peak_reserved_mib": int(result["train_peak_reserved_mib"]),
        "train_peak_allocated_mib": int(result["train_peak_allocated_mib"]),
        "n_train_utterances": int(result["n_train_utterances"]),
        "steps_per_epoch": int(result["steps_per_epoch"]),
        "config_sha256": report["config_sha256"],
    }
    sources = {
        f"{name}/run_report.json": sha256_of(report_path),
        f"{name}/trainer_state.json": sha256_of(state_path),
    }
    return row, sources


def summary_of(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Ringkasan lintas run yang dikutip pada prosa naskah."""
    early = [row for row in rows if row["stop_reason"] == STOP_EARLY]
    peak = max(rows, key=lambda row: row["train_peak_reserved_mib"])
    hours = [row["train_runtime_hours"] for row in rows]
    return {
        "n_runs": len(rows),
        "n_early_stopping": len(early),
        "n_max_steps": len(rows) - len(early),
        "max_steps_runs": [row["run_name"] for row in rows if row["stop_reason"] == STOP_MAX_STEPS],
        "early_stopping_final_lr_fraction_range": [
            min(row["final_lr_fraction_of_peak"] for row in early),
            max(row["final_lr_fraction_of_peak"] for row in early),
        ],
        "train_runtime_hours_range": [min(hours), max(hours)],
        "train_peak_reserved_mib_max": peak["train_peak_reserved_mib"],
        "train_peak_reserved_mib_max_run": peak["run_name"],
    }


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Rekap status pelatihan kesepuluh konfigurasi.")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR,
                        help="Folder laporan run; bawaan outputs/runs.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Folder keluaran; bawaan outputs/results.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    print("=" * 78, flush=True)
    print("Rekap status pelatihan kesepuluh konfigurasi", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, perhitungan deterministik", flush=True)
    print(f"Folder run         : {args.runs_dir}", flush=True)
    print("=" * 78, flush=True)

    rows: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for model in MODELS:
        for level in LEVELS:
            row, run_sources = summarize_run(model, level, args.runs_dir)
            rows.append(row)
            sources.update(run_sources)
    print(f"Run dibaca         : {len(rows)}; seluruh pemeriksaan kesamaan antarberkas lolos", flush=True)

    print("-" * 78, flush=True)
    print(f"{'run':17s} {'langkah':>7s} {'epoch':>6s} {'berhenti':>14s} {'terbaik':>7s} "
          f"{'WERval%':>7s} {'LR%':>4s} {'jam':>5s} {'s/lgk':>6s} {'MiB':>5s}", flush=True)
    for row in rows:
        print(f"{row['run_name']:17s} {row['last_step']:7d} {row['epoch']:6.1f} {row['stop_reason']:>14s} "
              f"{row['best_step']:7d} {row['best_validation_wer'] * 100:7.1f} "
              f"{row['final_lr_fraction_of_peak'] * 100:4.0f} {row['train_runtime_hours']:5.1f} "
              f"{row['seconds_per_step']:6.2f} {row['train_peak_reserved_mib']:5d}", flush=True)

    summary = summary_of(rows)
    lr_low, lr_high = summary["early_stopping_final_lr_fraction_range"]
    hours_low, hours_high = summary["train_runtime_hours_range"]
    print("-" * 78, flush=True)
    print(f"Early stopping     : {summary['n_early_stopping']} run, learning rate akhir "
          f"{lr_low * 100:.0f} sampai {lr_high * 100:.0f} persen dari puncak", flush=True)
    print(f"Batas max_steps    : {summary['n_max_steps']} run ({', '.join(summary['max_steps_runs'])})", flush=True)
    print(f"Durasi per run     : {hours_low:.1f} sampai {hours_high:.1f} jam", flush=True)
    print(f"Puncak VRAM        : {summary['train_peak_reserved_mib_max']} MiB tercadang "
          f"({summary['train_peak_reserved_mib_max_run']})", flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": None,
        "sources": "run_report.json dan trainer_state.json yang ditulis train_asr.py",
        "definitions": {
            "final_learning_rate": "learning_rate terakhir pada log_history trainer_state.json, "
                                   "tercatat pada langkah terakhir",
            "train_runtime_hours": "train_runtime_seconds, mencakup evaluasi validation dan "
                                   "penyimpanan checkpoint",
            "seconds_per_step": "seconds_per_step laporan run, tanpa evaluasi dan penyimpanan",
        },
        "summary": summary,
        "runs": rows,
        "input_sha256": sources,
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    write_csv(args.results_dir / TABLE_NAME, TABLE_HEADER,
              [[row[column] for column in TABLE_HEADER] for row in rows])

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
