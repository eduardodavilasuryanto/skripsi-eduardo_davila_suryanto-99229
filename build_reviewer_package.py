"""Paket pemeriksa: kode, konfigurasi, lingkungan, pembagian, hipotesis, keluaran, dan log beserta manifest (Revisi 01 poin 24).

Tujuan
------
Pemeriksa perlu dapat mengaudit perhitungan setiap tabel tanpa melatih ulang
model. Skrip ini mengemas seluruh berkas repositori yang dilacak git ditambah
berkas yang belum dilacak tetapi tidak diabaikan `.gitignore`, sehingga bobot
model dan *checkpoint* tidak ikut. Setiap berkas dicatat pada manifest beserta
ukuran dan SHA-256-nya, dan ringkasannya memuat *commit* acuan, berkas yang
belum di-commit, revisi model prapelatihan dari konfigurasi, serta pemetaan
tabel naskah ke skrip dan keluarannya.

Masukan
-------
- Repositori `_01_python-scripts\\` pada keadaan berkas kerja.

Keluaran
--------
- `<out-dir>/paket-pemeriksa-<tanggal>.zip` : seluruh berkas beserta
  `MANIFEST.csv` dan `RINGKASAN.json` di dalamnya.
- `<out-dir>/paket-pemeriksa-<tanggal>.manifest.csv` dan `.json` : salinan di
  luar arsip.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u build_reviewer_package.py 2>&1 | Tee-Object -FilePath logs/build_reviewer_package.log
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "_15_dokumen-revisi" / "paket-pemeriksa"
GRID_CONFIGS = ["configs/trial/whisper_d1_trial.yaml", "configs/trial/xlsr_d1_trial.yaml",
                *[f"configs/{m}_d{i}.yaml" for m in ("whisper", "xlsr") for i in range(2, 6)]]

# Tabel naskah -> skrip -> keluaran, sama dengan komentar sinkronisasi bab-4.md dan lampiran.md serta
# Tabel 3.16 bab-3.md.
TABLE_MAP = [
    ("Tabel 3.4, 3.5", "build_text_disjoint_split.py, build_nested_subsets.py",
     "outputs/build_text_disjoint_split_report.json, outputs/build_nested_subsets_report.json"),
    ("Tabel 3.11 sampai 3.13", "log_environment.py", "outputs/environment_report.json"),
    ("Tabel 4.1", "summarize_runs.py", "outputs/results/run_status.csv"),
    ("Tabel 4.2, 4.3", "bootstrap_metrics.py", "outputs/results/bootstrap_metrics_*.csv"),
    ("Tabel 4.4, 4.5, Gambar 4.1", "fit_scaling_curve.py", "outputs/results/scaling_curve_*"),
    ("Tabel 4.6", "robustness_diacritics.py", "outputs/results/robustness_diacritics*"),
    ("Tabel 4.7", "error_analysis.py", "outputs/results/error_analysis_report.json"),
    ("Tabel 3.16, Subbab 4.3 sampai 4.5.3", "sepuluh skrip analisis tambahan Revisi 01",
     "outputs/results/{audit_*,analysis_speaker_coverage_*,cluster_bootstrap_*,curve_model_comparison_*,"
     "data_requirement*,h1_inference_check_*,error_boundary*,sensitivity_defective_test_*}, "
     "outputs/compare_subset_corpus_report.json"),
    # Gambar B.1 dan B.2 digambar render_dinamika_pelatihan.py dan render_proses_run.py di
    # _06_dokumen-skripsi/_02_figure-generation/, di luar repositori, dari keluaran yang sama.
    ("Subbab 4.1.2 sampai 4.1.5, Tabel B.1 sampai B.8, Gambar B.1 dan B.2", "analyze_training_process.py",
     "outputs/results/training_process_report.json, outputs/results/training_process_*.csv, "
     "outputs/results/training_process_appendix_b1_durations.csv sampai _b8_snapshot_groups.csv"),
    # Revisi pembimbing (25 November 2026), tanpa tabel baru; metode pada Subbab 3.6.7 sampai 3.6.10 bab-3.md.
    ("Subbab 4.1.6, 4.5.4 sampai 4.5.6, 4.6.2",
     "audit_audio_annotations.py, audit_speech_register.py, audit_affix_errors.py, benchmark_inference.py",
     "outputs/results/{audit_audio_annotations_*,speech_register_*,audit_affix_errors_*,"
     "benchmark_inference_whisper_d5.json,benchmark_inference_xlsr_d5.json}"),
]


# --------------------------------------------------------------------------
# Pengumpulan
# --------------------------------------------------------------------------


def git(*args: str) -> str:
    """Keluaran perintah git pada repositori ini."""
    return subprocess.run(["git", "-C", str(SCRIPT_DIR), *args], capture_output=True, check=True).stdout.decode("utf-8")


def collect_files() -> list[str]:
    """Berkas terlacak dan berkas belum terlacak yang tidak diabaikan, terurut."""
    listed = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split("\0")
    # Log skrip ini sendiri masih ditulis saat pengemasan, sehingga tidak ikut.
    return sorted(path for path in listed
                  if path and (SCRIPT_DIR / path).is_file() and path != "logs/build_reviewer_package.log")


def sha256(data: bytes) -> str:
    """SHA-256 heksadesimal."""
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Paket pemeriksa beserta manifest.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Folder keluaran paket.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    now = datetime.now(timezone.utc).astimezone()
    timestamp = now.isoformat(timespec="seconds")
    stem = f"paket-pemeriksa-{now:%Y%m%d}"

    print("=" * 78, flush=True)
    print("Paket pemeriksa (Revisi 01 poin 24)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, pengemasan deterministik", flush=True)
    print("=" * 78, flush=True)

    head = git("rev-parse", "HEAD").strip()
    untracked = sorted(p for p in git("ls-files", "--others", "--exclude-standard", "-z").split("\0") if p)
    modified = sorted(p for p in git("diff", "--name-only", "-z").split("\0") if p)
    files = collect_files()

    manifest: list[dict[str, Any]] = []
    for path in files:
        data = (SCRIPT_DIR / path).read_bytes()
        status = "belum dilacak" if path in untracked else "diubah sejak commit" if path in modified else "terlacak"
        manifest.append({"path": path, "bytes": len(data), "sha256": sha256(data), "git_status": status})

    revisions = {}
    for config_path in GRID_CONFIGS:
        config = yaml.safe_load((SCRIPT_DIR / config_path).read_text(encoding="utf-8"))
        revisions[config["run_name"]] = {"pretrained": config["pretrained"], "revision": config["revision"],
                                         "config_sha256": sha256((SCRIPT_DIR / config_path).read_bytes())}

    summary = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "repository_head": head,
        "n_files": len(manifest),
        "total_bytes": sum(item["bytes"] for item in manifest),
        "untracked_files": untracked,
        "modified_files": modified,
        "excluded": "berkas yang diabaikan .gitignore, yaitu outputs/models/, checkpoint, dan __pycache__",
        "model_revisions": revisions,
        "table_map": [{"tabel": t, "skrip": s, "keluaran": o} for t, s, o in TABLE_MAP],
        "verification": "hitung SHA-256 setiap berkas lalu cocokkan dengan MANIFEST.csv",
    }

    manifest_csv = io.StringIO()
    writer = csv.DictWriter(manifest_csv, fieldnames=["path", "bytes", "sha256", "git_status"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(manifest)
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.out_dir / f"{stem}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in manifest:
            info = zipfile.ZipInfo(f"_01_python-scripts/{item['path']}", date_time=(2026, 11, 15, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (SCRIPT_DIR / item["path"]).read_bytes())
        archive.writestr("MANIFEST.csv", manifest_csv.getvalue())
        archive.writestr("RINGKASAN.json", summary_json)
    (args.out_dir / f"{stem}.manifest.csv").write_text(manifest_csv.getvalue(), encoding="utf-8", newline="")
    (args.out_dir / f"{stem}.json").write_text(summary_json, encoding="utf-8")

    elapsed = round(time.perf_counter() - started, 1)
    print(f"Commit acuan       : {head}", flush=True)
    print(f"Berkas             : {len(manifest)} ({summary['total_bytes'] / 1e6:.1f} MB sebelum kompresi)", flush=True)
    print(f"Belum dilacak      : {len(untracked)}; diubah sejak commit: {len(modified)}", flush=True)
    print(f"Arsip              : {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB), SHA-256 "
          f"{sha256(zip_path.read_bytes())}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
