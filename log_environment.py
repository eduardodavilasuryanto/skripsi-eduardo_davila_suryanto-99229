"""Mendeteksi dan mencatat lingkungan eksekusi penelitian ke berkas laporan dan log.

Tujuan
------
Menyediakan satu sumber tunggal
bagi nilai lingkungan yang dilaporkan pada naskah BAB III Subbab 3.5, sekaligus
menyediakan fungsi yang dapat dipanggil skrip pelatihan agar setiap *run*
meninggalkan jejak lingkungannya sendiri.

Keluarannya mencakup tiga lapis informasi:

* Lapis 1, perangkat keras fisik: GPU, VRAM, jumlah GPU, CPU dan jumlah *core*,
  RAM sistem, jenis penyimpanan, serta sistem operasi dan versinya.
* Lapis 2, lapisan perangkat lunak yang terikat perangkat keras: versi driver
  NVIDIA, CUDA *runtime*, cuDNN, PyTorch beserta *build* CUDA-nya, presisi, status
  TF32, dan *flag* determinisme.
* Lapis 3, dicatat otomatis setiap *run*: *timestamp*, *random seed*, VRAM puncak
  terpakai, dan durasi eksekusi.

Catatan mengenai TF32. PyTorch memberi nilai bawaan yang berbeda bagi kedua kanal
pemakaian TF32, yaitu `torch.backends.cuda.matmul.allow_tf32` bernilai `False`
sedangkan `torch.backends.cudnn.allow_tf32` bernilai `True`. Skrip ini melaporkan
keduanya secara terpisah, karena melaporkannya sebagai satu status tunggal adalah
penyederhanaan yang keliru.

Masukan
-------
Tidak memerlukan berkas masukan. Seluruh nilai dibaca dari sistem melalui
`platform`, `psutil`, `torch`, `nvidia-smi`, dan Windows PowerShell.

Keluaran
--------
* `outputs/environment_report.json` berisi laporan lengkap dalam bentuk JSON.
* Ringkasan terbaca manusia ke stdout, yang diarahkan ke berkas log oleh perintah
  penjalanan di bawah.

Nilai yang tidak dapat dibaca ditulis sebagai `null` pada JSON dan sebagai
`TIDAK TERBACA` pada ringkasan, disertai alasannya. Nilai tersebut tidak pernah
diisi dari perkiraan.

Cara menjalankan
----------------
    python -u log_environment.py 2>&1 | Tee-Object -FilePath logs/log_environment.log

Opsi:

    python -u log_environment.py --seed 42 --output outputs/environment_report.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

SCRIPT_DIR: Path = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH: Path = SCRIPT_DIR / "outputs" / "environment_report.json"
DEFAULT_SEED: int = 42

BYTES_PER_GIB: float = 1024.0 ** 3
BYTES_PER_MIB: float = 1024.0 ** 2
SUBPROCESS_TIMEOUT_S: int = 30

UNREADABLE: str = "TIDAK TERBACA"

NVIDIA_SMI_FIELDS: tuple[str, ...] = (
    "name",
    "driver_version",
    "memory.total",
    "compute_cap",
)

POWERSHELL_STORAGE_COMMAND: str = (
    "$disks = Get-PhysicalDisk | Select-Object FriendlyName, MediaType, "
    "@{n='SizeGB';e={[math]::Round($_.Size/1GB,1)}}; "
    "$vols = Get-Volume | Where-Object { $_.DriveLetter } | Select-Object "
    "DriveLetter, FileSystemType, "
    "@{n='SizeGB';e={[math]::Round($_.Size/1GB,1)}}; "
    "[pscustomobject]@{PhysicalDisks=@($disks); Volumes=@($vols)} | "
    "ConvertTo-Json -Depth 4 -Compress"
)

#: Pustaka yang versinya dilaporkan pada naskah BAB III Tabel 3.13.
RESEARCH_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "tokenizers",
    "safetensors",
    "soundfile",
    "jiwer",
    "bitsandbytes",
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "statsmodels",
    "matplotlib",
)


def set_seed(seed: int) -> None:
    """Menetapkan *random seed* pada seluruh pustaka yang dipakai penelitian ini.

    Dipanggil meskipun skrip ini tidak melakukan penarikan sampel, karena butir 10
    konvensi penulisan skrip mewajibkan *seed* ditetapkan eksplisit dan dicatat
    pada setiap eksekusi. Fungsi ini juga menjadi rujukan tunggal bagi skrip
    pelatihan agar prosedur penetapan *seed*-nya identik.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_command(args: list[str], timeout_s: int = SUBPROCESS_TIMEOUT_S) -> str | None:
    """Menjalankan perintah eksternal dan mengembalikan stdout, atau `None` bila gagal.

    Kegagalan tidak pernah dilemparkan sebagai eksepsi, karena ketiadaan satu nilai
    lingkungan tidak boleh menggagalkan pencatatan nilai lainnya.
    """
    try:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def format_cudnn_version(raw: int | None) -> str | None:
    """Mengubah bilangan versi cuDNN dari `torch` menjadi notasi bertitik.

    cuDNN memakai dua skema penyandian. Seri 9 ke atas memakai lima digit, misalnya
    `92400` berarti 9.24.0, sedangkan seri sebelumnya memakai empat digit, misalnya
    `8907` berarti 8.9.7.
    """
    if raw is None:
        return None
    if raw >= 10_000:
        major, minor, patch = raw // 10_000, (raw % 10_000) // 100, raw % 100
    else:
        major, minor, patch = raw // 1_000, (raw % 1_000) // 100, raw % 100
    return f"{major}.{minor}.{patch}"


def read_package_versions(packages: tuple[str, ...]) -> dict[str, str | None]:
    """Membaca versi terpasang setiap paket, `None` bila paket tidak terpasang."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def query_nvidia_smi() -> dict[str, str] | None:
    """Membaca identitas GPU melalui `nvidia-smi`, atau `None` bila tidak tersedia."""
    query = ",".join(NVIDIA_SMI_FIELDS)
    raw = run_command(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    )
    if raw is None:
        return None
    first_gpu = raw.splitlines()[0]
    values = [item.strip() for item in first_gpu.split(",")]
    if len(values) != len(NVIDIA_SMI_FIELDS):
        return None
    return dict(zip(NVIDIA_SMI_FIELDS, values))


def query_storage() -> dict[str, Any] | None:
    """Membaca jenis media penyimpanan melalui PowerShell, atau `None` bila gagal.

    Pemetaan dari huruf drive ke disk fisik sengaja tidak dilakukan, karena
    pemetaan tersebut tidak dapat diandalkan pada konfigurasi dengan beberapa
    partisi. Yang dilaporkan adalah daftar disk fisik beserta jenis medianya dan
    daftar volume, sehingga pembaca dapat menyimpulkan sendiri tanpa skrip ini
    mengklaim lebih daripada yang dapat dipastikannya.
    """
    raw = run_command(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL_STORAGE_COMMAND]
    )
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def read_cpu_name() -> str | None:
    """Membaca nama dagang CPU dari registry Windows.

    `platform.processor()` pada Windows hanya mengembalikan penanda keluarga,
    misalnya `Intel64 Family 6 Model 191 Stepping 2`, yang tidak layak dikutip
    pada naskah. Registry memuat nama sesungguhnya. Pada sistem non-Windows atau
    bila kunci registry tidak terbaca, fungsi ini jatuh kembali ke
    `platform.processor()`.
    """
    try:
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        name = str(value).strip()
        if name:
            return name
    except (ImportError, OSError, ValueError):
        pass
    return platform.processor() or None


def collect_layer1_hardware() -> dict[str, Any]:
    """Mengumpulkan Lapis 1, yaitu perangkat keras fisik dan sistem operasi."""
    memory = psutil.virtual_memory()
    smi = query_nvidia_smi()

    gpu: dict[str, Any] = {
        "jumlah_gpu_terdeteksi_torch": torch.cuda.device_count(),
        "nama": None,
        "vram_total_mib": None,
        "compute_capability": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu["nama"] = props.name
        gpu["vram_total_mib"] = round(props.total_memory / BYTES_PER_MIB)
        gpu["compute_capability"] = f"sm_{props.major}{props.minor}"
        gpu["jumlah_streaming_multiprocessor"] = props.multi_processor_count
    elif smi is not None:
        gpu["nama"] = smi.get("name")
        gpu["vram_total_mib"] = smi.get("memory.total")
        gpu["compute_capability"] = smi.get("compute_cap")

    return {
        "gpu": gpu,
        "cpu": {
            "nama": read_cpu_name(),
            "penanda_keluarga": platform.processor() or None,
            "core_fisik": psutil.cpu_count(logical=False),
            "core_logis": psutil.cpu_count(logical=True),
        },
        "ram_sistem": {
            "total_gib": round(memory.total / BYTES_PER_GIB, 2),
            "tersedia_saat_pencatatan_gib": round(memory.available / BYTES_PER_GIB, 2),
        },
        "penyimpanan": query_storage(),
        "sistem_operasi": {
            "platform": platform.platform(),
            "sistem": platform.system(),
            "rilis": platform.release(),
            "versi": platform.version(),
            "arsitektur": platform.machine(),
            "nama_host": socket.gethostname(),
        },
    }


def collect_layer2_software() -> dict[str, Any]:
    """Mengumpulkan Lapis 2, yaitu lapisan perangkat lunak yang terikat perangkat keras."""
    smi = query_nvidia_smi()
    cuda_available = torch.cuda.is_available()

    return {
        "python": {
            "versi": platform.python_version(),
            "implementasi": platform.python_implementation(),
            "executable": sys.executable,
        },
        "driver_nvidia": {
            "versi": smi.get("driver_version") if smi else None,
            "sumber": "nvidia-smi" if smi else UNREADABLE,
        },
        "cuda": {
            "tersedia": cuda_available,
            "runtime_versi_torch": torch.version.cuda,
            "cudnn_versi_mentah": torch.backends.cudnn.version() if cuda_available else None,
            "cudnn_versi": format_cudnn_version(
                torch.backends.cudnn.version() if cuda_available else None
            ),
        },
        "pytorch": {
            "versi": torch.__version__,
            "build_cuda": torch.version.cuda,
            "versi_lengkap": f"{torch.__version__}",
        },
        "presisi": {
            "dtype_bawaan": str(torch.get_default_dtype()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "bf16_didukung": torch.cuda.is_bf16_supported() if cuda_available else None,
        },
        "tf32": {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "catatan": (
                "Kedua kanal dilaporkan terpisah karena nilai bawaannya berbeda. "
                "Melaporkannya sebagai satu status tunggal adalah penyederhanaan "
                "yang keliru."
            ),
        },
        "determinisme": {
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "pustaka": read_package_versions(RESEARCH_PACKAGES),
    }


def collect_layer3_runtime(seed: int, started_at: float) -> dict[str, Any]:
    """Mengumpulkan Lapis 3, yaitu jejak eksekusi yang dicatat pada setiap *run*.

    VRAM puncak bernilai nol apabila fungsi ini dipanggil oleh skrip ini sendiri,
    karena tidak ada tensor yang dialokasikan. Nilainya menjadi bermakna ketika
    fungsi ini dipanggil skrip pelatihan pada akhir sebuah *run*.
    """
    peak_allocated = peak_reserved = None
    if torch.cuda.is_available():
        peak_allocated = round(torch.cuda.max_memory_allocated() / BYTES_PER_MIB, 2)
        peak_reserved = round(torch.cuda.max_memory_reserved() / BYTES_PER_MIB, 2)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timestamp_lokal": datetime.now().astimezone().isoformat(timespec="seconds"),
        "random_seed": seed,
        "vram_puncak_teralokasi_mib": peak_allocated,
        "vram_puncak_tercadang_mib": peak_reserved,
        "durasi_pengumpulan_detik": round(time.perf_counter() - started_at, 3),
    }


def collect_environment(seed: int, started_at: float) -> dict[str, Any]:
    """Merakit laporan lengkap ketiga lapis menjadi satu struktur.

    Fungsi inilah yang dipanggil skrip pelatihan agar setiap *run* meninggalkan
    jejak lingkungannya sendiri tanpa menduplikasi kode pembacaan.
    """
    return {
        "skrip": Path(__file__).name,
        "lapis_1_perangkat_keras": collect_layer1_hardware(),
        "lapis_2_perangkat_lunak_terikat_perangkat_keras": collect_layer2_software(),
        "lapis_3_jejak_eksekusi": collect_layer3_runtime(seed, started_at),
    }


def _show(label: str, value: Any, width: int = 34) -> str:
    """Membentuk satu baris ringkasan dengan label selebar tetap."""
    rendered = UNREADABLE if value is None else str(value)
    return f"  {label:<{width}}: {rendered}"


def render_summary(report: dict[str, Any]) -> str:
    """Membentuk ringkasan terbaca manusia dari laporan."""
    layer1 = report["lapis_1_perangkat_keras"]
    layer2 = report["lapis_2_perangkat_lunak_terikat_perangkat_keras"]
    layer3 = report["lapis_3_jejak_eksekusi"]

    lines: list[str] = ["=" * 78, "LINGKUNGAN EKSEKUSI PENELITIAN", "=" * 78]

    lines += ["", "LAPIS 1. PERANGKAT KERAS FISIK"]
    gpu = layer1["gpu"]
    lines += [
        _show("GPU", gpu["nama"]),
        _show("VRAM total (MiB)", gpu["vram_total_mib"]),
        _show("Compute capability", gpu["compute_capability"]),
        _show("Jumlah GPU", gpu["jumlah_gpu_terdeteksi_torch"]),
        _show("CPU", layer1["cpu"]["nama"]),
        _show("Core fisik / logis", f"{layer1['cpu']['core_fisik']} / {layer1['cpu']['core_logis']}"),
        _show("RAM sistem (GiB)", layer1["ram_sistem"]["total_gib"]),
        _show("Sistem operasi", layer1["sistem_operasi"]["platform"]),
    ]
    storage = layer1["penyimpanan"]
    if storage is None:
        lines.append(_show("Penyimpanan", None))
    else:
        disks = storage.get("PhysicalDisks") or []
        if isinstance(disks, dict):
            disks = [disks]
        for index, disk in enumerate(disks, start=1):
            detail = f"{disk.get('MediaType')}, {disk.get('SizeGB')} GB, {disk.get('FriendlyName')}"
            lines.append(_show(f"Disk fisik {index}", detail))

    lines += ["", "LAPIS 2. PERANGKAT LUNAK TERIKAT PERANGKAT KERAS"]
    lines += [
        _show("Python", layer2["python"]["versi"]),
        _show("Driver NVIDIA", layer2["driver_nvidia"]["versi"]),
        _show("CUDA tersedia", layer2["cuda"]["tersedia"]),
        _show("CUDA runtime (torch)", layer2["cuda"]["runtime_versi_torch"]),
        _show("cuDNN", layer2["cuda"]["cudnn_versi"]),
        _show("PyTorch", layer2["pytorch"]["versi"]),
        _show("dtype bawaan", layer2["presisi"]["dtype_bawaan"]),
        _show("float32_matmul_precision", layer2["presisi"]["float32_matmul_precision"]),
        _show("bf16 didukung", layer2["presisi"]["bf16_didukung"]),
        _show("TF32 matmul", layer2["tf32"]["cuda_matmul_allow_tf32"]),
        _show("TF32 cuDNN", layer2["tf32"]["cudnn_allow_tf32"]),
        _show("cudnn.deterministic", layer2["determinisme"]["cudnn_deterministic"]),
        _show("cudnn.benchmark", layer2["determinisme"]["cudnn_benchmark"]),
        _show("deterministic_algorithms", layer2["determinisme"]["use_deterministic_algorithms"]),
    ]

    lines += ["", "  Versi pustaka penelitian:"]
    for name, version in layer2["pustaka"].items():
        lines.append(_show(f"  {name}", version))

    lines += ["", "LAPIS 3. JEJAK EKSEKUSI"]
    lines += [
        _show("Timestamp lokal", layer3["timestamp_lokal"]),
        _show("Random seed", layer3["random_seed"]),
        _show("VRAM puncak teralokasi (MiB)", layer3["vram_puncak_teralokasi_mib"]),
        _show("Durasi pengumpulan (detik)", layer3["durasi_pengumpulan_detik"]),
    ]
    lines += ["", "=" * 78]
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_path: Path) -> None:
    """Menuliskan laporan sebagai JSON ber-*encoding* UTF-8 eksplisit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    """Membaca argumen baris perintah."""
    parser = argparse.ArgumentParser(
        description="Mencatat lingkungan eksekusi penelitian ke JSON dan stdout."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed yang ditetapkan dan dicatat (bawaan {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Lokasi berkas laporan JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Titik masuk skrip."""
    started_at = time.perf_counter()
    args = parse_args()

    set_seed(args.seed)
    report = collect_environment(seed=args.seed, started_at=started_at)

    print(render_summary(report))
    write_report(report, args.output)
    print(f"\nLaporan JSON ditulis ke: {args.output}")

    if not report["lapis_2_perangkat_lunak_terikat_perangkat_keras"]["cuda"]["tersedia"]:
        print(
            "\nPERINGATAN: CUDA tidak terdeteksi. Nilai lapis 1 dan 2 yang "
            "bergantung pada GPU tidak terbaca."
        )


if __name__ == "__main__":
    main()
