"""Pemeriksa angka paper terhadap keluaran skrip penelitian.

Tujuan
------
Memastikan setiap angka pada paper `_06_dokumen-skripsi/_04_paper/paper.md`
berasal dari keluaran skrip penelitian, bukan disalin tangan dari naskah
skripsi. Atas keputusan pengguna 4 November 2026, paper memuat tabel dan
daftar pustakanya sendiri sebagai pengecualian terhadap aturan Sesi 9 bahwa
tabel hanya hidup pada bab naskah. Skrip ini adalah pengaman pengecualian itu.

Tiga jenis pemeriksaan dijalankan atas bagian paper pada `CHECKED_SECTIONS`:

1. Klaim prosa. Setiap teks yang diharapkan dibentuk dari nilai sumber dengan
   format angka naskah, yaitu koma desimal dan titik ribuan, lalu harus muncul
   pada bagian paper yang ditentukan. Klaim juga dapat bersyarat: bila syarat
   pada sumber tidak terpenuhi, klaim gagal walaupun teksnya ada.
2. Tabel. Isi setiap tabel paper dicocokkan sel demi sel dengan tabel yang
   dibangun dari sumber, dan tabel yang tidak dikenal skrip ini menggagalkan
   pemeriksaan.
3. Cakupan. Setiap bilangan pada bagian yang diperiksa harus tercakup oleh
   salah satu klaim. Bilangan tanpa klaim menggagalkan pemeriksaan, sehingga
   angka tanpa sumber tidak dapat masuk paper diam-diam. Bilangan di dalam
   tabel, blok persamaan, sitasi, rujukan silang, dan nama model tidak
   dihitung sebagai bilangan prosa.

4. Rujukan. Setiap entri Daftar Pustaka harus sama persis dengan satu entri
   registri `DraftResearch.md` Bab 14, entri bernomor 1 sampai N tanpa celah,
   setiap nomor yang disitasi memiliki entri dan setiap entri disitasi, serta
   urutan kemunculan pertama sitasi adalah 1, 2, 3, dan seterusnya. Pemetaan
   nomor registri ke nomor paper diturunkan dari kecocokan teks entri itu.
   Klaim ditulis dengan nomor registri, lalu diterjemahkan lewat pemetaan
   tersebut sebelum dicari pada paper, sehingga sitasi yang menunjuk rujukan
   keliru ikut menggagalkan klaim.
5. Abstrak. Abstrak dan *Abstract* masing-masing paling banyak 250 kata di luar
   baris kata kunci, dan masing-masing memuat 5 kata kunci. Angka *Abstract*
   ditulis dengan format Inggris, yaitu titik desimal dan koma ribuan.

Selain itu diperiksa akhiran baris CRLF. Bagian yang diperiksa bertambah
setiap tahap penulisan paper.

Beberapa bilangan bukan keluaran skrip penelitian, yaitu spesifikasi model
dan corpus yang tercatat pada `bab-3.md`. Bilangan itu ditulis sebagai
konstanta dengan sumbernya, dan klaimnya ditandai sebagai spesifikasi luar.

Skrip hanya membaca, deterministik tanpa random seed, dan tidak memakai GPU.

Masukan
-------
- `../_06_dokumen-skripsi/_04_paper/paper.md` : paper, di luar repositori.
- `outputs/build_text_disjoint_split_report.json`
- `outputs/build_nested_subsets_report.json`
- `outputs/audit_transcript_leakage_report.json`
- `outputs/measure_component_durations_report.json`
- `outputs/results/bootstrap_metrics_report.json`, `bootstrap_metrics_configs.csv`,
  dan `bootstrap_metrics_h1.csv`
- `outputs/results/scaling_curve_report.json` dan `scaling_curve_parameters.csv`
- `outputs/results/run_status.csv`
- `outputs/results/robustness_diacritics_report.json`
- `outputs/results/error_analysis_report.json`
- `outputs/results/<run>_test_metrics.json` kesepuluh run grid
- `outputs/runs/<run>/run_report.json` kesepuluh run grid, serta
  `trainer_state.json` keempat run uji determinisme
- `../_02_context-MD-files/DraftResearch.md`, `bab-1.md`, dan `bab-2.md` :
  hulu angka penelitian terdahulu yang dikutip paper. `DraftResearch.md` Bab 14
  sekaligus registri rujukan bagi pemeriksaan Daftar Pustaka.
- `configs/trial/*_d1_trial.yaml`, `configs/*_d{2..5}.yaml`, dan
  `configs/trial/*_det_{a,b}.yaml`
- `train_asr.py` dan `run_inference.py` pada commit `5fda5f9`, lewat `git show`
- Nilai bawaan `TrainingArguments` dan `GenerationConfig` pada transformers yang
  terpasang
- `generation_config.json` Whisper *small* pada revisi yang dikunci, di cache
  Hugging Face lokal
- `../_03_dataset/asr_javanese_0/LICENSE`

Keluaran
--------
- `outputs/verify_paper_numbers_report.json` : hasil setiap klaim, tabel, dan
  bilangan yang tidak tercakup, beserta SHA-256 paper dan setiap sumber.
- Ringkasan ke stdout, untuk disalurkan ke berkas log. Kode keluar 1 bila ada
  yang gagal.

Perintah
--------
    python -u verify_paper_numbers.py 2>&1 |
        Tee-Object -FilePath logs/verify_paper_numbers.log
"""

from __future__ import annotations

import argparse
import ast
import csv
import dataclasses
import hashlib
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Sequence

import yaml
from transformers import GenerationConfig, TrainingArguments

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER = SCRIPT_DIR.parent / "_06_dokumen-skripsi" / "_04_paper" / "paper.md"
OUTPUTS = SCRIPT_DIR / "outputs"
RESULTS = OUTPUTS / "results"
REPORT_PATH = OUTPUTS / "verify_paper_numbers_report.json"
LICENSE_PATH = SCRIPT_DIR.parent / "_03_dataset" / "asr_javanese_0" / "LICENSE"
WHISPER_CACHE = Path.home() / ".cache" / "huggingface" / "hub" / "models--openai--whisper-small"

# Bagian paper yang sudah ditulis dan diperiksa. Tahap 2: Metode Penelitian.
# Tahap 3: pengantar Bagian 3 dan Bagian 3.1 sampai 3.5. Tahap 4: Bagian 3.6, 3.7, dan 4.
# Tahap 5: Bagian 1 Pendahuluan. Tahap 6: Abstrak, Abstract, serta Ketersediaan Data dan Kode.
CHECKED_SECTIONS = ("abstrak", "abstract", "1", "2.1", "2.2", "2.3", "2.4", "2.5", "3", "3.1", "3.2", "3.3",
                    "3.4", "3.5", "3.6", "3.7", "4", "ketersediaan")
# Judul bagian tanpa nomor dan kunci bagiannya.
NAMED_SECTIONS = {"Abstrak": "abstrak", "Abstract": "abstract", "Ucapan Terima Kasih": "terima_kasih",
                  "Ketersediaan Data dan Kode": "ketersediaan", "Daftar Pustaka": "pustaka"}
ABSTRACT_SECTIONS = {"abstrak": "**Kata kunci:**", "abstract": "**Keywords:**"}
ABSTRACT_MAX_WORDS = 250
N_KEYWORDS = 5
# Naskah hulu bagi angka penelitian terdahulu; angka literatur pada paper harus ada di salah satunya.
LITERATURE_SOURCES = ("DraftResearch.md", "bab-1.md", "bab-2.md")
REGISTRY_SOURCE = "DraftResearch.md"   # Bab 14, registri tunggal nomor rujukan

PIPELINE_COMMIT = "5fda5f9"
PIPELINE_FILES = ("train_asr.py", "run_inference.py", "asr_common.py")
MODELS = ("whisper", "xlsr")
LEVELS = ("d1", "d2", "d3", "d4", "d5")
LEVEL_KEYS_DIFFERING = frozenset({"run_name", "train_subset"})
DET_PAIRS = (("whisper_det_a", "whisper_det_b"), ("xlsr_det_a", "xlsr_det_b"))
OPTIMIZER_DEFAULTS = ("weight_decay", "max_grad_norm")
XLSR_ZERO_OVERRIDES = ("attention_dropout", "hidden_dropout", "feat_proj_dropout", "layerdrop")

WARNING_ATTENTION_TEXT = "Memory Efficient attention defaults to a non-deterministic algorithm"
WARNING_ATTENTION_FIX = "use_deterministic_algorithms(True, warn_only=False)"
WARNING_CTC = "ctc_loss_backward_gpu"
WARNING_CTC_TEXT = "does not have a deterministic implementation"

# Spesifikasi luar, bukan keluaran skrip penelitian.
N_CORPUS_ARCHIVES = 16                 # arsip SLR35 di OpenSLR; bab-3.md Subbab 3.1
PARAMS_MILLIONS = {"whisper": 244, "xlsr": 315, "whisper_medium": 769}  # bab-3.md Subbab 3.4.1
WHISPER_WINDOW_SECONDS = 30            # jendela masukan Whisper [3]; bab-3.md Subbab 3.1
SPEC_SOURCE = "spesifikasi luar, bab-3.md"

NUMBER_WORDS = {1: "satu", 2: "dua", 3: "tiga", 4: "empat", 5: "lima", 7: "tujuh", 8: "delapan",
                10: "sepuluh", 16: "enam belas", 20: "dua puluh"}
SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MINUS = "−"
STOP_LABELS = {"early_stopping": "*Early stopping*", "max_steps": "Batas langkah"}
MODEL_LABELS = {"whisper": "Whisper *small*", "xlsr": "XLS-R 300M"}

SECTION_HEADING = re.compile(r"^#{2,3} (\d+(?:\.\d+)?)\.? ")
NAMED_HEADING = re.compile(r"^## (.+)$")
TABLE_CAPTION = re.compile(r"^\*\*Tabel (\d+) (.+)\*\*$")
CITATION = re.compile(r"\[(\d+)\]")
REFERENCE_ENTRY = re.compile(r"^\[(\d+)\] (.+)$")
REGISTRY_ENTRY = re.compile(r"^\[(\d+)\]\t(.+)$")
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
NUMBER_TOKEN = re.compile(r"(?<![\w.,/+])\d+(?:[.,]\d+)*")
MASKED_PATTERNS = (
    re.compile(r"```.*?```", re.S),                                   # blok persamaan
    re.compile(r"!\[[^\]]*\]\([^)]*\)"),                              # tautan gambar
    re.compile(r"\[\d+\]"),                                           # sitasi bernomor
    re.compile(r"\b(?:Bagian|Tabel|Gambar|[Pp]ersamaan)\s+\(?\d+(?:\.\d+)?\)?"      # rujukan silang
               r"(?:\s*(?:,|dan|sampai)\s*\(?\d+(?:\.\d+)?\)?)*"),
    re.compile(r"SHA-256"),
    re.compile(r"XLS-R 300M"),
    re.compile(r"\(\d\)"),                                            # penomor (1), (2), (3) di dalam kalimat
)


# --------------------------------------------------------------------------
# Format angka naskah
# --------------------------------------------------------------------------


def fmt_dec(value: float, places: int) -> str:
    """Bilangan dengan koma desimal dan titik ribuan, dibulatkan setengah ke atas."""
    quantum = Decimal(1).scaleb(-places)
    rounded = Decimal(repr(float(value))).quantize(quantum, rounding=ROUND_HALF_UP)
    sign, digits = (MINUS, f"{-rounded:f}") if rounded < 0 else ("", f"{abs(rounded):f}")
    integer, _, fraction = digits.partition(".")
    grouped = f"{int(integer):,}".replace(",", ".")
    return sign + grouped + ("," + fraction if fraction else "")


def pct(value: float, places: int) -> str:
    """Proporsi sebagai persen atau poin persentase."""
    return fmt_dec(100 * value, places)


def interval(low: float, high: float, places: int, scale: float = 100.0) -> str:
    """Selang dalam kurung siku, misalnya [40,2; 42,3]."""
    return f"[{fmt_dec(scale * low, places)}; {fmt_dec(scale * high, places)}]"


def signed(value: float, places: int) -> str:
    """Bilangan bertanda; nilai positif diberi tanda plus."""
    text = fmt_dec(value, places)
    return "+" + text if not text.startswith(MINUS) and text.strip("0,") else text


def level_pct(value: float) -> str:
    """Taraf selang dalam persen tanpa nol di belakang, misalnya 98,75 atau 99."""
    text = pct(value, 2)
    return text.rstrip("0").rstrip(",") if "," in text else text


def fmt_int(value: float) -> str:
    """Bilangan bulat dengan titik ribuan."""
    return fmt_dec(value, 0)


def en(text: str) -> str:
    """Bilangan berformat naskah diubah ke format Inggris: titik desimal, koma ribuan."""
    return text.translate(str.maketrans({".": ",", ",": "."}))


def fmt_sci(value: float) -> str:
    """Notasi ilmiah satu angka penting, misalnya 1 × 10⁻⁵."""
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa} × 10{str(int(exponent)).translate(SUPERSCRIPT)}"


def word(number: int) -> str:
    """Bilangan dalam kata, hanya bagi bilangan yang dipakai paper."""
    return NUMBER_WORDS[number]


# --------------------------------------------------------------------------
# Pembacaan sumber
# --------------------------------------------------------------------------


def sha256_of(path: Path) -> str:
    """Hash SHA-256 isi berkas."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, used: dict[str, str]) -> Any:
    """Berkas JSON, sekaligus mencatat SHA-256-nya."""
    used[str(path.relative_to(SCRIPT_DIR.parent))] = sha256_of(path)
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path, used: dict[str, str]) -> list[dict[str, str]]:
    """Berkas CSV sebagai daftar baris, sekaligus mencatat SHA-256-nya."""
    used[str(path.relative_to(SCRIPT_DIR.parent))] = sha256_of(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_yaml(path: Path, used: dict[str, str]) -> dict[str, Any]:
    """Berkas konfigurasi YAML, sekaligus mencatat SHA-256-nya."""
    used[str(path.relative_to(SCRIPT_DIR.parent))] = sha256_of(path)
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def git_show(commit: str, path: str) -> str:
    """Isi berkas pada sebuah commit."""
    output = subprocess.run(["git", "-C", str(SCRIPT_DIR), "show", f"{commit}:{path}"],
                            capture_output=True, check=True).stdout
    return output.decode("utf-8")


def unchanged_since(commit: str, path: str) -> bool:
    """Apakah berkas kerja identik dengan berkas pada commit."""
    return subprocess.run(["git", "-C", str(SCRIPT_DIR), "diff", "--quiet", commit, "--", path],
                          capture_output=True).returncode == 0


def run_name(model: str, level: str) -> str:
    """Nama run grid; D1 berasal dari run percobaan Milestone 1."""
    return f"{model}_{level}_trial" if level == "d1" else f"{model}_{level}"


def config_path(model: str, level: str) -> Path:
    """Berkas konfigurasi run grid."""
    if level == "d1":
        return SCRIPT_DIR / "configs" / "trial" / f"{model}_d1_trial.yaml"
    return SCRIPT_DIR / "configs" / f"{model}_{level}.yaml"


def ctc_vocab(source: str) -> list[str]:
    """Nilai konstanta `CTC_VOCAB` pada kode sumber, dievaluasi dari AST."""
    def evaluate(node: ast.AST) -> list[str]:
        if isinstance(node, ast.List):
            return [element.value for element in node.elts if isinstance(element, ast.Constant)]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "list"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Constant)):
            return list(node.args[0].value)
        raise ValueError(f"Bentuk CTC_VOCAB tidak dikenal: {ast.dump(node)[:80]}")

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "CTC_VOCAB" for target in node.targets):
            return evaluate(node.value)
    raise ValueError("CTC_VOCAB tidak ditemukan")


def losses(state: dict[str, Any]) -> list[float]:
    """Deret loss training pada riwayat log Trainer."""
    return [entry["loss"] for entry in state["log_history"] if "loss" in entry]


def load_sources() -> dict[str, Any]:
    """Seluruh sumber angka paper."""
    used: dict[str, str] = {}
    src: dict[str, Any] = {"used": used}
    src["split"] = load_json(OUTPUTS / "build_text_disjoint_split_report.json", used)
    src["nested"] = load_json(OUTPUTS / "build_nested_subsets_report.json", used)
    src["audit"] = load_json(OUTPUTS / "audit_transcript_leakage_report.json", used)
    src["components"] = load_json(OUTPUTS / "measure_component_durations_report.json", used)
    src["bootstrap"] = load_json(RESULTS / "bootstrap_metrics_report.json", used)
    src["scaling"] = load_json(RESULTS / "scaling_curve_report.json", used)
    src["run_status"] = {(row["model"], row["level"]): row for row in load_csv(RESULTS / "run_status.csv", used)}
    src["metrics"] = {(row["model"], row["level"]): row
                      for row in load_csv(RESULTS / "bootstrap_metrics_configs.csv", used)}
    src["h1"] = {row["level"]: row for row in load_csv(RESULTS / "bootstrap_metrics_h1.csv", used)}
    src["parameters"] = {(row["model_spec"], row["quantity"]): row
                         for row in load_csv(RESULTS / "scaling_curve_parameters.csv", used)}
    src["robustness"] = load_json(RESULTS / "robustness_diacritics_report.json", used)
    src["errors"] = load_json(RESULTS / "error_analysis_report.json", used)
    src["test_metrics"] = {(m, lv): load_json(RESULTS / f"{run_name(m, lv)}_test_metrics.json", used)["metrics"]
                           for m in MODELS for lv in LEVELS}
    src["configs"] = {(m, lv): load_yaml(config_path(m, lv), used) for m in MODELS for lv in LEVELS}
    src["runs"] = {(m, lv): load_json(OUTPUTS / "runs" / run_name(m, lv) / "run_report.json", used)
                   for m in MODELS for lv in LEVELS}
    src["det_configs"] = {name: load_yaml(SCRIPT_DIR / "configs" / "trial" / f"{name}.yaml", used)
                          for pair in DET_PAIRS for name in pair}
    src["det_states"] = {name: load_json(OUTPUTS / "runs" / name / "trainer_state.json", used)
                         for pair in DET_PAIRS for name in pair}
    src["train_asr_source"] = git_show(PIPELINE_COMMIT, "train_asr.py")
    used[f"train_asr.py@{PIPELINE_COMMIT}"] = hashlib.sha256(
        src["train_asr_source"].encode("utf-8")).hexdigest()
    fields = {field.name: field.default for field in dataclasses.fields(TrainingArguments)}
    src["training_defaults"] = {name: fields[name] for name in OPTIMIZER_DEFAULTS}
    src["run_inference_source"] = git_show(PIPELINE_COMMIT, "run_inference.py")
    used[f"run_inference.py@{PIPELINE_COMMIT}"] = hashlib.sha256(
        src["run_inference_source"].encode("utf-8")).hexdigest()
    revision = src["configs"][("whisper", "d1")]["revision"]
    generation_path = WHISPER_CACHE / "snapshots" / revision / "generation_config.json"
    used[f"generation_config.json whisper-small@{revision[:8]}"] = sha256_of(generation_path)
    # Mode generasi ditanyakan ke pustaka itu sendiri: pada transformers 5.17.0 nilai bawaan num_beams
    # adalah None, dan get_generation_mode memilih greedy bila num_beams None atau 1 tanpa do_sample.
    src["whisper_generation_mode"] = str(GenerationConfig.from_pretrained(generation_path.parent)
                                         .get_generation_mode().value)
    used[str(LICENSE_PATH.relative_to(SCRIPT_DIR.parent))] = sha256_of(LICENSE_PATH)
    src["license"] = LICENSE_PATH.read_text(encoding="utf-8")
    manuscripts = SCRIPT_DIR.parent / "_02_context-MD-files"
    src["literature"] = ""
    for name in LITERATURE_SOURCES:
        path = manuscripts / name
        used[str(path.relative_to(SCRIPT_DIR.parent))] = sha256_of(path)
        src["literature"] += path.read_text(encoding="utf-8") + "\n"
    src["registry"] = read_registry(manuscripts / REGISTRY_SOURCE)
    return src


def read_registry(path: Path) -> dict[int, str]:
    """Entri registri rujukan `[n]<tab>teks`; nomor ganda menggagalkan pembacaan."""
    registry: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = REGISTRY_ENTRY.match(line)
        if match:
            number = int(match.group(1))
            if number in registry:
                raise ValueError(f"Nomor registri [{number}] ganda pada {path.name}")
            registry[number] = match.group(2)
    return registry


# --------------------------------------------------------------------------
# Klaim dan tabel yang diharapkan
# --------------------------------------------------------------------------


@dataclasses.dataclass
class Claim:
    """Teks yang harus muncul pada satu bagian paper, beserta sumbernya."""

    section: str
    text: str
    source: str
    holds: bool = True


def same_or(first: str, second: str) -> str:
    """Sel kolom XLS-R: "Sama" bila nilainya sama dengan kolom Whisper."""
    return "Sama" if first == second else second


def build_claims(src: dict[str, Any]) -> list[Claim]:
    """Seluruh klaim prosa bagian yang diperiksa."""
    split, nested, components = src["split"], src["nested"], src["components"]["parts"]
    work = split["working_subset_duration_statistics"]
    filtered = split["filtered_subset_duration_statistics"]
    exclusion, parts, checks = split["exclusion"], split["parts"], split["checks"]
    levels = nested["levels"]
    pool_hours = nested["pool"]["hours"]
    populations = {pop["population"]: pop for pop in src["audit"]["populations"]}
    full = populations["corpus_penuh"]["base_statistics"]
    subset_split = populations["subset_kerja"]["split_strategies"]
    configs, runs = src["configs"], src["runs"]
    w1, x1 = configs[("whisper", "d1")], configs[("xlsr", "d1")]
    boot, scaling = src["bootstrap"], src["scaling"]
    vocab = ctc_vocab(src["train_asr_source"])
    letters = [token for token in vocab if len(token) == 1 and token.isalpha()]
    specials = [token for token in vocab if token.startswith("<")]

    (format_key, format_count), = split["audio_format"]["format_counts"].items()
    hz, channels, bits = map(int, re.fullmatch(r"FLAC (\d+) Hz (\d+) kanal PCM_(\d+)", format_key).groups())
    audio = f"FLAC {hz / 1000:g} kHz {'mono' if channels == 1 else channels} {bits}-bit"

    effective = {cfg["effective_batch_size"] for cfg in configs.values()}
    eff = effective.pop() if len(effective) == 1 else None
    min_steps, max_steps = w1["early_stopping_min_steps"], w1["max_steps"]
    gpus = {json.dumps(run["environment"]["lapis_1_perangkat_keras"]["gpu"], sort_keys=True)
            for run in runs.values()}
    gpu = json.loads(next(iter(gpus)))
    libraries = {json.dumps(run["environment"]["lapis_2_perangkat_lunak_terikat_perangkat_keras"]["pustaka"],
                            sort_keys=True) for run in runs.values()}
    lib = json.loads(next(iter(libraries)))
    xlsr_first_wer = [run["result"]["validation_history"][0]["wer"]
                      for (model, _), run in runs.items() if model == "xlsr"]
    det_identical = all(
        {k: v for k, v in src["det_configs"][a].items() if k != "run_name"}
        == {k: v for k, v in src["det_configs"][b].items() if k != "run_name"} for a, b in DET_PAIRS)
    det_differs = all(losses(src["det_states"][a]) != losses(src["det_states"][b]) for a, b in DET_PAIRS)
    ctc_warning_ok = all(
        any(WARNING_CTC in w and WARNING_CTC_TEXT in w for w in run["determinism_warnings"])
        for (model, _), run in runs.items() if model == "xlsr")
    attention_warning_ok = all(
        any(WARNING_ATTENTION_TEXT in w and WARNING_ATTENTION_FIX in w for w in run["determinism_warnings"])
        and run["numerics_state_at_end"]["use_deterministic_algorithms"] is True
        and run["numerics_state_at_end"]["deterministic_warn_only"] is True
        for run in runs.values())
    identical_levels = all(
        {k: v for k, v in configs[(m, lv)].items() if k not in LEVEL_KEYS_DIFFERING}
        == {k: v for k, v in configs[(m, "d1")].items() if k not in LEVEL_KEYS_DIFFERING}
        for m in MODELS for lv in LEVELS)
    confidence = boot["confidence"]
    tail = 100 * (1 - confidence) / 2
    hours = [scaling["hours"][lv] for lv in LEVELS]
    constraints = scaling["fitting"]["constraints"]
    aic = scaling["aic"]
    targets = [nested["level_target_hours"][lv] for lv in LEVELS[:-1]] + [round(pool_hours)]
    leaks = [checks[key] for key in ("test_transcripts_in_train_pct", "test_transcripts_in_validation_pct",
                                     "validation_transcripts_in_train_pct")]

    split_src = "build_text_disjoint_split_report.json"
    nested_src = "build_nested_subsets_report.json"
    audit_src = "audit_transcript_leakage_report.json"
    run_src = "run_report.json kesepuluh run grid"
    cfg_src = "konfigurasi YAML kesepuluh run grid"
    return [
        # 2.1 Rancangan Penelitian
        Claim("2.1", f"{', '.join(fmt_int(t) for t in targets[:-1])}, dan {fmt_int(targets[-1])} jam",
              f"{nested_src}: level_target_hours dan pool.hours dibulatkan"),
        Claim("2.1", f"menghasilkan {word(len(MODELS) * len(LEVELS))} konfigurasi", "MODELS x LEVELS",
              holds=len(runs) == len(MODELS) * len(LEVELS)),
        Claim("2.1", f"sebelum {word(sum(lv != 'd1' for _, lv in runs))} dari {word(len(runs))} konfigurasi "
              "dilatih", "run grid di luar D1 percobaan"),

        # 2.2 Data dan Pembagian Data
        Claim("2.2", "berlisensi Creative Commons Attribution-ShareAlike 4.0 [9]", "LICENSE corpus",
              holds=src["license"].startswith("Attribution-ShareAlike 4.0 International")),
        Claim("2.2", f"memuat {fmt_int(full['n_utterances'])} utterance dari {fmt_int(full['n_unique_speakers'])} "
              f"penutur yang terbagi atas {word(N_CORPUS_ARCHIVES)} arsip",
              f"{audit_src}: corpus_penuh.base_statistics; jumlah arsip {SPEC_SOURCE}"),
        Claim("2.2", f"memakai {word(len(split['working_subset_archives']))} arsip, yaitu "
              f"{fmt_int(split['n_flac_files'])} utterance atau {fmt_dec(work['total_hours'], 2)} jam audio",
              f"{split_src}: working_subset_archives, n_flac_files, total_hours",
              holds=work["n_utterances"] == split["n_flac_files"] and split["n_flac_without_metadata"] == 0),
        Claim("2.2", f"sebesar {fmt_int(100 * len(split['working_subset_archives']) / N_CORPUS_ARCHIVES)}% "
              "dari corpus penuh", f"{split_src} dan jumlah arsip {SPEC_SOURCE}"),
        Claim("2.2", f"berformat {audio}", f"{split_src}: audio_format",
              holds=format_count == split["n_flac_files"]),
        Claim("2.2", f"berdurasi {fmt_dec(work['max_seconds'], 2)} detik", f"{split_src}: max_seconds",
              holds=work["max_seconds"] < WHISPER_WINDOW_SECONDS),
        Claim("2.2", f"selebar {WHISPER_WINDOW_SECONDS} detik [3]", SPEC_SOURCE),
        Claim("2.2", f"disaring dari {fmt_int(exclusion['n_excluded'])} utterance", f"{split_src}: exclusion"),
        Claim("2.2", f"Sebanyak {fmt_int(exclusion['reason_counts']['memuat_angka'])} utterance dikeluarkan "
              "karena transkripnya memuat angka", f"{split_src}: exclusion.reason_counts"),
        Claim("2.2", f"dan {fmt_int(exclusion['reason_counts']['durasi_di_bawah_1_5_detik'])} utterance "
              f"dikeluarkan karena durasinya di bawah {fmt_dec(split['min_duration_seconds'], 1)} detik",
              f"{split_src}: exclusion.reason_counts, min_duration_seconds",
              holds=sum(exclusion["reason_counts"].values()) == exclusion["n_excluded"]),
        Claim("2.2", f"Sisanya, {fmt_int(filtered['n_utterances'])} utterance", f"{split_src}: filtered",
              holds=filtered["n_utterances"] == work["n_utterances"] - exclusion["n_excluded"]),
        Claim("2.2", f"porsi data test {fmt_dec(100 * src['audit']['test_fraction'], 1)}%, porsi data "
              f"validation {fmt_dec(100 * src['audit']['val_fraction'], 1)}%, dan "
              f"{word(len(src['audit']['seeds']))} *seed*", f"{audit_src}: test_fraction, val_fraction, seeds"),
        Claim("2.2", f"rata-rata {fmt_dec(subset_split['random_utterance']['transcript_leak_mean'], 2)}% "
              "utterance data test", f"{audit_src}: subset_kerja random_utterance"),
        Claim("2.2", f"menurunkannya menjadi {fmt_dec(subset_split['speaker_disjoint']['transcript_leak_mean'], 2)}%",
              f"{audit_src}: subset_kerja speaker_disjoint"),
        Claim("2.2", f"menjadi {fmt_int(split['n_transcript_groups'])} grup", f"{split_src}: n_transcript_groups",
              holds=split["n_transcript_groups"] == filtered["n_unique_transcripts"]),
        Claim("2.2", f"diacak dengan *seed* {split['split_seed']}", f"{split_src}: split_seed"),
        Claim("2.2", f"sampai kuota {fmt_int(split['test_quota'])} utterance terpenuhi dan ke data validation "
              f"sampai kuota {fmt_int(split['validation_quota'])} utterance terpenuhi",
              f"{split_src}: test_quota, validation_quota",
              holds=checks["test_meets_quota"] and checks["validation_meets_quota"]),
        Claim("2.2", f"sekitar {fmt_int(round(parts['test']['hours']))} jam bagi data test dan "
              f"{fmt_int(round(parts['validation']['hours']))} jam bagi data validation",
              f"{split_src}: parts.*.hours dibulatkan"),
        Claim("2.2", f"tetap sekitar {fmt_int(round(parts['train_pool']['hours']))} jam",
              f"{split_src}: parts.train_pool.hours dibulatkan"),
        Claim("2.2", f"kebocoran transkrip {fmt_dec(max(leaks), 2)}% di antara ketiga bagian",
              f"{split_src}: checks", holds=all(value == 0 for value in leaks)),
        Claim("2.2", f"satu komponen berisi {fmt_dec(components['largest']['pct_utterances'], 2)}% utterance, "
              f"satu komponen berisi {fmt_dec(components['second']['pct_utterances'], 2)}% utterance, dan "
              f"{fmt_int(components['remaining']['n_components'])} komponen kecil yang seluruhnya hanya berisi "
              f"{fmt_int(components['remaining']['n_utterances'])} utterance atau "
              f"{fmt_dec(components['remaining']['minutes'], 1)} menit audio",
              "measure_component_durations_report.json: parts",
              holds=all(src["components"]["checks"].values())),
        Claim("2.2", f"membuat tingkat {fmt_int(levels['d4']['target_hours'])} atau {fmt_int(round(pool_hours))} "
              "jam mustahil disusun", "measure_component_durations_report.json: hours_left_if_test",
              holds=components["largest"]["hours_left_if_test"] < levels["d4"]["target_hours"]
              and components["second"]["hours_left_if_test"] < pool_hours),
        Claim("2.2", f"{fmt_dec(checks['test_speakers_in_train_pct'], 2)}% utterance data test berasal dari "
              "penutur yang juga muncul pada *training pool*", f"{split_src}: checks.test_speakers_in_train_pct"),
        Claim("2.2", f"diacak satu kali dengan *seed* {nested['subset_seed']}", f"{nested_src}: subset_seed",
              holds=all(nested["checks"][key] for key in
                        ("levels_nested", "targets_met_at_first_crossing", "full_level_equals_pool"))),
        Claim("2.2", f"secara bersarang dengan *seed* {nested['subset_seed']}", f"{nested_src}: subset_seed"),
        Claim("2.2", f"dari {fmt_dec(levels['d1']['test_speakers_in_level_pct'], 2)}% pada D1 menjadi "
              f"{fmt_dec(levels['d5']['test_speakers_in_level_pct'], 2)}% pada D5",
              f"{nested_src}: levels.*.test_speakers_in_level_pct"),

        # 2.3 Normalisasi Teks dan Audio
        Claim("2.3", f"inventaris {fmt_int(len(letters))} huruf Latin ditambah spasi",
              f"CTC_VOCAB pada train_asr.py@{PIPELINE_COMMIT}"),
        Claim("2.3", f"sudah berformat {hz / 1000:g} kHz mono", f"{split_src}: audio_format",
              holds=channels == 1 and format_count == split["n_flac_files"]),

        # 2.4 Model dan Konfigurasi Pelatihan
        Claim("2.4", f"`{w1['pretrained']}` dengan sekitar {PARAMS_MILLIONS['whisper']} juta parameter [3]",
              f"pretrained pada {cfg_src}; jumlah parameter {SPEC_SOURCE}"),
        Claim("2.4", f"`{x1['pretrained']}` dengan sekitar {PARAMS_MILLIONS['xlsr']} juta parameter [4]",
              f"pretrained pada {cfg_src}; jumlah parameter {SPEC_SOURCE}"),
        Claim("2.4", f"berukuran sekitar {PARAMS_MILLIONS['whisper_medium']} juta parameter", SPEC_SOURCE),
        Claim("2.4", f"XLS-R tetap sekitar {fmt_int(100 * (PARAMS_MILLIONS['xlsr'] / PARAMS_MILLIONS['whisper'] - 1))}% "
              "lebih besar", f"turunan jumlah parameter {SPEC_SOURCE}"),
        Claim("2.4", "Berkas konfigurasi satu model hanya berbeda pada nama *run* dan subset training", cfg_src,
              holds=identical_levels),
        Claim("2.4", "dimuat hanya dari cache lokal", f"train_asr.py@{PIPELINE_COMMIT}: local_files_only",
              holds=re.search(r"""local_files_only["']?\s*[:=]\s*True""", src["train_asr_source"]) is not None),
        Claim("2.4", f"Kosakata CTC terdiri atas {fmt_int(len(letters))} huruf, pembatas kata, dan "
              f"{word(len(specials))} token khusus", f"CTC_VOCAB pada train_asr.py@{PIPELINE_COMMIT}",
              holds=vocab.count("|") == 1 and len(letters) + 1 + len(specials) == len(vocab)),
        Claim("2.4", f"*Batch* efektif ditetapkan {eff} utterance per langkah *optimizer*", cfg_src,
              holds=eff is not None),
        Claim("2.4", f"merata-ratakan tepat {eff} utterance", f"{run_src}: n_train_utterances_per_epoch",
              holds=eff is not None and all(run["result"]["n_train_utterances_per_epoch"] % eff == 0
                                            for run in runs.values())),
        Claim("2.4", f"Batas bawah {fmt_int(min_steps)} langkah", f"{cfg_src}: early_stopping_min_steps"),
        Claim("2.4", f"bertahan pada {fmt_int(100 * max(xlsr_first_wer))}%",
              f"{run_src}: validation_history XLS-R langkah pertama", holds=max(xlsr_first_wer) == 1.0),
        Claim("2.4", f"Batas atas {fmt_int(max_steps)} langkah", f"{cfg_src}: max_steps"),
        Claim("2.4", f"paling banyak {fmt_int(eff * max_steps)} utterance terproses",
              f"{cfg_src}: effective_batch_size x max_steps", holds=eff is not None),
        Claim("2.4", f"sampai nol tepat pada langkah {fmt_int(max_steps)}", f"{cfg_src}: lr_scheduler_type",
              holds=all(cfg["lr_scheduler_type"] == "linear" and cfg["max_steps"] == max_steps
                        for cfg in configs.values())),
        Claim("2.4", f"{word(gpu['jumlah_gpu_terdeteksi_torch'])} GPU laptop "
              f"{gpu['nama'].removesuffix(' Laptop GPU')} dengan VRAM {fmt_int(gpu['vram_total_mib'])} MiB",
              f"{run_src}: environment.gpu", holds=len(gpus) == 1 and gpu["nama"].endswith(" Laptop GPU")),
        Claim("2.4", f"PyTorch {lib['torch']}, transformers {lib['transformers']}, dan jiwer {lib['jiwer']}",
              f"{run_src}: environment.pustaka", holds=len(libraries) == 1),
        Claim("2.4", f"Pelatihan memakai {w1['precision']} *autocast*, TF32 dinonaktifkan", f"{cfg_src} dan {run_src}",
              holds=all(cfg["precision"] == w1["precision"] and cfg["tf32"] is False for cfg in configs.values())
              and all(not run["numerics_state_at_end"]["cuda_matmul_allow_tf32"]
                      and not run["numerics_state_at_end"]["cudnn_allow_tf32"] for run in runs.values())),
        Claim("2.4", "Lintasan mundur CTC pada XLS-R tidak memiliki implementasi deterministik",
              f"{run_src}: determinism_warnings", holds=ctc_warning_ok),
        Claim("2.4", "lintasan mundur *memory-efficient attention* pada kedua model memakai algoritme "
              "nondeterministik bawaannya, karena determinisme ditetapkan dengan `warn_only=True`",
              f"{run_src}: determinism_warnings dan numerics_state_at_end", holds=attention_warning_ok),
        Claim("2.4", f"{word(len(DET_PAIRS[0])).capitalize()} *run* uji berkonfigurasi identik pada setiap model "
              "menghasilkan deret loss yang tidak identik", "trainer_state.json run uji determinisme",
              holds=det_identical and det_differs),

        # 2.5 Evaluasi dan Analisis Statistik
        Claim("2.5", f"Selang kepercayaan {fmt_int(100 * confidence)}% bagi setiap WER dan CER",
              "bootstrap_metrics_report.json: confidence"),
        Claim("2.5", f"Sebanyak {fmt_int(parts['test']['n_utterances'])} utterance data test ditarik ulang dengan "
              f"pengembalian dalam {fmt_int(boot['n_bootstrap'])} iterasi",
              "bootstrap_metrics_report.json: n_bootstrap, configs.n_utterances",
              holds=all(cfg["n_utterances"] == parts["test"]["n_utterances"] for cfg in boot["configs"])),
        Claim("2.5", f"persentil {fmt_dec(tail, 1)} dan {fmt_dec(100 - tail, 1)}",
              "bootstrap_metrics_report.json dan scaling_curve_report.json: confidence, interval_method",
              holds=boot["interval_method"] == scaling["interval_method"] == "persentil"
              and scaling["confidence"] == confidence),
        Claim("2.5", f"ditarik dengan *seed* {boot['seed']}", "bootstrap_metrics_report.json: seed",
              holds=boot["seed"] == scaling["seed"] and boot["n_bootstrap"] == scaling["n_bootstrap"]),
        Claim("2.5", f"sebaran {fmt_int(boot['n_bootstrap'])} selisih", "bootstrap_metrics_report.json: n_bootstrap"),
        Claim("2.5", f"berada di bawah {fmt_dec(boot['h1']['alpha'], 2)}", "bootstrap_metrics_report.json: h1",
              holds="Holm" in boot["h1"]["correction"] and boot["h1"]["n_tests"] == len(LEVELS)),
        Claim("2.5", f"dilaporkan bersama selang {fmt_int(100 * confidence)}%",
              "bootstrap_metrics_report.json: confidence"),
        Claim("2.5", f"hanya menghasilkan {word(len(LEVELS))} titik pengamatan", "LEVELS"),
        Claim("2.5", f"hanya menyisakan {word(len(LEVELS) - 3)} derajat kebebasan residual", "LEVELS - 3 parameter"),
        Claim("2.5", f"menyisakan {word(2 * len(LEVELS) - 5)} derajat kebebasan residual",
              "2 x LEVELS - 5 parameter", holds=len(scaling["full_sample"]) == 3),
        Claim("2.5", f"yaitu {'; '.join(fmt_dec(h, 3) for h in hours[:-1])}; dan {fmt_dec(hours[-1], 3)} jam",
              "scaling_curve_report.json: hours",
              holds=all(scaling["hours"][lv] == levels[lv]["hours"] for lv in LEVELS)),
        Claim("2.5", "`scipy.optimize.least_squares` di bawah kendala `E₀ ≥ 0` dan `a > 0`",
              "scaling_curve_report.json: fitting",
              holds=scaling["fitting"]["method"].startswith("scipy.optimize.least_squares")
              and "E0 >= 0" in constraints and "a >= 1e-09" in constraints),
        Claim("2.5", "AIC = n ln(RSS/n) + 2K", "scaling_curve_report.json: aic.formula",
              holds=aic["formula"].startswith("n ln(RSS/n) + 2K")),
        Claim("2.5", "K jumlah parameter ditambah satu", "scaling_curve_report.json: aic.formula",
              holds="K = jumlah parameter + 1" in aic["formula"]),
        Claim("2.5", f"lebih rendah lebih dari {fmt_int(aic['threshold'])}", "scaling_curve_report.json: aic.threshold"),
        Claim("2.5", "kasus khusus `E₀ = 0`", "scaling_curve_report.json: full_sample.pure_power_law",
              holds="pure_power_law" in scaling["full_sample"]),
        Claim("2.5", f"selang {fmt_int(100 * confidence)}% `c` model itu", "scaling_curve_report.json: confidence"),
        Claim("2.5", f"selang {fmt_int(100 * confidence)}% selisih `c_W − c_X`",
              "scaling_curve_report.json: confidence"),
    ]


def build_tables(src: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Tabel yang diharapkan, sel demi sel, beserta syarat sumbernya."""
    split, nested = src["split"], src["nested"]
    parts, levels = split["parts"], nested["levels"]
    table1_rows = [
        ["Data test", fmt_int(parts["test"]["n_utterances"]), fmt_dec(parts["test"]["hours"], 2), "Tidak berlaku"],
        ["Data validation", fmt_int(parts["validation"]["n_utterances"]),
         fmt_dec(parts["validation"]["hours"], 2), "Tidak berlaku"],
    ]
    for level in LEVELS:
        label = "D5 (*training pool*)" if level == "d5" else level.upper()
        table1_rows.append([label, fmt_int(levels[level]["n_utterances"]), fmt_dec(levels[level]["hours"], 2),
                            fmt_dec(levels[level]["test_speakers_in_level_pct"], 2)])

    configs, runs = src["configs"], src["runs"]
    w, x = configs[("whisper", "d1")], configs[("xlsr", "d1")]
    vocab = ctc_vocab(src["train_asr_source"])
    defaults = src["training_defaults"]
    accumulation = {m: {runs[(m, lv)]["result"]["gradient_accumulation_steps"] for lv in LEVELS} for m in MODELS}
    overrides = x["model_overrides"]

    def schedule(cfg: dict[str, Any]) -> str:
        return (f"*Warmup* {fmt_int(cfg['warmup_steps'])} langkah, lalu turun linear sampai nol pada langkah "
                f"{fmt_int(cfg['max_steps'])}")

    def optimizer(cfg: dict[str, Any]) -> str:
        return (f"AdamW, *weight decay* {fmt_int(defaults['weight_decay'])}, *gradient clipping* norma "
                f"{fmt_dec(defaults['max_grad_norm'], 1)}")

    def batch(cfg: dict[str, Any], model: str) -> str:
        return f"{cfg['per_device_train_batch_size']} × {next(iter(accumulation[model]))}"

    def evaluation(cfg: dict[str, Any]) -> str:
        return (f"WER validation setiap {cfg['eval_steps']} langkah; berhenti sesudah "
                f"{cfg['early_stopping_patience']} evaluasi tanpa perbaikan; paling sedikit "
                f"{fmt_int(cfg['early_stopping_min_steps'])} dan paling banyak {fmt_int(cfg['max_steps'])} langkah")

    def precision(cfg: dict[str, Any]) -> str:
        return f"{cfg['precision']}, TF32 nonaktif, *seed* {cfg['seed']}"

    table2_rows = [
        ["Revisi model prapelatihan", f"`{w['revision'][:8]}`", f"`{x['revision'][:8]}`"],
        ["Keluaran dan fungsi loss", "*Decoder* autoregresif bawaan, *cross-entropy*",
         f"Kepala CTC baru dengan kosakata tetap {len(vocab)} token, loss CTC dirata-ratakan"],
        ["*Learning rate* puncak", fmt_sci(w["learning_rate"]), fmt_sci(x["learning_rate"])],
        ["Jadwal *learning rate*", schedule(w), same_or(schedule(w), schedule(x))],
        ["*Optimizer*", optimizer(w), same_or(optimizer(w), optimizer(x))],
        ["*Batch* per langkah × akumulasi gradien", batch(w, "whisper"), batch(x, "xlsr")],
        ["Evaluasi dan penghentian", evaluation(w), same_or(evaluation(w), evaluation(x))],
        ["Regularisasi di dalam model", "Nilai bawaan model",
         f"*Dropout* dan *layerdrop* {fmt_int(overrides['layerdrop'])}; `mask_time_prob` "
         f"{fmt_dec(overrides['mask_time_prob'], 2)}"],
        ["Dekode saat evaluasi", f"*Greedy*, paling banyak {w['generation_max_new_tokens']} token",
         "*Greedy CTC decoding* tanpa model bahasa eksternal"],
        ["Presisi dan *seed*", precision(w), same_or(precision(w), precision(x))],
    ]
    source = src["train_asr_source"]
    table2_conditions = {
        "keluarga model sesuai kolom": w["model_family"] == "whisper" and x["model_family"] == "xlsr",
        "loss CTC dirata-ratakan": overrides.get("ctc_loss_reduction") == "mean",
        "optimizer adamw_torch": all(cfg["optim"] == "adamw_torch" for cfg in configs.values()),
        "weight decay dan clipping tidak ditimpa train_asr.py":
            not any(name in source for name in OPTIMIZER_DEFAULTS),
        "jadwal linear": all(cfg["lr_scheduler_type"] == "linear" for cfg in configs.values()),
        "batch efektif = batch per langkah x akumulasi": all(
            len(accumulation[m]) == 1 and configs[(m, "d1")]["per_device_train_batch_size"]
            * next(iter(accumulation[m])) == configs[(m, "d1")]["effective_batch_size"] for m in MODELS),
        "Whisper tanpa override model": w["model_overrides"] == {},
        "dropout dan layerdrop XLS-R nol": all(overrides[key] == 0 for key in XLSR_ZERO_OVERRIDES),
        "TF32 nonaktif": all(cfg["tf32"] is False for cfg in configs.values()),
        "Whisper greedy pada inference data test": re.search(r"num_beams\s*=\s*1\b", src["run_inference_source"])
            is not None,
        "Whisper greedy pada evaluasi validation": "num_beams" not in source and "do_sample" not in source
            and src["whisper_generation_mode"] == "greedy_search",
        "XLS-R memakai argmax CTC": "argmax" in src["run_inference_source"],
    }
    return {
        "1": {"header": ["Bagian", "Jumlah utterance", "Durasi (jam)", "Cakupan penutur data test (%)"],
              "rows": table1_rows,
              "source": "build_text_disjoint_split_report.json dan build_nested_subsets_report.json",
              "conditions": {"subset bersarang dan D5 sama dengan training pool": all(
                  nested["checks"][key] for key in ("levels_nested", "full_level_equals_pool"))}},
        "2": {"header": ["Butir", "Whisper *small*", "XLS-R 300M"],
              "rows": table2_rows,
              "source": f"konfigurasi YAML, run_report.json, train_asr.py@{PIPELINE_COMMIT}, "
                        "nilai bawaan TrainingArguments",
              "conditions": table2_conditions},
    }


def build_result_claims(src: dict[str, Any]) -> list[Claim]:
    """Klaim prosa pengantar Bagian 3 dan Bagian 3.1 sampai 3.5."""
    status, metrics, h1, params = src["run_status"], src["metrics"], src["h1"], src["parameters"]
    scaling, boot, split, nested = src["scaling"], src["bootstrap"], src["split"], src["nested"]
    max_steps = src["configs"][("whisper", "d1")]["max_steps"]
    n_boot, confidence = boot["n_bootstrap"], boot["confidence"]
    n_test = split["parts"]["test"]["n_utterances"]
    n_validation = split["parts"]["validation"]["n_utterances"]
    pool_hours = nested["pool"]["hours"]

    def value(row: dict[str, str], key: str) -> float:
        return float(row[key])

    capped = {key for key, row in status.items() if row["stop_reason"] == "max_steps"}
    early = [row for key, row in status.items() if key not in capped]
    early_fracs = [value(row, "final_lr_fraction_of_peak") for row in early]
    hours = [value(row, "train_runtime_hours") for row in status.values()]
    reserved = [int(row["train_peak_reserved_mib"]) for row in status.values()]
    vram = src["runs"][("whisper", "d1")]["environment"]["lapis_1_perangkat_keras"]["gpu"]["vram_total_mib"]
    best_capped = sorted({int(status[key]["best_step"]) for key in capped})
    val_wer = {key: value(row, "best_validation_wer") for key, row in status.items()}

    wer = {key: value(row, "wer") for key, row in metrics.items()}
    cer = {key: value(row, "cer") for key, row in metrics.items()}
    wer_ci = {key: (value(row, "wer_ci_low"), value(row, "wer_ci_high")) for key, row in metrics.items()}
    diff = {lv: value(row, "wer_diff") for lv, row in h1.items()}
    cer_diff = {lv: value(row, "cer_diff") for lv, row in h1.items()}
    cer_ci = {lv: (value(row, "cer_diff_ci_low"), value(row, "cer_diff_ci_high")) for lv, row in h1.items()}
    holm_ci = {lv: (value(row, "holm_ci_low"), value(row, "holm_ci_high")) for lv, row in h1.items()}
    p_holm = {value(row, "p_holm") for row in h1.values()}
    rises = [(m, LEVELS[i]) for m in MODELS for i in range(len(LEVELS) - 1)
             if wer[(m, LEVELS[i + 1])] > wer[(m, LEVELS[i])]]
    n_words = {cfg["n_reference_words"] for cfg in boot["configs"]}

    shared = scaling["full_sample"]["shared_e0"]
    residual = {(p["model"], p["level"]): 100 * p["residual"] for p in shared["points"]}
    bootstrap_shared = scaling["bootstrap"]["shared_e0"]
    aic = scaling["aic"]["values"]

    def param(spec: str, quantity: str) -> tuple[float, float, float]:
        row = params[(spec, quantity)]
        return value(row, "estimate"), value(row, "ci_low"), value(row, "ci_high")

    e0, e0_low, _ = param("shared_e0", "E0")
    a_w, _, a_w_high = param("shared_e0", "a_whisper")
    a_x, a_x_low, _ = param("shared_e0", "a_xlsr")
    c_w, c_w_low, _ = param("shared_e0", "c_whisper")
    c_x, c_x_low, _ = param("shared_e0", "c_xlsr")
    dc, dc_low, dc_high = param("shared_e0", "c_whisper_minus_c_xlsr")
    dcs, dcs_low, dcs_high = param("separate_e0", "c_whisper_minus_c_xlsr")
    pure_c_w = float(params[("pure_power_law", "c_whisper")]["estimate"])
    pure_c_x = float(params[("pure_power_law", "c_xlsr")]["estimate"])

    robust = {(c["model"], c["level"]): c for c in src["robustness"]["configs"]}
    increase = {key: c["wer_kept_minus_main"] for key, c in robust.items()}
    cer_increase = [c["cer_kept_minus_main"] for c in robust.values()]
    smallest = min(increase, key=increase.get)
    largest = pct(max(increase.values()), 1)
    vocab = ctc_vocab(src["train_asr_source"])

    errors = src["errors"]["configs"]
    profile = {(c["model"], c["level"]): c for c in errors.values()}
    top_n = src["errors"]["top_n"]
    shares = {key: c["share_of_errors"] for key, c in profile.items()}
    no_error = {key: c["utterances"]["share_without_error"] for key, c in profile.items()}
    long_share = profile[("whisper", "d1")]["utterances"]["share_hypothesis_longer_than_ratio"]
    long_ratio = profile[("whisper", "d1")]["utterances"]["long_hypothesis_ratio"]

    def deletions(key: tuple[str, str]) -> list[str]:
        return [token for token, _ in profile[key]["top_deletions"]]

    def substitutions(key: tuple[str, str]) -> list[str]:
        return [pair for pair, _ in profile[key]["top_substitutions"]]

    def has_pair(key: tuple[str, str], first: str, second: str) -> bool:
        return f"{first} -> {second}" in substitutions(key) or f"{second} -> {first}" in substitutions(key)

    d5 = [("whisper", "d5"), ("xlsr", "d5")]
    merged = any("cuba cuba" in w["reference"] and "cubacuba" in w["hypothesis"]
                 for key, c in profile.items() if key[0] == "xlsr" for w in c["worst_utterances"])
    capped_levels = sorted(lv for _, lv in capped)

    run_src, metric_src, h1_src = "run_status.csv", "bootstrap_metrics_configs.csv", "bootstrap_metrics_h1.csv"
    curve_src, robust_src, error_src = "scaling_curve_report.json", "robustness_diacritics_report.json", \
        "error_analysis_report.json"
    return [
        # 3 Pengantar
        Claim("3", f"yaitu {fmt_int(n_test)} utterance hasil pembagian *text-disjoint*",
              "build_text_disjoint_split_report.json: parts.test"),

        # 3.1 Pelaksanaan Pelatihan
        Claim("3.1", "tanpa satu pun hipotesis kosong", "<run>_test_metrics.json: n_empty_hypothesis",
              holds=all(m["n_empty_hypothesis"] == 0 and m["n_utterances"] == n_test
                        for m in src["test_metrics"].values())),
        Claim("3.1", f"{word(len(capped)).capitalize()} *run* XLS-R, yaitu D1, D3, dan D4, tidak pernah memenuhi "
              f"syarat *early stopping* dan berhenti pada batas {fmt_int(max_steps)} langkah, ketika *learning rate* "
              "sudah turun sampai nol", run_src,
              holds={m for m, _ in capped} == {"xlsr"} and capped_levels == ["d1", "d3", "d4"]
              and all(pct(value(status[key], "final_lr_fraction_of_peak"), 0) == "0"
                      and int(status[key]["last_step"]) == max_steps for key in capped)),
        Claim("3.1", f"Ke{word(len(early))} *run* lainnya berhenti oleh *early stopping* ketika *learning rate* masih "
              f"{pct(min(early_fracs), 0)} sampai {pct(max(early_fracs), 0)}% dari puncaknya", run_src,
              holds=all(row["stop_reason"] == "early_stopping" for row in early)),
        Claim("3.1", f"pada langkah {fmt_int(best_capped[0])} atau {fmt_int(best_capped[-1])}", run_src,
              holds=len(best_capped) == 2),
        Claim("3.1", f"performa pada anggaran {fmt_int(max_steps)} langkah", "konfigurasi YAML: max_steps"),
        Claim("3.1", f"WER validation terbaik XLS-R pada D5 ({pct(val_wer[('xlsr', 'd5')], 1)}%) juga lebih tinggi "
              f"daripada pada D4 ({pct(val_wer[('xlsr', 'd4')], 1)}%)", run_src,
              holds=val_wer[("xlsr", "d5")] > val_wer[("xlsr", "d4")]),
        Claim("3.1", "padahal data D5 lebih dari dua kali lipat data D4", "build_nested_subsets_report.json",
              holds=nested["levels"]["d5"]["hours"] > 2 * nested["levels"]["d4"]["hours"]),
        Claim("3.1", f"*run* D5 berhenti ketika *learning rate* masih "
              f"{pct(value(status[('xlsr', 'd5')], 'final_lr_fraction_of_peak'), 0)}% dari puncaknya", run_src,
              holds=("xlsr", "d5") not in capped),
        Claim("3.1", f"berlangsung {fmt_dec(min(hours), 1)} sampai {fmt_dec(max(hours), 1)} jam termasuk evaluasi "
              "validation", f"{run_src}: train_runtime_hours"),
        Claim("3.1", f"paling tinggi {fmt_int(max(reserved))} MiB, di bawah kapasitas terpasang",
              f"{run_src}: train_peak_reserved_mib", holds=max(reserved) < vram),
        Claim("3.1", f"WER validation diukur pada {fmt_int(n_validation)} utterance validation",
              "run_report.json: n_validation_utterances",
              holds=all(run["result"]["n_validation_utterances"] == n_validation for run in src["runs"].values())),

        # 3.2 Performa Kedua Model pada Setiap Tingkat Data
        Claim("3.2", "Whisper *small* mencapai WER lebih rendah daripada XLS-R 300M pada kelima tingkat data",
              metric_src, holds=all(wer[("whisper", lv)] < wer[("xlsr", lv)] for lv in LEVELS)),
        Claim("3.2", f"Pada D1, WER Whisper {pct(wer[('whisper', 'd1')], 1)}% dan XLS-R {pct(wer[('xlsr', 'd1')], 1)}%, "
              f"sedangkan pada D5 keduanya turun menjadi {pct(wer[('whisper', 'd5')], 1)}% dan "
              f"{pct(wer[('xlsr', 'd5')], 1)}%", metric_src),
        Claim("3.2", f"Nilai *p* sesudah koreksi Holm sebesar {fmt_dec(next(iter(p_holm)), 3)} pada kelima tingkat, "
              f"yaitu batas terkecil yang dapat dihasilkan {fmt_int(n_boot)} iterasi", h1_src,
              holds=len(p_holm) == 1
              and abs(next(iter(p_holm)) - min(1.0, len(LEVELS) * 2 / (n_boot + 1))) < 1e-12),
        Claim("3.2", "selang selisih pada taraf Holm tidak memuat nol", h1_src,
              holds=all(high < 0 or low > 0 for low, high in holm_ci.values())),
        Claim("3.2", f"selang kepercayaan {fmt_int(100 * confidence)}% *bootstrap*, kecuali pada selisih WER",
              "bootstrap_metrics_report.json: confidence"),
        Claim("3.2", f"yaitu {', '.join(level_pct(float(h1[lv]['holm_ci_level'])) + '%' for lv in LEVELS[:-1])}, "
              f"dan {level_pct(float(h1[LEVELS[-1]]['holm_ci_level']))}% berturut-turut dari D1 sampai D5", h1_src),
        Claim("3.2", f"Data test berisi {fmt_int(n_test)} utterance dengan {fmt_int(next(iter(n_words)))} kata "
              "referensi", "bootstrap_metrics_report.json: configs", holds=len(n_words) == 1),
        Claim("3.2", f"Antara D1 dan D3, WER Whisper turun {pct(wer[('whisper', 'd1')] - wer[('whisper', 'd3')], 1)} "
              f"poin persentase dan WER XLS-R {pct(wer[('xlsr', 'd1')] - wer[('xlsr', 'd3')], 1)} poin, sedangkan "
              f"antara D3 dan D5 penurunannya tinggal {pct(wer[('whisper', 'd3')] - wer[('whisper', 'd5')], 1)} dan "
              f"{pct(wer[('xlsr', 'd3')] - wer[('xlsr', 'd5')], 1)} poin", metric_src),
        Claim("3.2", f"yaitu dari {pct(wer[('xlsr', 'd4')], 1)}% pada D4 menjadi {pct(wer[('xlsr', 'd5')], 1)}% pada D5, "
              "searah dengan WER validation pada Tabel 3", f"{metric_src} dan {run_src}",
              holds=rises == [("xlsr", "d4")] and val_wer[("xlsr", "d5")] > val_wer[("xlsr", "d4")]),
        Claim("3.2", f"Selisih itu terbesar pada D1 dan D2, yaitu {pct(-diff['d1'], 1)} dan {pct(-diff['d2'], 1)} poin "
              f"persentase, lalu menyempit menjadi {pct(-diff['d3'], 1)} dan {pct(-diff['d4'], 1)} poin pada D3 dan "
              f"D4, dan melebar lagi menjadi {pct(-diff['d5'], 1)} poin pada D5", h1_src,
              holds=all(d < 0 for d in diff.values())
              and min(-diff["d1"], -diff["d2"]) > -diff["d5"] > max(-diff["d3"], -diff["d4"])),
        Claim("3.2", "CER memperlihatkan urutan yang sama dengan WER pada D1, D2, dan D5, tetapi tidak pada D3 dan D4",
              h1_src, holds=all(cer_diff[lv] < 0 for lv in ("d1", "d2", "d5"))
              and all(cer_diff[lv] > 0 for lv in ("d3", "d4"))),
        Claim("3.2", f"yaitu {pct(cer[('whisper', 'd3')], 1)}% pada D3 serta {pct(cer[('whisper', 'd4')], 1)}% dan "
              f"{pct(cer[('xlsr', 'd4')], 1)}% pada D4, dan selang selisih CER-nya memuat nol", f"{metric_src} dan {h1_src}",
              holds=pct(cer[("whisper", "d3")], 1) == pct(cer[("xlsr", "d3")], 1)
              and all(low < 0 < high for lv, (low, high) in cer_ci.items() if lv in ("d3", "d4"))),

        # 3.3 Hubungan Jumlah Data dan WER
        Claim("3.3", f"`E₀` bersama sebesar {pct(e0, 1)}%, eksponen {fmt_dec(c_w, 3)} bagi Whisper, dan eksponen "
              f"{fmt_dec(c_x, 3)} bagi XLS-R", "scaling_curve_parameters.csv: shared_e0"),
        Claim("3.3", f"Selang {fmt_int(100 * confidence)}% kedua eksponen seluruhnya berada di atas nol",
              "scaling_curve_parameters.csv: shared_e0",
              holds=c_w_low > 0 and c_x_low > 0 and scaling["decisions"]["h2_supported_for_both_models"]),
        Claim("3.3", f"seluruh {fmt_int(bootstrap_shared['n_used'])} pencocokan ulang *bootstrap* berhasil",
              f"{curve_src}: bootstrap.shared_e0",
              holds=bootstrap_shared["n_failed"] == 0 and bootstrap_shared["n_used"] == n_boot),
        Claim("3.3", f"dalam rentang {fmt_int(nested['level_target_hours']['d1'])} sampai {fmt_int(round(pool_hours))} jam",
              "build_nested_subsets_report.json"),
        Claim("3.3", f"selang kepercayaan {fmt_int(100 * confidence)}% persentil dari {fmt_int(n_boot)} iterasi "
              "*bootstrap*", f"{curve_src}: confidence, n_bootstrap",
              holds=scaling["interval_method"] == "persentil" and scaling["n_bootstrap"] == n_boot),
        Claim("3.3", f"Karena `D` bernilai {fmt_int(scaling['hours']['d1'])} jam pada D1", f"{curve_src}: hours",
              holds=scaling["hours"]["d1"] == 1.0),
        Claim("3.3", f"yaitu {pct(a_w, 1)} poin persentase bagi Whisper dan {pct(a_x, 1)} poin bagi XLS-R, dan selang "
              "kedua koefisien tidak tumpang tindih", "scaling_curve_parameters.csv: shared_e0",
              holds=a_w_high < a_x_low),
        Claim("3.3", f"Titik Whisper menyimpang paling jauh "
              f"{fmt_dec(max(abs(residual[('whisper', lv)]) for lv in LEVELS), 1)} poin persentase dari kurvanya",
              f"{curve_src}: full_sample.shared_e0.points"),
        Claim("3.3", f"Ketiga *run* yang mencapai batas {fmt_int(max_steps)} langkah berada di bawah kurva, yaitu D1, D3, "
              f"dan D4 dengan residual {signed(residual[('xlsr', 'd1')], 1)}, {signed(residual[('xlsr', 'd3')], 1)}, "
              f"dan {signed(residual[('xlsr', 'd4')], 1)} poin", f"{curve_src} dan {run_src}",
              holds=capped_levels == ["d1", "d3", "d4"]
              and all(residual[("xlsr", lv)] < 0 for lv in capped_levels)),
        Claim("3.3", f"berada di atasnya dengan residual {signed(residual[('xlsr', 'd2')], 1)} dan "
              f"{signed(residual[('xlsr', 'd5')], 1)} poin", f"{curve_src} dan {run_src}",
              holds=all(residual[("xlsr", lv)] > 0 and ("xlsr", lv) not in capped for lv in ("d2", "d5"))),
        Claim("3.3", "Residual XLS-R pada D2 sampai D5 melampaui dua kali setengah lebar selang WER titiknya",
              f"{curve_src} dan {metric_src}",
              holds=all(abs(residual[("xlsr", lv)]) > 100 * (wer_ci[("xlsr", lv)][1] - wer_ci[("xlsr", lv)][0])
                        for lv in ("d2", "d3", "d4", "d5"))),
        Claim("3.3", f"AIC-nya {fmt_dec(aic['separate_e0'] - aic['shared_e0'], 3)} lebih tinggi", f"{curve_src}: aic",
              holds=0 < aic["separate_e0"] - aic["shared_e0"] <= scaling["aic"]["threshold"]
              and scaling["aic"]["primary_model"] == "shared_e0"),
        Claim("3.3", f"memberi AIC {fmt_dec(aic['shared_e0'] - aic['pure_power_law'], 3)} lebih rendah daripada `E₀` "
              "bersama", f"{curve_src}: aic", holds=aic["pure_power_law"] < aic["shared_e0"]),
        Claim("3.3", "Menurut AIC, bentuk dengan `E₀` bersama tidak terbedakan dari kedua bentuk pembandingnya",
              f"{curve_src}: aic",
              holds=all(abs(aic[spec] - aic["shared_e0"]) <= scaling["aic"]["threshold"]
                        for spec in ("separate_e0", "pure_power_law"))),
        Claim("3.3", "selang `E₀` bersama tidak memuat nol, dan tidak satu pun iterasi *bootstrap* berakhir pada batas "
              "`E₀ = 0`", f"{curve_src}: bootstrap.shared_e0",
              holds=e0_low > 0 and not scaling["decisions"]["e0_ci_reaches_zero"]
              and all(share == 0 for share in bootstrap_shared["share_E0_at_zero_bound"].values())),

        # 3.4 Perbandingan Data Efficiency
        Claim("3.4", f"selisih `c_W − c_X` sebesar {fmt_dec(dc, 3)} dengan selang {fmt_int(100 * confidence)}% "
              f"{interval(dc_low, dc_high, 3, 1)}, yang memuat nol, sehingga H3 tidak diterima",
              "scaling_curve_parameters.csv: shared_e0",
              holds=dc_low < 0 < dc_high and not scaling["decisions"]["h3_c_difference_ci_excludes_zero"]),
        Claim("3.4", f"dengan selisih {fmt_dec(dcs, 3)} dan selang yang jauh lebih lebar, yaitu "
              f"{interval(dcs_low, dcs_high, 3, 1)}", "scaling_curve_parameters.csv: separate_e0",
              holds=dcs_low < 0 < dcs_high and dcs_high - dcs_low > dc_high - dc_low),
        Claim("3.4", "Pada *power law* murni, eksponen XLS-R justru sedikit lebih besar daripada eksponen Whisper",
              "scaling_curve_parameters.csv: pure_power_law", holds=pure_c_x > pure_c_w),

        # 3.5 Uji Ketegaran dan Analisis Kesalahan
        Claim("3.5", "Mempertahankan diakritik menaikkan WER kedua model pada setiap tingkat, dan kenaikannya "
              "bertambah seiring bertambahnya data", robust_src,
              holds=all(v > 0 for v in increase.values()) and all(
                  round(100 * increase[(m, LEVELS[i])], 1) <= round(100 * increase[(m, LEVELS[i + 1])], 1)
                  for m in MODELS for i in range(len(LEVELS) - 1))),
        Claim("3.5", f"dari {pct(increase[smallest], 1)} poin persentase pada Whisper D1 sampai {largest} poin pada D4 "
              "dan D5 kedua model", robust_src,
              holds=smallest == ("whisper", "d1")
              and all(pct(increase[(m, lv)], 1) == largest for m in MODELS for lv in ("d4", "d5"))),
        Claim("3.5", f"mengubah {fmt_int(src['robustness']['n_references_changed'])} dari {fmt_int(n_test)} referensi "
              "data test", robust_src),
        Claim("3.5", f"yaitu {pct(min(cer_increase), 1)} sampai {pct(max(cer_increase), 1)} poin persentase",
              robust_src),
        Claim("3.5", "kosakata CTC XLS-R tidak memuat huruf berdiakritik dan hipotesis Whisper memuat huruf é paling "
              "banyak satu kali pada setiap konfigurasi", f"{robust_src} dan CTC_VOCAB",
              holds=all(token.isascii() for token in vocab)
              and all(not c["hypothesis_non_ascii_letters"] for key, c in robust.items() if key[0] == "xlsr")
              and all(set(c["hypothesis_non_ascii_letters"]) <= {"U+00E9 LATIN SMALL LETTER E WITH ACUTE"}
                      and all(n <= 1 for n in c["hypothesis_non_ascii_letters"].values())
                      for key, c in robust.items() if key[0] == "whisper")),
        Claim("3.5", "urutan nilai titik kedua model tetap sama dengan hasil utama pada setiap tingkat, termasuk "
              "kenaikan WER XLS-R dari D4 ke D5", robust_src,
              holds=all(robust[("whisper", lv)]["wer_kept"] < robust[("xlsr", lv)]["wer_kept"] for lv in LEVELS)
              and robust[("xlsr", "d5")]["wer_kept"] > robust[("xlsr", "d4")]["wer_kept"]),
        Claim("3.5", f"yaitu {pct(min(s['substitutions'] for s in shares.values()), 1)} sampai "
              f"{pct(max(s['substitutions'] for s in shares.values()), 1)}% dari seluruh galat", error_src,
              holds=all(s["substitutions"] == max(s.values()) for s in shares.values())),
        Claim("3.5", "Whisper lebih banyak menyisipkan daripada menghapus kata, sedangkan XLS-R sebaliknya", error_src,
              holds=all((s["insertions"] > s["deletions"]) == (key[0] == "whisper") for key, s in shares.items())),
        Claim("3.5", f"dari {pct(no_error[('whisper', 'd1')], 1)}% menjadi {pct(no_error[('whisper', 'd5')], 1)}% pada "
              f"Whisper dan dari {pct(no_error[('xlsr', 'd1')], 1)}% menjadi {pct(no_error[('xlsr', 'd5')], 1)}% pada "
              "XLS-R", error_src),
        Claim("3.5", f"yaitu hipotesis yang lebih dari {word(int(long_ratio))} kali panjang referensinya, pada "
              f"{pct(long_share, 2)}% utterance", error_src, holds=long_ratio == int(long_ratio)),
        Claim("3.5", "*siti* menjadi *city* dan *akting* menjadi *acting*", error_src,
              holds={"siti -> city", "akting -> acting"} <= set(substitutions(("whisper", "d1")))),
        Claim("3.5", "Keluaran berulang semacam itu tidak tampak lagi pada D5", error_src,
              holds=profile[("whisper", "d5")]["utterances"]["share_hypothesis_longer_than_ratio"] == 0),
        Claim("3.5", "*cuba cuba* menjadi *cubacuba*", f"{error_src}: worst_utterances XLS-R", holds=merged),
        Claim("3.5", f"*atus* dan *loro* termasuk {word(top_n)} kata yang paling sering terhapus pada kedua model, "
              "demikian pula *ewu* pada Whisper, tetapi ketiganya tidak lagi muncul di antara penghapusan tersering "
              "pada D5", error_src,
              holds=all({"atus", "loro"} <= set(deletions((m, "d1"))) for m in MODELS)
              and "ewu" in deletions(("whisper", "d1"))
              and not any(w in deletions(key) for key in d5 for w in ("atus", "loro", "ewu"))),
        Claim("3.5", f"sebagian besar dari {word(top_n)} pasangan substitusi tersering kedua model berupa varian ejaan",
              error_src, holds=all(len(substitutions(key)) == top_n for key in d5)),
        Claim("3.5", "*limo* dan *lima*, *biyasane* dan *biasane*, serta *daerah* dan *dhaerah*", error_src,
              holds=all(has_pair(key, a, b) for key in d5
                        for a, b in (("limo", "lima"), ("biyasane", "biasane"), ("daerah", "dhaerah")))),
        Claim("3.5", "*nduweni* yang ditulis *duweni* dan sebaliknya", error_src,
              holds=any({"nduweni -> duweni", "duweni -> nduweni"} <= set(substitutions(key)) for key in d5)),
    ]


def build_discussion_claims(src: dict[str, Any]) -> list[Claim]:
    """Klaim prosa Bagian 3.6, 3.7, dan 4."""
    status, metrics, h1, params, nested = src["run_status"], src["metrics"], src["h1"], src["parameters"], \
        src["nested"]
    literature, split = src["literature"], src["split"]
    n_test = split["parts"]["test"]["n_utterances"]
    targets = nested["level_target_hours"]
    pool = fmt_int(round(nested["pool"]["hours"]))
    wer = {key: float(row["wer"]) for key, row in metrics.items()}
    diff = {lv: float(row["wer_diff"]) for lv, row in h1.items()}
    cer_zero = all(float(h1[lv]["cer_diff_ci_low"]) < 0 < float(h1[lv]["cer_diff_ci_high"]) for lv in ("d3", "d4"))
    holm_excludes = {lv: float(row["holm_ci_high"]) < 0 or float(row["holm_ci_low"]) > 0 for lv, row in h1.items()}
    capped = {key for key, row in status.items() if row["stop_reason"] == "max_steps"}
    e0 = float(params[("shared_e0", "E0")]["estimate"])
    e0_low = float(params[("shared_e0", "E0")]["ci_low"])
    c_w = float(params[("shared_e0", "c_whisper")]["estimate"])
    c_x = float(params[("shared_e0", "c_xlsr")]["estimate"])
    c_w_low = float(params[("shared_e0", "c_whisper")]["ci_low"])
    c_x_low = float(params[("shared_e0", "c_xlsr")]["ci_low"])
    a_w = float(params[("shared_e0", "a_whisper")]["estimate"])
    a_x = float(params[("shared_e0", "a_xlsr")]["estimate"])
    dc = params[("shared_e0", "c_whisper_minus_c_xlsr")]
    residual_w = max(abs(100 * p["residual"]) for p in src["scaling"]["full_sample"]["shared_e0"]["points"]
                     if p["model"] == "whisper")
    leak = {pop["population"]: pop for pop in src["audit"]["populations"]}["subset_kerja"]["split_strategies"]
    halving = {pct(2 ** -c, 0) for c in (c_w, c_x)}
    smallest = min(diff, key=lambda lv: abs(diff[lv]))
    largest_two = sorted(diff, key=lambda lv: abs(diff[lv]))[-2:]
    decline_steep = all(wer[(m, "d1")] - wer[(m, "d3")] > wer[(m, "d3")] - wer[(m, "d5")] for m in MODELS)
    aic = src["scaling"]["aic"]
    floor_indistinct = abs(aic["values"]["pure_power_law"] - aic["values"]["shared_e0"]) <= aic["threshold"]
    d1, d2, d3, d4 = (fmt_int(targets[lv]) for lv in ("d1", "d2", "d3", "d4"))
    span = f"rentang {d1} sampai {pool} jam"

    def cited(*fragments: str) -> bool:
        return all(fragment in literature for fragment in fragments)

    lit_src = "angka penelitian terdahulu pada " + ", ".join(LITERATURE_SOURCES)
    metric_src, curve_src = "bootstrap_metrics_configs.csv", "scaling_curve_parameters.csv"
    return [
        # 3.6 Pembahasan
        Claim("3.6", f"masih {pct(float(status[('xlsr', 'd5')]['final_lr_fraction_of_peak']), 0)}% dari puncaknya, "
              "sedangkan D4 menjalani jadwal sampai selesai", "run_status.csv",
              holds=("xlsr", "d4") in capped and ("xlsr", "d5") not in capped),
        Claim("3.6", f"kemampuan kedua model pada {pool} jam data", "build_nested_subsets_report.json"),
        Claim("3.6", "Wav2Vec2 XLSR-53 sebesar 23,62% dan Whisper *small* sebesar 28,67% pada sekitar 6 jam data "
              "OpenSLR [10]", lit_src, holds=cited("XLSR-53", "23,62%", "28,67%", "sekitar 6 jam")),
        Claim("3.6", f"baik pada {d2} jam maupun {d3} jam", metric_src,
              holds=all(wer[("whisper", lv)] < wer[("xlsr", lv)] for lv in ("d2", "d3"))),
        Claim("3.6", span, "build_nested_subsets_report.json"),
        Claim("3.6", f"yaitu 14,97% dengan Whisper *small* pada sekitar 100 jam data OpenSLR [40], dibanding "
              f"{pct(wer[('whisper', 'd5')], 1)}% pada {pool} jam di sini", f"{lit_src} dan {metric_src}",
              holds=cited("14,97%", "sekitar 100 jam")),
        Claim("3.6", "WER 13,77% pada Whisper large-v2 [7]", lit_src, holds=cited("13,77%", "large-v2")),
        Claim("3.6", "model bahasa 5-gram menurunkan WER XLS-R [8]", lit_src, holds=cited("5-gram")),
        Claim("3.6", f"antara {d1} dan {d3} jam lalu melandai", metric_src, holds=decline_steep),
        Claim("3.6", f"simpangan paling jauh {fmt_dec(residual_w, 1)} poin persentase", "scaling_curve_report.json"),
        Claim("3.6", f"Nilai `E₀` sebesar {pct(e0, 1)}%", curve_src),
        Claim("3.6", "bahwa 38,6% kasus galat tinggi pada corpus mereka", lit_src, holds=cited("38,6%")),
        Claim("3.6", f"dengan faktor 2^(−c), yaitu menjadi kira-kira {next(iter(halving))}% dari nilai sebelumnya bagi "
              "kedua model", curve_src, holds=len(halving) == 1),
        Claim("3.6", f"koefisien `a` sebesar {pct(a_w, 1)}% lawan {pct(a_x, 1)}% pada XLS-R", curve_src),
        Claim("3.6", f"selisih WER-nya terhadap XLS-R terbesar pada {d1} dan {d2} jam", "bootstrap_metrics_h1.csv",
              holds=sorted(largest_two) == ["d1", "d2"]),
        Claim("3.6", f"Pada {d3} sampai {d4} jam, akurasi tidak lagi menjadi pembeda yang kuat",
              "bootstrap_metrics_h1.csv",
              holds=cer_zero and max(abs(diff["d3"]), abs(diff["d4"])) < min(abs(diff["d1"]), abs(diff["d2"]))),
        Claim("3.6", f"dengan porsi data test {fmt_dec(100 * src['audit']['test_fraction'], 1)}%, pembagian "
              "*speaker-disjoint* hanya menurunkan kebocoran transkrip dari "
              f"{fmt_dec(leak['random_utterance']['transcript_leak_mean'], 2)}% menjadi "
              f"{fmt_dec(leak['speaker_disjoint']['transcript_leak_mean'], 2)}%", "audit_transcript_leakage_report.json"),

        # 3.7 Keterbatasan
        Claim("3.7", f"data test sebesar {fmt_int(n_test)} utterance cukup bagi H1", "bootstrap_metrics_h1.csv",
              holds=all(holm_excludes.values())),
        Claim("3.7", f"selisih terkecil, yaitu {pct(abs(diff[smallest]), 1)} poin pada D4, tetap terbedakan pada taraf "
              "Holm", "bootstrap_metrics_h1.csv", holds=smallest == "d4" and holm_excludes["d4"]),
        Claim("3.7", f"{word(len(capped))} *run* menjalani penurunan *learning rate* sampai nol, sedangkan D2 dan D5 "
              "berhenti ketika *learning rate* masih tinggi", "run_status.csv",
              holds=len(capped) == 3 and not {("xlsr", "d2"), ("xlsr", "d5")} & capped),
        Claim("3.7", f"sekitar {PARAMS_MILLIONS['whisper']} juta lawan {PARAMS_MILLIONS['xlsr']} juta", SPEC_SOURCE),
        Claim("3.7", f"naik dari {fmt_dec(nested['levels']['d1']['test_speakers_in_level_pct'], 2)}% pada D1 menjadi "
              f"{fmt_dec(nested['levels']['d5']['test_speakers_in_level_pct'], 2)}% pada D5",
              "build_nested_subsets_report.json"),
        Claim("3.7", span, "build_nested_subsets_report.json"),
        Claim("3.7", f"{word(len(split['working_subset_archives']))} dari {word(N_CORPUS_ARCHIVES)} arsip SLR35",
              f"build_text_disjoint_split_report.json; jumlah arsip {SPEC_SOURCE}"),

        # 4 Simpulan
        Claim("4", f"Pada data test {fmt_int(n_test)} utterance, Whisper mencapai WER lebih rendah pada kelima tingkat, "
              f"yaitu {pct(wer[('whisper', 'd1')], 1)}% lawan {pct(wer[('xlsr', 'd1')], 1)}% pada {d1} jam dan "
              f"{pct(wer[('whisper', 'd5')], 1)}% lawan {pct(wer[('xlsr', 'd5')], 1)}% pada {pool} jam", metric_src,
              holds=all(wer[("whisper", lv)] < wer[("xlsr", lv)] for lv in LEVELS)),
        Claim("4", f"paling besar pada {d1} dan {d2} jam dan menyempit pada {d3} dan {d4} jam, ketika CER kedua model "
              "setara", "bootstrap_metrics_h1.csv", holds=sorted(largest_two) == ["d1", "d2"] and cer_zero),
        Claim("4", span, "build_nested_subsets_report.json"),
        Claim("4", f"eksponen {fmt_dec(c_w, 3)} bagi Whisper dan {fmt_dec(c_x, 3)} bagi XLS-R yang selangnya berada di "
              "atas nol", curve_src, holds=c_w_low > 0 and c_x_low > 0),
        Claim("4", f"Batas bawah bersama sebesar {pct(e0, 1)}% memiliki selang yang tidak memuat nol", curve_src,
              holds=e0_low > 0),
        Claim("4", "menurut AIC bentuk berbatas bawah tidak terbedakan dari *power law* murni",
              "scaling_curve_report.json: aic", holds=floor_indistinct),
        Claim("4", f"Selisih eksponen sebesar {fmt_dec(float(dc['estimate']), 3)} memiliki selang yang memuat nol",
              curve_src, holds=float(dc["ci_low"]) < 0 < float(dc["ci_high"])),
        Claim("4", f"sekurang-kurangnya pada {d3} dan {d4} jam", "build_nested_subsets_report.json"),
        Claim("4", f"Sapuan data di atas {pool} jam", "build_nested_subsets_report.json"),
    ]


def build_intro_claims(src: dict[str, Any]) -> list[Claim]:
    """Klaim prosa Bagian 1 Pendahuluan; hampir seluruhnya angka penelitian terdahulu."""
    literature, nested = src["literature"], src["nested"]
    span = f"rentang {fmt_int(nested['level_target_hours']['d1'])} sampai {fmt_int(round(nested['pool']['hours']))} jam"
    lit_src = "angka penelitian terdahulu pada " + ", ".join(LITERATURE_SOURCES)

    def cited(*fragments: str) -> bool:
        return all(fragment in literature for fragment in fragments)

    entries = (
        ("Long Form Sensus Penduduk 2020", ("Long Form Sensus Penduduk 2020",)),
        ("42,12% penduduk berumur lima tahun ke atas", ("42,12%", "lima tahun ke atas")),
        ("42,19% di lingkungan tetangga", ("42,19%",)),
        ("sekitar 90 juta orang [9]", ("sekitar 90 juta orang [9]",)),
        ("sekitar 680.000 jam data wicara multibahasa", ("680.000 jam data wicara multibahasa",)),
        ("setengah juta jam data dari 128 bahasa", ("setengah juta jam dari 128 bahasa",)),
        ("kerangka wav2vec 2.0", ("kerangka wav2vec 2.0",)),
        ("sebesar 89,40% pada data bahasa Jawa, yang turun menjadi 13,77% sesudah *fine-tuning* Whisper large-v2 [7]",
         ("89,40%", "13,77%", "large-v2 [7]")),
        ("WER 92,08% pada Whisper *medium* dan 74,62% pada Whisper large-v3", ("92,08%", "74,62%", "large-v3")),
        ("sekitar 60 jam [41]", ("sekitar 60 jam",)),
        ("WER 14,97% dengan Whisper *small* pada sekitar 100 jam data OpenSLR [40]",
         ("sekitar 100 jam data OpenSLR", "14,97% [40]")),
        ("model bahasa 5-gram KenLM", ("5-gram KenLM",)),
        ("sekitar 6 jam data OpenSLR, Wav2Vec2 XLSR-53", ("data sekitar 6 jam, Wav2Vec2 XLSR-53",)),
        ("WER di bawah 13% sekitar 50 jam dan di bawah 10% sekitar 200 jam [12]",
         ("di bawah 13% tercapai sekitar 50 jam", "di bawah 10% sekitar 200 jam")),
        ("Wav2Vec2 XLSR-53 unggul pada data Telugu", ("XLSR-53",)),
        ("sapuan data 134 sampai 23.000 jam", ("134 hingga 23.000 jam",)),
    )
    claims = [Claim("1", text, lit_src, holds=cited(*fragments)) for text, fragments in entries]
    claims.append(Claim("1", span, "build_nested_subsets_report.json"))
    return claims


def build_front_back_claims(src: dict[str, Any]) -> list[Claim]:
    """Klaim Abstrak, *Abstract*, serta Ketersediaan Data dan Kode.

    Abstrak hanya memuat klaim yang sudah ada pada Bagian 3.2 sampai 4, sehingga
    syaratnya sama dengan syarat klaim padanannya di badan paper. *Abstract*
    memakai format angka Inggris lewat `en`.
    """
    metrics, h1, params, nested = src["metrics"], src["h1"], src["parameters"], src["nested"]
    aic = src["scaling"]["aic"]
    n_test = src["split"]["parts"]["test"]["n_utterances"]
    confidence = src["bootstrap"]["confidence"]
    alpha = src["bootstrap"]["h1"]["alpha"]
    wer = {key: float(row["wer"]) for key, row in metrics.items()}
    w1, x1, w5, x5 = (pct(wer[key], 1) for key in
                      (("whisper", "d1"), ("xlsr", "d1"), ("whisper", "d5"), ("xlsr", "d5")))
    d1, pool = fmt_int(nested["level_target_hours"]["d1"]), fmt_int(round(nested["pool"]["hours"]))

    def estimate(quantity: str, key: str = "estimate") -> float:
        return float(params[("shared_e0", quantity)][key])

    c_w, c_x, e0 = fmt_dec(estimate("c_whisper"), 3), fmt_dec(estimate("c_xlsr"), 3), pct(estimate("E0"), 1)
    level = fmt_int(100 * confidence)
    whisper_lower = all(wer[("whisper", lv)] < wer[("xlsr", lv)] for lv in LEVELS)
    h1_all = all((float(row["holm_ci_high"]) < 0 or float(row["holm_ci_low"]) > 0)
                 and float(row["p_holm"]) < alpha for row in h1.values())
    dc_zero = estimate("c_whisper_minus_c_xlsr", "ci_low") < 0 < estimate("c_whisper_minus_c_xlsr", "ci_high")
    e0_positive = estimate("E0", "ci_low") > 0
    floor_indistinct = abs(aic["values"]["pure_power_law"] - aic["values"]["shared_e0"]) <= aic["threshold"]
    lower_curve = whisper_lower and estimate("a_whisper") < estimate("a_xlsr")
    no_lm = "argmax" in src["run_inference_source"]

    metric_src, h1_src, curve_src = "bootstrap_metrics_configs.csv", "bootstrap_metrics_h1.csv", \
        "scaling_curve_parameters.csv"
    nested_src, aic_src = "build_nested_subsets_report.json", "scaling_curve_report.json: aic"
    return [
        # Abstrak
        Claim("abstrak", f"dalam rentang {d1} sampai {pool} jam", nested_src),
        Claim("abstrak", f"Pada data test {fmt_int(n_test)} utterance, Whisper mencapai WER lebih rendah pada kelima "
              f"tingkat, yaitu {w1}% lawan {x1}% pada {d1} jam dan {w5}% lawan {x5}% pada {pool} jam", metric_src,
              holds=whisper_lower),
        Claim("abstrak", "selisih keduanya berbeda dari nol pada setiap tingkat", h1_src, holds=h1_all),
        Claim("abstrak", f"Eksponen `c` sebesar {c_w} bagi Whisper dan {c_x} bagi XLS-R tidak terbedakan, karena selang "
              f"{level}% selisihnya memuat nol", curve_src, holds=dc_zero),
        Claim("abstrak", f"Batas bawah bersama sebesar {e0}% memiliki selang yang tidak memuat nol", curve_src,
              holds=e0_positive),
        Claim("abstrak", "menurut kriteria informasi Akaike (AIC) bentuk berbatas bawah tidak terbedakan dari *power "
              "law* murni", aic_src, holds=floor_indistinct),
        Claim("abstrak", "keunggulan Whisper terletak pada kurva yang lebih rendah di seluruh rentang",
              f"{metric_src} dan {curve_src}", holds=lower_curve),
        Claim("abstrak", "dengan XLS-R tanpa model bahasa eksternal", f"run_inference.py@{PIPELINE_COMMIT}",
              holds=no_lm),

        # Abstract
        Claim("abstract", f"from {d1} to {pool} hours", nested_src),
        Claim("abstract", f"On a test set of {en(fmt_int(n_test))} utterances, Whisper achieves a lower WER at all five "
              f"levels, {en(w1)}% versus {en(x1)}% at {d1} hour and {en(w5)}% versus {en(x5)}% at {pool} hours",
              metric_src, holds=whisper_lower),
        Claim("abstract", "the difference is significant at every level", h1_src, holds=h1_all),
        Claim("abstract", f"The exponents `c` of {en(c_w)} for Whisper and {en(c_x)} for XLS-R are indistinguishable, "
              f"as the {level}% interval of their difference contains zero", curve_src, holds=dc_zero),
        Claim("abstract", f"The shared floor of {en(e0)}% has an interval that excludes zero", curve_src,
              holds=e0_positive),
        Claim("abstract", "by the Akaike information criterion (AIC) the floored form is indistinguishable from a pure "
              "power law", aic_src, holds=floor_indistinct),
        Claim("abstract", "Whisper's advantage lies in a lower curve across the whole range",
              f"{metric_src} dan {curve_src}", holds=lower_curve),
        Claim("abstract", "XLS-R without an external language model", f"run_inference.py@{PIPELINE_COMMIT}",
              holds=no_lm),

        # Ketersediaan Data dan Kode
        Claim("ketersediaan", "dengan lisensi Creative Commons Attribution-ShareAlike 4.0 [9]", "LICENSE corpus",
              holds=src["license"].startswith("Attribution-ShareAlike 4.0 International")),
    ]


def build_result_tables(src: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Tabel 3 sampai 5 yang diharapkan, sel demi sel."""
    status, metrics, h1, params, scaling = (src["run_status"], src["metrics"], src["h1"], src["parameters"],
                                            src["scaling"])
    table3_rows = []
    for model in MODELS:
        for level in LEVELS:
            row = status[(model, level)]
            table3_rows.append([MODEL_LABELS[model], level.upper(), fmt_int(int(row["last_step"])),
                                fmt_int(int(row["best_step"])), STOP_LABELS[row["stop_reason"]],
                                pct(float(row["final_lr_fraction_of_peak"]), 0),
                                pct(float(row["best_validation_wer"]), 1)])

    def metric(model: str, level: str, name: str) -> str:
        row = metrics[(model, level)]
        return (f"{pct(float(row[name]), 1)} "
                f"{interval(float(row[name + '_ci_low']), float(row[name + '_ci_high']), 1)}")

    table4_rows = []
    for level in LEVELS:
        row = h1[level]
        table4_rows.append([
            level.upper(), metric("whisper", level, "wer"), metric("xlsr", level, "wer"),
            f"{pct(float(row['wer_diff']), 1)} {interval(float(row['holm_ci_low']), float(row['holm_ci_high']), 1)}",
            metric("whisper", level, "cer"), metric("xlsr", level, "cer"),
            f"{pct(float(row['cer_diff']), 2)} "
            f"{interval(float(row['cer_diff_ci_low']), float(row['cer_diff_ci_high']), 2)}",
        ])

    def estimate(spec: str, quantity: str, places: int, scale: float, with_interval: bool = True) -> str:
        row = params[(spec, quantity)]
        text = fmt_dec(scale * float(row["estimate"]), places)
        return f"{text} {interval(float(row['ci_low']), float(row['ci_high']), places, scale)}" \
            if with_interval else text

    full, aic = scaling["full_sample"], scaling["aic"]["values"]
    table5_rows = [
        ["Jumlah parameter", *(str(full[spec]["n_parameters"]) for spec in
                               ("shared_e0", "separate_e0", "pure_power_law"))],
        ["`E₀` Whisper (%)", estimate("shared_e0", "E0", 1, 100), estimate("separate_e0", "E0_whisper", 1, 100),
         "Ditetapkan 0"],
        ["`E₀` XLS-R (%)", estimate("shared_e0", "E0", 1, 100), estimate("separate_e0", "E0_xlsr", 1, 100),
         "Ditetapkan 0"],
        ["`a` Whisper (%)", estimate("shared_e0", "a_whisper", 1, 100), "Tidak disajikan", "Tidak disajikan"],
        ["`a` XLS-R (%)", estimate("shared_e0", "a_xlsr", 1, 100), "Tidak disajikan", "Tidak disajikan"],
        ["`c` Whisper", estimate("shared_e0", "c_whisper", 3, 1), estimate("separate_e0", "c_whisper", 3, 1, False),
         estimate("pure_power_law", "c_whisper", 3, 1, False)],
        ["`c` XLS-R", estimate("shared_e0", "c_xlsr", 3, 1), estimate("separate_e0", "c_xlsr", 3, 1, False),
         estimate("pure_power_law", "c_xlsr", 3, 1, False)],
        ["`c_W − c_X`", estimate("shared_e0", "c_whisper_minus_c_xlsr", 3, 1),
         estimate("separate_e0", "c_whisper_minus_c_xlsr", 3, 1), "Tidak tersedia"],
        ["Jumlah kuadrat residual (poin²)", *(fmt_dec(10_000 * full[spec]["rss"], 2) for spec in
                                              ("shared_e0", "separate_e0", "pure_power_law"))],
        ["Selisih AIC terhadap `E₀` bersama", "0", signed(aic["separate_e0"] - aic["shared_e0"], 3),
         signed(aic["pure_power_law"] - aic["shared_e0"], 3)],
    ]
    return {
        "3": {"header": ["Model", "Tingkat", "Langkah terakhir", "Langkah terbaik", "Penghentian",
                         "*Learning rate* akhir (% puncak)", "WER validation terbaik (%)"],
              "rows": table3_rows, "source": "run_status.csv",
              "conditions": {"status penghentian dikenal": all(row["stop_reason"] in STOP_LABELS
                                                              for row in status.values())}},
        "4": {"header": ["Tingkat", "WER Whisper *small* (%)", "WER XLS-R 300M (%)", "Selisih WER (poin)",
                         "CER Whisper *small* (%)", "CER XLS-R 300M (%)", "Selisih CER (poin)"],
              "rows": table4_rows, "source": "bootstrap_metrics_configs.csv dan bootstrap_metrics_h1.csv",
              "conditions": {"selisih Whisper dikurangi XLS-R":
                             src["bootstrap"]["h1"]["difference"] == "whisper - xlsr"}},
        "5": {"header": ["Besaran", "`E₀` bersama", "`E₀` terpisah", "*Power law* murni"],
              "rows": table5_rows, "source": "scaling_curve_parameters.csv dan scaling_curve_report.json",
              "conditions": {"E0 bersama sama bagi kedua model":
                             full["shared_e0"]["curve"]["E0_whisper"] == full["shared_e0"]["curve"]["E0_xlsr"],
                             "power law murni tanpa E0": "E0" not in full["pure_power_law"]["parameters"]}},
    }


# --------------------------------------------------------------------------
# Pembacaan paper
# --------------------------------------------------------------------------


def read_paper(path: Path) -> tuple[list[str], dict[str, int]]:
    """Baris paper tanpa akhiran baris, beserta cacah jenis akhiran baris."""
    raw = path.read_bytes()
    crlf = raw.count(b"\r\n")
    endings = {"crlf": crlf, "lone_lf": raw.count(b"\n") - crlf, "lone_cr": raw.count(b"\r") - crlf}
    return raw.decode("utf-8").replace("\r\n", "\n").split("\n"), endings


def split_sections(lines: Sequence[str]) -> dict[str, list[str]]:
    """Isi setiap bagian bernomor atau bernama, dari judulnya sampai judul berikutnya.

    Komentar HTML dikosongkan lebih dulu agar catatan sinkronisasi tidak ikut
    terbaca sebagai isi bagian.
    """
    text = HTML_COMMENT.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), "\n".join(lines))
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.split("\n"):
        numbered, named = SECTION_HEADING.match(line), NAMED_HEADING.match(line)
        if numbered:
            current = numbered.group(1)
        elif named and named.group(1) in NAMED_SECTIONS:
            current = NAMED_SECTIONS[named.group(1)]
        elif line.startswith("#"):
            current = None
            continue
        elif current is not None:
            sections[current].append(line)
            continue
        else:
            continue
        if current in sections:
            raise ValueError(f"Bagian {current} muncul dua kali")
        sections[current] = []
    return sections


def parse_tables(lines: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Setiap tabel bernomor: judul, kepala, dan baris isi."""
    def cells(row: str) -> list[str]:
        return [cell.strip() for cell in row.strip().strip("|").split("|")]

    tables: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines):
        match = TABLE_CAPTION.match(line)
        if not match:
            continue
        body = index + 1
        while body < len(lines) and not lines[body].strip():
            body += 1
        rows = []
        while body < len(lines) and lines[body].startswith("|"):
            rows.append(cells(lines[body]))
            body += 1
        if match.group(1) in tables:
            raise ValueError(f"Tabel {match.group(1)} muncul dua kali")
        tables[match.group(1)] = {"title": match.group(2), "header": rows[0] if rows else [],
                                  "rows": rows[2:], "line": index + 1}
    return tables


def mask(text: str) -> str:
    """Teks dengan bagian yang bukan prosa diganti spasi, posisi karakter tetap."""
    masked_lines = [" " * len(line) if line.startswith(("|", "#")) else line for line in text.split("\n")]
    masked = "\n".join(masked_lines)
    for pattern in MASKED_PATTERNS:
        masked = pattern.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), masked)
    return masked


# --------------------------------------------------------------------------
# Pemeriksaan
# --------------------------------------------------------------------------


def check_claims(claims: Sequence[Claim], sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Kehadiran dan syarat setiap klaim pada bagiannya."""
    results = []
    for claim in claims:
        text = "\n".join(sections.get(claim.section, []))
        found = claim.text in text
        results.append({**dataclasses.asdict(claim), "found": found, "ok": found and claim.holds})
    return results


def uncovered_numbers(claims: Sequence[Claim], sections: dict[str, list[str]]) -> list[dict[str, str]]:
    """Bilangan prosa pada bagian yang diperiksa yang tidak tercakup klaim mana pun."""
    uncovered = []
    for key in CHECKED_SECTIONS:
        text = "\n".join(sections.get(key, []))
        spans = []
        for claim in claims:
            if claim.section != key:
                continue
            start = text.find(claim.text)
            while start >= 0:
                spans.append((start, start + len(claim.text)))
                start = text.find(claim.text, start + 1)
        for token in NUMBER_TOKEN.finditer(mask(text)):
            if not any(low <= token.start() and token.end() <= high for low, high in spans):
                context = text[max(0, token.start() - 40):token.end() + 20].replace("\n", " ")
                uncovered.append({"section": key, "token": token.group(0), "context": context})
    return uncovered


def check_tables(expected: dict[str, dict[str, Any]], found: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Kecocokan setiap tabel yang diharapkan dan keberadaan tabel yang tidak dikenal."""
    results = []
    for number, table in expected.items():
        actual = found.get(number)
        mismatches = []
        if actual is None:
            mismatches.append("tabel tidak ditemukan")
        else:
            if actual["header"] != table["header"]:
                mismatches.append(f"kepala: {actual['header']} lawan {table['header']}")
            if len(actual["rows"]) != len(table["rows"]):
                mismatches.append(f"jumlah baris {len(actual['rows'])} lawan {len(table['rows'])}")
            for row_index, (shown, wanted) in enumerate(zip(actual["rows"], table["rows"]), start=1):
                for column, (cell, value) in enumerate(zip(shown, wanted), start=1):
                    if cell != value:
                        mismatches.append(f"baris {row_index} kolom {column}: '{cell}' lawan '{value}'")
                if len(shown) != len(wanted):
                    mismatches.append(f"baris {row_index}: {len(shown)} kolom lawan {len(wanted)}")
        failed = [name for name, passed in table["conditions"].items() if not passed]
        results.append({"table": number, "source": table["source"], "mismatches": mismatches,
                        "failed_conditions": failed, "ok": not mismatches and not failed})
    for number in sorted(set(found) - set(expected)):
        results.append({"table": number, "source": None, "mismatches": ["tabel tidak dikenal pemeriksa"],
                        "failed_conditions": [], "ok": False})
    return results


def check_references(lines: Sequence[str], registry: dict[int, str]) -> dict[str, Any]:
    """Daftar Pustaka terhadap registri dan sitasi pada badan paper.

    Badan paper adalah seluruh teks sebelum judul Daftar Pustaka, tanpa komentar
    HTML. Pemetaan registri ke paper hanya berisi entri yang teksnya sama persis
    dengan tepat satu entri registri.
    """
    text = HTML_COMMENT.sub(lambda match: re.sub(r"[^\n]", " ", match.group(0)), "\n".join(lines))
    body, found, references = text.partition("\n## Daftar Pustaka\n")
    problems: list[str] = []
    entries: dict[int, str] = {}
    if not found:
        problems.append("judul Daftar Pustaka tidak ditemukan")
    for line in references.split("\n"):
        if line.startswith("#"):
            break
        if not line.strip():
            continue
        match = REFERENCE_ENTRY.match(line)
        if not match:
            problems.append(f"baris Daftar Pustaka tidak dikenal: {line[:60]}")
            continue
        number = int(match.group(1))
        if number in entries:
            problems.append(f"entri [{number}] ganda")
        entries[number] = match.group(2)
    if sorted(entries) != list(range(1, len(entries) + 1)):
        problems.append(f"nomor entri bukan 1 sampai {len(entries)} tanpa celah: {sorted(entries)}")

    by_text: dict[str, list[int]] = {}
    for number, entry in registry.items():
        by_text.setdefault(entry, []).append(number)
    mapping: dict[int, int] = {}
    for number, entry in sorted(entries.items()):
        matches = by_text.get(entry, [])
        if len(matches) != 1:
            problems.append(f"entri [{number}] tidak sama persis dengan tepat satu entri registri "
                            f"({len(matches)} kecocokan)")
        elif matches[0] in mapping:
            problems.append(f"entri [{number}] dan [{mapping[matches[0]]}] sama-sama registri [{matches[0]}]")
        else:
            mapping[matches[0]] = number

    citations = [int(match.group(1)) for match in CITATION.finditer(body)]
    order: list[int] = []
    for number in citations:
        if number not in order:
            order.append(number)
    if order != list(range(1, len(order) + 1)):
        problems.append(f"urutan kemunculan pertama bukan 1, 2, 3, dan seterusnya: {order}")
    for number in sorted(set(order) - set(entries)):
        problems.append(f"[{number}] disitasi tanpa entri Daftar Pustaka")
    for number in sorted(set(entries) - set(order)):
        problems.append(f"entri [{number}] tidak pernah disitasi")
    return {"n_entries": len(entries), "n_citations": len(citations), "first_appearance_order": order,
            "registry_to_paper": {str(key): value for key, value in sorted(mapping.items())},
            "problems": problems, "ok": not problems}


def localize(claims: Sequence[Claim], mapping: dict[int, int]) -> list[Claim]:
    """Klaim dengan nomor sitasi registri diganti nomor paper secara serentak.

    Nomor registri tanpa entri pada paper diganti penanda yang tidak mungkin ada
    pada paper, sehingga klaimnya gagal ditemukan.
    """
    def replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        return f"[{mapping[number]}]" if number in mapping else f"[registri {number} tanpa entri]"

    return [dataclasses.replace(claim, text=CITATION.sub(replace, claim.text)) for claim in claims]


def check_abstracts(sections: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Batas kata dan jumlah kata kunci Abstrak serta *Abstract*."""
    results = []
    for key, keyword_label in ABSTRACT_SECTIONS.items():
        lines = sections.get(key, [])
        keyword_lines = [line for line in lines if line.startswith(keyword_label)]
        prose = " ".join(line for line in lines if not line.startswith(keyword_label))
        n_words = len(prose.split())
        n_keywords = (len([item for item in keyword_lines[0].removeprefix(keyword_label).split(";") if item.strip()])
                      if len(keyword_lines) == 1 else 0)
        results.append({"section": key, "words": n_words, "max_words": ABSTRACT_MAX_WORDS,
                        "keywords": n_keywords, "expected_keywords": N_KEYWORDS,
                        "ok": 0 < n_words <= ABSTRACT_MAX_WORDS and n_keywords == N_KEYWORDS})
    return results


def ascii_safe(text: str) -> str:
    """Teks yang aman dicetak ke PowerShell berhalaman kode cp1252."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Pemeriksa angka paper terhadap keluaran skrip.")
    parser.add_argument("--paper", type=Path, default=PAPER,
                        help="Berkas paper; bawaan ../_06_dokumen-skripsi/_04_paper/paper.md.")
    parser.add_argument("--report", type=Path, default=REPORT_PATH,
                        help="Berkas laporan; bawaan outputs/verify_paper_numbers_report.json.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines, endings = read_paper(args.paper)
    sections = split_sections(lines)
    src = load_sources()
    references = check_references(lines, src["registry"])
    mapping = {int(key): value for key, value in references["registry_to_paper"].items()}
    claims = localize(build_front_back_claims(src) + build_intro_claims(src) + build_claims(src)
                      + build_result_claims(src) + build_discussion_claims(src), mapping)

    print("=" * 78, flush=True)
    print("Pemeriksaan angka paper terhadap keluaran skrip", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print("Random seed        : tidak dipakai, pemeriksaan deterministik", flush=True)
    print(f"Paper              : {args.paper.name}", flush=True)
    print(f"Akhiran baris      : {endings['crlf']} CRLF, {endings['lone_lf']} LF tunggal, "
          f"{endings['lone_cr']} CR tunggal", flush=True)
    print(f"Bagian diperiksa   : {', '.join(CHECKED_SECTIONS)}", flush=True)
    print(f"Sumber dibaca      : {len(src['used'])} berkas", flush=True)
    print("=" * 78, flush=True)

    missing_sections = [key for key in CHECKED_SECTIONS if not "".join(sections.get(key, [])).strip()]
    claim_results = check_claims(claims, sections)
    table_results = check_tables({**build_tables(src), **build_result_tables(src)}, parse_tables(lines))
    uncovered = uncovered_numbers(claims, sections)
    abstracts = check_abstracts(sections)
    pipeline = {name: unchanged_since(PIPELINE_COMMIT, name) for name in PIPELINE_FILES}

    for result in claim_results:
        if not result["ok"]:
            reason = "teks tidak ditemukan" if not result["found"] else "syarat sumber tidak terpenuhi"
            print(f"KLAIM GAGAL  [{result['section']}] {reason}: {ascii_safe(result['text'])}", flush=True)
    for result in table_results:
        status = "COCOK" if result["ok"] else "GAGAL"
        print(f"Tabel {result['table']:3s} {status}", flush=True)
        for item in result["mismatches"] + [f"syarat: {name}" for name in result["failed_conditions"]]:
            print(f"  {ascii_safe(item)}", flush=True)
    for item in uncovered:
        print(f"TANPA SUMBER [{item['section']}] {item['token']}: ...{ascii_safe(item['context'])}...", flush=True)
    for key in missing_sections:
        print(f"BAGIAN KOSONG [{key}]", flush=True)
    for problem in references["problems"]:
        print(f"RUJUKAN      {ascii_safe(problem)}", flush=True)
    for result in abstracts:
        print(f"Abstrak {result['section']:9s}: {result['words']} kata (batas {result['max_words']}), "
              f"{result['keywords']} kata kunci{'' if result['ok'] else '  GAGAL'}", flush=True)
    for name, same in pipeline.items():
        if not same:
            print(f"Catatan: {name} pada berkas kerja berbeda dari {PIPELINE_COMMIT}", flush=True)

    n_claims_ok = sum(result["ok"] for result in claim_results)
    endings_ok = endings["lone_lf"] == 0 and endings["lone_cr"] == 0 and endings["crlf"] > 0
    all_ok = (endings_ok and not missing_sections and n_claims_ok == len(claim_results)
              and all(result["ok"] for result in table_results) and not uncovered
              and references["ok"] and all(result["ok"] for result in abstracts))
    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": None,
        "paper": args.paper.name,
        "paper_sha256": sha256_of(args.paper),
        "line_endings": endings,
        "line_endings_ok": endings_ok,
        "checked_sections": list(CHECKED_SECTIONS),
        "missing_sections": missing_sections,
        "sources_sha256": src["used"],
        "pipeline_unchanged_since": {"commit": PIPELINE_COMMIT, "files": pipeline},
        "claims": claim_results,
        "tables": table_results,
        "uncovered_numbers": uncovered,
        "references": references,
        "abstracts": abstracts,
        "all_ok": all_ok,
        "elapsed_seconds": elapsed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("=" * 78, flush=True)
    print(f"Klaim cocok        : {n_claims_ok} dari {len(claim_results)}", flush=True)
    print(f"Tabel cocok        : {sum(r['ok'] for r in table_results)} dari {len(table_results)}", flush=True)
    print(f"Bilangan tanpa klaim: {len(uncovered)}", flush=True)
    print(f"Rujukan            : {references['n_entries']} entri, {references['n_citations']} sitasi, "
          f"{'cocok dengan registri' if references['ok'] else 'ADA MASALAH'}", flush=True)
    print(f"Hasil              : {'seluruh angka cocok' if all_ok else 'ADA YANG GAGAL'}", flush=True)
    print(f"Laporan ditulis ke : {args.report}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
