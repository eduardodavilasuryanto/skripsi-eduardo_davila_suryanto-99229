"""Audit batas 128 token keluaran Whisper dan uji sensitivitasnya (Revisi 01 poin 18).

Tujuan
------
Memeriksa apakah batas `generation_max_new_tokens` 128 pada dekode Whisper
(`bab-3.md` Tabel 3.14) memotong keluaran pada data test, dan seberapa besar
pengaruhnya terhadap WER dan CER bila batas dinaikkan. XLS-R dengan dekode
CTC tidak memiliki batas padanan; panjang keluarannya hanya dibatasi jumlah
kerangka akustik.

Prosedur
--------
1. Panjang keluaran setiap hipotesis Whisper data test dihitung sebagai jumlah
   token teks hasil tokenisasi ulang `hypothesis_raw` dengan tokenizer Whisper
   *small* revisi terkunci. Token prompt (`<|startoftranscript|>`, bahasa,
   tugas, `<|notimestamps|>`) tidak termasuk `max_new_tokens`, sedangkan
   `<|endoftext|>` termasuk, sehingga keluaran yang berakhir normal paling
   banyak 127 token teks; keluaran sepanjang batas dianggap terpotong.
   Tokenisasi ulang memberi tokenisasi kanonik, yang dapat berbeda dari urutan
   token yang dibangkitkan model; karena itu status terpotong dipastikan pada
   langkah 3.
2. Panjang token referensi mentah data test dan validation dihitung dengan
   tokenizer yang sama, untuk menunjukkan apakah batas itu dapat memotong
   transkrip yang benar.
3. Dengan `--regenerate`, setiap utterance yang mencapai batas didekode ulang
   dengan model terbaik konfigurasinya memakai fungsi `transcribe` milik
   `run_inference.py` tanpa perubahan, sekali pada batas 128 untuk memastikan
   hipotesis tersimpan terulang, dan sekali pada batas 444, yaitu batas
   tertinggi yang dapat diterima Whisper (448 posisi dikurangi 4 token
   prompt). Panjang keluaran diukur dari token yang benar-benar dibangkitkan.
4. WER dan CER setiap konfigurasi yang terdampak dihitung ulang dengan
   hipotesis batas 444 menggantikan hipotesis tersimpan, beserta selisih H1
   terhadap XLS-R pada tingkat yang sama dengan *paired bootstrap* yang sama
   dengan `bootstrap_metrics.py` (seed 42, 10.000 iterasi).

Hasil utama pada `outputs/results/` tidak disentuh. Berkas ini hanya
menambah audit dan sensitivitas.

Masukan
-------
- `outputs/hypotheses/<run>_test.tsv` kesepuluh konfigurasi.
- `outputs/splits/test.tsv` dan `validation.tsv` : transkrip mentah.
- `configs/trial/whisper_d1_trial.yaml`, `configs/whisper_d{2..5}.yaml`, dan
  `outputs/models/<run>/` : hanya dengan `--regenerate`.

Keluaran
--------
- `outputs/results/audit_whisper_token_limit_report.json`
- `outputs/results/audit_whisper_token_limit_utterances.csv` : utterance yang
  mencapai batas beserta panjangnya pada kedua batas.
- Ringkasan ke stdout, untuk disalurkan ke berkas log.

Perintah
--------
    python -u audit_whisper_token_limit.py --regenerate 2>&1 |
        Tee-Object -FilePath logs/audit_whisper_token_limit.log
"""

from __future__ import annotations

import argparse
import copy
import csv
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analysis_common import (
    ANALYSIS_SEED,
    LEVELS,
    N_BOOTSTRAP,
    RESULTS_DIR,
    SCRIPT_DIR,
    SPLIT_DIR,
    bootstrap_ratios,
    find_config,
    load_configs,
    percentile_interval,
    sha256_of,
    utterance_counts,
    write_csv,
    write_json,
)
from asr_common import corpus_metrics, normalize_text, read_split
from log_environment import collect_environment, set_seed

# --------------------------------------------------------------------------
# Konstanta
# --------------------------------------------------------------------------

REPORT_NAME = "audit_whisper_token_limit_report.json"
UTTERANCE_TABLE_NAME = "audit_whisper_token_limit_utterances.csv"

WHISPER_REPO = "openai/whisper-small"
WHISPER_REVISION = "973afd24965f72e36ca33b3055d56a652f456b4d"

# Batas pada konfigurasi grid dan batas tertinggi yang diterima Whisper small.
GRID_LIMIT = 128
MAX_TARGET_POSITIONS = 448
PROMPT_TOKENS = 4
SENSITIVITY_LIMIT = MAX_TARGET_POSITIONS - PROMPT_TOKENS

WHISPER_CONFIGS: dict[str, Path] = {
    "d1": SCRIPT_DIR / "configs" / "trial" / "whisper_d1_trial.yaml",
    **{level: SCRIPT_DIR / "configs" / f"whisper_{level}.yaml" for level in LEVELS[1:]},
}


# --------------------------------------------------------------------------
# Tokenisasi
# --------------------------------------------------------------------------


def load_tokenizer() -> Any:
    """Tokenizer Whisper small dari cache lokal pada revisi yang dikunci."""
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained(WHISPER_REPO, revision=WHISPER_REVISION, local_files_only=True)


def text_tokens(tokenizer: Any, text: str) -> int:
    """Jumlah token teks kanonik, tanpa token khusus."""
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def length_summary(values: Sequence[int]) -> dict[str, Any]:
    """Ringkasan sebaran panjang token."""
    array = np.asarray(values)
    return {
        "n": int(array.size),
        "max": int(array.max()),
        "p99": float(np.percentile(array, 99)),
        "median": float(np.median(array)),
        "n_at_or_above_grid_limit": int(np.sum(array >= GRID_LIMIT)),
    }


# --------------------------------------------------------------------------
# Dekode ulang
# --------------------------------------------------------------------------


def regenerate(level: str, utt_ids: Sequence[str], tokenizer: Any) -> dict[str, dict[str, Any]]:
    """Mendekode ulang utterance terpilih pada batas grid dan batas sensitivitas."""
    import torch

    from run_inference import load_model, transcribe
    from train_asr import apply_numerics, load_config, prevent_background_downloads

    config, config_sha256 = load_config(WHISPER_CONFIGS[level])
    prevent_background_downloads()
    apply_numerics(config)
    set_seed(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA tidak tersedia")
    model, processor = load_model(config)
    utterances = [utt for utt in read_split(SPLIT_DIR / "test.tsv") if utt.utt_id in set(utt_ids)]

    results: dict[str, dict[str, Any]] = {utt.utt_id: {} for utt in utterances}
    for limit in (GRID_LIMIT, SENSITIVITY_LIMIT):
        variant = copy.deepcopy(config)
        variant["generation_max_new_tokens"] = limit
        texts = transcribe(variant, model, processor, utterances)
        for utt in utterances:
            results[utt.utt_id][limit] = {
                "hypothesis_raw": texts[utt.utt_id],
                "text_tokens": text_tokens(tokenizer, texts[utt.utt_id]),
            }
    del model
    torch.cuda.empty_cache()
    for item in results.values():
        item["config_sha256"] = config_sha256
    return results


# --------------------------------------------------------------------------
# Titik masuk
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Menyusun argumen baris perintah."""
    parser = argparse.ArgumentParser(description="Audit batas token keluaran Whisper.")
    parser.add_argument("--regenerate", action="store_true",
                        help="Dekode ulang utterance yang mencapai batas (memerlukan GPU).")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                        help="Folder keluaran; bawaan outputs/results.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Titik masuk skrip."""
    args = parse_args(argv)
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    set_seed(ANALYSIS_SEED)

    print("=" * 78, flush=True)
    print("Audit batas token keluaran Whisper (Revisi 01 poin 18)", flush=True)
    print(f"Waktu mulai        : {timestamp}", flush=True)
    print(f"Random seed        : {ANALYSIS_SEED} (bootstrap), konfigurasi run bagi dekode ulang", flush=True)
    print(f"Batas grid         : {GRID_LIMIT}, batas sensitivitas {SENSITIVITY_LIMIT}", flush=True)
    print(f"Dekode ulang       : {'ya' if args.regenerate else 'tidak'}", flush=True)
    print("=" * 78, flush=True)

    tokenizer = load_tokenizer()
    configs = load_configs()

    references: dict[str, dict[str, Any]] = {}
    for split in ("test", "validation"):
        lengths = [text_tokens(tokenizer, utt.transcript) for utt in read_split(SPLIT_DIR / f"{split}.tsv")]
        references[split] = length_summary(lengths)
        print(f"Referensi {split:10s}: maks {references[split]['max']} token, "
              f"{references[split]['n_at_or_above_grid_limit']} >= {GRID_LIMIT}", flush=True)

    outputs: dict[str, Any] = {}
    reaching: dict[str, list[str]] = {}
    for config in configs:
        if config.model != "whisper":
            words = [len(hyp.split()) for hyp in config.hypotheses]
            outputs[config.run_name] = {"max_hypothesis_words": int(max(words))}
            continue
        lengths = [text_tokens(tokenizer, raw) for raw in config.hypotheses_raw]
        summary = length_summary(lengths)
        hit = [utt for utt, n in zip(config.ids, lengths, strict=True) if n >= GRID_LIMIT]
        summary["utterances_at_limit"] = hit
        summary["max_hypothesis_words"] = int(max(len(hyp.split()) for hyp in config.hypotheses))
        outputs[config.run_name] = summary
        reaching[config.level] = hit
        print(f"{config.run_name:18s}: maks {summary['max']} token, {len(hit)} mencapai batas {hit}", flush=True)

    sensitivity: dict[str, Any] = {}
    table_rows: list[list[Any]] = []
    if args.regenerate:
        print("-" * 78, flush=True)
        wer_columns, len_columns, labels = [], [], []
        for level, utt_ids in reaching.items():
            if not utt_ids:
                continue
            whisper = configs[find_config(configs, "whisper", level)]
            xlsr = configs[find_config(configs, "xlsr", level)]
            regenerated = regenerate(level, utt_ids, tokenizer)
            stored = dict(zip(whisper.ids, whisper.hypotheses_raw, strict=True))
            reproduced = {utt: regenerated[utt][GRID_LIMIT]["hypothesis_raw"] == stored[utt] for utt in utt_ids}

            replaced = list(whisper.hypotheses)
            index = {utt: i for i, utt in enumerate(whisper.ids)}
            for utt in utt_ids:
                replaced[index[utt]] = normalize_text(regenerated[utt][SENSITIVITY_LIMIT]["hypothesis_raw"])
            metrics = corpus_metrics(whisper.references, replaced)
            words = utterance_counts(whisper.references, replaced, "word")

            wer_columns += [whisper.words.errors, words.errors, xlsr.words.errors]
            len_columns += [whisper.words.lengths, words.lengths, xlsr.words.lengths]
            labels.append(level)
            sensitivity[level] = {
                "run_name": whisper.run_name,
                "config_sha256": next(iter(regenerated.values()))["config_sha256"],
                "reproduced_at_grid_limit": reproduced,
                "wer_grid": whisper.words.rate, "wer_sensitivity": metrics["wer"],
                "cer_grid": whisper.chars.rate, "cer_sensitivity": metrics["cer"],
                "xlsr_wer": xlsr.words.rate,
            }
            for utt in utt_ids:
                n_ref = len(whisper.references[index[utt]].split())
                table_rows.append([
                    whisper.run_name, utt, n_ref,
                    text_tokens(tokenizer, stored[utt]),
                    regenerated[utt][GRID_LIMIT]["text_tokens"],
                    regenerated[utt][SENSITIVITY_LIMIT]["text_tokens"],
                    len(normalize_text(stored[utt]).split()),
                    len(normalize_text(regenerated[utt][SENSITIVITY_LIMIT]["hypothesis_raw"]).split()),
                    reproduced[utt],
                ])
            print(f"{whisper.run_name}: dekode ulang 128 sama dengan tersimpan: {reproduced}", flush=True)
            print(f"  WER {whisper.words.rate:.4f} -> {metrics['wer']:.4f}, "
                  f"CER {whisper.chars.rate:.4f} -> {metrics['cer']:.4f}", flush=True)

        if labels:
            boot = bootstrap_ratios(np.column_stack(wer_columns), np.column_stack(len_columns),
                                    N_BOOTSTRAP, ANALYSIS_SEED)
            for position, level in enumerate(labels):
                grid, sens, xlsr = boot[:, 3 * position], boot[:, 3 * position + 1], boot[:, 3 * position + 2]
                item = sensitivity[level]
                item["h1_diff_grid"] = item["wer_grid"] - item["xlsr_wer"]
                item["h1_diff_grid_ci"] = list(percentile_interval(grid - xlsr))
                item["h1_diff_sensitivity"] = item["wer_sensitivity"] - item["xlsr_wer"]
                item["h1_diff_sensitivity_ci"] = list(percentile_interval(sens - xlsr))
                item["wer_change_ci"] = list(percentile_interval(sens - grid))
                print(f"  H1 {level}: {item['h1_diff_grid']:+.4f} -> {item['h1_diff_sensitivity']:+.4f} "
                      f"[{item['h1_diff_sensitivity_ci'][0]:+.4f}, {item['h1_diff_sensitivity_ci'][1]:+.4f}]",
                      flush=True)

    elapsed = round(time.perf_counter() - started, 1)
    report = {
        "generated_at": timestamp,
        "script": Path(__file__).name,
        "seed": ANALYSIS_SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "tokenizer": {"repo": WHISPER_REPO, "revision": WHISPER_REVISION},
        "grid_limit": GRID_LIMIT,
        "sensitivity_limit": SENSITIVITY_LIMIT,
        "length_definition": ("token teks kanonik hasil tokenisasi ulang, tanpa token prompt dan "
                              "<|endoftext|>; keluaran normal paling banyak batas dikurangi satu"),
        "references": references,
        "outputs": outputs,
        "regenerated": args.regenerate,
        "sensitivity": sensitivity,
        "input_sha256": {
            **{f"{c.run_name}_test.tsv": c.hypotheses_sha256 for c in configs},
            "test.tsv": sha256_of(SPLIT_DIR / "test.tsv"),
            "validation.tsv": sha256_of(SPLIT_DIR / "validation.tsv"),
        },
        "environment": collect_environment(ANALYSIS_SEED, started),
        "elapsed_seconds": elapsed,
    }
    report_path = args.results_dir / REPORT_NAME
    write_json(report_path, report)
    if table_rows:
        write_csv(args.results_dir / UTTERANCE_TABLE_NAME,
                  ["run_name", "id", "reference_words", "stored_text_tokens", "regenerated_128_text_tokens",
                   f"regenerated_{SENSITIVITY_LIMIT}_text_tokens", "stored_hypothesis_words",
                   f"hypothesis_{SENSITIVITY_LIMIT}_words", "reproduced_at_128"], table_rows)

    print("=" * 78, flush=True)
    print(f"Laporan ditulis ke : {report_path}", flush=True)
    print(f"Durasi eksekusi    : {elapsed} detik", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
