"""Menilai WER/CER menurut tingkat tutur teks rujukan tanpa memuat model.

Masukan: configs/speech_register_audit.json, test.tsv, label rujukan dan
hipotesis sembilan konfigurasi lain serta Whisper D5 (grid lengkap).
Keluaran: CSV sampel, CSV metrik, JSON laporan di outputs/results.
Perintah (dari repositori):
    python -u audit_speech_register.py sample
    python -u audit_speech_register.py score 2>&1 | Tee-Object -FilePath logs/audit_speech_register.log
Label diisikan pada speech_register_labels.csv sebelum menjalankan score.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from analysis_common import RESULTS_DIR, load_configs, sha256_of, write_csv, write_json
from asr_common import Utterance, read_split

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "outputs" / "splits" / "test.tsv"
CONFIG = ROOT / "configs" / "speech_register_audit.json"
LABELS = ROOT / "configs" / "speech_register_labels.csv"
SAMPLE = RESULTS_DIR / "speech_register_sample.csv"
REPORT = RESULTS_DIR / "speech_register_report.json"
METRICS = RESULTS_DIR / "speech_register_metrics.csv"
COHORTS = ("acak", "tambahan_kandidat_krama")


def select(rows: list[Utterance], cfg: dict[str, Any]) -> list[tuple[str, str, Utterance]]:
    """Pilih sampel acak dan pengayaan terpisah tanpa membaca hipotesis."""
    rng = random.Random(cfg["seed"])
    markers = set(cfg["krama_screening_markers"])
    base = sorted(rows, key=lambda r: r.utt_id)
    first = rng.sample(base, cfg["random_sample_size"])
    chosen = {item.utt_id for item in first}
    pool = [item for item in base if item.utt_id not in chosen
            and markers.intersection(item.text.split())]
    second = rng.sample(pool, min(len(pool), cfg["krama_enrichment_size"]))
    tagged = [(COHORTS[0], row) for row in first] + [(COHORTS[1], row) for row in second]
    rng.shuffle(tagged)
    return [(str(i), cohort, row) for i, (cohort, row) in enumerate(tagged, 1)]


def labels_for(path: Path, selected: list[tuple[str, str, Utterance]],
               cfg: dict[str, Any]) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    by_number = {r["number"]: r for r in labels}
    if len(by_number) != len(labels) or set(by_number) != {n for n, _, _ in selected}:
        raise ValueError("Label rows and selected sample IDs do not match")
    result = {}
    for n, _, row in selected:
        label = by_number[n]
        if label["id"] != row.utt_id or label["register"] not in cfg["categories"]:
            raise ValueError(f"Invalid label or mismatched ID at row {n}")
        result[row.utt_id] = label["register"]
    return result


def score(selected: list[tuple[str, str, Utterance]], cfg: dict[str, Any]) -> None:
    labels = labels_for(LABELS, selected, cfg)
    configs = load_configs()
    first = configs[0]
    positions = {u: i for i, u in enumerate(first.ids)}
    assert all(first.references[positions[r.utt_id]] == r.text for _, _, r in selected)
    metrics = []
    counts = []
    for cohort in (*COHORTS, "gabungan_deskriptif"):
        eligible = [r for _, c, r in selected if c == cohort or cohort == "gabungan_deskriptif"]
        for register in cfg["categories"]:
            group = [r for r in eligible if labels[r.utt_id] == register]
            ix = np.array([positions[r.utt_id] for r in group], dtype=np.intp)
            n_words = int(first.words.lengths[ix].sum())
            n_chars = int(first.chars.lengths[ix].sum())
            summary = {"cohort": cohort, "register": register, "utterances": len(group),
                       "speakers": len({r.speaker_id for r in group}),
                       "prompts": len({r.text for r in group}),
                       "reference_words": n_words, "reference_characters": n_chars}
            counts.append(summary)
            if not group:
                continue
            for c in configs:
                word_errors = int(c.words.errors[ix].sum())
                char_errors = int(c.chars.errors[ix].sum())
                metrics.append({**summary, "model": c.model, "level": c.level,
                                "word_errors": word_errors, "character_errors": char_errors,
                                "wer": word_errors / n_words, "cer": char_errors / n_chars})
    report = {"test_n": len(first.ids), "config": cfg, "config_sha256": sha256_of(CONFIG),
              "test_sha256": sha256_of(TEST), "labels_sha256": sha256_of(LABELS),
              "hypotheses_sha256": {c.run_name: c.hypotheses_sha256 for c in configs},
              "annotator": "single assistant, text only; no human validation or agreement estimate",
              "n_selected": len(selected), "cohort_sizes": dict(Counter(c for _, c, _ in selected)),
              "counts": counts, "metrics": metrics}
    write_json(REPORT, report)
    columns = list(metrics[0])
    write_csv(METRICS, columns, [[m[k] for k in columns] for m in metrics])
    print(json.dumps(counts, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sample", "score"))
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = read_split(TEST)
    if len(rows) != 3100 or len({r.utt_id for r in rows}) != len(rows):
        raise ValueError("Unexpected test split")
    print(f"Started {datetime.now(timezone.utc).isoformat()}; seed={cfg['seed']}; stage={args.stage}")
    print(f"Config SHA-256 {sha256_of(CONFIG)}; test SHA-256 {sha256_of(TEST)}")
    picks = select(rows, cfg)
    if args.stage == "sample":
        write_csv(SAMPLE, ("number", "id", "reference"),
                  [(n, r.utt_id, r.transcript) for n, _, r in picks])
        print(f"Prepared {len(picks)} references without hypotheses or cohort columns")
    else:
        score(picks, cfg)
        print(f"Completed {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
