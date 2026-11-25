"""Audit audio dan anotasi SLR35 pascaeksperimen tanpa pelatihan atau inference ulang.

Masukan: pembagian dan hipotesis beku, FLAC lokal, konfigurasi audit JSON.
Keluaran: laporan JSON, ringkasan per arsip/bagian, indikator per utterance,
antrean pemeriksaan dengar, dan WER deskriptif per kelompok data test.
Perintah: python -u audit_audio_annotations.py 2>&1 | Tee-Object -FilePath logs/audit_audio_annotations.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from analysis_common import RESULTS_DIR, load_configs, sha256_of, write_csv, write_json
from asr_common import DEFAULT_DATASET_ROOT, flac_path, read_split

ROOT = Path(__file__).resolve().parent
SPLITS = ROOT / "outputs" / "splits"
CONFIG = ROOT / "configs" / "audio_annotation_audit.json"
PARTS = ("train_pool", "validation", "test", "excluded")
EXPECTED = (41293, 1900, 3100, 16)
FORMAT = ("FLAC", 16000, 1, "PCM_16")
STEM = "audit_audio_annotations"


def dbfs(rms: float, floor: float) -> float:
    return float(20 * np.log10(max(rms, 10 ** (floor / 20))))


def audio_metrics(samples: np.ndarray, rate: int, cfg: dict[str, Any]) -> dict[str, Any]:
    """Hitung taraf bingkai relatif terhadap skala penuh, bukan SNR."""
    x = samples.astype(np.float64) / 32768
    size = rate * cfg["frame_ms"] // 1000
    n_frames = len(x) // size
    if not n_frames:
        raise ValueError("No complete audio frame")
    frames = x[:n_frames * size].reshape(n_frames, size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    low, high = np.percentile(rms, [cfg["lower_percentile"], cfg["upper_percentile"]])
    low_db = dbfs(low, cfg["dbfs_floor"])
    high_db = dbfs(high, cfg["dbfs_floor"])
    return {
        "low_frame_dbfs": low_db,
        "high_frame_dbfs": high_db,
        "frame_contrast_db": high_db - low_db,
        "near_full_scale_samples": int(np.count_nonzero(abs(x) >= cfg["near_full_scale_fraction"])),
        "pcm_sha256": hashlib.sha256(samples.astype("<i2", copy=False).tobytes()).hexdigest(),
    }


def spelling_key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flags = Counter(f for r in rows for f in r["flags"])
    return {
        "n": len(rows),
        "low_frame_dbfs_q10_q50_q90": np.percentile(
            [r["low_frame_dbfs"] for r in rows], [10, 50, 90]
        ).tolist(),
        "frame_contrast_db_median": float(np.median([r["frame_contrast_db"] for r in rows])),
        "near_full_scale_utterances": sum(r["near_full_scale_samples"] > 0 for r in rows),
        "flag_counts": dict(sorted(flags.items())),
    }


def stored_wer(test: list[dict[str, Any]], results: Path) -> dict[str, Any]:
    """Pakai batas kuintil empiris (kelompok ekor) dan referensi tetap."""
    by_id = {r["id"]: r for r in test}
    q20, q80 = np.quantile([r["low_frame_dbfs"] for r in test], [0.2, 0.8])
    if q20 >= q80:
        raise ValueError("No separation between test tails")
    masks = {
        "lower": {r["id"] for r in test if r["low_frame_dbfs"] <= q20},
        "upper": {r["id"] for r in test if r["low_frame_dbfs"] >= q80},
    }
    configs = load_configs(results_dir=results)
    rows = []
    for c in configs:
        if set(c.ids) != set(by_id) or any(
            by_id[i]["normalized"] != ref for i, ref in zip(c.ids, c.references, strict=True)
        ):
            raise ValueError(f"Frozen reference mismatch: {c.run_name}")
        for group, ids in masks.items():
            keep = np.array([i in ids for i in c.ids])
            n_errors = int(c.words.errors[keep].sum())
            n_words = int(c.words.lengths[keep].sum())
            rows.append({"run": c.run_name, "group": group, "n": int(keep.sum()),
                         "words": n_words, "errors": n_errors, "wer": n_errors / n_words})
    return {"thresholds_dbfs": [float(q20), float(q80)], "rows": rows,
            "hypotheses_sha256": {c.run_name: c.hypotheses_sha256 for c in configs}}


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        columns = list(rows[0])
        write_csv(path, columns, [[r[k] for k in columns] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    print(f"Started {datetime.now(timezone.utc).isoformat()}; config={args.config}")
    print(f"Config SHA-256={sha256_of(args.config)}; seed={cfg['seed']}; CPU only")
    print(json.dumps(cfg, sort_keys=True))
    rng = random.Random(cfg["seed"])
    rows = []
    formats = Counter()
    by_pcm = defaultdict(list)
    spellings = defaultdict(list)
    seen = set()
    for part, expected in zip(PARTS, EXPECTED, strict=True):
        utterances = read_split(SPLITS / f"{part}.tsv")
        if len(utterances) != expected:
            raise ValueError(f"Unexpected {part} size: {len(utterances)}")
        for i, utt in enumerate(utterances, 1):
            uid = utt.utt_id
            if uid in seen:
                raise ValueError(f"Duplicate split ID: {uid}")
            seen.add(uid)
            path = flac_path(uid, args.dataset_root)
            with sf.SoundFile(path) as handle:
                fmt = (handle.format, handle.samplerate, handle.channels, handle.subtype)
                if fmt != FORMAT:
                    raise ValueError(f"Unexpected audio format for {uid}: {fmt}")
                pcm = handle.read(dtype="int16")
                duration = len(pcm) / handle.samplerate
                if abs(duration - utt.duration) > 0.00051:
                    raise ValueError(f"Split duration mismatch for {uid}")
                features = audio_metrics(pcm, handle.samplerate, cfg)
            formats[" ".join(map(str, fmt))] += 1
            text = unicodedata.normalize("NFC", utt.transcript).casefold()
            text = " ".join(text.split())
            n_words = len(utt.text.split())
            wps = n_words / duration
            flags = []
            if text == cfg["known_recorder_message"].casefold():
                flags.append("recorder_message")
            if not utt.text:
                flags.append("empty_normalized_text")
            if n_words >= cfg["min_normalized_words_for_rate_flag"]:
                if wps > cfg["max_audio_words_per_second_flag"]:
                    flags.append("high_reference_word_rate")
                if wps < cfg["min_audio_words_per_second_flag"]:
                    flags.append("low_reference_word_rate")
            if uid in cfg["confirmed_test_defect_ids"]:
                if part != "test":
                    raise ValueError(f"Confirmed case absent from test: {uid}")
                flags.append("previously_confirmed_test_defect")
            row = {"id": uid, "split": part, "archive": int(uid[0], 16),
                   "speaker_id": utt.speaker_id, "duration_s": duration,
                   "normalized_words_per_s": wps, **features,
                   "transcript": utt.transcript, "normalized": utt.text, "flags": flags}
            rows.append(row)
            if part != "excluded":
                by_pcm[features["pcm_sha256"]].append(row)
                spellings[spelling_key(text)].append((text, row))
            if i in (expected,) or i % 5000 == 0:
                print(f"Scanned {part}: {i}/{expected}")

    if not set(cfg["confirmed_test_defect_ids"]) <= seen:
        raise ValueError("Previously confirmed test IDs not found")
    conflicting_pcm = [group for group in by_pcm.values()
                       if len({r["normalized"] for r in group}) > 1]
    for group in conflicting_pcm:
        for row in group:
            row["flags"].append("identical_pcm_distinct_normalized_transcripts")
    variant_groups = [entries for entries in spellings.values()
                      if len({text for text, _ in entries}) > 1]
    for entries in variant_groups:
        for _, row in entries:
            row["flags"].append("same_prompt_differing_diacritics")

    used = [r for r in rows if r["split"] != "excluded"]
    selected = defaultdict(set)
    groups = []
    for (split, archive), batch in groupby(
        sorted(used, key=lambda r: (r["split"], r["archive"], r["id"])),
        key=lambda r: (r["split"], r["archive"]),
    ):
        group = list(batch)
        groups.append({"split": split, "archive": archive, **summarize(group)})
        for r in rng.sample(group, min(len(group), cfg["random_examples_per_archive_split"])):
            selected[r["id"]].add("random_in_stratum")
        highest = sorted(group, key=lambda r: (-r["low_frame_dbfs"], r["id"]))
        for r in highest[:cfg["high_floor_examples_per_archive_split"]]:
            selected[r["id"]].add("upper_acoustic_candidates")
    for r in used:
        for flag in r["flags"]:
            if flag != "same_prompt_differing_diacritics":
                selected[r["id"]].add(flag)
    queue = [{"id": r["id"], "split": r["split"], "archive": r["archive"],
              "reasons": ";".join(sorted(selected[r["id"]])),
              "transcript": r["transcript"], "audio_review": "pending"}
             for r in used if r["id"] in selected]

    test = [r for r in used if r["split"] == "test"]
    wer = stored_wer(test, RESULTS_DIR)
    report = {
        "configuration": cfg, "configuration_sha256": sha256_of(args.config),
        "split_sha256": {p: sha256_of(SPLITS / f"{p}.tsv") for p in PARTS},
        "format_counts": dict(formats), "audited": len(rows), "included": len(used),
        "split_sizes": dict(zip(PARTS, EXPECTED, strict=True)),
        "included_summary": summarize(used), "archive_split": groups,
        "diacritic_variant_groups": len(variant_groups),
        "identical_pcm_groups": sum(len(g) > 1 for g in by_pcm.values()),
        "identical_pcm_conflicting_normalized_transcript_groups": len(conflicting_pcm),
        "recorder_message_ids": [r["id"] for r in used if "recorder_message" in r["flags"]],
        "test_wer": wer, "human_listening_done_by_this_audit": False,
        "pending_listening_queue_size": len(queue),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / f"{STEM}_utterances.csv",
               [{k: ";".join(v) if k == "flags" else v for k, v in r.items()
                 if k not in {"transcript", "normalized", "pcm_sha256"}} for r in rows])
    write_rows(args.output_dir / f"{STEM}_review_queue.csv", queue)
    write_rows(args.output_dir / f"{STEM}_test_wer.csv", wer["rows"])
    write_json(args.output_dir / f"{STEM}_report.json", report)
    print(json.dumps({k: report[k] for k in (
        "format_counts", "included_summary", "diacritic_variant_groups",
        "identical_pcm_groups", "identical_pcm_conflicting_normalized_transcript_groups",
        "pending_listening_queue_size",
    )}, sort_keys=True))
    print(f"Completed {datetime.now(timezone.utc).isoformat()}; outputs={args.output_dir}")


if __name__ == "__main__":
    main()
