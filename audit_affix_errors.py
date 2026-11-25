"""Uraikan galat kata, karakter, digraf, dan afiks dari hipotesis data test.

Masukan: hipotesis dan laporan inference beku, configs/affix_error_audit.json.
Keluaran: laporan JSON, rekap CSV, dan contoh galat afiks per utterance.
Perintah: python -u audit_affix_errors.py 2>&1 | Tee-Object -FilePath logs/audit_affix_errors.log
Tanpa pelatihan, inference, audio, maupun konversi grafem-ke-fonem.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jiwer
from rapidfuzz.distance import Levenshtein

from analysis_common import LEVELS, RESULTS_DIR, load_configs, sha256_of, write_csv, write_json
from asr_common import WER_TRANSFORM

CONFIG = Path(__file__).resolve().parent / "configs" / "affix_error_audit.json"
STEM = "audit_affix_errors"
CLASSES = ("exact", "absent", "listed_replacement", "other_or_unresolved")


def affix_status(hyp: str, side: str, affix: str, remainder: str,
                 alternatives: dict[str, list[str]]) -> str:
    """Hanya lokalisasi galat yang menyisakan sisa kata secara persis."""
    def form(part: str) -> str:
        return part + remainder if side == "prefix" else remainder + part

    if hyp == form(affix):
        return "exact"
    if hyp == remainder:
        return "absent"
    if any(hyp == form(alt) for alt in alternatives[affix]):
        return "listed_replacement"
    return "other_or_unresolved"


def audit() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    print(f"Started {datetime.now(timezone.utc).isoformat()}; deterministic, no random seed")
    print(f"Config {CONFIG}; SHA-256 {sha256_of(CONFIG)}")
    print(json.dumps(cfg, sort_keys=True))
    digraphs = set(cfg["digraphs"])
    pattern = re.compile("|".join(map(re.escape, cfg["digraphs"])) + "|.")
    lexicon = {}
    for side in ("prefix", "suffix"):
        for affix, remainders in cfg[f"{side}es"].items():
            for rest in remainders:
                word = affix + rest if side == "prefix" else rest + affix
                if word in lexicon:
                    raise ValueError(f"Duplicate affix annotation: {word}")
                lexicon[word] = (side, affix, rest)
    configs = load_configs()
    # Kedua penyelarasan karakter memakai spasi sebagaimana CER utama.
    first = configs[0]
    reference_tokens = [pattern.findall(ref) for ref in first.references]
    ref_digraphs = Counter(t for ref in reference_tokens for t in ref if t in digraphs)
    unit_count = sum(len(t) for t in reference_tokens)
    vocab = Counter(word for ref in first.references for word in ref.split())
    denominators = Counter()
    for word, (_, affix, _) in lexicon.items():
        denominators[affix] += vocab[word]
    rows, cases, graphemes = [], [], []
    for c in configs:
        word_alignments = jiwer.process_words(
            list(c.references), list(c.hypotheses), reference_transform=WER_TRANSFORM,
            hypothesis_transform=WER_TRANSFORM,
        )
        counts = Counter()
        grapheme_confusions = Counter()
        char_ops = Counter()
        unit_errors = 0
        for uid, ref, hyp, ref_units, anchored in zip(
            c.ids, c.references, c.hypotheses, reference_tokens,
            word_alignments.alignments, strict=True,
        ):
            char_ops.update(ed.tag for ed in Levenshtein.editops(ref, hyp))
            hyp_units = pattern.findall(hyp)
            edits = Levenshtein.editops(ref_units, hyp_units)
            unit_errors += len(edits)
            for ed in edits:
                src = ref_units[ed.src_pos] if ed.tag != "insert" else ""
                dst = hyp_units[ed.dest_pos] if ed.tag != "delete" else ""
                if src in digraphs or dst in digraphs:
                    grapheme_confusions[(src, dst)] += 1
            ref_words, hyp_words = ref.split(), hyp.split()
            for chunk in anchored:
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    word = ref_words[i]
                    if word not in lexicon:
                        continue
                    side, affix, rest = lexicon[word]
                    if chunk.type == "equal":
                        observed = word
                    elif (chunk.type == "substitute"
                          and chunk.ref_end_idx - chunk.ref_start_idx == 1
                          and chunk.hyp_end_idx - chunk.hyp_start_idx == 1):
                        observed = hyp_words[chunk.hyp_start_idx]
                    else:
                        observed = None
                    status = (affix_status(observed, side, affix, rest, cfg["alternatives"])
                              if observed is not None else "other_or_unresolved")
                    counts[(affix, status)] += 1
                    if status in ("absent", "listed_replacement"):
                        cases.append([c.run_name, uid, i, affix, rest, word, observed, status])
        if sum(char_ops.values()) != int(c.chars.errors.sum()):
            raise ValueError(f"Character edit count differs from primary CER: {c.run_name}")
        for affix in sorted(denominators):
            n = denominators[affix]
            values = {k: counts[(affix, k)] for k in CLASSES}
            if sum(values.values()) != n:
                raise ValueError(f"Affix denominator mismatch: {c.run_name}, {affix}")
            rows.append({"model": c.model, "level": c.level, "affix": affix,
                         "reference_tokens": n, **values})
        graphemes.append({
            "model": c.model, "level": c.level, "wer": c.words.rate, "cer": c.chars.rate,
            "character_operations": dict(char_ops), "grouped_grapheme_errors": unit_errors,
            "grouped_grapheme_rate": unit_errors / unit_count,
            "digraph_confusions": [{"reference": a, "hypothesis": b, "count": n}
                                  for (a, b), n in sorted(grapheme_confusions.items())],
        })
    keys = list(rows[0])
    write_csv(RESULTS_DIR / f"{STEM}_counts.csv", keys, [[r[k] for k in keys] for r in rows])
    write_csv(RESULTS_DIR / f"{STEM}_cases.csv",
              ("run", "id", "reference_word_index", "affix", "remainder", "reference", "hypothesis", "status"),
              cases)
    write_json(RESULTS_DIR / f"{STEM}_report.json", {
        "config_sha256": sha256_of(CONFIG), "configuration": cfg,
        "hypotheses_sha256": {c.run_name: c.hypotheses_sha256 for c in configs},
        "test_utterances": len(first.ids), "levels": LEVELS,
        "grouped_grapheme_denominator": unit_count, "reference_digraphs": dict(ref_digraphs),
        "lexicon_types_observed": sum(vocab[word] > 0 for word in lexicon),
        "unobserved_lexicon_types": sorted(word for word in lexicon if not vocab[word]),
        "affix_counts": rows, "graphemes": graphemes,
    })
    print(f"Completed {datetime.now(timezone.utc).isoformat()}; denominators={dict(denominators)}")


if __name__ == "__main__":
    audit()
