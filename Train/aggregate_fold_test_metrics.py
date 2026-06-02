"""
Aggregate fold-level test metrics across model variants.

This script scans folders like:
  Train/runs/fold1_sequential/test_metrics.json
  ...
  Train/runs/fold5_mamba_only/test_metrics.json

Then writes one JSON report containing:
  - metrics per fold and per model variant
  - mean metrics per model variant
  - both non-beam and beam summaries

Requested metrics tracked:
  CER, WER, WAR, SAR, SD

Notes on mapping used in this project:
    - WAR = word accuracy (word_acc_micro when present, else 1 - WER).
    - SAR = sentence accuracy (sentence_acc when present).
    - SD = standard deviation of per-sample WER.
        Non-beam SD is computed from test_predictions.csv when available.
        Beam SD is computed from beam_predictions in test_metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ["sequential", "bigru_only", "parallel", "mamba_only"]
TARGET_METRICS = ["cer", "wer", "war", "sar", "sd"]
MEAN_KEYS = ["cer", "wer", "war", "sar", "sd_within_fold", "sd_between_folds"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate fold test_metrics into one JSON summary.")
    parser.add_argument("--runs_dir", type=str, default="Train/runs", help="Directory containing fold run folders.")
    parser.add_argument("--start_fold", type=int, default=1, help="Start fold index (inclusive).")
    parser.add_argument("--end_fold", type=int, default=5, help="End fold index (inclusive).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model variants to aggregate.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="Train/runs/comparison/fold_metrics_summary.json",
        help="Output path for aggregated JSON.",
    )
    return parser.parse_args()


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, a_token in enumerate(seq_a, start=1):
        curr = [i] + [0] * len(seq_b)
        for j, b_token in enumerate(seq_b, start=1):
            cost = 0 if a_token == b_token else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def std_population(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)

def compute_kenlm_sar_from_csv(kenlm_predictions_test: Path) -> float | None:
    if not kenlm_predictions_test.exists():
        return None

    exact = 0
    total = 0
    with kenlm_predictions_test.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ref = row.get("ref")
            hyp = row.get("kenlm")
            if ref is None or hyp is None:
                continue
            total += 1
            if normalize_text(ref) == normalize_text(hyp):
                exact += 1

    if total == 0:
        return None
    return exact / total

def compute_non_beam_sd_from_csv(test_predictions_csv: Path) -> float | None:
    if not test_predictions_csv.exists():
        return None

    wers: list[float] = []
    with test_predictions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word_er = safe_float(row.get("word_er"))
            if word_er is not None:
                wers.append(word_er)

    return std_population(wers)


def compute_beam_sd_from_predictions(pred_rows: Any) -> float | None:
    if not isinstance(pred_rows, list) or not pred_rows:
        return None

    wers: list[float] = []
    for row in pred_rows:
        if not isinstance(row, dict):
            continue
        ref = row.get("ref")
        hyp = row.get("beam")
        if ref is None or hyp is None:
            continue

        ref_words = normalize_text(ref).split()
        hyp_words = normalize_text(hyp).split()
        word_len = max(len(ref_words), 1)
        word_ed = levenshtein(ref_words, hyp_words)
        wers.append(word_ed / word_len)

    return std_population(wers)


def pick_first(metrics: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        if key in metrics:
            val = safe_float(metrics.get(key))
            if val is not None:
                return val
    return None


def extract_non_beam_metrics(raw: dict[str, Any], run_dir: Path) -> dict[str, float | None]:
    cer = pick_first(raw, ["cer", "char_er", "char_error_rate"])
    wer = pick_first(raw, ["wer", "word_er", "word_error_rate"])
    war = pick_first(raw, ["word_acc_micro", "word_accuracy", "war"])
    sar = pick_first(raw, ["sentence_acc", "sentence_accuracy", "sar"])
    sd_within = compute_non_beam_sd_from_csv(run_dir / "test_predictions.csv")

    if war is None and wer is not None:
        war = 1.0 - wer

    return {
        "cer": cer,
        "wer": wer,
        "war": war,
        "sar": sar,
        "sd_within_fold": sd_within,
    }


def extract_beam_metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    beam = raw.get("beam_metrics")
    if not isinstance(beam, dict):
        beam = {}

    cer = pick_first(beam, ["cer", "char_er", "char_error_rate"])
    wer = pick_first(beam, ["wer", "word_er", "word_error_rate"])
    war = pick_first(beam, ["word_acc_micro", "word_accuracy", "war"])
    sar = pick_first(beam, ["sentence_acc", "sentence_accuracy", "sar"])
    sd_within = compute_beam_sd_from_predictions(raw.get("beam_predictions"))

    if war is None and wer is not None:
        war = 1.0 - wer

    if sar is None:
        preds = raw.get("beam_predictions")
        if isinstance(preds, list) and preds:
            exact = 0
            total = 0
            for row in preds:
                if not isinstance(row, dict):
                    continue
                ref = row.get("ref")
                hyp = row.get("beam")
                if ref is None or hyp is None:
                    continue
                total += 1
                if normalize_text(ref) == normalize_text(hyp):
                    exact += 1
            if total > 0:
                sar = exact / total

    return {
        "cer": cer,
        "wer": wer,
        "war": war,
        "sar": sar,
        "sd_within_fold": sd_within,
    }

def extract_kenlm_metrics(raw: dict[str, Any], run_dir: Path) -> dict[str, float | None]:
    kenlm = raw.get("kenlm_metrics")
    if not isinstance(kenlm, dict):
        kenlm = {}
    
    cer = pick_first(kenlm, ["cer", "char_er", "char_error_rate"])
    wer = pick_first(kenlm, ["wer", "word_er", "word_error_rate"])
    war = pick_first(kenlm, ["word_acc_micro", "word_accuracy", "war"])
    sar = pick_first(kenlm, ["sentence_acc", "sentence_accuracy", "sar"])

    if war is None and wer is not None:
        war = 1.0 - wer

    if sar is None:
        sar = compute_kenlm_sar_from_csv(run_dir / "kenlm_predictions_test.csv")

    return {
        "cer": cer,
        "wer": wer,
        "war": war,
        "sar": sar,
    }


def mean_ignore_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def compute_means(per_fold_metrics: list[dict[str, float | None]]) -> dict[str, float | None]:
    return {
        metric: mean_ignore_none([fold.get(metric) for fold in per_fold_metrics])
        for metric in TARGET_METRICS
    }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_report(
    runs_dir: Path,
    models: list[str],
    start_fold: int,
    end_fold: int,
) -> dict[str, Any]:
    folds = list(range(start_fold, end_fold + 1))
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs_dir": str(runs_dir),
        "fold_range": [start_fold, end_fold],
        "models": {},
    }

    for model in models:
        model_entry: dict[str, Any] = {
            "folds": {},
            "means": {
                "non_beam": {k: None for k in MEAN_KEYS},
                "beam": {k: None for k in MEAN_KEYS},
                "kenlm": {k: None for k in MEAN_KEYS},
            },
            "counts": {
                "non_beam": 0,
                "beam": 0,
                "kenlm": 0,
                "requested_folds": len(folds),
            },
            "missing_folds": [],
        }

        non_beam_rows: list[dict[str, float | None]] = []
        beam_rows: list[dict[str, float | None]] = []
        kenlm_rows: list[dict[str, float | None]] = []

        for fold in folds:
            run_dir = runs_dir / f"fold{fold}_{model}"
            metrics_path = run_dir / "test_metrics.json"

            if not metrics_path.exists():
                model_entry["missing_folds"].append(fold)
                model_entry["folds"][f"fold{fold}"] = {
                    "exists": False,
                    "path": str(metrics_path),
                    "non_beam": {"cer": None, "wer": None, "war": None, "sar": None, "sd_within_fold": None},
                    "beam": {"cer": None, "wer": None, "war": None, "sar": None, "sd_within_fold": None},
                    "kenlm": {"cer": None, "wer": None, "war": None, "sar": None},
                }
                continue

            raw = read_json(metrics_path)
            non_beam = extract_non_beam_metrics(raw, run_dir)
            beam = extract_beam_metrics(raw)
            kenlm = extract_kenlm_metrics(raw, run_dir)

            model_entry["folds"][f"fold{fold}"] = {
                "exists": True,
                "path": str(metrics_path),
                "non_beam": non_beam,
                "beam": beam,
                "kenlm": kenlm,
            }

            if any(v is not None for v in non_beam.values()):
                non_beam_rows.append(non_beam)
            if any(v is not None for v in beam.values()):
                beam_rows.append(beam)
            if any(v is not None for v in kenlm.values()):
                kenlm_rows.append(kenlm)

        # compute mean metrics (including within-fold SD) and between-fold SD for WER
        def compute_model_means(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
            if not rows:
                return {k: None for k in MEAN_KEYS}
            result: dict[str, float | None] = {}
            result["cer"] = mean_ignore_none([r.get("cer") for r in rows])
            result["wer"] = mean_ignore_none([r.get("wer") for r in rows])
            result["war"] = mean_ignore_none([r.get("war") for r in rows])
            result["sar"] = mean_ignore_none([r.get("sar") for r in rows])
            result["sd_within_fold"] = mean_ignore_none([r.get("sd_within_fold") for r in rows])
            # between-fold SD: std of fold-level WERs
            wer_values = [v for v in (r.get("wer") for r in rows) if v is not None]
            result["sd_between_folds"] = std_population(wer_values)
            return result

        model_entry["means"]["non_beam"] = compute_model_means(non_beam_rows)
        model_entry["means"]["beam"] = compute_model_means(beam_rows)
        model_entry["means"]["kenlm"] = compute_model_means(kenlm_rows)
        model_entry["counts"]["non_beam"] = len(non_beam_rows)
        model_entry["counts"]["beam"] = len(beam_rows)
        model_entry["counts"]["kenlm"] = len(kenlm_rows)

        report["models"][model] = model_entry

    return report


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_path = Path(args.output_json)

    if args.end_fold < args.start_fold:
        raise ValueError("end_fold must be >= start_fold")

    report = build_report(
        runs_dir=runs_dir,
        models=list(args.models),
        start_fold=int(args.start_fold),
        end_fold=int(args.end_fold),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved aggregated metrics JSON: {out_path}")


if __name__ == "__main__":
    main()
