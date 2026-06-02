#!/usr/bin/env python3
"""Build a combined KenLM training corpus from Leipzig + lip-reading labels.

This script reuses the conservative cleaning rules from prepare_kenlm_corpus.py,
merges multiple sentence sources, de-duplicates the final corpus, and writes a
single cleaned text file suitable for KenLM training.

Typical usage:
  python Preprocess/build_combined_kenlm_corpus.py \
      --leipzig_input data/leipzig_news.txt \
      --manifest_csv LUMINA_preprocessed/manifest.csv \
      --output KenLM/combined_corpus.txt \
      --report KenLM/combined_corpus.report.json

If your manifest has a different sentence column, pass --text_column.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from prepare_kenlm_corpus import clean_line, tokenize_words


def iter_plain_text_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            yield line.rstrip("\n\r")


def iter_manifest_texts(manifest_csv: Path, text_column: str) -> Iterable[str]:
    with manifest_csv.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or text_column not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_csv} must contain a `{text_column}` column. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            text = (row.get(text_column) or "").strip()
            if text:
                yield text


def build_combined_corpus(
    leipzig_input: Path | None,
    manifest_csv: Path | None,
    text_column: str,
    output_path: Path,
    report_path: Path | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    token_counter = Counter()
    kept_lines: list[str] = []
    source_stats: dict[str, dict[str, int]] = {}

    def process_source(source_name: str, texts: Iterable[str]) -> None:
        source_stats[source_name] = {"total": 0, "kept": 0, "dropped": 0}
        for raw_line in texts:
            source_stats[source_name]["total"] += 1
            stats["total"] += 1
            cleaned = clean_line(raw_line)
            if cleaned is None:
                source_stats[source_name]["dropped"] += 1
                stats["dropped"] += 1
                continue
            source_stats[source_name]["kept"] += 1
            stats["kept"] += 1
            kept_lines.append(cleaned)
            token_counter.update(tokenize_words(cleaned))

    if leipzig_input is not None:
        if not leipzig_input.exists():
            raise FileNotFoundError(f"Leipzig input not found: {leipzig_input}")
        process_source(f"leipzig:{leipzig_input.name}", iter_plain_text_lines(leipzig_input))

    if manifest_csv is not None:
        if not manifest_csv.exists():
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_csv}")
        process_source(f"manifest:{manifest_csv.name}", iter_manifest_texts(manifest_csv, text_column))

    unique_lines = list(dict.fromkeys(kept_lines))
    stats["deduped"] = len(unique_lines)
    stats["duplicates_removed"] = len(kept_lines) - len(unique_lines)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in unique_lines:
            handle.write(line + "\n")

    report = {
        "output": str(output_path),
        "sources": source_stats,
        "total_lines": stats["total"],
        "kept_lines": stats["kept"],
        "dropped_lines": stats["dropped"],
        "duplicates_removed": stats["duplicates_removed"],
        "final_lines": stats["deduped"],
        "top_tokens": token_counter.most_common(100),
        "settings": {
            "lowercase": True,
            "min_words": 5,
            "max_words": 45,
            "alpha_min": 0.60,
            "digit_max": 0.30,
            "text_column": text_column,
        },
    }

    if report_path is not None:
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a combined cleaned corpus for KenLM.")
    parser.add_argument("--leipzig_input", default=None, help="Plain text / TSV Leipzig corpus input")
    parser.add_argument("--manifest_csv", default=None, help="LUMINA manifest CSV containing sentence text")
    parser.add_argument("--text_column", default="text", help="Sentence column name in the manifest CSV")
    parser.add_argument("--output", required=True, help="Output cleaned combined corpus text file")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    leipzig_input = Path(args.leipzig_input) if args.leipzig_input else None
    manifest_csv = Path(args.manifest_csv) if args.manifest_csv else None
    if leipzig_input is None and manifest_csv is None:
        raise ValueError("Provide at least one input source: --leipzig_input and/or --manifest_csv")

    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None

    report = build_combined_corpus(
        leipzig_input=leipzig_input,
        manifest_csv=manifest_csv,
        text_column=args.text_column,
        output_path=output_path,
        report_path=report_path,
    )

    print(f"Saved combined cleaned corpus to: {output_path}")
    print(f"Total lines      : {report['total_lines']}")
    print(f"Kept lines       : {report['kept_lines']}")
    print(f"Dropped lines    : {report['dropped_lines']}")
    print(f"Duplicates removed: {report['duplicates_removed']}")
    print(f"Final lines      : {report['final_lines']}")
    if report_path is not None:
        print(f"Saved report to  : {report_path}")


if __name__ == "__main__":
    main()
