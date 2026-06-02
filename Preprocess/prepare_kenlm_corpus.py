"""Clean Indonesian text corpora for KenLM training.

The cleaner is intentionally conservative:
- lowercase everything
- normalize Unicode punctuation and whitespace
- drop obvious fragments, boilerplate, URLs, and noisy numeric-only lines
- keep only lines that look like real Indonesian sentences

Usage:
    python Preprocess/prepare_kenlm_corpus.py \
        --input data/leipzig_news.txt \
        --output data/leipzig_news.cleaned.txt \
        --report data/leipzig_news.report.json

The input may be a TSV/TTX style file with a leading numeric ID followed by a tab
and the sentence text, or a plain text file with one sentence per line.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


STOPWORDS = {
    "yang",
    "dan",
    "di",
    "ke",
    "dari",
    "untuk",
    "pada",
    "dengan",
    "itu",
    "ini",
    "akan",
    "tidak",
    "karena",
    "atau",
    "sebagai",
    "dalam",
    "juga",
    "lebih",
    "sudah",
    "sangat",
    "para",
    "kami",
    "kita",
    "mereka",
    "oleh",
    "saat",
}

LEADING_ID_RE = re.compile(r"^\s*\d+(?:\t+|\s{2,})")
URL_RE = re.compile(r"(?:https?://|www\.|%[0-9A-Fa-f]{2}|\.pdf\b)", re.IGNORECASE)
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f\u200b\u200c\u200d\ufeff]")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")
TOKEN_STRIP_CHARS = ".,!?;:\"'()[]{}-"


def strip_leading_id(text: str) -> str:
    return LEADING_ID_RE.sub("", text, count=1)


def normalize_quotes_and_dashes(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2026": "...",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_RE.sub(" ", text)
    text = normalize_quotes_and_dashes(text)
    text = text.lower()
    text = text.replace("\t", " ")
    text = MULTISPACE_RE.sub(" ", text).strip()
    return text


def contains_letter(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def alpha_digit_stats(text: str) -> tuple[float, float]:
    letters = sum(1 for ch in text if ch.isalpha())
    digits = sum(1 for ch in text if ch.isdigit())
    non_space = sum(1 for ch in text if not ch.isspace())
    if non_space == 0:
        return 0.0, 0.0
    return letters / non_space, digits / non_space


def tokenize_words(text: str) -> list[str]:
    tokens: list[str] = []
    for raw_token in re.split(r"\s+", text):
        token = raw_token.strip(TOKEN_STRIP_CHARS)
        if token:
            tokens.append(token)
    return tokens


def strip_outer_quotes(text: str) -> str:
    if len(text) >= 2:
        pairs = [("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")]
        for left, right in pairs:
            if text.startswith(left) and text.endswith(right):
                return text[1:-1].strip()
    return text


def has_balanced_brackets(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    for left, right in pairs:
        if text.count(left) != text.count(right):
            return False
    quote_count = text.count('"')
    if quote_count % 2 != 0:
        return False
    return True


def looks_like_real_sentence(text: str, min_words: int, max_words: int, alpha_min: float, digit_max: float) -> bool:
    if not text:
        return False
    if URL_RE.search(text):
        return False
    if REPEATED_CHAR_RE.search(text):
        return False

    words = tokenize_words(text)
    if not (min_words <= len(words) <= max_words):
        return False

    alpha_ratio, digit_ratio = alpha_digit_stats(text)
    if alpha_ratio < alpha_min or digit_ratio > digit_max:
        return False

    if not contains_letter(text):
        return False

    if not has_balanced_brackets(text):
        return False

    if not any(word in STOPWORDS for word in words):
        return False

    alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
    if len(alpha_words) < 3:
        return False

    symbol_count = sum(1 for ch in text if ch in "•▪·|/\\")
    if symbol_count >= 3:
        return False

    if len(text) < 20:
        return False

    return True


def clean_line(raw_line: str) -> str | None:
    line = raw_line.rstrip("\n\r")
    line = strip_leading_id(line)
    line = normalize_text(line)
    line = strip_outer_quotes(line)
    line = MULTISPACE_RE.sub(" ", line).strip(" \"'.,;:!?()-")
    line = " ".join(tokenize_words(line))
    if not line:
        return None
    if not looks_like_real_sentence(line, min_words=5, max_words=45, alpha_min=0.60, digit_max=0.30):
        return None
    return line


def clean_corpus(input_path: Path, output_path: Path, report_path: Path | None = None) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    token_counter = Counter()
    kept_lines: list[str] = []

    with input_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            stats["total"] += 1
            cleaned = clean_line(raw_line)
            if cleaned is None:
                stats["dropped"] += 1
                continue
            stats["kept"] += 1
            kept_lines.append(cleaned)
            token_counter.update(tokenize_words(cleaned))

    unique_lines = list(dict.fromkeys(kept_lines))
    stats["deduped"] = len(unique_lines)
    stats["duplicates_removed"] = len(kept_lines) - len(unique_lines)

    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in unique_lines:
            handle.write(line + "\n")

    report = {
        "input": str(input_path),
        "output": str(output_path),
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
        },
    }

    if report_path is not None:
        with report_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Indonesian text for KenLM training.")
    parser.add_argument("--input", required=True, help="Input TSV/plain-text corpus file")
    parser.add_argument("--output", required=True, help="Output cleaned corpus text file")
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    report = clean_corpus(input_path, output_path, report_path)
    print(f"Saved cleaned corpus to: {output_path}")
    print(f"Total lines   : {report['total_lines']}")
    print(f"Kept lines    : {report['kept_lines']}")
    print(f"Dropped lines : {report['dropped_lines']}")
    print(f"Deduped lines  : {report['final_lines']}")
    if report_path is not None:
        print(f"Saved report to: {report_path}")


if __name__ == "__main__":
    main()