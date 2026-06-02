#!/usr/bin/env python3
"""Post-process lip-reading predictions with KenLM.

This script is meant to be used after your model has already produced greedy
predictions (for example from Train/test.py). It reads a CSV with at least:
  - ref: reference sentence
  - hyp: model hypothesis / greedy prediction

It then writes a new CSV with an additional `kenlm_hyp` column and prints
before/after CER and WER.

Example:
  python Train/kenlm_postprocess.py \
      --input_csv Train/runs/lumina_sequential/test_predictions.csv \
      --output_csv Train/runs/lumina_sequential/test_predictions_kenlm.csv \
      --lm_path KenLM/lm.binary \
      --unigram_path KenLM/clean_corpus.txt \
      --per_word_k 10 --beam_size 200 --topk 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kenlm_correct import correct_sentence, load_top_unigrams
from utils import compute_cer, compute_wer


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process model predictions with KenLM.")
    parser.add_argument("--input_csv", type=str, required=True, help="CSV containing ref/hyp columns")
    parser.add_argument("--output_csv", type=str, required=True, help="Where to write corrected predictions")
    parser.add_argument("--lm_path", type=str, default="KenLM/lm.binary", help="KenLM binary path")
    parser.add_argument("--unigram_path", type=str, required=True, help="Cleaned corpus used for candidate words")
    parser.add_argument("--per_word_k", type=int, default=8, help="Candidates per token")
    parser.add_argument("--beam_size", type=int, default=200, help="Beam size used during correction")
    parser.add_argument("--topk", type=int, default=1, help="How many corrected candidates to store per row")
    parser.add_argument("--top_unigrams", type=int, default=20000, help="How many frequent corpus words to load")
    parser.add_argument("--cutoff", type=float, default=0.6, help="difflib cutoff for token candidate generation")
    parser.add_argument(
        "--conservative_weight",
        type=float,
        default=0.0,
        help="Penalty per token change to keep corrections close to the input.",
    )
    parser.add_argument(
        "--max_token_changes",
        type=int,
        default=None,
        help="Hard limit on the number of token changes allowed versus the input.",
    )
    parser.add_argument("--sentence_col", type=str, default="hyp", help="Column to correct")
    parser.add_argument("--ref_col", type=str, default="ref", help="Reference column")
    args = parser.parse_args()

    try:
        import kenlm
    except Exception:
        print("kenlm Python bindings not available. Install with `conda install -c conda-forge kenlm` or `pip install ./python` from KenLM repo.")
        raise

    input_path = Path(args.input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    unigram_path = Path(args.unigram_path)
    if not unigram_path.exists():
        raise FileNotFoundError(f"Unigram corpus not found: {unigram_path}")

    lm_path = Path(args.lm_path)
    if not lm_path.exists():
        raise FileNotFoundError(f"KenLM binary not found: {lm_path}")

    df = pd.read_csv(input_path)
    if args.ref_col not in df.columns or args.sentence_col not in df.columns:
        raise ValueError(f"Input CSV must contain columns `{args.ref_col}` and `{args.sentence_col}`")

    model = kenlm.Model(str(lm_path))
    top_unigrams = load_top_unigrams(unigram_path, top_n=args.top_unigrams)
    print(f"Loaded {len(top_unigrams)} top unigrams")

    refs = df[args.ref_col].astype(str).tolist()
    noisy = df[args.sentence_col].astype(str).tolist()

    before_cer = compute_cer(refs, noisy)
    before_wer = compute_wer(refs, noisy)
    print(f"Before correction: CER={before_cer:.4f} WER={before_wer:.4f}")

    corrected = []
    corrected_best = []
    for sent in noisy:
        top = correct_sentence(
            model,
            sent,
            top_unigrams,
            per_word_k=args.per_word_k,
            beam_size=args.beam_size,
            topk=max(1, args.topk),
            cutoff=args.cutoff,
            conservative_weight=args.conservative_weight,
            max_token_changes=args.max_token_changes,
        )
        corrected.append(top)
        corrected_best.append(top[0][0] if top else sent)

    after_cer = compute_cer(refs, corrected_best)
    after_wer = compute_wer(refs, corrected_best)
    print(f"After correction : CER={after_cer:.4f} WER={after_wer:.4f}")
    print(f"Delta CER        : {after_cer - before_cer:+.4f}")
    print(f"Delta WER        : {after_wer - before_wer:+.4f}")

    out = df.copy()
    out["kenlm_hyp"] = corrected_best
    out["kenlm_topk"] = [
        " | ".join([f"{sent}||{score:.6f}||{perp:.4f}" for sent, score, perp in row])
        for row in corrected
    ]
    out.to_csv(args.output_csv, index=False)
    print(f"Saved corrected CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
