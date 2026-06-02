#!/usr/bin/env python3
"""Simple KenLM tester.

Usage examples:
  python Train/kenlm_test.py --lm_path KenLM/lm.binary --sentence "ini adalah contoh kalimat"
  python Train/kenlm_test.py --lm_path KenLM/lm.binary --sentence "ini adalah contoh" --unigram_path KenLM/clean_corpus.txt --topk 10

This prints: model order, log10 score, perplexity, and (optionally) best next-word candidates.
"""
import argparse
import math
from pathlib import Path
from collections import Counter


def compute_perplexity(model, sentence: str):
    total_log10 = 0.0
    n_words = 0
    try:
        for log10p, _, _ in model.full_scores(sentence, bos=True, eos=True):
            total_log10 += log10p
            n_words += 1
    except Exception:
        # fallback: use single score as approximation
        total_log10 = model.score(sentence, bos=True, eos=True)
        # approximate n_words by whitespace
        n_words = max(1, len(sentence.split()))

    total_ln = total_log10 * math.log(10)
    perplexity = math.exp(- total_ln / max(1, n_words))
    return perplexity, total_log10, n_words


def main():
    parser = argparse.ArgumentParser(description="Test KenLM binary on sentences and candidate continuations.")
    parser.add_argument("--lm_path", type=str, default="KenLM/lm.binary", help="Path to KenLM binary")
    parser.add_argument("--sentence", type=str, default="ini adalah contoh kalimat untuk diuji", help="Sentence to score")
    parser.add_argument("--unigram_path", type=str, default=None, help="Optional cleaned corpus to extract candidate next words")
    parser.add_argument("--topk", type=int, default=10, help="Number of top candidate continuations to show")
    parser.add_argument("--candidates", type=int, default=200, help="Number of candidate words to consider from corpus (most common)")
    args = parser.parse_args()

    try:
        import kenlm
    except Exception as e:
        print("kenlm Python bindings not available. Install with `conda install -c conda-forge kenlm` or `pip install ./python` from KenLM repo.")
        raise

    lm_path = Path(args.lm_path)
    if not lm_path.exists():
        print(f"LM file not found: {lm_path}")
        return

    print(f"Loading KenLM model from: {lm_path}")
    model = kenlm.Model(str(lm_path))
    try:
        order = model.order()
    except Exception:
        order = None
    print(f"Model order: {order}")

    sent = args.sentence.strip()
    print(f"\nSentence: {sent}")

    score = model.score(sent, bos=True, eos=True)
    perplexity, total_log10, n_words = compute_perplexity(model, sent)
    print(f"log10 score (sentence): {score:.6f}")
    print(f"words counted (approx): {n_words}")
    print(f"perplexity (approx): {perplexity:.6f}")

    # Candidate next-word ranking
    if args.unigram_path:
        u_path = Path(args.unigram_path)
        if not u_path.exists():
            print(f"Unigram corpus not found: {u_path}")
            return
        # build frequency counter
        ctr = Counter()
        with u_path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for w in line.strip().split():
                    if not w:
                        continue
                    ctr[w] += 1
        candidates = [w for w, _ in ctr.most_common(args.candidates)]
        print(f"\nLoaded {len(candidates)} candidate words from {u_path} (top {args.candidates})")

        # score each continuation and compute delta
        base_score = score
        scored = []
        for w in candidates:
            test = sent + " " + w
            s = model.score(test, bos=True, eos=True)
            delta = s - base_score
            scored.append((w, s, delta))

        scored.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop {args.topk} candidate continuations (by delta log10 score):")
        for word, s, delta in scored[: args.topk]:
            print(f"  {word:20s}  score={s:.6f}  delta={delta:.6f}")
    else:
        print("\nNo unigram_path provided; skipping candidate ranking.")


if __name__ == '__main__':
    main()
