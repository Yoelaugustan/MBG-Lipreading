#!/usr/bin/env python3
"""kenlm_correct.py

Beam-based candidate corrector using KenLM.
Given a noisy sentence, propose per-token candidates (from a unigram corpus)
and run a beam search that keeps top-scoring partial sentences by LM score.

Example:
  python Train/kenlm_correct.py --lm_path KenLM/lm.binary --unigram_path KenLM/clean_corpus.txt \
      --sentence "in adelah conth kahlimat" --per_word_k 10 --beam_size 200 --topk 5

Outputs the top-k corrected sentences with LM scores and perplexities.
"""
import argparse
from collections import Counter
import difflib
import math
from pathlib import Path
from typing import List, Tuple


def compute_perplexity(model, sentence: str):
    total_log10 = 0.0
    n_words = 0
    try:
        for log10p, _, _ in model.full_scores(sentence, bos=True, eos=True):
            total_log10 += log10p
            n_words += 1
    except Exception:
        total_log10 = model.score(sentence, bos=True, eos=True)
        n_words = max(1, len(sentence.split()))
    total_ln = total_log10 * math.log(10)
    perplexity = math.exp(- total_ln / max(1, n_words))
    return perplexity, total_log10, n_words


def load_top_unigrams(unigram_path: Path, top_n: int = 20000) -> List[str]:
    ctr = Counter()
    with unigram_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            for w in line.strip().split():
                if not w:
                    continue
                ctr[w] += 1
    return [w for w, _ in ctr.most_common(top_n)]


def token_edit_distance(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def candidates_for_token(token: str, top_unigrams: List[str], per_word_k: int = 8, cutoff: float = 0.6) -> List[str]:
    # always include the original token as a candidate
    cand = [token]
    matches = difflib.get_close_matches(token, top_unigrams, n=per_word_k, cutoff=cutoff)
    for m in matches:
        if m not in cand:
            cand.append(m)
    return cand[:per_word_k]


def correct_sentence(
    model,
    noisy_sentence: str,
    top_unigrams: List[str],
    per_word_k: int = 8,
    beam_size: int = 200,
    topk: int = 5,
    cutoff: float = 0.6,
    conservative_weight: float = 0.0,
    max_token_changes: int | None = None,
) -> List[Tuple[str, float, float]]:
    """Return top corrected sentences as (sentence, lm_log10_score, perplexity).

    The correction is made milder by penalizing changes away from the noisy
    input. Set `conservative_weight` > 0 to prefer small edits, and/or
    `max_token_changes` to hard-limit how many tokens may change.
    """
    tokens = noisy_sentence.split()
    beams: List[Tuple[str, float]] = [("", 0.0)]
    for step_idx, tok in enumerate(tokens, start=1):
        next_beams = []
        cands = candidates_for_token(tok, top_unigrams, per_word_k=per_word_k, cutoff=cutoff)
        for sent_prefix, _ in beams:
            prefix = sent_prefix.strip()
            for c in cands:
                new_sent = c if prefix == "" else prefix + " " + c
                try:
                    s = model.score(new_sent, bos=True, eos=False)
                except Exception:
                    s = model.score(new_sent, bos=True, eos=True)
                if conservative_weight > 0.0:
                    partial_changes = token_edit_distance(tokens[:step_idx], new_sent.split())
                    s -= conservative_weight * partial_changes
                next_beams.append((new_sent, s))
        next_beams.sort(key=lambda x: x[1], reverse=True)
        beams = next_beams[:beam_size]

    scored = []
    for sent, s in beams[:topk]:
        changes = token_edit_distance(tokens, sent.split())
        if max_token_changes is not None and changes > max_token_changes:
            continue
        if conservative_weight > 0.0:
            s -= conservative_weight * changes
        perp, _, _ = compute_perplexity(model, sent)
        scored.append((sent, s, perp))
    return scored


def beam_search_correct(
    model,
    tokens: List[str],
    top_unigrams: List[str],
    per_word_k: int = 8,
    beam_size: int = 200,
    conservative_weight: float = 0.0,
    max_token_changes: int | None = None,
) -> List[Tuple[str, float]]:
    # beams are tuples (sentence_str, lm_score)
    beams: List[Tuple[str, float]] = [("", 0.0)]
    for step_idx, tok in enumerate(tokens, start=1):
        next_beams = []
        cands = candidates_for_token(tok, top_unigrams, per_word_k=per_word_k)
        for sent_prefix, prefix_score in beams:
            prefix = sent_prefix.strip()
            for c in cands:
                if prefix == "":
                    new_sent = c
                else:
                    new_sent = prefix + " " + c
                # score partial sentence; do not force EOS so partials score reasonably
                try:
                    s = model.score(new_sent, bos=True, eos=False)
                except Exception:
                    s = model.score(new_sent, bos=True, eos=True)
                if conservative_weight > 0.0:
                    partial_changes = token_edit_distance(tokens[:step_idx], new_sent.split())
                    s -= conservative_weight * partial_changes
                next_beams.append((new_sent, s))
        # keep top beam_size by score
        next_beams.sort(key=lambda x: x[1], reverse=True)
        beams = next_beams[:beam_size]
    # final beams are full sentences; return them
    if max_token_changes is not None:
        beams = [
            (sent, score)
            for sent, score in beams
            if token_edit_distance(tokens, sent.split()) <= max_token_changes
        ]
    return beams


def correct_sentences(
    model,
    noisy_sentences: List[str],
    top_unigrams: List[str],
    per_word_k: int = 8,
    beam_size: int = 200,
    topk: int = 1,
    cutoff: float = 0.6,
    conservative_weight: float = 0.0,
    max_token_changes: int | None = None,
) -> List[List[Tuple[str, float, float]]]:
    """Correct multiple sentences."""
    return [
        correct_sentence(
            model,
            sent,
            top_unigrams,
            per_word_k=per_word_k,
            beam_size=beam_size,
            topk=topk,
            cutoff=cutoff,
            conservative_weight=conservative_weight,
            max_token_changes=max_token_changes,
        )
        for sent in noisy_sentences
    ]


def main():
    parser = argparse.ArgumentParser(description="Correct a noisy sentence using KenLM and unigram candidates")
    parser.add_argument("--lm_path", type=str, default="KenLM/lm.binary", help="KenLM binary path")
    parser.add_argument("--unigram_path", type=str, required=True, help="Cleaned corpus for unigram candidates")
    parser.add_argument("--sentence", type=str, default=None, help="Noisy sentence to correct")
    parser.add_argument("--input_file", type=str, default=None, help="File with one noisy sentence per line")
    parser.add_argument("--per_word_k", type=int, default=8, help="Candidates per token")
    parser.add_argument("--beam_size", type=int, default=200, help="Beam size (pruning) during search")
    parser.add_argument("--topk", type=int, default=5, help="How many top corrections to show")
    parser.add_argument("--top_unigrams", type=int, default=20000, help="How many top unigrams to load from corpus")
    parser.add_argument("--cutoff", type=float, default=0.6, help="difflib cutoff for close matches (0..1)")
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
    args = parser.parse_args()

    try:
        import kenlm
    except Exception:
        print("kenlm Python bindings not available. Install with `conda install -c conda-forge kenlm` or `pip install ./python` from KenLM repo.")
        raise

    lm_path = Path(args.lm_path)
    if not lm_path.exists():
        print(f"LM binary not found: {lm_path}")
        return

    uni_path = Path(args.unigram_path)
    if not uni_path.exists():
        print(f"Unigram corpus not found: {uni_path}")
        return

    model = kenlm.Model(str(lm_path))
    top_unigrams = load_top_unigrams(uni_path, top_n=args.top_unigrams)
    print(f"Loaded {len(top_unigrams)} top unigrams")

    # gather sentences to correct
    sentences = []
    if args.sentence:
        sentences.append(args.sentence.strip())
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8", errors="ignore") as fh:
            for ln in fh:
                if ln.strip():
                    sentences.append(ln.strip())
    if not sentences:
        print("No input sentences provided. Use --sentence or --input_file.")
        return

    for noisy in sentences:
        tokens = noisy.split()
        print("\nNoisy:", noisy)
        beams = beam_search_correct(
            model,
            tokens,
            top_unigrams,
            per_word_k=args.per_word_k,
            beam_size=args.beam_size,
            conservative_weight=args.conservative_weight,
            max_token_changes=args.max_token_changes,
        )
        # compute final scores and perplexities for topk beams
        out = []
        for sent, s in beams[: args.topk]:
            perp, total_log10, n_words = compute_perplexity(model, sent)
            out.append((sent, s, perp))
        print(f"Top {args.topk} corrections:")
        for sent, s, perp in out:
            print(f"  {sent}\n    LM_log10={s:.6f}  perplexity={perp:.4f}")


if __name__ == '__main__':
    main()
