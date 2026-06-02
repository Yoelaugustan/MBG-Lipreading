"""
kenlm_decoder.py - Integrate KenLM binary LM with pyctcdecode for beam decoding.

Uspython Train/kenlm_decoder.py --split val --checkpoint Train/runs/luminpython Train/kenlm_decoder.py --split val --checkpoint Train/runs/lumina_sequential/best.pt --beam_widths 25 --lm_alphas 0.4 --debuge:
    from kenlm_decoder import KenLMDecoder
    decoder = KenLMDecoder(vocab_path="LUMINA_preprocessed/vocab.json", 
                            lm_path="KenLM/lm.binary")
    hyp = decoder.decode(log_probs, beam_width=50, lm_alpha=0.8)

Or validate/tune on a split:
    python kenlm_decoder.py --split val --checkpoint Train/runs/lumina_sequential/best.pt \
        --beam_width 50 --lm_alpha 0.8
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from pyctcdecode import build_ctcdecoder
    HAS_PYCTCDECODE = True
except ImportError:
    HAS_PYCTCDECODE = False
    print("Warning: pyctcdecode not installed. Install with: pip install pyctcdecode")

from config import get_config
from dataset import build_dataloaders
from model import LUMINAModel
from utils import compute_cer, compute_wer, greedy_ctc_decode, levenshtein


def load_unigrams_from_corpus(
    corpus_path: str | Path,
    allowed_chars: set[str],
    max_words: int | None = None,
) -> list[str]:
    """Load unique words from a cleaned KenLM corpus for pyctcdecode unigrams.

    Words are filtered so every character can be produced by the acoustic model.
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        return []

    seen = set()
    unigrams: list[str] = []
    with corpus_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            for word in line.strip().split():
                word = word.strip()
                if not word or word in seen:
                    continue
                # Skip special tokens and purely numeric tokens
                if word == "<unk>" or word.isdigit():
                    continue
                # Ensure every character in the word is representable by the acoustic alphabet
                if any(ch not in allowed_chars for ch in word):
                    continue
                # Accept the word
                seen.add(word)
                unigrams.append(word)
                if max_words is not None and len(unigrams) >= max_words:
                    return unigrams
    return unigrams


class KenLMDecoder:
    """Wrapper around pyctcdecode + KenLM binary for CTC decoding."""

    def __init__(
        self,
        vocab_path: str | Path = "LUMINA_preprocessed/vocab.json",
        lm_path: str | Path = "KenLM/lm.binary",
        unigram_path: str | Path | None = None,
        alpha: float = 0.5,
        beta: float = 1.5,
        debug: bool = False,
        **kwargs  # passed to build_ctcdecoder
    ):
        """
        Initialize the KenLM decoder.
        
        Args:
            vocab_path: Path to vocab.json (character -> index mapping).
            lm_path: Path to compiled KenLM .binary file.
            **kwargs: Additional arguments for build_ctcdecoder (e.g., alpha, beta, unk_score_offset).
        """
        if not HAS_PYCTCDECODE:
            raise ImportError(
                "pyctcdecode is required. Install with: pip install pyctcdecode"
            )

        # Load vocab and build character list in index order
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        # Build idx -> char and char_list in index order
        self.idx_to_char = {v: k for k, v in vocab.items()}
        max_idx = max(vocab.values()) if vocab else 0
        self.char_list = [""] * (max_idx + 1)
        for char, idx in vocab.items():
            self.char_list[idx] = char

        # Decoder-side labels exclude the explicit blank token.
        # pyctcdecode treats the blank as implicit, so the model outputs are reordered
        # to [<unk>, space, a, b, ..., z, <blank>] before decoding.
        self.decoder_labels = [token for token in self.char_list if token != "<blank>"]
        self.allowed_chars = {token for token in self.decoder_labels if len(token) == 1}
        self.debug = debug
        self._debug_printed = False

        unigrams = None
        if unigram_path is not None:
            unigrams = load_unigrams_from_corpus(unigram_path, allowed_chars=self.allowed_chars)
            print(f"Loaded {len(unigrams)} unigrams from {unigram_path}")
            if len(unigrams) == 0:
                print("Warning: no unigrams retained after filtering; proceeding without unigrams.")
                unigrams = None

        # Verify that we can read the LM binary
        lm_path = Path(lm_path)
        if not lm_path.exists():
            raise FileNotFoundError(f"KenLM binary not found: {lm_path}")

        try:
            self.decoder = build_ctcdecoder(
                self.decoder_labels,
                str(lm_path),
                unigrams=unigrams,
                alpha=alpha,
                beta=beta,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build ctcdecoder with {lm_path}. "
                f"Make sure the LM binary is valid and was built from the same corpus. Error: {e}"
            )

    def decode(
        self,
        log_probs: np.ndarray,
        beam_width: int = 50,
    ) -> str:
        """
        Decode log probabilities using KenLM + beam search.
        
        Args:
            log_probs: Shape [T, V] numpy array of log probabilities.
            beam_width: Beam width for search.
            lm_alpha: Weight for LM score (higher = more LM influence).
            
        Returns:
            Decoded text string.
        """
        # Reorder from model vocab order [<blank>, <unk>, space, a, b, ..., z]
        # to decoder order [<unk>, space, a, b, ..., z, <blank>].
        orig_logits = np.asarray(log_probs, dtype=float)
        logits = orig_logits.copy()
        if logits.ndim != 2:
            raise ValueError(f"Expected a 2D [T, V] array for one sample, got shape {logits.shape}")
        if logits.shape[1] == len(self.char_list):
            logits = np.concatenate([logits[:, 1:], logits[:, :1]], axis=1)
        elif logits.shape[1] != len(self.decoder_labels) + 1:
            raise ValueError(
                f"Unexpected vocab size {logits.shape[1]}. Expected {len(self.char_list)} from the model "
                f"or {len(self.decoder_labels) + 1} after reordering."
            )

        # pyctcdecode expects probabilities per timestep. Detect whether `logits`
        # are already probabilities (rows sum to ~1), otherwise apply a stable
        # softmax per timestep to convert scores/log-probs -> probs.
        probs = None
        if np.all((logits >= 0) & (logits <= 1)) and np.allclose(logits.sum(axis=1), 1.0, atol=1e-3):
            probs = logits
            converted = False
        else:
            # stable softmax
            m = logits.max(axis=1, keepdims=True)
            exps = np.exp(logits - m)
            sums = exps.sum(axis=1, keepdims=True)
            probs = exps / sums
            converted = True

        # One-time debug print showing top tokens and greedy baseline for first sample
        if self.debug and not self._debug_printed:
            try:
                col_labels = self.decoder_labels + ["<blank>"]
                first_probs = probs[0]
                topk = np.argsort(first_probs)[-8:][::-1]
                print("[kenlm_decoder debug] sample shape:", probs.shape)
                print("[kenlm_decoder debug] converted_to_probs:", converted)
                print("[kenlm_decoder debug] top tokens (first timestep):")
                for idx in topk:
                    lbl = col_labels[idx] if idx < len(col_labels) else f"idx_{idx}"
                    print(f"  {lbl}: {first_probs[idx]:.4f}")
                # compute greedy string from original model ordering
                pred_idxs = orig_logits.argmax(axis=1).tolist()
                s = []
                prev = None
                for p in pred_idxs:
                    if p == prev:
                        continue
                    prev = p
                    if p == 0:
                        continue
                    s.append(self.idx_to_char.get(int(p), ""))
                greedy_sample = "".join(s)
                print("[kenlm_decoder debug] greedy (sample):", greedy_sample)
            except Exception as e:
                print("[kenlm_decoder debug] failed to print diagnostics:", e)
            self._debug_printed = True

        try:
            text = self.decoder.decode(probs, beam_width=beam_width)
            return text
        except Exception as e:
            print(f"Decoding failed for a batch: {e}. Falling back to empty string.")
            return ""


def decode_batch(
    model,
    loader,
    device,
    decoder: KenLMDecoder,
    beam_width: int = 50,
    use_amp: bool = True,
):
    """
    Run KenLM decoding on a full dataset loader.
    
    Returns:
        refs: List of reference texts.
        hyps: List of hypothesis texts.
        sample_data: List of dicts with per-sample metrics.
    """
    model.eval()
    refs = []
    hyps = []
    sample_data = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[KenLM decode bw={beam_width}]", ncols=100):
            videos = batch["videos"].to(device)
            labels = batch["labels"]
            input_lengths = batch["input_lengths"]
            label_lengths = batch["label_lengths"]
            batch_refs = batch["texts"]

            # Forward pass
            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=use_amp):
                    log_probs = model(videos)
            else:
                log_probs = model(videos)

            # Decode each sample in the batch. Model output is [T, B, V], so we
            # need to slice along the batch axis to get one [T, V] sequence per sample.
            log_probs_cpu = log_probs.cpu().numpy()
            batch_hyps = []
            for i in range(log_probs_cpu.shape[1]):
                log_prob_t = log_probs_cpu[:, i, :]
                hyp = decoder.decode(log_prob_t, beam_width=beam_width)
                batch_hyps.append(hyp)

            # One-time per-batch debug: show ref, greedy (from argmax), and LM hyp for a few samples
            if getattr(decoder, "debug", False) and not getattr(decoder, "_batch_debug_printed", False):
                try:
                    n_show = min(5, len(batch_hyps))
                    print("[kenlm_decoder debug] sample diagnostics (first batch):")
                    for i in range(n_show):
                        ref = batch_refs[i]
                        # compute greedy using the shared utility to match baseline
                        batch_greedy = greedy_ctc_decode(log_probs, decoder.idx_to_char, blank=0)
                        greedy_sample = batch_greedy[i]
                        lm_sample = batch_hyps[i]
                        print(f"  REF: {ref}")
                        print(f"  GREEDY: {greedy_sample}")
                        print(f"  LM_HYP: {lm_sample}")
                        print("  ---")
                except Exception as e:
                    print("[kenlm_decoder debug] failed to print batch diagnostics:", e)
                decoder._batch_debug_printed = True

            refs.extend(batch_refs)
            hyps.extend(batch_hyps)

            # Compute per-sample metrics
            for ref, hyp in zip(batch_refs, batch_hyps):
                char_ed = levenshtein(list(ref), list(hyp))
                char_len = max(len(ref), 1)
                word_ref = ref.split()
                word_hyp = hyp.split()
                word_ed = levenshtein(word_ref, word_hyp)
                word_len = max(len(word_ref), 1)

                sample_data.append({
                    "ref": ref,
                    "hyp": hyp,
                    "char_cer": char_ed / char_len,
                    "word_wer": word_ed / word_len,
                })

    return refs, hyps, sample_data


def main():
    parser = argparse.ArgumentParser(
        description="Validate KenLM decoder on a split with optional hyperparameter sweep."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Train/runs/lumina_sequential/best.pt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="LUMINA_preprocessed/vocab.json",
        help="Path to vocab.json.",
    )
    parser.add_argument(
        "--lm_path",
        type=str,
        default="KenLM/lm.binary",
        help="Path to KenLM binary file.",
    )
    parser.add_argument(
        "--unigram_path",
        type=str,
        default="KenLM/clean_corpus.txt",
        help="Path to the cleaned LM corpus used for unigram extraction.",
    )
    parser.add_argument(
        "--beam_widths",
        type=int,
        nargs="+",
        default=[25, 50, 100],
        help="Beam widths to try.",
    )
    parser.add_argument(
        "--lm_alphas",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="LM alpha weights to try.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Train/runs/lumina_sequential",
        help="Output directory for results.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use AMP.")
    parser.add_argument("--debug", action="store_true", default=False, help="Print debug diagnostics for the decoder.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config and model
    cfg = get_config()
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    
    # Load vocab to get vocab_size
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    model = LUMINAModel(vocab_size=vocab_size, cfg=cfg)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Build dataloaders (returns train_loader, val_loader, test_loader, vocab)
    train_loader, val_loader, test_loader, ds_vocab = build_dataloaders(cfg)
    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loaders[args.split]
    print(f"Evaluating on {args.split} split with {len(loader.dataset)} samples")

    # Initialize decoder
    print(f"Loading KenLM decoder from {args.lm_path}...")
    decoder = KenLMDecoder(vocab_path=args.vocab_path, lm_path=args.lm_path, unigram_path=args.unigram_path, debug=args.debug)
    print(f"Decoder labels: {decoder.decoder_labels}")
    print(f"Decoder label count: {len(decoder.decoder_labels)} (blank is implicit)")
    print("KenLM decoder ready.")

    # Run greedy baseline
    print("\n=== Greedy Baseline (no LM) ===")
    from utils import greedy_ctc_decode
    idx_to_char = {v: k for k, v in vocab.items()}
    greedy_refs = []
    greedy_hyps = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="[greedy]", ncols=100):
            videos = batch["videos"].to(device)
            if device.type == "cuda":
                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    log_probs = model(videos)
            else:
                log_probs = model(videos)
            batch_hyps = greedy_ctc_decode(log_probs, idx_to_char, blank=0)
            greedy_hyps.extend(batch_hyps)
            greedy_refs.extend(batch["texts"])

    greedy_cer = compute_cer(greedy_refs, greedy_hyps)
    greedy_wer = compute_wer(greedy_refs, greedy_hyps)
    print(f"Greedy CER: {greedy_cer:.4f}, WER: {greedy_wer:.4f}")

    greedy_rows = []
    for ref, hyp in zip(greedy_refs, greedy_hyps):
        char_ed = levenshtein(list(ref), list(hyp))
        char_len = max(len(ref), 1)
        word_ref = ref.split()
        word_hyp = hyp.split()
        word_ed = levenshtein(word_ref, word_hyp)
        word_len = max(len(word_ref), 1)
        greedy_rows.append({
            "greedy_ref": ref,
            "greedy_hyp": hyp,
            "greedy_char_cer": char_ed / char_len,
            "greedy_word_wer": word_ed / word_len,
        })

    # Sweep beam widths and LM alphas
    print(f"\n=== KenLM Sweep ===")
    results = []
    per_sample_rows = []

    decoder_cache: dict[float, KenLMDecoder] = {}

    for beam_width in args.beam_widths:
        for lm_alpha in args.lm_alphas:
            if lm_alpha not in decoder_cache:
                decoder_cache[lm_alpha] = KenLMDecoder(
                    vocab_path=args.vocab_path,
                    lm_path=args.lm_path,
                    unigram_path=args.unigram_path,
                    alpha=lm_alpha,
                    debug=args.debug,
                )
            sweep_decoder = decoder_cache[lm_alpha]

            refs, hyps, sample_data = decode_batch(
                model,
                loader,
                device,
                sweep_decoder,
                beam_width=beam_width,
                use_amp=args.use_amp,
            )
            cer = compute_cer(refs, hyps)
            wer = compute_wer(refs, hyps)
            results.append({
                "beam_width": beam_width,
                "lm_alpha": lm_alpha,
                "cer": cer,
                "wer": wer,
            })
            for greedy_row, row in zip(greedy_rows, sample_data):
                per_sample_rows.append({
                    "beam_width": beam_width,
                    "lm_alpha": lm_alpha,
                    **greedy_row,
                    **row,
                })
            print(f"  bw={beam_width:3d} α={lm_alpha:.1f} => CER={cer:.4f} WER={wer:.4f} (vs greedy: Δ_CER={cer-greedy_cer:+.4f} Δ_WER={wer-greedy_wer:+.4f})")

    # Find best settings
    best = min(results, key=lambda x: x["cer"])
    print(f"\nBest validation CER: {best['cer']:.4f} at beam_width={best['beam_width']}, lm_alpha={best['lm_alpha']}")
    
    # Save sweep results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_file = output_dir / "kenlm_sweep.json"
    with open(sweep_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved sweep results to {sweep_file}")

    if per_sample_rows:
        per_sample_df = pd.DataFrame(per_sample_rows)
        per_sample_csv = output_dir / "kenlm_predictions.csv"
        per_sample_df.to_csv(per_sample_csv, index=False)
        print(f"Saved per-sample predictions to {per_sample_csv}")


if __name__ == "__main__":
    main()
