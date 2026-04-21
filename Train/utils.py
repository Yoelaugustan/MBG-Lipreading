"""
utils.py — CTC greedy decode + CER / WER metrics.
"""
import torch


# ──────────────────────────────────────────────────────────────────────────────
# GREEDY CTC DECODE
# ──────────────────────────────────────────────────────────────────────────────
def greedy_ctc_decode(log_probs: torch.Tensor, idx_to_char: dict, blank: int = 0) -> list[str]:
    """
    log_probs : [T, B, V]    (model output — already log-softmaxed)
    returns   : list[str]    length B
    """
    T, B, V = log_probs.shape
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()   # [B, T]

    decoded = []
    for seq in preds:
        chars, prev = [], -1
        for p in seq:
            if p != prev and p != blank:
                ch = idx_to_char.get(p, "")
                if ch in ("<blank>", "<unk>"):
                    continue
                chars.append(ch)
            prev = p
        decoded.append("".join(chars))
    return decoded


# ──────────────────────────────────────────────────────────────────────────────
# EDIT DISTANCE (Levenshtein)
# ──────────────────────────────────────────────────────────────────────────────
def levenshtein(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(cur[j - 1] + 1,       # insertion
                         prev[j] + 1,           # deletion
                         prev[j - 1] + cost)    # substitution
        prev = cur
    return prev[-1]


# ──────────────────────────────────────────────────────────────────────────────
# CER / WER
# ──────────────────────────────────────────────────────────────────────────────
def compute_cer(refs: list[str], hyps: list[str]) -> float:
    total_err, total_len = 0, 0
    for r, h in zip(refs, hyps):
        total_err += levenshtein(list(r), list(h))
        total_len += max(len(r), 1)
    return total_err / total_len


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    total_err, total_len = 0, 0
    for r, h in zip(refs, hyps):
        r_words = r.split()
        h_words = h.split()
        total_err += levenshtein(r_words, h_words)
        total_len += max(len(r_words), 1)
    return total_err / total_len