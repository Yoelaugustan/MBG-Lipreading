"""
dataset.py — LUMINA sentence-level lip reading dataset.

Reads preprocessed .pt files (shape [T, C, H, W], float16) produced by
preprocess_lumina.py and encodes text into character-index CTC targets.
"""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────────────────────────────────────
# LUMINA DATASET
# ──────────────────────────────────────────────────────────────────────────────
class LUMINADataset(Dataset):
    """
    Returns a dict per sample:
        video         : float32 tensor [T, C, H', W']  (augmented if train)
        label         : long tensor   [L]              (character indices)
        label_length  : int
        input_length  : int           = T
        text          : str           (raw sentence, for logging/CER)
    """

    def __init__(self, manifest_df: pd.DataFrame, vocab: dict, cfg, split: str = "train"):
        self.df      = manifest_df.reset_index(drop=True)
        self.vocab   = vocab
        self.cfg     = cfg
        self.split   = split
        self.is_train = (split == "train")

        self.idx_to_char = {v: k for k, v in vocab.items()}
        self.unk_idx     = vocab["<unk>"]

    def __len__(self) -> int:
        return len(self.df)

    def _encode(self, text: str) -> torch.Tensor:
        # Character-level encoding. Lowercased; unknown chars → <unk>.
        ids = [self.vocab.get(ch, self.unk_idx) for ch in text.lower()]
        return torch.tensor(ids, dtype=torch.long)

    # ── Augmentation ──────────────────────────────────────────────────────────
    def _augment(self, video: torch.Tensor) -> torch.Tensor:
        # video: [T, C, H, W]
        T, C, H, W = video.shape
        cfg = self.cfg

        # Random crop (spatial)
        if cfg.aug_random_crop:
            cs = cfg.aug_crop_size
            x1 = random.randint(0, W - cs)
            y1 = random.randint(0, H - cs)
            video = video[:, :, y1:y1 + cs, x1:x1 + cs]
        else:
            # Center crop to cfg.aug_crop_size for shape consistency with train
            cs = cfg.aug_crop_size
            y1 = (H - cs) // 2
            x1 = (W - cs) // 2
            video = video[:, :, y1:y1 + cs, x1:x1 + cs]

        # Horizontal flip
        if cfg.aug_horizontal_flip and random.random() < 0.5:
            video = torch.flip(video, dims=[-1])

        # Temporal masking (zero out a short contiguous block of frames)
        if cfg.aug_time_mask_p > 0 and random.random() < cfg.aug_time_mask_p:
            mask_len = random.randint(1, cfg.aug_time_mask_max)
            t0 = random.randint(0, T - mask_len)
            video[t0:t0 + mask_len] = 0.0

        return video

    def _center_crop(self, video: torch.Tensor) -> torch.Tensor:
        T, C, H, W = video.shape
        cs = self.cfg.aug_crop_size
        y1 = (H - cs) // 2
        x1 = (W - cs) // 2
        return video[:, :, y1:y1 + cs, x1:x1 + cs]

    # ── __getitem__ ───────────────────────────────────────────────────────────
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        video = torch.load(row["pt_path"], weights_only=True).float()   # [T, C, H, W]   (from float16)

        video = self._augment(video) if self.is_train else self._center_crop(video)

        label = self._encode(row["text"])
        return {
            "video"       : video,
            "label"       : label,
            "label_length": label.size(0),
            "input_length": video.size(0),
            "text"        : row["text"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# COLLATE  (CTC expects flat labels + length vectors)
# ──────────────────────────────────────────────────────────────────────────────
def ctc_collate_fn(batch: list[dict]) -> dict:
    videos        = torch.stack([b["video"] for b in batch])                 # [B, T, C, H, W]
    input_lengths = torch.tensor([b["input_length"] for b in batch], dtype=torch.long)
    label_lengths = torch.tensor([b["label_length"] for b in batch], dtype=torch.long)
    labels        = torch.cat([b["label"] for b in batch], dim=0)            # flat  [sum(L)]
    texts         = [b["text"] for b in batch]
    return {
        "videos"        : videos,
        "labels"        : labels,
        "input_lengths" : input_lengths,
        "label_lengths" : label_lengths,
        "texts"         : texts,
    }


# ──────────────────────────────────────────────────────────────────────────────
# DATALOADER BUILDERS
# ──────────────────────────────────────────────────────────────────────────────
def load_vocab(vocab_path: str) -> dict:
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_speaker_split(manifest_df: pd.DataFrame, val_speakers, test_speakers):
    """
    Speaker-independent split.
    Returns (train_df, val_df, test_df).
    """
    val_spk  = set(val_speakers)
    test_spk = set(test_speakers)

    train_df = manifest_df[~manifest_df["speaker_id"].isin(val_spk | test_spk)]
    val_df   = manifest_df[manifest_df["speaker_id"].isin(val_spk)]
    test_df  = manifest_df[manifest_df["speaker_id"].isin(test_spk)]
    return train_df, val_df, test_df


def build_dataloaders(cfg):
    manifest = pd.read_csv(cfg.manifest_csv)
    vocab    = load_vocab(cfg.vocab_json)

    train_df, val_df, test_df = build_speaker_split(
        manifest, cfg.val_speakers, cfg.test_speakers
    )

    print(f"Split sizes — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")

    train_ds = LUMINADataset(train_df, vocab, cfg, split="train")
    val_ds   = LUMINADataset(val_df,   vocab, cfg, split="val")
    test_ds  = LUMINADataset(test_df,  vocab, cfg, split="test")

    common = dict(
        batch_size  = cfg.batch_size,
        num_workers = cfg.num_workers,
        collate_fn  = ctc_collate_fn,
        pin_memory  = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **common)

    return train_loader, val_loader, test_loader, vocab