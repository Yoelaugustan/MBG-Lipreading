"""
LUMINA Dataset Preprocessing Script
=====================================
Extracts lip ROI from each video using MediaPipe Face Mesh,
normalizes with ImageNet stats, and saves as float16 .pt tensors.

Output structure:
    LUMINA_preprocessed/
        female/
            P11_S1.pt  ...
        male/
            P01_S1.pt  ...
        manifest.csv       <- maps every .pt to its label + metadata
        vocab.json         <- character vocab for CTC decoder
        failed_videos.csv  <- any videos that failed (if any)

Usage:
    pip install mediapipe opencv-python-headless tqdm openpyxl
    python preprocess_lumina.py
"""

import os
import re
import json
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp
from tqdm import tqdm

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  ←  edit these paths before running
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    "dataset_root"  : "/home/flamz/Pre-thesis/LUMINA_Dataset", # Modify Path
    "output_root"   : "LUMINA_preprocessed",
    "label_file"    : "/home/flamz/Pre-thesis/LUMINA_Dataset/list_of_sentence.xlsx", # Modify Path
    "log_file"      : "preprocessing.log",

    # Video settings
    "num_frames"    : 84,      # fixed temporal length  (pad / truncate to this)
    "roi_size"      : 88,      # spatial H = W for lip crop (88×88)
    "roi_padding"   : 0.35,    # fractional padding around lip bounding box

    # MediaPipe
    "min_detection_confidence": 0.5,
    "min_tracking_confidence" : 0.5,
}

# ImageNet normalization (pretrained ResNet-18 frontend expects these)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# MediaPipe Face Mesh — outer lip contour landmark indices
# Covers the full outer boundary of both upper and lower lip
LIP_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
]


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def setup_logging(log_file: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [%(levelname)-8s]  %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# LABEL LOADING
# ──────────────────────────────────────────────────────────────────────────────
def load_labels(xlsx_path: str) -> dict[int, str]:
    """
    Reads list_of_sentence.xlsx.
    Returns {sentence_number (int): sentence_text (str)}.
    """
    df = pd.read_excel(xlsx_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Flexible column matching
    num_col  = next(c for c in df.columns if "number" in c or "num" in c)
    text_col = next(c for c in df.columns if "text" in c)

    labels = {
        int(row[num_col]): str(row[text_col]).strip()
        for _, row in df.iterrows()
    }
    return labels


# ──────────────────────────────────────────────────────────────────────────────
# VIDEO DISCOVERY
# ──────────────────────────────────────────────────────────────────────────────
def collect_videos(dataset_root: str) -> list[dict]:
    """
    Walks LUMINA_Dataset/female/video/ and male/video/.
    Parses filenames of the form  P01_S12.MP4  →  speaker=P01, sentence=12.
    Returns a list of record dicts.
    """
    root    = Path(dataset_root)
    records = []

    for gender in ("female", "male"):
        video_dir = root / gender / "video"
        if not video_dir.exists():
            logging.warning(f"Directory not found, skipping: {video_dir}")
            continue

        mp4_files = sorted(
            list(video_dir.glob("*.MP4")) + list(video_dir.glob("*.mp4"))
        )

        for fp in mp4_files:
            m = re.match(r"(P\d+)_S(\d+)", fp.stem, re.IGNORECASE)
            if not m:
                logging.warning(f"Unrecognised filename pattern, skipping: {fp.name}")
                continue
            records.append({
                "path"        : fp,
                "gender"      : gender,
                "speaker_id"  : m.group(1).upper(),
                "sentence_num": int(m.group(2)),
            })

    logging.info(f"Discovered {len(records)} videos "
                 f"({sum(1 for r in records if r['gender']=='female')} female, "
                 f"{sum(1 for r in records if r['gender']=='male')} male)")
    return records


# ──────────────────────────────────────────────────────────────────────────────
# LIP ROI EXTRACTION  (single frame)
# ──────────────────────────────────────────────────────────────────────────────
def extract_lip_roi(
    frame_rgb : np.ndarray,
    face_mesh,
    roi_size  : int,
    padding   : float,
) -> np.ndarray | None:
    """
    Runs MediaPipe on one RGB frame.
    Returns (roi_size, roi_size, 3) uint8 array, or None if no face found.
    """
    h, w = frame_rgb.shape[:2]
    result = face_mesh.process(frame_rgb)

    if not result.multi_face_landmarks:
        return None

    lm = result.multi_face_landmarks[0].landmark
    xs = [lm[i].x * w for i in LIP_OUTER]
    ys = [lm[i].y * h for i in LIP_OUTER]

    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)

    x1 = max(0, int(min(xs) - bw * padding))
    y1 = max(0, int(min(ys) - bh * padding))
    x2 = min(w, int(max(xs) + bw * padding))
    y2 = min(h, int(max(ys) + bh * padding))

    crop = frame_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


# ──────────────────────────────────────────────────────────────────────────────
# FULL VIDEO PROCESSING
# ──────────────────────────────────────────────────────────────────────────────
def process_video(video_path: Path, face_mesh, cfg: dict) -> torch.Tensor | None:
    """
    Reads all frames from a video, extracts lip ROI per frame,
    pads / truncates to cfg['num_frames'], normalizes, and returns
    a float16 tensor of shape [T, C, H, W] = [84, 3, 88, 88].

    Returns None if the video cannot be opened or yields zero frames.
    """
    T        = cfg["num_frames"]
    roi_size = cfg["roi_size"]
    padding  = cfg["roi_padding"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frames     = []
    last_valid = None   # carry-forward buffer for missed MediaPipe detections

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop = extract_lip_roi(rgb, face_mesh, roi_size, padding)

        if crop is not None:
            last_valid = crop
        elif last_valid is not None:
            # MediaPipe missed this frame — reuse the previous valid crop
            crop = last_valid
        else:
            # No valid frame yet (first frames with no detection)
            # Fall back to a simple center-bottom crop of the raw frame
            fh, fw = rgb.shape[:2]
            cx = fw // 2
            cy = int(fh * 0.78)           # lip sits roughly 78% down the face
            half = roi_size // 2
            y1 = max(0, cy - half)
            y2 = min(fh, cy + half)
            x1 = max(0, cx - half)
            x2 = min(fw, cx + half)
            crop = cv2.resize(rgb[y1:y2, x1:x2], (roi_size, roi_size))

        frames.append(crop)

    cap.release()

    if len(frames) == 0:
        return None

    # Stack → [actual_T, H, W, C]  then permute → [actual_T, C, H, W]
    tensor = torch.from_numpy(np.stack(frames, axis=0)).float()
    tensor = tensor.permute(0, 3, 1, 2) / 255.0            # [T, C, H, W] in [0,1]

    # ImageNet normalisation (for pretrained ResNet-18 frontend)
    tensor = (tensor - _MEAN) / _STD

    # Temporal padding / truncation to exactly T frames
    actual_T = tensor.shape[0]
    if actual_T < T:
        pad    = torch.zeros(T - actual_T, 3, roi_size, roi_size)
        tensor = torch.cat([tensor, pad], dim=0)
    else:
        tensor = tensor[:T]

    # Save as float16 → halves disk usage with negligible precision loss
    return tensor.half()


# ──────────────────────────────────────────────────────────────────────────────
# VOCAB BUILDER
# ──────────────────────────────────────────────────────────────────────────────
def build_vocab(texts: list[str]) -> dict[str, int]:
    """
    Character-level vocabulary for CTC.
    Reserved tokens:
        0  →  <blank>   (CTC blank token — must stay at index 0)
        1  →  <unk>     (unknown character)
        2  →  <space>   (word boundary)
        3+ →  a–z and other characters, sorted
    """
    chars = set()
    for t in texts:
        chars.update(t.lower())
    chars.discard(" ")

    vocab = {"<blank>": 0, "<unk>": 1, " ": 2}
    for idx, ch in enumerate(sorted(chars), start=3):
        vocab[ch] = idx

    return vocab


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = CONFIG
    setup_logging(cfg["log_file"])
    logging.info("LUMINA Preprocessing — starting")
    logging.info(f"Settings: frames={cfg['num_frames']}, "
                 f"roi={cfg['roi_size']}x{cfg['roi_size']}, "
                 f"padding={cfg['roi_padding']}")

    # Output directories
    out_root = Path(cfg["output_root"])
    (out_root / "female").mkdir(parents=True, exist_ok=True)
    (out_root / "male").mkdir(parents=True, exist_ok=True)

    # ── Labels ────────────────────────────────────────────────────────────────
    logging.info(f"Loading labels from: {cfg['label_file']}")
    labels = load_labels(cfg["label_file"])
    logging.info(f"  → {len(labels)} sentence labels loaded")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab      = build_vocab(list(labels.values()))
    vocab_path = out_root / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logging.info(f"Vocabulary saved ({len(vocab)} tokens) → {vocab_path}")

    # ── Discover videos ───────────────────────────────────────────────────────
    records = collect_videos(cfg["dataset_root"])
    if not records:
        logging.error("No videos found — check CONFIG['dataset_root']")
        return

    # ── MediaPipe Face Mesh (single instance, reused per frame) ──────────────
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode        = False,   # video mode: faster tracking
        max_num_faces            = 1,
        refine_landmarks         = True,    # enables detailed lip landmarks
        min_detection_confidence = cfg["min_detection_confidence"],
        min_tracking_confidence  = cfg["min_tracking_confidence"],
    )

    manifest_rows = []
    failed_rows   = []

    # ── Main loop ─────────────────────────────────────────────────────────────
    for rec in tqdm(records, desc="Preprocessing videos", unit="vid", ncols=90):
        sentence_num = rec["sentence_num"]
        text = labels.get(sentence_num)

        if text is None:
            logging.warning(f"No label for S{sentence_num} — skipping {rec['path'].name}")
            failed_rows.append({"file": rec["path"].name, "reason": "missing_label"})
            continue

        out_path = out_root / rec["gender"] / (rec["path"].stem + ".pt")

        # Resume support: skip already-processed files
        if out_path.exists():
            manifest_rows.append({
                "pt_path"     : str(out_path),
                "gender"      : rec["gender"],
                "speaker_id"  : rec["speaker_id"],
                "sentence_num": sentence_num,
                "text"        : text,
            })
            continue

        tensor = process_video(rec["path"], face_mesh, cfg)

        if tensor is None:
            logging.error(f"Processing failed: {rec['path'].name}")
            failed_rows.append({"file": rec["path"].name, "reason": "processing_error"})
            continue

        torch.save(tensor, out_path)
        manifest_rows.append({
            "pt_path"     : str(out_path),
            "gender"      : rec["gender"],
            "speaker_id"  : rec["speaker_id"],
            "sentence_num": sentence_num,
            "text"        : text,
        })

    face_mesh.close()

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest_path = out_root / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    logging.info(f"Manifest saved ({len(manifest_rows)} entries) → {manifest_path}")

    # ── Failed videos ─────────────────────────────────────────────────────────
    if failed_rows:
        failed_path = out_root / "failed_videos.csv"
        pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
        logging.warning(f"{len(failed_rows)} videos failed — see {failed_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    logging.info("─" * 55)
    logging.info(f"  Processed  : {len(manifest_rows)}")
    logging.info(f"  Failed     : {len(failed_rows)}")
    logging.info(f"  Output dir : {out_root.resolve()}")
    logging.info("─" * 55)
    logging.info("Preprocessing complete.")


if __name__ == "__main__":
    main()