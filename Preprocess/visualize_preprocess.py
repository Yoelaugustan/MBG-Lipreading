"""
visualize_preprocessing.py — generate paper figure images showing each step of
the LUMINA preprocessing pipeline applied to a single representative frame.

Outputs (saved to figures/preprocessing/):
    1_raw_frame.png              — original video frame, full resolution
    2_mediapipe_landmarks.png    — same frame with 20 lip landmarks overlaid
    3_bounding_box.png           — same frame with the 35%-padded lip bbox
    4_lip_crop_color.png         — cropped lip region, still RGB
    5_grayscale.png              — cropped lip region, converted to grayscale
    6_resized_88x88.png          — resized to model input resolution
    7_normalized.png             — final normalized tensor visualized as image

Each image is saved at 300 DPI without margins/axes — paper-ready.
You can pick whichever subset works best for your figure layout.

Usage:
    python visualize_preprocessing.py
    python visualize_preprocessing.py path/to/video.MP4
    python visualize_preprocessing.py path/to/video.MP4 --frame 42

Requires:
    pip install opencv-python-headless mediapipe matplotlib numpy
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np


# ─── Config (matches your training preprocessing) ────────────────────────────
ROI_SIZE     = 88
ROI_PADDING  = 0.35
NORM_MEAN    = 0.421
NORM_STD     = 0.165
DPI          = 300

# MediaPipe outer-lip landmark indices (same as preprocess_lumina.py)
LIP_OUTER = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
]


# ─── Save helper — no axes, no padding, paper-ready ──────────────────────────
def save_clean(image: np.ndarray, path: Path, cmap: str = None) -> None:
    """Save a numpy image as a PNG without axes, ticks, or whitespace."""
    h, w = image.shape[:2]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=DPI)
    ax  = fig.add_axes([0, 0, 1, 1])  # no margins
    ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"  saved: {path}")


# ─── Step-by-step preprocessing with intermediate captures ───────────────────
def visualize_pipeline(video_path: Path, frame_idx: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Read the raw frame ───────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        print(f"[warn] frame {frame_idx} out of range — using middle frame instead")
        frame_idx = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr_frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    h, w = rgb_frame.shape[:2]

    print(f"\nProcessing frame {frame_idx} of {video_path.name} ({w}×{h} px)")

    # Save 1: raw frame
    save_clean(rgb_frame, output_dir / "1_raw_frame.png")

    # ── Step 2: MediaPipe lip landmark detection ─────────────────────────────
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    result = face_mesh.process(rgb_frame)
    face_mesh.close()

    if not result.multi_face_landmarks:
        raise RuntimeError(f"No face detected in frame {frame_idx} — try a different frame")

    landmarks = result.multi_face_landmarks[0].landmark
    xs = [landmarks[i].x * w for i in LIP_OUTER]
    ys = [landmarks[i].y * h for i in LIP_OUTER]

    # Save 2: frame with landmarks overlaid as red dots
    frame_with_landmarks = rgb_frame.copy()
    for x, y in zip(xs, ys):
        cv2.circle(frame_with_landmarks, (int(x), int(y)), radius=3,
                   color=(220, 30, 30), thickness=-1)
    save_clean(frame_with_landmarks, output_dir / "2_mediapipe_landmarks.png")

    # ── Step 3: Bounding box with 35% padding ────────────────────────────────
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    x1 = max(0, int(min(xs) - bw * ROI_PADDING))
    y1 = max(0, int(min(ys) - bh * ROI_PADDING))
    x2 = min(w, int(max(xs) + bw * ROI_PADDING))
    y2 = min(h, int(max(ys) + bh * ROI_PADDING))

    # Save 3: frame with bounding box overlaid in green
    frame_with_bbox = rgb_frame.copy()
    cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2),
                  color=(40, 200, 70), thickness=3)
    save_clean(frame_with_bbox, output_dir / "3_bounding_box.png")

    # ── Step 4: Crop the lip region (still color) ────────────────────────────
    lip_color = rgb_frame[y1:y2, x1:x2]
    save_clean(lip_color, output_dir / "4_lip_crop_color.png")

    # ── Step 5: Convert to grayscale ─────────────────────────────────────────
    lip_gray = cv2.cvtColor(lip_color, cv2.COLOR_RGB2GRAY)
    save_clean(lip_gray, output_dir / "5_grayscale.png", cmap="gray")

    # ── Step 6: Resize to 88×88 ──────────────────────────────────────────────
    lip_resized = cv2.resize(lip_gray, (ROI_SIZE, ROI_SIZE),
                             interpolation=cv2.INTER_LINEAR)
    save_clean(lip_resized, output_dir / "6_resized_88x88.png", cmap="gray")

    # ── Step 7: Normalize (visualized — clipped to displayable range) ────────
    # Real normalization produces values roughly in [-2.5, 2.5] which can't be
    # shown directly. We map back to [0, 1] for display purposes only.
    normalized = (lip_resized.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    # Map for visualization: clip to [-2, 2], then rescale to [0, 1]
    norm_display = np.clip(normalized, -2.0, 2.0)
    norm_display = (norm_display + 2.0) / 4.0
    save_clean(norm_display, output_dir / "7_normalized.png", cmap="gray")

    print(f"\nDone. {7} images saved to {output_dir.resolve()}")


# ─── CLI ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate preprocessing visualization images for the paper."
    )
    parser.add_argument(
        "video", nargs="?",
        default="LUMINA_Dataset/male/video/P01_S1.MP4",
        help="Path to a raw LUMINA video file",
    )
    parser.add_argument(
        "--frame", type=int, default=42,
        help="Which frame index to use (default: 42, near the middle of a 84-frame clip)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="figures/preprocessing",
        help="Where to save the output images",
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"[error] Video not found: {video_path}")
        print("Pass a valid path as the first argument.")
        sys.exit(1)

    visualize_pipeline(
        video_path=video_path,
        frame_idx=args.frame,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()