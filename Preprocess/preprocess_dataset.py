# preprocess_dataset.py
import os
import torch
from pathlib import Path
from tqdm import tqdm
import cv2
import sys

train_folder_path = str(Path(__file__).resolve().parent.parent / "Train")
if train_folder_path not in sys.path:
    sys.path.append(train_folder_path)


from train_lip_reading import LipROIExtractor, VideoAugmentation, Config

config = Config()
extractor = LipROIExtractor(output_size=112)
augmentation = VideoAugmentation(mode='val', img_size=112)  # No augmentation for preprocessing

# Output directory
PREPROCESSED_DIR = Path("preprocessed_videos")
PREPROCESSED_DIR.mkdir(exist_ok=True)

def preprocess_video(video_path):
    """Extract and save lip ROIs for a video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract lip ROI
        lip_roi = extractor.extract_lip_roi(frame)
        lip_roi_gray = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply normalization (no augmentation)
        augmented_frame = augmentation(lip_roi_gray)
        frames.append(augmented_frame)
    
    cap.release()
    
    # Stack and sample to 84 frames
    if len(frames) > 0:
        frames = torch.stack(frames)
        if len(frames) != 84:
            indices = torch.linspace(0, len(frames) - 1, 84).long()
            frames = frames[indices]
        return frames
    return None

# Process all videos
video_files = list(Path(config.DATASET_PATH).rglob("*.MP4"))

for video_file in tqdm(video_files, desc="Preprocessing"):
    output_path = PREPROCESSED_DIR / f"{video_file.stem}.pt"
    
    if output_path.exists():
        continue
    
    frames = preprocess_video(str(video_file))
    if frames is not None:
        torch.save(frames, output_path)

print(f"✓ Preprocessed {len(video_files)} videos to {PREPROCESSED_DIR}")