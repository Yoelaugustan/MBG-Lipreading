import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# For Mamba - you need to install: pip install mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    print("Warning: mamba-ssm not installed. Install with: pip install mamba-ssm")
    MAMBA_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =============================================================P========
# CONFIGURATION
# =====================================================================

class Config:
    # Paths
    DATASET_PATH = "C:\\Users\\Yoel\\Documents\\Binus\\Pre-Thesis\\LUMINA (Linguistic Unified Multimodal Indonesian Natural Audio-Visual)"  # CHANGE THIS
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"
    
    # Data
    IMG_SIZE = 112  # Lip ROI size
    SEQUENCE_LENGTH = 84  # Fixed from EDA
    NUM_FRAMES_SAMPLE = 84  # Can downsample if needed (e.g., 42)
    
    # Model Architecture
    CNN_BACKBONE = 'resnet18'  # or 'resnet34'
    CNN_PRETRAINED = True
    MAMBA_LAYERS = 6
    MAMBA_D_MODEL = 512
    MAMBA_D_STATE = 16
    BIGRU_HIDDEN = 512
    BIGRU_LAYERS = 2
    DROPOUT = 0.3
    
    # Training
    BATCH_SIZE = 8
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    GRAD_CLIP = 1.0
    
    # Optimization
    OPTIMIZER = 'adamw'
    LR_SCHEDULER = 'cosine'  # 'cosine' or 'step'
    WARMUP_EPOCHS = 5
    
    # Data Split (speaker-independent)
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Hardware
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Vocabulary (Indonesian characters + special tokens)
    VOCAB = list("abcdefghijklmnopqrstuvwxyz .,?!'")
    BLANK_TOKEN = '<BLANK>'
    
    def __init__(self):
        self.vocab_to_idx = {char: idx for idx, char in enumerate(self.VOCAB)}
        self.vocab_to_idx[self.BLANK_TOKEN] = len(self.VOCAB)
        self.idx_to_vocab = {idx: char for char, idx in self.vocab_to_idx.items()}
        self.num_classes = len(self.vocab_to_idx)
        
        # Create directories
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

config = Config()
print(f"Device: {config.DEVICE}")
print(f"Vocabulary size: {config.num_classes}")
print(f"Mamba available: {MAMBA_AVAILABLE}")

# =====================================================================
# DATA PREPROCESSING & AUGMENTATION
# =====================================================================

class LipROIExtractor:
    """Extract lip region from face video frames using MediaPipe"""
    
    def __init__(self, output_size=112):
        import mediapipe as mp
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        self.output_size = output_size
        
        # Lip landmark indices (MediaPipe Face Mesh)
        self.LIP_INDICES = list(range(61, 68)) + list(range(146, 181)) + \
                           list(range(185, 191)) + list(range(314, 318)) + \
                           list(range(375, 381)) + list(range(402, 406))
    
    def extract_lip_roi(self, frame):
        """Extract lip ROI from a single frame"""
        h, w = frame.shape[:2]
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face mesh
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            # If detection fails, return center crop
            center_x, center_y = w // 2, int(h * 0.7)
            x1 = max(0, center_x - self.output_size // 2)
            y1 = max(0, center_y - self.output_size // 2)
            x2 = min(w, x1 + self.output_size)
            y2 = min(h, y1 + self.output_size)
            return cv2.resize(frame[y1:y2, x1:x2], (self.output_size, self.output_size))
        
        # Get lip landmarks
        landmarks = results.multi_face_landmarks[0]
        lip_points = []
        
        for idx in self.LIP_INDICES:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                lip_points.append([int(lm.x * w), int(lm.y * h)])
        
        lip_points = np.array(lip_points)
        
        # Get bounding box with margin
        x_min, y_min = lip_points.min(axis=0)
        x_max, y_max = lip_points.max(axis=0)
        
        margin = 0.3  # 30% margin
        width = x_max - x_min
        height = y_max - y_min
        
        x_min = max(0, int(x_min - width * margin))
        y_min = max(0, int(y_min - height * margin))
        x_max = min(w, int(x_max + width * margin))
        y_max = min(h, int(y_max + height * margin))
        
        # Crop and resize
        lip_roi = frame[y_min:y_max, x_min:x_max]
        lip_roi = cv2.resize(lip_roi, (self.output_size, self.output_size))
        
        return lip_roi
    
    def __del__(self):
        self.face_mesh.close()


class VideoAugmentation:
    """Augmentation pipeline for video frames"""
    
    def __init__(self, mode='train', img_size=112):
        self.mode = mode
        
        if mode == 'train':
            self.transform = A.Compose([
                A.RandomCrop(height=int(img_size * 0.9), width=int(img_size * 0.9), p=0.5),
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(mean=[0.5], std=[0.5]),  # Grayscale normalization
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2()
            ])
    
    def __call__(self, frame):
        """Apply augmentation to a single frame (grayscale)"""
        # Ensure grayscale
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Add channel dimension for albumentation
        frame = np.expand_dims(frame, axis=-1)
        
        # Apply transform
        augmented = self.transform(image=frame)
        return augmented['image']


# =====================================================================
# DATASET
# =====================================================================

class LUMINADataset(Dataset):
    """LUMINA Dataset for Lip Reading"""
    
    def __init__(self, video_files, labels, mode='train', config=None):
        """
        Args:
            video_files: List of video file paths
            labels: List of text labels
            mode: 'train', 'val', or 'test'
            config: Configuration object
        """
        self.video_files = video_files
        self.labels = labels
        self.mode = mode
        self.config = config or Config()
        
        # Initialize preprocessors
        self.lip_extractor = LipROIExtractor(output_size=self.config.IMG_SIZE)
        self.augmentation = VideoAugmentation(mode=mode, img_size=self.config.IMG_SIZE)
        
        print(f"{mode.upper()} Dataset: {len(self.video_files)} samples")
    
    def text_to_indices(self, text):
        """Convert text to character indices"""
        text = text.lower().strip()
        indices = []
        for char in text:
            if char in self.config.vocab_to_idx:
                indices.append(self.config.vocab_to_idx[char])
        return torch.LongTensor(indices)
    
    def load_video(self, video_path):
        """Load video and extract lip ROI from each frame"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract lip ROI
            lip_roi = self.lip_extractor.extract_lip_roi(frame)
            
            # Convert to grayscale
            lip_roi_gray = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply augmentation
            augmented_frame = self.augmentation(lip_roi_gray)
            
            frames.append(augmented_frame)
        
        cap.release()
        
        # Stack frames: (T, C, H, W)
        frames = torch.stack(frames)
        
        # Temporal sampling if needed
        if len(frames) != self.config.NUM_FRAMES_SAMPLE:
            indices = torch.linspace(0, len(frames) - 1, self.config.NUM_FRAMES_SAMPLE).long()
            frames = frames[indices]
        
        return frames
    
    def __len__(self):
        return len(self.video_files)
    
    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label_text = self.labels[idx]
        
        # Load video frames
        frames = self.load_video(video_path)  # (T, C, H, W)
        
        # Convert label to indices
        label_indices = self.text_to_indices(label_text)
        
        return {
            'frames': frames,
            'label': label_indices,
            'label_length': len(label_indices),
            'video_path': video_path
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences"""
    frames = torch.stack([item['frames'] for item in batch])  # (B, T, C, H, W)
    
    # Pad labels to same length
    labels = [item['label'] for item in batch]
    label_lengths = torch.LongTensor([item['label_length'] for item in batch])
    
    # Pad labels
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    
    return {
        'frames': frames,
        'labels': labels_padded,
        'label_lengths': label_lengths,
        'input_lengths': torch.LongTensor([frames.size(1)] * len(batch))
    }


# =====================================================================
# MODEL ARCHITECTURE
# =====================================================================

class ResNet3D_Frontend(nn.Module):
    """3D ResNet for spatiotemporal feature extraction"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Simple 3D CNN (you can replace with torchvision.models.video.r3d_18)
        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        # ResNet blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                                stride=(1, stride, stride), padding=1))
        layers.append(nn.BatchNorm3d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(num_blocks - 1):
            layers.append(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)  # (B, C, T, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, C, T)
        x = x.permute(0, 2, 1)  # (B, T, C)
        
        return x


class MambaEncoder(nn.Module):
    """Mamba-based temporal encoder"""
    
    def __init__(self, d_model=512, n_layers=6, d_state=16):
        super().__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm not installed. Install with: pip install mamba-ssm")
        
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state) 
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: (B, T, D)
        for layer in self.layers:
            x = layer(x) + x  # Residual connection
        x = self.norm(x)
        return x


class BiGRUDecoder(nn.Module):
    """Bidirectional GRU decoder"""
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super().__init__()
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x: (B, T, D)
        x, _ = self.bigru(x)  # (B, T, hidden*2)
        x = self.dropout(x)
        x = self.fc(x)  # (B, T, num_classes)
        return F.log_softmax(x, dim=-1)


class LipReadingModel(nn.Module):
    """Complete Lip Reading Model: 3D CNN + Mamba + Bi-GRU + CTC"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Visual Frontend
        self.cnn_frontend = ResNet3D_Frontend(pretrained=config.CNN_PRETRAINED)
        
        # Mamba Encoder
        if MAMBA_AVAILABLE:
            self.mamba_encoder = MambaEncoder(
                d_model=config.MAMBA_D_MODEL,
                n_layers=config.MAMBA_LAYERS,
                d_state=config.MAMBA_D_STATE
            )
        else:
            # Fallback to Transformer if Mamba not available
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.MAMBA_D_MODEL,
                nhead=8,
                dim_feedforward=2048,
                dropout=config.DROPOUT,
                batch_first=True
            )
            self.mamba_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.MAMBA_LAYERS)
        
        # Bi-GRU Decoder
        self.bigru_decoder = BiGRUDecoder(
            input_size=config.MAMBA_D_MODEL,
            hidden_size=config.BIGRU_HIDDEN,
            num_layers=config.BIGRU_LAYERS,
            num_classes=config.num_classes,
            dropout=config.DROPOUT
        )
    
    def forward(self, x):
        # x: (B, T, C, H, W)
        
        # Extract visual features
        x = self.cnn_frontend(x)  # (B, T, 512)
        
        # Temporal encoding with Mamba
        x = self.mamba_encoder(x)  # (B, T, 512)
        
        # Sequence prediction with Bi-GRU
        x = self.bigru_decoder(x)  # (B, T, num_classes)
        
        return x  # Log probabilities


# =====================================================================
# TRAINING UTILITIES
# =====================================================================

class CTCLoss(nn.Module):
    """CTC Loss wrapper"""
    
    def __init__(self):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: (B, T, C) - log probabilities from model
            targets: (B, S) - target sequences (padded)
            input_lengths: (B,) - actual lengths of inputs
            target_lengths: (B,) - actual lengths of targets
        """
        # CTC expects (T, B, C)
        log_probs = log_probs.permute(1, 0, 2)
        
        # Flatten targets (remove padding)
        targets_flat = []
        for i, length in enumerate(target_lengths):
            targets_flat.extend(targets[i, :length].tolist())
        targets_flat = torch.LongTensor(targets_flat).to(log_probs.device)
        
        loss = self.ctc_loss(log_probs, targets_flat, input_lengths, target_lengths)
        return loss


class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_loss = 0.0
        self.count = 0
    
    def update(self, loss, batch_size):
        self.total_loss += loss * batch_size
        self.count += batch_size
    
    def get_average(self):
        return self.total_loss / self.count if self.count > 0 else 0.0


def decode_predictions(log_probs, config):
    """Greedy CTC decoding"""
    predictions = log_probs.argmax(dim=-1)  # (B, T)
    decoded = []
    
    for pred in predictions:
        # Remove consecutive duplicates and blanks
        pred_chars = []
        prev_idx = None
        
        for idx in pred.tolist():
            if idx != config.vocab_to_idx[config.BLANK_TOKEN] and idx != prev_idx:
                if idx in config.idx_to_vocab:
                    pred_chars.append(config.idx_to_vocab[idx])
            prev_idx = idx
        
        decoded.append(''.join(pred_chars))
    
    return decoded


# =====================================================================
# TRAINER
# =====================================================================

class Trainer:
    """Training engine"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Loss
        self.criterion = CTCLoss()
        
        # Optimizer
        if config.OPTIMIZER == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config.LEARNING_RATE,
                weight_decay=config.WEIGHT_DECAY
            )
        
        # Learning rate scheduler
        if config.LR_SCHEDULER == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        
        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.WARMUP_EPOCHS
        )
        
        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS} [TRAIN]")
        
        for batch in pbar:
            # Move to device
            frames = batch['frames'].to(self.config.DEVICE)
            labels = batch['labels'].to(self.config.DEVICE)
            label_lengths = batch['label_lengths']
            input_lengths = batch['input_lengths']
            
            # Forward pass
            self.optimizer.zero_grad()
            log_probs = self.model(frames)  # (B, T, C)
            
            # Compute loss
            loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            
            # Update weights
            self.optimizer.step()
            
            # Track metrics
            metrics.update(loss.item(), frames.size(0))
            
            # Update progress bar
            pbar.set_postfix({'loss': f"{metrics.get_average():.4f}"})
        
        return metrics.get_average()
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        metrics = MetricsTracker()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.NUM_EPOCHS} [VAL]")
        
        with torch.no_grad():
            for batch in pbar:
                # Move to device
                frames = batch['frames'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                label_lengths = batch['label_lengths']
                input_lengths = batch['input_lengths']
                
                # Forward pass
                log_probs = self.model(frames)
                
                # Compute loss
                loss = self.criterion(log_probs, labels, input_lengths, label_lengths)
                
                # Track metrics
                metrics.update(loss.item(), frames.size(0))
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{metrics.get_average():.4f}"})
        
        return metrics.get_average()
    
    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config.__dict__
        }
        
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filename='checkpoint.pth'):
        """Load model checkpoint"""
        filepath = os.path.join(self.config.CHECKPOINT_DIR, filename)
        checkpoint = torch.load(filepath, map_location=self.config.DEVICE)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ Checkpoint loaded: {filepath}")
    
    def train(self):
        """Full training loop"""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        print(f"Device: {self.config.DEVICE}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print("="*60 + "\n")
        
        for epoch in range(self.config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if epoch < self.config.WARMUP_EPOCHS:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model! Val Loss: {val_loss:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning Rate
        plt.subplot(1, 2, 2)
        plt.plot(self.history['learning_rate'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.LOG_DIR, 'training_curves.png'), dpi=300)
        print(f"✓ Training curves saved: {self.config.LOG_DIR}/training_curves.png")
        plt.show()


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def prepare_data_splits(dataset_path, config):
    """
    Prepare train/val/test splits
    Returns: train_files, train_labels, val_files, val_labels, test_files, test_labels
    """
    # This is a placeholder - you need to implement based on your dataset structure
    # Read video files and their corresponding labels
    
    male_path = Path(dataset_path) / 'male'
    female_path = Path(dataset_path) / 'female'
    
    all_videos = []
    all_labels = []
    all_speakers = []
    
    # Scan dataset
    for gender_folder in [male_path, female_path]:
        if not gender_folder.exists():
            continue
        
        # Video files are in male/video/ or female/video/ subfolder
        video_subfolder = gender_folder / 'video'
        if video_subfolder.exists():
            video_files = list(video_subfolder.glob('*.mp4'))
        else:
            # Fallback: check directly in gender folder
            video_files = list(gender_folder.glob('*.mp4'))
        
        for video_file in video_files:
            # Extract speaker ID
            match = re.search(r'P(\d+)', video_file.name)
            speaker_id = f"P{match.group(1)}" if match else "Unknown"
            
            all_videos.append(str(video_file))
            all_speakers.append(speaker_id)
            
            # TODO: Load actual labels from annotation file
            # For now, placeholder
            all_labels.append("placeholder text")
    
    # Create DataFrame
    df = pd.DataFrame({
        'video': all_videos,
        'label': all_labels,
        'speaker': all_speakers
    })
    
    # Speaker-independent split
    unique_speakers = df['speaker'].unique()
    np.random.shuffle(unique_speakers)
    
    n_train = int(len(unique_speakers) * config.TRAIN_RATIO)
    n_val = int(len(unique_speakers) * config.VAL_RATIO)
    
    train_speakers = unique_speakers[:n_train]
    val_speakers = unique_speakers[n_train:n_train+n_val]
    test_speakers = unique_speakers[n_train+n_val:]
    
    train_df = df[df['speaker'].isin(train_speakers)]
    val_df = df[df['speaker'].isin(val_speakers)]
    test_df = df[df['speaker'].isin(test_speakers)]
    
    print(f"Data split:")
    print(f"  Train: {len(train_df)} videos from {len(train_speakers)} speakers")
    print(f"  Val: {len(val_df)} videos from {len(val_speakers)} speakers")
    print(f"  Test: {len(test_df)} videos from {len(test_speakers)} speakers")
    
    return (train_df['video'].tolist(), train_df['label'].tolist(),
            val_df['video'].tolist(), val_df['label'].tolist(),
            test_df['video'].tolist(), test_df['label'].tolist())


def main():
    """Main training pipeline"""
    
    # Prepare data
    print("Preparing data splits...")
    train_files, train_labels, val_files, val_labels, test_files, test_labels = \
        prepare_data_splits(config.DATASET_PATH, config)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = LUMINADataset(train_files, train_labels, mode='train', config=config)
    val_dataset = LUMINADataset(val_files, val_labels, mode='val', config=config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nInitializing model...")
    model = LipReadingModel(config).to(config.DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, config)
    
    # Train
    trainer.train()
    
    print("\n✓ Training completed successfully!")


if __name__ == "__main__":
    main()
