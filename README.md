# LUMINA Lip Reading - Training Pipeline

Complete PyTorch implementation for Indonesian sentence-level lip reading using **3D CNN + Mamba + Bi-GRU + CTC**.

## Architecture

```
Input Video (84 frames, 250×150)
    ↓
MediaPipe Face Mesh (Lip ROI Extraction → 112×112)
    ↓
Data Augmentation (rotation, brightness, crop, noise)
    ↓
3D ResNet (Spatiotemporal Feature Extraction)
    ↓
Mamba Encoder (6 layers, d=512)
    ↓
Bi-GRU (2 layers, hidden=512)
    ↓
CTC Decoder
    ↓
Output Text
```

---

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt

# Install PyTorch with CUDA (for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Mamba

```bash
# Mamba requires specific installation
pip install mamba-ssm causal-conv1d

# If installation fails, you can use Transformer fallback (automatic in code)
```

### 3. Prepare Dataset

**Directory structure:**
```
LUMINA/
├── male/
│   ├── video/
│   │   ├── P01_S1.mp4
│   │   ├── P01_S2.mp4
│   │   ├── P02_S1.mp4
│   │   └── ...
│   └── audio/
│       ├── P01_S1.wav
│       └── ...
├── female/
│   ├── video/
│   │   ├── P01_S1.mp4
│   │   ├── P01_S2.mp4
│   │   └── ...
│   └── audio/
│       ├── P01_S1.wav
│       └── ...
└── annotations.txt  # (You need to create this)
```

**Annotations format:**
```
P01_S1.mp4|tunjuk merah angka dua di depan a satu
P01_S2.mp4|letakkan hijau angka lima di bawah b tiga
...
```

---

## Configuration

Edit the `Config` class in `train_lip_reading.py`:

```python
class Config:
    # Paths
    DATASET_PATH = "/path/to/LUMINA"  # ← CHANGE THIS
    
    # Model
    MAMBA_LAYERS = 6  # Number of Mamba layers
    MAMBA_D_MODEL = 512  # Hidden dimension
    BIGRU_LAYERS = 2  # Bi-GRU layers
    
    # Training
    BATCH_SIZE = 8  # Adjust based on GPU memory
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
```

---

## Training

### Quick Start

```bash
python train_lip_reading.py
```

### What Happens During Training

1. **Data Loading:**
   - Scans `male/` and `female/` folders
   - Extracts speaker IDs from filenames (P01, P02, etc.)
   - Creates speaker-independent train/val/test splits (70/15/15)

2. **Preprocessing (automatic per batch):**
   - MediaPipe Face Mesh detects facial landmarks
   - Crops lip region (112×112)
   - Converts to grayscale
   - Applies augmentation (train mode only)

3. **Training Loop:**
   - Shows progress bar with loss per epoch
   - Validates after each epoch
   - Saves best model based on validation loss
   - Auto-saves checkpoint every 10 epochs

4. **Outputs:**
   - `checkpoints/best_model.pth` - Best model
   - `checkpoints/final_model.pth` - Final model
   - `logs/training_curves.png` - Loss curves
   - Checkpoints every 10 epochs

---

## Model Checkpoints

### Saved Files

```
checkpoints/
├── best_model.pth          # Best validation loss
├── final_model.pth         # After all epochs
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_20.pth
└── ...
```

### Checkpoint Contents

Each checkpoint contains:
- Model state dict
- Optimizer state
- Learning rate scheduler state
- Training history
- Configuration

### Resume Training

```python
# Load checkpoint and continue training
trainer = Trainer(model, train_loader, val_loader, config)
trainer.load_checkpoint('checkpoint_epoch_50.pth')
trainer.train()  # Continues from epoch 51
```

---

## Data Augmentation

Applied **only during training**:

| Augmentation | Parameters | Probability |
|--------------|------------|-------------|
| Random Crop | 90% of original | 0.5 |
| Horizontal Flip | - | 0.5 |
| Rotation | ±5° | 0.3 |
| Brightness/Contrast | ±20% | 0.5 |
| Gaussian Noise | var 10-50 | 0.3 |
| Normalization | mean=0.5, std=0.5 | 1.0 |

---

## Hardware Requirements

### Minimum

- GPU: 8GB VRAM (GTX 1070, RTX 2060)
- RAM: 16GB
- Storage: 50GB

### Recommended

- GPU: 16GB+ VRAM (RTX 3090, A100)
- RAM: 32GB+
- Storage: 100GB SSD

### Expected Training Time

| GPU | Batch Size | Time per Epoch | Total (100 epochs) |
|-----|------------|----------------|-------------------|
| RTX 2060 | 4 | ~45 min | ~75 hours |
| RTX 3090 | 8 | ~20 min | ~33 hours |
| A100 | 16 | ~10 min | ~17 hours |

---

## Memory Optimization

If you run out of GPU memory:

1. **Reduce batch size:**
   ```python
   config.BATCH_SIZE = 4  # or 2
   ```

2. **Use gradient accumulation:**
   ```python
   accumulation_steps = 4
   # Effective batch size = BATCH_SIZE × accumulation_steps
   ```

3. **Downsample frames:**
   ```python
   config.NUM_FRAMES_SAMPLE = 42  # Instead of 84
   ```

4. **Smaller model:**
   ```python
   config.MAMBA_LAYERS = 4  # Instead of 6
   config.MAMBA_D_MODEL = 256  # Instead of 512
   ```

---

## Monitoring Training

### Progress Bar

Training shows real-time progress:
```
Epoch 25/100 [TRAIN]: 100%|████████| 1250/1250 [18:23<00:00, loss: 2.3456]
Epoch 25/100 [VAL]:   100%|████████| 250/250 [03:45<00:00, loss: 2.1234]

Epoch 25/100
  Train Loss: 2.3456
  Val Loss: 2.1234
  LR: 0.000087
  ✓ New best model! Val Loss: 2.1234
```

### TensorBoard (Optional)

Add to training loop:
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/experiment_1')

# In train_epoch():
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

Then view:
```bash
tensorboard --logdir=runs
```

---

## Inference

### Single Video Prediction

```python
import torch
from train_lip_reading import LipReadingModel, Config, LipROIExtractor

# Load model
config = Config()
model = LipReadingModel(config).to(config.DEVICE)
checkpoint = torch.load('checkpoints/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Process video
extractor = LipROIExtractor(output_size=112)
# ... (extract frames, preprocess)

# Predict
with torch.no_grad():
    log_probs = model(frames)
    prediction = decode_predictions(log_probs, config)
    print(f"Prediction: {prediction[0]}")
```

---

## Troubleshooting

### Issue: `mamba-ssm` installation fails

**Solution:** Code automatically falls back to Transformer encoder
```
Warning: mamba-ssm not installed. Using Transformer fallback.
```

### Issue: CUDA out of memory

**Solutions:**
1. Reduce `BATCH_SIZE`
2. Reduce `NUM_FRAMES_SAMPLE`
3. Use smaller model (fewer layers, smaller hidden dim)

### Issue: MediaPipe can't detect face

**Solution:** Fallback to center crop is automatic
```python
# In extract_lip_roi():
if not results.multi_face_landmarks:
    # Returns center crop automatically
```

### Issue: CTC Loss becomes NaN

**Solutions:**
1. Check label lengths > 0
2. Reduce learning rate
3. Use gradient clipping (already enabled)
4. Check for inf/nan in inputs

---

## Citation

If you use this code, please cite:

```bibtex
@misc{lumina_lipreading_2026,
  author = {Yoel},
  title = {Indonesian Sentence-Level Lip Reading with Mamba},
  year = {2026},
  publisher = {GitHub},
}
```

---

## License

MIT License - feel free to use for research and commercial applications.

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email].
