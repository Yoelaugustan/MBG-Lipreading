# Sentence-Level Lip Reading for Bahasa Indonesia
---

## Repository Structure

```
.
├── EDA/                          # Exploratory data analysis notebooks
├── LUMINA_Dataset/               # Raw video files (gitignored — user must provide)
├── LUMINA_preprocessed/          # Preprocessed .pt tensors (gitignored — generated locally)
│   ├── female/
│   ├── male/
│   ├── manifest.csv
│   └── vocab.json
├── Preprocess/
│   └── preprocess_dataset.py     # MediaPipe lip extraction pipeline
├── Train/
│   ├── config.py                 # Centralized hyperparameter configuration
│   ├── dataset.py                # PyTorch Dataset and DataLoader builders
│   ├── model.py                  # LUMINAModel and LipFrontend definitions
│   ├── train.py                  # Training entry point
│   ├── utils.py                  # CTC decoding and CER/WER metrics
│   ├── plot_history.py           # Training curve visualization
│   └── runs/                     # Checkpoints and training logs (gitignored)
│       └── lumina_exp1/
│           ├── best.pt           # Best validation CER checkpoint
│           ├── latest.pt         # Most recent epoch checkpoint
│           ├── ckpt_epoch{N}.pt  # Periodic backups
│           └── history.json      # Per-epoch metrics
├── .gitignore
├── README.md
└── requirements.txt
```

The `LUMINA_Dataset/`, `LUMINA_preprocessed/`, and `Train/runs/` directories are excluded from version control via `.gitignore`. They must be created locally — the dataset by download from the official source, the preprocessed tensors by running the preprocessing script, and the run directory by training the model.

---

## Installation

This project was developed and trained on **Windows Subsystem for Linux 2 (WSL2)** with Ubuntu and an NVIDIA GPU. Native Linux is also fully supported. Native Windows is **not recommended** because the `mamba-ssm` package requires CUDA-specific compilation that is significantly easier to set up under Linux/WSL.

### 1. WSL setup (Windows users only)

If you are on Windows, install WSL2 with Ubuntu and verify that your NVIDIA GPU is accessible from within WSL.

```powershell
# In PowerShell (as Administrator)
wsl --install -d Ubuntu-22.04
```

After installation, install the latest NVIDIA driver for Windows (CUDA-on-WSL is supported by the standard NVIDIA Game Ready / Studio drivers — no separate Linux driver is needed inside WSL). Then verify the GPU is visible from inside WSL:

```bash
nvidia-smi
```

If you do not see your GPU listed, update your Windows NVIDIA driver and restart WSL with `wsl --shutdown` from PowerShell.

### 2. Obtain the LUMINA dataset

The LUMINA dataset is **not included** in this repository. Download it from the official LUMINA source and extract it so that the resulting `LUMINA_Dataset/` folder sits at the root of the project, following the directory layout described in the [Dataset](#https://data.mendeley.com/datasets/8fw93k4rny/4) section.

> Replace this paragraph with the official LUMINA download link when available.

### 3. Clone the repository

```bash
git clone https://github.com/Yoelaugustan/Pre-thesis.git
cd Pre-thesis
```

### 4. Create a Python environment

This project requires Python 3.10+. A conda environment is recommended:

```bash
conda create -n lipreading python=3.10
conda activate lipreading
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

The `mamba-ssm` package depends on `causal-conv1d`, which can be installation-sensitive. If `pip install -r requirements.txt` fails on the Mamba-related packages, install them manually with the no-build-isolation flag:

```bash
pip install causal-conv1d --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

## Usage

### Step 1 — Preprocessing

After placing the raw `LUMINA_Dataset/` directory at the project root, run:

```bash
python Preprocess/preprocess_dataset.py
```

This generates `LUMINA_preprocessed/` containing one `.pt` tensor per video, a `manifest.csv` mapping every tensor to its speaker identifier, gender, sentence number, and ground-truth text, and a `vocab.json` describing the character-level vocabulary. Preprocessing is resumable — re-running the script skips already-processed videos.

### Step 2 — Training

From the project root:

```bash
python Train/train.py
```

Training writes checkpoints and per-epoch metrics to the directory specified by `Config.output_dir` (default `Train/runs/lumina_exp1/`). The `best.pt` checkpoint is updated whenever validation CER improves.

### Step 3 — Resume

If training is interrupted, resume from the most recent checkpoint:

```bash
python Train/train.py --resume Train/runs/lumina_exp1/latest.pt
```

The optimizer state, learning rate scheduler state, AMP gradient scaler, best-CER tracker, and early stopping counter are all restored. The `--resume` argument requires a valid path; an invalid path will raise `FileNotFoundError` rather than silently restarting from epoch 1.

### Step 4 — Visualize training curves

```bash
python Train/plot_history.py
```

This reads `runs/lumina_exp1/history.json` and produces side-by-side loss and CER plots for both training and validation splits.

