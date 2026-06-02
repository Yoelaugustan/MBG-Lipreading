# Sentence-Level Lip Reading for Bahasa Indonesia

---

## Repository Structure

```
.
├── EDA/                                  # Exploratory data analysis notebooks
├── LUMINA_Dataset/                       # Raw video files (gitignored — user must provide)
├── LUMINA_preprocessed/                  # Preprocessed .pt tensors (gitignored — generated locally)
│   ├── female/
│   ├── male/
│   ├── manifest.csv
│   └── vocab.json
├── Preprocess/
│   └── preprocess_dataset.py             # MediaPipe lip extraction pipeline
├── Train/
│   ├── config.py                         # Centralized hyperparameter configuration
│   ├── dataset.py                        # PyTorch Dataset and DataLoader builders
│   ├── model.py                          # LUMINAModel + four architecture variants
│   ├── train.py                          # Training entry point
│   ├── utils.py                          # CTC decoding and CER/WER metrics
│   ├── plot_history.py                   # Single-run training curve visualization
│   ├── plot_variants.py                  # Multi-variant ablation comparison plots
│   └── runs/                             # Checkpoints and training logs (gitignored)
│       ├── lumina_sequential/            # Main model — sequential Mamba → Bi-GRU
│       │   ├── best.pt                   # Best validation CER checkpoint
│       │   ├── latest.pt                 # Most recent epoch checkpoint
│       │   └── history.json              # Per-epoch metrics
│       ├── lumina_parallel/              # Ablation — parallel Mamba + Bi-GRU
│       │   ├── best.pt
│       │   ├── history.json
│       │   ├── latest.pt
│       │   └── training_curves.png
│       ├── lumina_bigru_only/            # Ablation — Bi-GRU only (no Mamba)
│       │   ├── best.pt
│       │   ├── history.json
│       │   ├── latest.pt
│       │   └── training_curves.png
│       ├── lumina_mamba_only/            # Ablation — Mamba only (no Bi-GRU)
│       │   ├── best.pt
│       │   ├── history.json
│       │   └── latest.pt
│       └── comparison/                   # Cross-variant comparison outputs
│           ├── comparison.csv            # Best metrics per variant
│           ├── variants_best.png         # Bar chart of best CER/WER per variant
│           └── variants_curves.png       # Side-by-side training curves
├── .gitignore
├── README.md
└── requirements.txt
```

The `LUMINA_Dataset/`, `LUMINA_preprocessed/`, and `Train/runs/` directories are excluded from version control via `.gitignore`. They must be created locally — the dataset by download from the official source, the preprocessed tensors by running the preprocessing script, and the run directory by training the model.

The `Train/runs/` folder contains four subdirectories corresponding to the architecture variants evaluated in the ablation study. The main model is **sequential** (Mamba → Bi-GRU), which was selected based on empirical ablation results showing it outperforms parallel composition, Bi-GRU alone, and Mamba alone. The other three variants are retained for reproducibility of the ablation study.

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

The LUMINA dataset is **not included** in this repository. Download it from the official LUMINA source and extract it so that the resulting `LUMINA_Dataset/` folder sits at the root of the project, following the directory layout described in the Dataset section.

> https://data.mendeley.com/datasets/8fw93k4rny/4

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

### Step 2 — Training the main model

From the project root:

```bash
python Train/train.py
```

This trains the **sequential** variant (Mamba → Bi-GRU), which is the default configured in `Train/config.py`. Training writes checkpoints and per-epoch metrics to the directory specified by `Config.output_dir` (default `Train/runs/lumina_sequential/`). The `best.pt` checkpoint is updated whenever validation CER improves.

### Step 3 — Resume

If training is interrupted, resume from the most recent checkpoint:

```bash
python Train/train.py --resume Train/runs/lumina_sequential/latest.pt
```

The optimizer state, learning rate scheduler state, AMP gradient scaler, best-CER tracker, and early stopping counter are all restored. The `--resume` argument requires a valid path; an invalid path will raise `FileNotFoundError` rather than silently restarting from epoch 1.

### Step 4 — Training the ablation variants (optional)

Three additional architecture variants are implemented for comparison against the main sequential model. Each can be trained by overriding the `--variant` and `--output_dir` arguments:

```bash
# Parallel variant — Mamba + Bi-GRU in parallel
python Train/train.py --variant parallel \
    --output_dir Train/runs/lumina_parallel

# Bi-GRU only — no Mamba
python Train/train.py --variant bigru_only \
    --output_dir Train/runs/lumina_bigru_only

# Mamba only — no Bi-GRU
python Train/train.py --variant mamba_only \
    --output_dir Train/runs/lumina_mamba_only
```

All variants share the same frontend, hyperparameters, and training loop — only the temporal backend differs, so results are directly comparable.

### Step 5 — Visualize training curves

For a single variant's training curves (loss and CER, train and val):

```bash
python Train/plot_history.py
```

This reads `runs/lumina_sequential/history.json` by default. To plot a different variant, pass the path as an argument.

### Step 6 — Compare variants (ablation study)

After training multiple variants, generate comparison plots and a summary table:

```bash
python Train/plot_variants.py
```

This produces three outputs in `Train/runs/comparison/`:

- **`variants_curves.png`** — 2×2 grid overlaying train/val loss and train/val CER curves across all variants
- **`variants_best.png`** — horizontal bar chart ranking variants by best validation CER and WER
- **`comparison.csv`** — table with the best epoch, loss, CER, and WER for each variant

The script automatically detects which variants have been trained by looking for `history.json` files in each `lumina_<variant>/` subdirectory. Variants without completed training are skipped with a warning.
## Config
`config.py` may need to be modified as the path for the dataset etc. might be different per user.
CONFIG section in `preprocess_dataset.py` needs to also be modified according to the path.

## KenLM Runbook

This project uses lowercase text at evaluation time, so the LM corpus should be cleaned and lowercased too. The current cleaner is [Preprocess/prepare_kenlm_corpus.py](Preprocess/prepare_kenlm_corpus.py). Keep LUMINA transcripts out of the LM if you want the decoding experiment to stay unbiased.

### 1. Clean the external corpus

Use the Leipzig/news corpus or another external Indonesian text source:

```bash
python Preprocess/prepare_kenlm_corpus.py \
    --input KenLM/ind_news_2024_1M-sentences.txt \
    --output KenLM/clean_corpus.txt \
    --report KenLM/clean_corpus.report.json
```

The cleaner lowercases text, strips Leipzig-style numeric IDs, removes obvious noise such as URLs and fragment-like lines, deduplicates exact repeats, and strips punctuation from token boundaries before writing the final corpus.

### 2. Inspect the cleaned corpus

Before training KenLM, check the report and a small random sample of the cleaned lines:

```bash
python -c "import random; from pathlib import Path; lines=Path('KenLM/clean_corpus.txt').read_text(encoding='utf-8').splitlines(); print('\n'.join(random.sample(lines, min(20, len(lines)))))"
```

You want normal Indonesian sentence flow, not many fragments, URLs, or list-like entries.

### 3. Train KenLM

If you don't have KenLM lmplz and build_binary installed yet: (more info: [KenLM Documentation](https://kheafield.com/code/kenlm/))
```bash
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake --build . --target lmplz build_binary -j$(nproc)
make -j$(nproc) lmplz build_binary
```

Start with a 5-gram model if the cleaned corpus is large enough:

```bash
lmplz -o 5 < KenLM/clean_corpus.txt > KenLM/lm.arpa
build_binary KenLM/lm.arpa KenLM/lm.binary
```

If memory is tight, start with `-o 3` and compare validation WER/CER later.

### 4. Install decoder dependencies

In the `lipreading` conda environment, install the decoder packages:

```bash
pip install pyctcdecode
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 5. Install pyctcdecode and validate vocabulary alignment

The model vocabulary from `LUMINA_preprocessed/vocab.json` will be loaded automatically. The decoder vocabulary must match the acoustic model token order **exactly**.

Verify by inspecting the vocab:

```bash
python -c "import json; v = json.load(open('LUMINA_preprocessed/vocab.json')); print('Vocab size:', len(v)); print('First 10:', {k: v[k] for k in list(v.keys())[:10]})"
```

Since predictions and evaluation text are already lowercase, the LM and decoder operate in lowercase mode. Do not reintroduce uppercase tokens in the pipeline.

### 6. Tune on validation split

Run a hyperparameter sweep over beam widths and LM weights using the validation split. A new script `Train/kenlm_decoder.py` automates this:

```bash
python Train/kenlm_decoder.py \
    --checkpoint Train/runs/lumina_sequential/best.pt \
    --split val \
    --vocab_path LUMINA_preprocessed/vocab.json \
    --lm_path KenLM/lm.binary \
    --beam_widths 25 50 100 \
    --lm_alphas 0.2 0.4 0.6 0.8 1.0 \
    --output_dir Train/runs/lumina_sequential
```

This will:
- Print greedy baseline (CER/WER with no LM)
- Try all combinations of beam_width × lm_alpha
- Report delta CER/WER vs. greedy for each setting
- Save a JSON sweep file: `Train/runs/lumina_sequential/kenlm_sweep.json`

Pick the setting with the **lowest validation CER** (or WER if your primary metric is WER). For example, if the sweep output shows:

```
bw= 50 α=0.8 => CER=0.1450 WER=0.2950 (vs greedy: Δ_CER=-0.0191 Δ_WER=-0.0442)
```

Then record `beam_width=50, lm_alpha=0.8` for the test run.

### 7. Evaluate on test split with locked settings

After validation tuning, run the test split **exactly once** with the best-tuned hyperparameters. Lock them and do not retune on test.

```bash
python Train/kenlm_decoder.py \
    --checkpoint Train/runs/lumina_sequential/best.pt \
    --split test \
    --vocab_path LUMINA_preprocessed/vocab.json \
    --lm_path KenLM/lm.binary \
    --beam_widths 50 \
    --lm_alphas 0.8 \
    --output_dir Train/runs/lumina_sequential
```

Report the final metrics:

- **Greedy baseline**: CER and WER on test (no LM)
- **KenLM + pyctcdecode**: CER and WER on test (with best-tuned settings)
- **Delta**: improvement in both metrics

Example output summary:

```
Greedy CER: 0.1641, WER: 0.3392
KenLM bw=50 α=0.8 CER: 0.1450, WER: 0.2950
Improvements: Δ_CER = -0.0191 (-11.6%), Δ_WER = -0.0442 (-13.0%)
```

### 8. Iterate only if needed

If the test CER/WER are still far from your targets, the usual fixes are:

- **More cleaned text**: Gather more external Indonesian news/Wikipedia and reprocess the corpus
- **Increase beam width**: Try 200–500 instead of 100 (slower but can help)
- **Retune LM weight**: Validate with a finer sweep around the best α found in step 6
- **Subword tokens later**: If character-level decoding remains brittle, transition to BPE/Wordpiece after validation (requires retraining the acoustic model)
- **Different smoothing**: Experiment with KenLM `--discount_fallback` flag (see KenLM documentation)