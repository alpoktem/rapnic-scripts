# RAPNIC Data Analysis - Setup Guide

## Directory Structure

```
DATA/
├── scripts/						   # This repository cloned
│   ├── 00_docs.md                     # This file
│   ├── 01_pull_recordings.sh          # Download audio from Firebase Storage
│   ├── 02_pull_users.py               # Download user metadata from Firestore
│   ├── 03_analyze_corpus.py           # Analyze audio recordings
│   ├── 04_demographics_analyzer.py    # Analyze user demographics
│   ├── 05_combine_corpus.py           # Combine multiple datasets
│   ├── 06_hf_upload.py                # Upload dataset to Hugging Face Hub
│   └── firebase-credentials.json      # Firebase credentials (required)
│
├── <DATA_PULL_NAME>/                  # e.g., PILOT, LREC-PAPER, PHASE2, etc.
│   ├── audioapp_recordings/           # Audio files by user
│   ├── Users.tsv                      # User metadata
│   ├── filtered_users.txt             # (auto-generated)
│   ├── downloaded_users.txt           # (auto-generated)
│   └── analysis_results/              # Analysis outputs
│
├── PILOT/                             # Example: First data pull
├── LREC-PAPER/                        # Example: Data pull for LREC paper
└── COMBINED/                          # Combined analysis across pulls
```

---

## Setup

### Setup working directory with scripts

```
mkdir DATA
cd DATA
git clone https://github.com/alpoktem/rapnic-scripts.git scripts
```

### Firebase Credentials

To download data from Firebase:

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Select your project
3. Go to **Project Settings** → **Service Accounts** tab
4. Click **Generate New Private Key**
5. Save the downloaded JSON file as `firebase-credentials.json` in the `scripts/` directory

**Security**: Never commit this file to version control.

### Required Software

```bash
# Google Cloud SDK (for step 1)
brew install google-cloud-sdk
gcloud auth login

# Python packages
pip install pandas firebase-admin datasets huggingface_hub

# Hugging Face authentication (for step 6)
huggingface-cli login
```

---

## Script Execution Order

### Step 1: Pull Recordings from Firebase

```bash
cd DATA
./scripts/01_pull_recordings.sh LREC-PAPER
```

Downloads audio recordings from Firebase Storage for users with ≥100 recordings. Resumable - can be run multiple times.

**Output:**
```
LREC-PAPER/
├── audioapp_recordings/E12345/
├── filtered_users.txt
└── downloaded_users.txt
```

---

### Step 2: Pull User Metadata from Firestore

```bash
cd DATA
python scripts/02_pull_users.py LREC-PAPER
```

Downloads user demographics from Firestore.

**Output:**
```
LREC-PAPER/Users.tsv
```

---

### Step 3: Analyze Corpus

```bash
cd DATA
python scripts/03_analyze_corpus.py LREC-PAPER
```

Analyzes audio files, detects duplicates, calculates durations.

**Output:**
```
LREC-PAPER/analysis_results/
├── corpus_stats.json
├── corpus_report.txt
├── recordings_clean.csv
├── recordings_other.csv
└── recordings_all.csv
```

**Parameters:**

Inside the script you'll find these parameters hard coded:
```
CUT_BACK_SECS = 0    # Amount of seconds to cut from each recording length (if there's a constant buffer)
LENGTH_THRESHOLD = 25  #Any recording above this amount of seconds would be discarded (to Other split)
DONT_COUNT_OVER_THRESHOLD = True 
```

---

### Step 4: Analyze Demographics

```bash
cd DATA
python scripts/04_demographics_analyzer.py LREC-PAPER
```

Analyzes user demographics and generates distributions.

**Output:**
```
LREC-PAPER/analysis_results/
├── demographics_stats.json
└── demographics_report.txt
```

---

### Step 5: Combine Datasets

```bash
cd DATA
python scripts/05_combine_corpus.py
```

Combines multiple datasets (PILOT, LREC-PAPER, etc.) into COMBINED directory.

**Output:**
```
COMBINED/analysis_results/
├── combined_corpus_stats.json
├── combined_demographics.json
└── recordings_all.csv
```

---

### Step 6: Upload to Hugging Face

```bash
cd DATA
python scripts/06_hf_upload.py LREC-PAPER --repo alp/rapnic-test --samples 10
```

Uploads dataset to Hugging Face Hub in parquet format with audio and metadata.

**Options:**
- `--repo` : Hugging Face repository ID
- `--samples N` : Upload only N samples per speaker (for testing)
- `--split` : Dataset split name (train/test/validation)
- `--private` : Make repository private
- `--use-clean-only` : Only upload clean recordings

**Examples:**

```bash
# Test upload (recommended first)
python scripts/06_hf_upload.py LREC-PAPER --repo alp/rapnic-test --samples 10

# Full dataset, clean only
python scripts/06_hf_upload.py LREC-PAPER --repo alp/rapnic-lrec --use-clean-only

# Private repository
python scripts/06_hf_upload.py PILOT --repo alp/rapnic-pilot --private
```

**Working with Combined Datasets:**

The script automatically detects combined datasets and resolves audio paths from multiple source directories:

```bash
# Upload combined dataset
python scripts/06_hf_upload.py COMBINED_PILOT_LREC-PAPER --repo alp/rapnic-combined --samples 10
```

When uploading COMBINED datasets:
- Script detects `data_pull` column automatically
- Resolves audio paths from source directories (PILOT, LREC-PAPER, etc.)
- Includes `data_pull` field in uploaded dataset
- Dataset card shows statistics per data pull

---

## Configuration

### Excluded Users
Test/invalid accounts excluded from all scripts:
```
E07516, E17994, E23241, E42131, E98931, E76038, E86686
```

### Parameters
- **Cut back**: 2 seconds removed from each recording
- **Length threshold**: 10 seconds
- **Minimum recordings**: 100 per user

---

## Troubleshooting

**"No Firebase credentials found"**
- Place `firebase-credentials.json` in `scripts/` directory

**"gsutil not found"**
- Install Google Cloud SDK and run `gcloud auth login`

**"Not logged in to Hugging Face"**
- Run `huggingface-cli login`

**Resume failed download**
```bash
./scripts/01_pull_recordings.sh LREC-PAPER  # Skips already downloaded
```

**Clear cache**
```bash
cd DATA/LREC-PAPER
rm filtered_users.txt downloaded_users.txt
```
