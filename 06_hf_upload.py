#!/usr/bin/env python3
"""
Hugging Face Dataset Upload Script
Uploads RAPNIC dataset to Hugging Face Hub with audio files and metadata

Usage: python 06_upload_to_huggingface.py <DATA_PULL_NAME> [OPTIONS]

Example:
  python 06_upload_to_huggingface.py LREC-PAPER --repo alp/rapnic-test
  python 06_upload_to_huggingface.py PILOT --repo alp/rapnic-test --samples 10

TODO: Upload other audio versions (wav16k, mp3 etc)
"""

import sys
import os
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio, Features, Value, ClassLabel, DatasetDict
from huggingface_hub import HfApi, login
import json

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Upload RAPNIC dataset to Hugging Face Hub',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'data_pull_name',
        type=str,
        help='Name of the data pull directory (e.g., LREC-PAPER, PILOT)'
    )
    
    parser.add_argument(
        '--repo',
        type=str,
        default='alp/rapnic-test',
        help='Hugging Face repository ID (default: alp/rapnic-test)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=None,
        help='Number of samples per speaker for testing (default: None, uploads all)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'test', 'validation'],
        help='Dataset split name (default: train)'
    )
    
    parser.add_argument(
        '--private',
        action='store_true',
        help='Make the repository private (default: public)'
    )
    
    parser.add_argument(
        '--use-clean-only',
        action='store_true',
        help='Only upload clean recordings (excludes duplicates and over-threshold)'
    )
    
    return parser.parse_args()

def verify_paths(data_pull_name):
    """Verify that required directories and files exist."""
    
    if not Path("scripts").exists():
        print("❌ ERROR: This script should be run from the DATA directory")
        print(f"   Current directory: {os.getcwd()}")
        print("   Expected to find: ./scripts/")
        sys.exit(1)
    
    data_dir = Path(data_pull_name)
    if not data_dir.exists():
        print(f"❌ ERROR: Data directory '{data_pull_name}' not found")
        sys.exit(1)
    
    analysis_dir = data_dir / "analysis_results"
    if not analysis_dir.exists():
        print(f"❌ ERROR: Analysis results not found. Please run analysis first:")
        print(f"   python scripts/03_analyze_corpus.py {data_pull_name}")
        sys.exit(1)
    
    # For COMBINED datasets, recordings_dir won't exist in the COMBINED folder
    # We'll verify audio paths later when we know if it's combined
    recordings_dir = data_dir / "audioapp_recordings"
    
    return data_dir, recordings_dir, analysis_dir


def load_metadata(analysis_dir, use_clean_only):
    """Load recording metadata from CSV files."""
    
    if use_clean_only:
        csv_file = analysis_dir / "recordings_clean.csv"
        print(f"Loading clean recordings only from: {csv_file}")
    else:
        csv_file = analysis_dir / "recordings_all.csv"
        print(f"Loading all recordings from: {csv_file}")
    
    if not csv_file.exists():
        print(f"❌ ERROR: Metadata CSV not found: {csv_file}")
        sys.exit(1)
    
    df = pd.read_csv(csv_file)
    print(f"✓ Loaded {len(df)} recordings from metadata")
    
    # Check if this is a combined dataset (has 'data_pull' column)
    is_combined = 'data_pull' in df.columns
    
    if is_combined:
        print(f"✓ Detected COMBINED dataset with {df['data_pull'].nunique()} data pulls:")
        for pull in df['data_pull'].unique():
            count = len(df[df['data_pull'] == pull])
            print(f"  - {pull}: {count} recordings")
    
    return df, is_combined

def filter_samples_per_speaker(df, samples_per_speaker):
    """Filter dataset to include only N samples per speaker for testing."""
    
    if samples_per_speaker is None:
        return df
    
    print(f"\nFiltering to {samples_per_speaker} samples per speaker...")
    
    # Group by speaker and take first N samples
    filtered_dfs = []
    for speaker_id, group in df.groupby('speaker_id'):
        sampled = group.head(samples_per_speaker)
        filtered_dfs.append(sampled)
    
    df_filtered = pd.concat(filtered_dfs, ignore_index=True)
    
    print(f"✓ Filtered from {len(df)} to {len(df_filtered)} recordings")
    print(f"  Speakers: {df['speaker_id'].nunique()} → {df_filtered['speaker_id'].nunique()}")
    
    return df_filtered

def prepare_dataset(df, data_pull_name, is_combined):
    """Prepare dataset for Hugging Face upload."""
    
    print("\nPreparing dataset...")
    
    # Add full audio file paths
    if is_combined:
        # For combined datasets, use data_pull column to resolve paths
        print("  Resolving audio paths from multiple data pulls...")
        df['audio_path'] = df.apply(
            lambda row: str(Path(row['data_pull']) / row['file_path']), 
            axis=1
        )
    else:
        # For single dataset, use data_pull_name
        df['audio_path'] = df['file_path'].apply(
            lambda x: str(Path(data_pull_name) / x)
        )
    
    # Verify all audio files exist
    missing_files = []
    for idx, row in df.iterrows():
        if not Path(row['audio_path']).exists():
            missing_files.append(row['audio_path'])
    
    if missing_files:
        print(f"⚠️  WARNING: {len(missing_files)} audio files not found")
        print(f"   First missing: {missing_files[0]}")
        # Remove missing files from dataset
        df = df[df['audio_path'].apply(lambda x: Path(x).exists())]
        print(f"   Continuing with {len(df)} valid recordings")
    
    # Prepare data dictionary for Hugging Face
    data_dict = {
        'audio': df['audio_path'].tolist(),
        'speaker_id': df['speaker_id'].tolist(),
        'filename': df['filename'].tolist(),
        'task_id': df['task_id'].tolist(),
        'prompt': df['prompt'].tolist(),
        'original_duration': df['original_duration'].tolist(),
        'trimmed_duration': df['trimmed_duration'].tolist(),
        'category': df['category'].tolist(),
        'reason': df['reason'].fillna('').tolist(),
        'age': df['age'].fillna('').tolist(),
        'gender': df['gender'].fillna('').tolist(),
        'disorder': df['disorder'].fillna('').tolist(),
        'dialect': df['dialect'].fillna('').tolist(),
        'province': df['province'].fillna('').tolist(),
        'city': df['city'].fillna('').tolist(),
        'hasHelper': df['hasHelper'].fillna(False).astype(bool).tolist(),
    }
    
    # Add data_pull column if it exists
    if is_combined:
        data_dict['data_pull'] = df['data_pull'].tolist()
    
    # Define features with proper types
    features = Features({
        'audio': Audio(sampling_rate=16000),
        'speaker_id': Value('string'),
        'filename': Value('string'),
        'task_id': Value('string'),
        'prompt': Value('string'),
        'original_duration': Value('float32'),
        'trimmed_duration': Value('float32'),
        'category': ClassLabel(names=['clean', 'duplicate', 'over_threshold']),
        'reason': Value('string'),
        'age': Value('string'),
        'gender': Value('string'), 
        'disorder': Value('string'),  
        'dialect': Value('string'),
        'province': Value('string'),
        'city': Value('string'),
        'hasHelper': Value('bool'),
    })
    
    # Add data_pull feature if it exists
    if is_combined:
        features['data_pull'] = Value('string')
    
    # Create dataset
    dataset = Dataset.from_dict(data_dict, features=features)
    
    print(f"✓ Dataset prepared with {len(dataset)} recordings")
    
    return dataset

def create_dataset_card(data_pull_name, df, use_clean_only, samples_per_speaker, is_combined):
    """Create a dataset card with metadata and description."""
    
    # Load corpus statistics
    analysis_dir = Path(data_pull_name) / "analysis_results"
    stats_file = analysis_dir / "corpus_stats.json"
    
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = {}
    
    # Calculate statistics
    num_speakers = df['speaker_id'].nunique()
    num_recordings = len(df)
    total_duration_hours = df['trimmed_duration'].sum() / 3600
    
    # Get disorder distribution
    disorder_dist = df['disorder'].value_counts().to_dict()
    gender_dist = df['gender'].value_counts().to_dict()
    dialect_dist = df['dialect'].value_counts().to_dict()
    
    # Build dataset name
    if is_combined:
        dataset_name = "RAPNIC Combined Dataset"
        data_pulls = df['data_pull'].unique().tolist()
        data_pull_list = []
        for pull in data_pulls:
            count = len(df[df['data_pull'] == pull])
            data_pull_list.append(f'- **{pull}**: {count} recordings')
        data_pull_info = f"\n### Data Pulls Included\n\n{chr(10).join(data_pull_list)}\n"
    else:
        dataset_name = f"RAPNIC Dataset - {data_pull_name}"
        data_pull_info = ""
    
    card_content = f"""---
language:
- ca
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
- audio-classification
size_categories:
- 1K<n<10K
pretty_name: {dataset_name}
tags:
- speech
- catalan
- accessibility
- speech-disorders
- cerebral-palsy
- down-syndrome
---

# {dataset_name}

## Dataset Description

RAPNIC (Reconeixement Automàtic de la Parla No Intel·ligible en Català) is a Catalan speech corpus collected from individuals with speech disorders, specifically cerebral palsy and Down syndrome.

This dataset was collected to develop and improve automatic speech recognition (ASR) systems that are accessible to people with speech disorders who speak Catalan.
{data_pull_info}
### Dataset Statistics

- **Speakers**: {num_speakers}
- **Recordings**: {num_recordings}
- **Total Duration**: {total_duration_hours:.2f} hours
- **Sampling Rate**: 16 kHz
- **Audio Format**: WAV
- **Language**: Catalan (multiple dialects)

### Disorder Distribution

{chr(10).join([f'- **{k}**: {v} recordings' for k, v in disorder_dist.items() if k])}

### Gender Distribution

{chr(10).join([f'- **{k}**: {v} recordings' for k, v in gender_dist.items() if k])}

### Dialect Distribution

{chr(10).join([f'- **{k}**: {v} recordings' for k, v in dialect_dist.items() if k])}

## Data Fields

- `audio`: Audio file (WAV format, 16 kHz)
- `speaker_id`: Unique identifier for each speaker (anonymized)
- `filename`: Original filename of the recording
- `task_id`: Task/prompt identifier
- `prompt`: Text that was read/spoken
- `original_duration`: Duration in seconds before preprocessing
- `trimmed_duration`: Duration in seconds after preprocessing (2s cut from end)
- `category`: Recording category (clean, duplicate, over_threshold)
- `reason`: Additional category information
- `age`: Age range of the speaker
- `gender`: Gender of the speaker
- `disorder`: Type of speech disorder
- `dialect`: Catalan dialect variety
- `province`: Province of residence
- `city`: City of residence
- `hasHelper`: Whether the speaker had assistance during recording{"" if not is_combined else chr(10) + "- `data_pull`: Source data collection phase (e.g., PILOT, LREC-PAPER)"}

## Data Collection

The data was collected using a web-based recording platform adapted from Google's Project Euphonia. Participants recorded themselves reading prompts displayed on the screen.

### Preprocessing

- Each recording has 2 seconds trimmed from the end to remove silence
- Duplicate recordings (same speaker, same task) were identified and marked
- Recordings over 10 seconds were flagged for review

### Data Splits

{"This is a test upload with " + str(samples_per_speaker) + " samples per speaker." if samples_per_speaker else "This dataset includes all available recordings."}
{"Only clean recordings (no duplicates or over-threshold recordings) are included." if use_clean_only else "This dataset includes all recordings (clean, duplicates, and over-threshold)."}

## Ethical Considerations

- All participants provided informed consent
- Data is anonymized (speaker IDs do not contain personally identifiable information)
- The dataset complies with GDPR regulations
- This dataset should be used to improve accessibility technology for people with speech disorders

## Citation

If you use this dataset, please cite:

```
[Citation information to be added]
```

## Contact

For questions or access requests, please contact: [contact information]

## License

This dataset is released under the Creative Commons Attribution 4.0 International License (CC-BY-4.0).
"""
    
    return card_content

def upload_to_hub(dataset, repo_id, split_name, dataset_card, private=False):
    """Upload dataset to Hugging Face Hub."""
    
    print(f"\n{'='*60}")
    print("Uploading to Hugging Face Hub")
    print(f"{'='*60}")
    print(f"Repository: {repo_id}")
    print(f"Split: {split_name}")
    print(f"Private: {private}")
    print()
    
    # Check if user is logged in
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face")
        print("   Please run: huggingface-cli login")
        sys.exit(1)
    
    # Create dataset dict with split
    dataset_dict = DatasetDict({
        split_name: dataset
    })
    
    print(f"\nUploading dataset...")
    print(f"  This may take a while depending on dataset size...")
    
    try:
        # Push to hub
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            private=private,
            token=True
        )
        
        print(f"\n✓ Dataset uploaded successfully!")
        
        # Upload dataset card
        print(f"\nUploading dataset card...")
        api.upload_file(
            path_or_fileobj=dataset_card.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=True
        )
        
        print(f"✓ Dataset card uploaded!")
        print(f"\n{'='*60}")
        print(f"🎉 Upload complete!")
        print(f"{'='*60}")
        print(f"View your dataset at:")
        print(f"  https://huggingface.co/datasets/{repo_id}")
        print()
        
    except Exception as e:
        print(f"\n❌ Error during upload: {str(e)}")
        sys.exit(1)

def main():
    args = parse_arguments()
    
    print("="*60)
    print("RAPNIC Dataset Upload to Hugging Face")
    print("="*60)
    print(f"Data Pull: {args.data_pull_name}")
    print(f"Repository: {args.repo}")
    print(f"Split: {args.split}")
    print(f"Samples per speaker: {args.samples or 'ALL'}")
    print(f"Use clean only: {args.use_clean_only}")
    print(f"Private: {args.private}")
    print()
    
    # Verify paths
    data_dir, recordings_dir, analysis_dir = verify_paths(args.data_pull_name)
    
    # Load metadata
    df, is_combined = load_metadata(analysis_dir, args.use_clean_only)
    
    # Filter samples if requested
    df = filter_samples_per_speaker(df, args.samples)
    
    # Prepare dataset
    dataset = prepare_dataset(df, args.data_pull_name, is_combined)
    
    # Create dataset card
    dataset_card = create_dataset_card(
        args.data_pull_name,
        df,
        args.use_clean_only,
        args.samples,
        is_combined
    )
    
    # Upload to hub
    upload_to_hub(
        dataset,
        args.repo,
        args.split,
        dataset_card,
        args.private
    )

if __name__ == "__main__":
    main()
