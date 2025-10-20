#!/usr/bin/env python3
"""
RAPNIC Corpus Combiner
Combines CSV manifests from multiple data pulls into unified CSVs.

Usage: python 05_combine_corpus.py <DATA_PULL_1> <DATA_PULL_2> [DATA_PULL_3] ...
Example: python 05_combine_corpus.py PILOT LREC-PAPER
         python 05_combine_corpus.py PILOT LREC-PAPER PHASE2
"""

import sys
import csv
from pathlib import Path

def load_recordings_csv(data_pull_name, category):
    """Load recordings CSV from a data pull and add data_pull column."""
    csv_file = Path(data_pull_name) / "analysis_results" / f"recordings_{category}.csv"
    
    if not csv_file.exists():
        print(f"⚠️  Warning: CSV file not found: {csv_file}")
        return []
    
    recordings = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['data_pull'] = data_pull_name  # Add data pull identifier
                recordings.append(row)
        return recordings
    except Exception as e:
        print(f"⚠️  Warning: Failed to load {csv_file}: {e}")
        return []

def combine_csvs(data_pull_names):
    """
    Combine CSV manifests from multiple data pulls.
    
    Args:
        data_pull_names: List of data pull directory names
    
    Returns:
        tuple: (combined_clean, combined_other, combined_all)
    """
    
    print("=" * 70)
    print("RAPNIC Corpus CSV Combiner")
    print("=" * 70)
    print(f"Combining data pulls: {', '.join(data_pull_names)}")
    print("")
    
    combined_clean = []
    combined_other = []
    combined_all = []
    
    for data_pull in data_pull_names:
        print(f"Loading CSVs from {data_pull}...")
        
        # Check if analysis results exist
        analysis_dir = Path(data_pull) / "analysis_results"
        if not analysis_dir.exists():
            print(f"❌ ERROR: Analysis results not found for {data_pull}")
            print(f"   Run: python scripts/03_analyze_corpus.py {data_pull}")
            sys.exit(1)
        
        clean = load_recordings_csv(data_pull, 'clean')
        other = load_recordings_csv(data_pull, 'other')
        all_recs = load_recordings_csv(data_pull, 'all')
        
        if not clean and not other and not all_recs:
            print(f"❌ ERROR: No CSV files found in {analysis_dir}")
            sys.exit(1)
        
        combined_clean.extend(clean)
        combined_other.extend(other)
        combined_all.extend(all_recs)
        
        print(f"  ✓ {len(clean)} clean, {len(other)} other, {len(all_recs)} total")
    
    print("")
    print(f"Combined totals:")
    print(f"  Clean: {len(combined_clean)} recordings")
    print(f"  Other: {len(combined_other)} recordings")
    print(f"  All: {len(combined_all)} recordings")
    
    return combined_clean, combined_other, combined_all

def save_combined_csvs(combined_clean, combined_other, combined_all, data_pull_names):
    """Save combined CSV files to a new directory."""
    
    # Create combined directory name
    combined_name = f"COMBINED_{'_'.join(data_pull_names)}"
    combined_dir = Path(combined_name)
    results_dir = combined_dir / "analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 70}")
    print(f"Saving combined CSVs to: {combined_dir}")
    print("")
    
    # Define CSV fields - get from first recording if available
    if combined_all:
        csv_fields = list(combined_all[0].keys())
    else:
        # Fallback to standard fields
        csv_fields = ['data_pull', 'speaker_id', 'filename', 'file_path', 'task_id', 
                     'category', 'reason', 'original_duration', 'trimmed_duration', 
                     'timestamp', 'prompt', 'province', 'city', 'age', 'gender', 
                     'disorder', 'dialect', 'hasHelper']
    
    # Save clean recordings CSV
    clean_csv = results_dir / "recordings_clean.csv"
    with open(clean_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(combined_clean)
    
    # Save other recordings CSV
    other_csv = results_dir / "recordings_other.csv"
    with open(other_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(combined_other)
    
    # Save all recordings CSV
    all_csv = results_dir / "recordings_all.csv"
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(combined_all)
    
    print(f"✓ Results saved:")
    print(f"  Clean: {clean_csv} ({len(combined_clean)} recordings)")
    print(f"  Other: {other_csv} ({len(combined_other)} recordings)")
    print(f"  All: {all_csv} ({len(combined_all)} recordings)")
    
    return combined_dir

def main():
    if len(sys.argv) < 3:
        print("ERROR: Need at least 2 data pulls to combine")
        print(f"Usage: {sys.argv[0]} <DATA_PULL_1> <DATA_PULL_2> [DATA_PULL_3] ...")
        print("Example: python 05_combine_corpus.py PILOT LREC-PAPER")
        print("         python 05_combine_corpus.py PILOT LREC-PAPER PHASE2")
        sys.exit(1)
    
    data_pull_names = sys.argv[1:]
    
    # Verify we're in the DATA directory
    if not Path("scripts").exists():
        print("❌ ERROR: This script should be run from the DATA directory")
        print(f"   Current directory: {Path.cwd()}")
        sys.exit(1)
    
    # Verify all data pulls exist
    for data_pull in data_pull_names:
        if not Path(data_pull).exists():
            print(f"❌ ERROR: Data pull directory not found: {data_pull}")
            sys.exit(1)
    
    # Combine the CSV files
    combined_clean, combined_other, combined_all = combine_csvs(data_pull_names)
    
    # Save combined CSVs
    combined_dir = save_combined_csvs(combined_clean, combined_other, combined_all, data_pull_names)
    
    print(f"\n{'=' * 70}")
    print(f"✅ Successfully combined {len(data_pull_names)} data pulls")
    print(f"\nNext steps:")
    print(f"  1. Generate statistics and export audio:")
    print(f"     python scripts/03_analyze_corpus.py {combined_dir} --export-audio")

if __name__ == "__main__":
    main()