#!/usr/bin/env python3
"""
RAPNIC Corpus Combiner
Combines multiple data pulls into a unified corpus with statistics and manifests.

Usage: python 05_combine_corpus.py <DATA_PULL_1> <DATA_PULL_2> [DATA_PULL_3] ...
Example: python 05_combine_corpus.py PILOT LREC-PAPER
         python 05_combine_corpus.py PILOT LREC-PAPER PHASE2
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

def load_corpus_stats(data_pull_name):
    """Load corpus statistics from a data pull."""
    stats_file = Path(data_pull_name) / "analysis_results" / "corpus_stats.json"
    
    if not stats_file.exists():
        print(f"❌ ERROR: Stats file not found: {stats_file}")
        print(f"   Make sure you've run: python scripts/03_analyze_corpus.py {data_pull_name}")
        return None
    
    try:
        with open(stats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ ERROR: Failed to load {stats_file}: {e}")
        return None

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

def combine_data_pulls(data_pull_names):
    """
    Combine multiple data pulls into unified statistics and manifests.
    
    Args:
        data_pull_names: List of data pull directory names
    
    Returns:
        tuple: (combined_stats, combined_clean, combined_other, combined_all)
    """
    
    print("=" * 70)
    print("RAPNIC Corpus Combiner")
    print("=" * 70)
    print(f"Combining data pulls: {', '.join(data_pull_names)}")
    print("")
    
    # Load stats from all data pulls
    all_stats = {}
    for data_pull in data_pull_names:
        print(f"Loading statistics from {data_pull}...")
        stats = load_corpus_stats(data_pull)
        if stats is None:
            sys.exit(1)
        all_stats[data_pull] = stats
        print(f"  ✓ {stats['recordings']['clean']} clean recordings, "
              f"{stats['duration']['clean']['total_hours']:.2f} hours")
    
    print("")
    
    # Combine recordings from all data pulls
    print("Combining recording manifests...")
    combined_clean = []
    combined_other = []
    combined_all = []
    
    for data_pull in data_pull_names:
        clean = load_recordings_csv(data_pull, 'clean')
        other = load_recordings_csv(data_pull, 'other')
        all_recs = load_recordings_csv(data_pull, 'all')
        
        combined_clean.extend(clean)
        combined_other.extend(other)
        combined_all.extend(all_recs)
        
        print(f"  ✓ {data_pull}: {len(clean)} clean, {len(other)} other, {len(all_recs)} total")
    
    print("")
    
    # Calculate combined statistics
    print("Calculating combined statistics...")
    
    # Aggregate speakers across data pulls
    all_speakers = set()
    for stats in all_stats.values():
        all_speakers.update(stats['speakers']['list'])
    
    # Aggregate recordings
    total_recordings_found = sum(s['recordings']['total_found'] for s in all_stats.values())
    total_unique = sum(s['recordings']['unique'] for s in all_stats.values())
    total_clean = len(combined_clean)
    total_duplicates = sum(s['recordings']['duplicates'] for s in all_stats.values())
    total_over_threshold = sum(s['recordings']['over_threshold'] for s in all_stats.values())
    total_other = len(combined_other)
    
    # Aggregate durations
    clean_duration_secs = sum(s['duration']['clean']['total_seconds'] for s in all_stats.values())
    other_duration_secs = sum(s['duration']['other']['total_seconds'] for s in all_stats.values())
    all_duration_secs = sum(s['duration']['all']['total_seconds'] for s in all_stats.values())
    
    duplicate_duration_secs = sum(s['duration']['other']['duplicates_seconds'] for s in all_stats.values())
    over_threshold_duration_secs = sum(s['duration']['other']['over_threshold_seconds'] for s in all_stats.values())
    
    # Per-speaker combined statistics
    speakers_combined = defaultdict(lambda: {
        'recordings_clean': 0,
        'duration_clean_seconds': 0,
        'recordings_other': 0,
        'duration_other_seconds': 0
    })
    
    for data_pull, stats in all_stats.items():
        for speaker_id, speaker_data in stats['speakers_detail'].items():
            speakers_combined[speaker_id]['recordings_clean'] += speaker_data['recordings_clean']
            speakers_combined[speaker_id]['duration_clean_seconds'] += speaker_data['duration_clean_seconds']
            speakers_combined[speaker_id]['recordings_other'] += speaker_data['recordings_other']
            speakers_combined[speaker_id]['duration_other_seconds'] += speaker_data['duration_other_seconds']
    
    # Convert to final format
    speakers_detail = {}
    for speaker_id, data in speakers_combined.items():
        speakers_detail[speaker_id] = {
            'recordings_clean': data['recordings_clean'],
            'duration_clean_seconds': round(data['duration_clean_seconds'], 2),
            'duration_clean_minutes': round(data['duration_clean_seconds'] / 60, 2),
            'duration_clean_hours': round(data['duration_clean_seconds'] / 3600, 2),
            'recordings_other': data['recordings_other'],
            'duration_other_seconds': round(data['duration_other_seconds'], 2),
            'duration_other_minutes': round(data['duration_other_seconds'] / 60, 2),
            'duration_other_hours': round(data['duration_other_seconds'] / 3600, 2)
        }
    
    # Create combined stats dictionary
    combined_stats = {
        'data_pulls': data_pull_names,
        'combined_name': '_'.join(data_pull_names),
        'speakers': {
            'total': len(all_speakers),
            'list': sorted(list(all_speakers))
        },
        'recordings': {
            'total_found': total_recordings_found,
            'unique': total_unique,
            'clean': total_clean,
            'duplicates': total_duplicates,
            'over_threshold': total_over_threshold,
            'other': total_other,
            'avg_per_speaker_clean': round(total_clean / len(all_speakers), 2) if len(all_speakers) > 0 else 0
        },
        'duration': {
            'clean': {
                'total_seconds': round(clean_duration_secs, 2),
                'total_minutes': round(clean_duration_secs / 60, 2),
                'total_hours': round(clean_duration_secs / 3600, 2),
                'avg_per_recording_seconds': round(clean_duration_secs / total_clean, 2) if total_clean > 0 else 0
            },
            'other': {
                'total_seconds': round(other_duration_secs, 2),
                'total_minutes': round(other_duration_secs / 60, 2),
                'total_hours': round(other_duration_secs / 3600, 2),
                'duplicates_seconds': round(duplicate_duration_secs, 2),
                'duplicates_minutes': round(duplicate_duration_secs / 60, 2),
                'duplicates_hours': round(duplicate_duration_secs / 3600, 2),
                'over_threshold_seconds': round(over_threshold_duration_secs, 2),
                'over_threshold_minutes': round(over_threshold_duration_secs / 60, 2),
                'over_threshold_hours': round(over_threshold_duration_secs / 3600, 2)
            },
            'all': {
                'total_seconds': round(all_duration_secs, 2),
                'total_minutes': round(all_duration_secs / 60, 2),
                'total_hours': round(all_duration_secs / 3600, 2)
            }
        },
        'speakers_detail': speakers_detail,
        'per_data_pull': {
            data_pull: {
                'speakers': stats['speakers']['total'],
                'recordings_clean': stats['recordings']['clean'],
                'duration_clean_hours': stats['duration']['clean']['total_hours']
            }
            for data_pull, stats in all_stats.items()
        }
    }
    
    # Print summary
    print("")
    print("=" * 70)
    print("COMBINED CORPUS SUMMARY")
    print("=" * 70)
    print(f"Data pulls combined: {', '.join(data_pull_names)}")
    print("")
    print(f"Total speakers: {len(all_speakers)}")
    print(f"Total recordings found: {total_recordings_found}")
    print(f"Clean recordings: {total_clean}")
    print(f"Other recordings: {total_other}")
    print("")
    print(f"CLEAN Corpus Duration:")
    print(f"  {combined_stats['duration']['clean']['total_hours']:.2f} hours")
    print(f"  {combined_stats['duration']['clean']['total_minutes']:.2f} minutes")
    print("")
    print(f"OTHER Recordings Duration:")
    print(f"  {combined_stats['duration']['other']['total_hours']:.2f} hours")
    print(f"  {combined_stats['duration']['other']['total_minutes']:.2f} minutes")
    print("")
    print(f"ALL Recordings Duration:")
    print(f"  {combined_stats['duration']['all']['total_hours']:.2f} hours")
    print(f"  {combined_stats['duration']['all']['total_minutes']:.2f} minutes")
    print("")
    print("Per Data Pull Breakdown:")
    for data_pull in data_pull_names:
        info = combined_stats['per_data_pull'][data_pull]
        print(f"  {data_pull}: {info['speakers']} speakers, "
              f"{info['recordings_clean']} clean recs, "
              f"{info['duration_clean_hours']:.2f} hours")
    print("=" * 70)
    
    return combined_stats, combined_clean, combined_other, combined_all

def save_combined_results(combined_stats, combined_clean, combined_other, combined_all):
    """Save combined results to a new directory."""
    
    # Create combined directory name
    combined_name = f"COMBINED_{'_'.join(combined_stats['data_pulls'])}"
    combined_dir = Path(combined_name)
    results_dir = combined_dir / "analysis_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving combined results to: {combined_dir}")
    
    # Save combined statistics JSON
    stats_file = results_dir / "corpus_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(combined_stats, f, indent=2, ensure_ascii=False)
    
    # Save comparison report
    report_file = results_dir / "corpus_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPNIC COMBINED CORPUS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data Pulls Combined: {', '.join(combined_stats['data_pulls'])}\n")
        f.write(f"\nTotal Speakers: {combined_stats['speakers']['total']}\n")
        f.write(f"Total Recordings Found: {combined_stats['recordings']['total_found']}\n")
        f.write(f"Clean Recordings: {combined_stats['recordings']['clean']}\n")
        f.write(f"Other Recordings: {combined_stats['recordings']['other']}\n")
        f.write(f"  - Duplicates: {combined_stats['recordings']['duplicates']}\n")
        f.write(f"  - Over Threshold: {combined_stats['recordings']['over_threshold']}\n")
        
        f.write(f"\n{'=' * 70}\n")
        f.write("DURATION STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CLEAN Corpus:\n")
        f.write(f"  {combined_stats['duration']['clean']['total_hours']:.2f} hours\n")
        f.write(f"  {combined_stats['duration']['clean']['total_minutes']:.2f} minutes\n")
        f.write(f"  {combined_stats['duration']['clean']['total_seconds']:.2f} seconds\n\n")
        
        f.write("OTHER Recordings:\n")
        f.write(f"  {combined_stats['duration']['other']['total_hours']:.2f} hours\n")
        f.write(f"  {combined_stats['duration']['other']['total_minutes']:.2f} minutes\n")
        f.write(f"  - Duplicates: {combined_stats['duration']['other']['duplicates_hours']:.2f} hours\n")
        f.write(f"  - Over Threshold: {combined_stats['duration']['other']['over_threshold_hours']:.2f} hours\n\n")
        
        f.write("ALL Recordings:\n")
        f.write(f"  {combined_stats['duration']['all']['total_hours']:.2f} hours\n")
        f.write(f"  {combined_stats['duration']['all']['total_minutes']:.2f} minutes\n\n")
        
        f.write(f"{'=' * 70}\n")
        f.write("PER DATA PULL BREAKDOWN\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Data Pull':<20} {'Speakers':<12} {'Clean Recs':<12} {'Duration (hrs)':<15}\n")
        f.write("-" * 70 + "\n")
        for data_pull in combined_stats['data_pulls']:
            info = combined_stats['per_data_pull'][data_pull]
            f.write(f"{data_pull:<20} {info['speakers']:<12} "
                   f"{info['recordings_clean']:<12} {info['duration_clean_hours']:<15.2f}\n")
        
        f.write(f"\n{'=' * 70}\n")
        f.write("PER-SPEAKER COMBINED DETAILS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"{'Speaker':<12} {'Clean Recs':<12} {'Clean Hrs':<12} {'Other Recs':<12} {'Other Hrs':<12}\n")
        f.write("-" * 70 + "\n")
        
        for speaker_id, speaker_stats in sorted(combined_stats['speakers_detail'].items()):
            f.write(f"{speaker_id:<12} {speaker_stats['recordings_clean']:<12} "
                   f"{speaker_stats['duration_clean_hours']:<12.2f} "
                   f"{speaker_stats['recordings_other']:<12} "
                   f"{speaker_stats['duration_other_hours']:<12.2f}\n")
    
    # Save combined CSV manifests
    csv_fields = ['data_pull', 'speaker_id', 'filename', 'file_path', 'task_id', 'category', 'reason',
                  'original_duration', 'trimmed_duration', 'timestamp', 'prompt',
                  'province', 'city', 'dialect', 'disorder', 'gender', 'age',
                  'accessDevices', 'hasHelper', 'numRecordings', 'numTasks', 'numCompletedTasks']
    
    # Clean recordings
    clean_csv = results_dir / "recordings_clean.csv"
    with open(clean_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(combined_clean)
    
    # Other recordings
    other_csv = results_dir / "recordings_other.csv"
    with open(other_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(combined_other)
    
    # All recordings
    all_csv = results_dir / "recordings_all.csv"
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(combined_all)
    
    print(f"\n✓ Results saved to: {results_dir}")
    print(f"  JSON stats: {stats_file}")
    print(f"  Report: {report_file}")
    print(f"  Clean recordings CSV: {clean_csv} ({len(combined_clean)} recordings)")
    print(f"  Other recordings CSV: {other_csv} ({len(combined_other)} recordings)")
    print(f"  All recordings CSV: {all_csv} ({len(combined_all)} recordings)")
    
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
    
    # Combine the data pulls
    combined_stats, combined_clean, combined_other, combined_all = combine_data_pulls(data_pull_names)
    
    # Save combined results
    combined_dir = save_combined_results(combined_stats, combined_clean, combined_other, combined_all)
    
    print(f"\n✅ Successfully combined {len(data_pull_names)} data pulls into: {combined_dir}")
    print(f"\nNext steps:")
    print(f"  - Review combined statistics: {combined_dir}/analysis_results/corpus_report.txt")
    print(f"  - Analyze combined data: {combined_dir}/analysis_results/recordings_clean.csv")

if __name__ == "__main__":
    main()