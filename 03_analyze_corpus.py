#!/usr/bin/env python3
"""
RAPNIC Corpus Analyzer
Analyzes audio recordings to calculate durations, detect duplicates, and generate statistics.
Creates detailed recording manifests for clean, other, and all recordings.

Usage: python 03_analyze_corpus.py <DATA_PULL_NAME>
Example: python 03_analyze_corpus.py LREC-PAPER

Note: skips open prompts! 
"""

import os
import json
import wave
import sys
import csv
from pathlib import Path
from collections import defaultdict
from datetime import timedelta
import pandas as pd

#CONSTANTS

CUT_BACK_SECS = 0
LENGTH_THRESHOLD = 25
DONT_COUNT_OVER_THRESHOLD = True

##########    

def get_wav_duration(wav_path):
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(str(wav_path), 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"⚠️  Error reading {wav_path}: {e}")
        return 0

def load_json_metadata(json_path):
    """Load metadata from JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {}

def load_users_data(data_pull_name):
    """Load user demographics from Users.tsv"""
    users_file = Path(data_pull_name) / "Users.tsv"
    
    if not users_file.exists():
        print(f"⚠️  Warning: Users.tsv not found at {users_file}")
        return {}
    
    try:
        df = pd.read_csv(users_file, sep='\t')
        # Create a dictionary keyed by User ID
        users_dict = df.set_index('User ID').to_dict('index')
        return users_dict
    except Exception as e:
        print(f"⚠️  Error loading Users.tsv: {e}")
        return {}

def analyze_corpus(data_pull_name, cut_back_secs=2, length_threshold=15, dont_count_over_threshold=True):
    """
    Analyze voice data collection and calculate total time.
    Creates manifests for clean, other, and all recordings.
    
    Args:
        data_pull_name: Name of the data pull directory
        cut_back_secs: Seconds to subtract from each recording length
        length_threshold: Threshold in seconds for filtering recordings
        dont_count_over_threshold: If True, exclude recordings over threshold
    
    Returns:
        tuple: (stats dict, clean_recordings list, other_recordings list, all_recordings list)
    """
    
    # Verify we're in the DATA directory
    if not Path("scripts").exists():
        print("❌ ERROR: This script should be run from the DATA directory")
        print(f"   Current directory: {os.getcwd()}")
        print("   Expected to find: ./scripts/")
        sys.exit(1)
    
    base_path = Path(data_pull_name) / "audioapp_recordings"
    
    if not base_path.exists():
        print(f"❌ ERROR: Recording directory not found: {base_path}")
        print(f"   Make sure you've run: ./scripts/01_pull_recordings.sh {data_pull_name}")
        sys.exit(1)
    
    # Load user demographics
    print("Loading user demographics...")
    users_data = load_users_data(data_pull_name)
    print(f"✓ Loaded demographics for {len(users_data)} users\n")
    
    # Statistics
    speaker_count = 0
    total_recordings = 0
    unique_recordings = 0
    duplicate_recordings = 0
    over_threshold_recordings = 0
    
    # Duration tracking
    clean_duration_secs = 0  # Deduplicated + within threshold
    duplicate_duration_secs = 0  # Duplicate recordings only
    over_threshold_duration_secs = 0  # Over threshold only
    other_duration_secs = 0  # Duplicates + over threshold
    all_duration_secs = 0  # Everything
    
    speakers_data = {}
    
    # Track duplicates by task ID
    task_recordings = defaultdict(list)
    
    # Lists to store detailed recording info for manifests
    clean_recordings = []
    other_recordings = []
    all_recordings = []
    
    print("=" * 70)
    print("RAPNIC Corpus Analyzer")
    print("=" * 70)
    print(f"Data pull: {data_pull_name}")
    print(f"Recording directory: {base_path}")
    print(f"Parameters:")
    print(f"  - Cut back: {cut_back_secs} seconds")
    print(f"  - Length threshold: {length_threshold}s")
    print(f"  - Exclude over threshold: {dont_count_over_threshold}")
    print("")
    print("Note: Clean = deduplicated + within threshold")
    print("      Other = duplicates + over threshold")
    print("")
    
    # Iterate through speaker directories
    for speaker_dir in sorted(base_path.iterdir()):
        if not speaker_dir.is_dir():
            continue
        
        # Skip hidden directories
        if speaker_dir.name.startswith('.'):
            continue
        
        speaker_id = speaker_dir.name
        speaker_count += 1
        
        print(f"Processing speaker: {speaker_id}")
        
        # Get user demographics
        user_demo = users_data.get(speaker_id, {})
        
        # Process all WAV files for this speaker
        for wav_file in speaker_dir.glob("*.wav"):
            total_recordings += 1
            
            # Get duration
            original_duration = get_wav_duration(wav_file)
            trimmed_duration = max(0, original_duration - cut_back_secs)
            
            # Get filename
            filename = wav_file.stem
            
            # Try to load JSON metadata
            json_file = wav_file.with_suffix('.json')
            task_id = None
            timestamp = 0
            prompt = ""
            task_type = ""
            
            if json_file.exists():
                metadata = load_json_metadata(json_file)
                task_id = metadata.get('task', None)
                timestamp = metadata.get('timestamp', 0)
                prompt = metadata.get('prompt', '')
                task_type = metadata.get('taskType', '')

            # If task type isn't prompt skip
            if task_type != 'prompt':
                continue
            
            # If no timestamp from JSON, use 0
            if timestamp == 0:
                timestamp = 0
            
            # If no task ID, fall back to using filename
            if not task_id:
                task_id = filename.split('_')[-1] if '_' in filename else filename
            
            # Create unique task key
            task_key = f"{speaker_id}_{task_id}"
            
            # Store recording info
            recording_info = {
                'speaker_id': speaker_id,
                'filename': wav_file.name,
                'file_path': str(wav_file.relative_to(Path(data_pull_name))),
                'task_id': task_id,
                'original_duration': original_duration,
                'trimmed_duration': trimmed_duration,
                'timestamp': timestamp,
                'prompt': prompt,
                # User demographics
                'province': user_demo.get('province', ''),
                'city': user_demo.get('city', ''),
                'dialect': user_demo.get('dialect', ''),
                'disorder': user_demo.get('disorder', ''),
                'gender': user_demo.get('gender', ''),
                'age': user_demo.get('age', ''),
                'accessDevices': user_demo.get('accessDevices', ''),
                'hasHelper': user_demo.get('hasHelper', ''),
                'numRecordings': user_demo.get('numRecordings', ''),
                'numTasks': user_demo.get('numTasks', ''),
                'numCompletedTasks': user_demo.get('numCompletedTasks', '')
            }
            
            task_recordings[task_key].append(recording_info)
        
        speakers_data[speaker_id] = {
            'recordings_clean': 0,
            'duration_clean': 0,
            'recordings_other': 0,
            'duration_other': 0
        }
    
    print(f"\n{'=' * 70}")
    print("Processing duplicates and applying filters...")
    print("")
    
    # Process each task group
    for task_key, recordings in task_recordings.items():
        speaker_id = recordings[0]['speaker_id']
        
        # Sort by timestamp to get the latest
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_recording = recordings[0]
        
        # Mark recording type
        is_duplicate = False
        is_over_threshold = False
        
        if len(recordings) > 1:
            # This is a duplicate situation
            duplicate_recordings += len(recordings) - 1
            
            # The latest is clean (potentially), others are duplicates
            for rec in recordings[1:]:
                rec['category'] = 'duplicate'
                rec['reason'] = 'Duplicate recording (older version)'
                duplicate_duration_secs += rec['trimmed_duration']
                other_duration_secs += rec['trimmed_duration']
                other_recordings.append(rec)
                all_recordings.append(rec)
                
                speakers_data[speaker_id]['recordings_other'] += 1
                speakers_data[speaker_id]['duration_other'] += rec['trimmed_duration']
        
        # Now check the latest recording
        unique_recordings += 1
        
        # Check if over threshold
        if dont_count_over_threshold and latest_recording['trimmed_duration'] > length_threshold:
            is_over_threshold = True
            over_threshold_recordings += 1
            over_threshold_duration_secs += latest_recording['trimmed_duration']
            other_duration_secs += latest_recording['trimmed_duration']
            
            latest_recording['category'] = 'over_threshold'
            latest_recording['reason'] = f"Over threshold ({latest_recording['trimmed_duration']:.2f}s > {length_threshold}s)"
            other_recordings.append(latest_recording)
            all_recordings.append(latest_recording)
            
            speakers_data[speaker_id]['recordings_other'] += 1
            speakers_data[speaker_id]['duration_other'] += latest_recording['trimmed_duration']
        else:
            # This is a clean recording
            clean_duration_secs += latest_recording['trimmed_duration']
            
            latest_recording['category'] = 'clean'
            latest_recording['reason'] = 'Clean (latest, within threshold)'
            clean_recordings.append(latest_recording)
            all_recordings.append(latest_recording)
            
            speakers_data[speaker_id]['recordings_clean'] += 1
            speakers_data[speaker_id]['duration_clean'] += latest_recording['trimmed_duration']
    
    # Calculate all duration
    all_duration_secs = clean_duration_secs + other_duration_secs
    
    # Calculate derived statistics
    clean_mins = clean_duration_secs / 60
    clean_hours = clean_duration_secs / 3600
    
    other_mins = other_duration_secs / 60
    other_hours = other_duration_secs / 3600
    
    all_mins = all_duration_secs / 60
    all_hours = all_duration_secs / 3600
    
    duplicate_mins = duplicate_duration_secs / 60
    duplicate_hours = duplicate_duration_secs / 3600
    
    over_threshold_mins = over_threshold_duration_secs / 60
    over_threshold_hours = over_threshold_duration_secs / 3600
    
    avg_duration_per_recording = clean_duration_secs / len(clean_recordings) if len(clean_recordings) > 0 else 0
    avg_recordings_per_speaker = len(clean_recordings) / speaker_count if speaker_count > 0 else 0
    
    # Generate statistics dictionary
    stats = {
        'data_pull_name': data_pull_name,
        'parameters': {
            'cut_back_seconds': cut_back_secs,
            'length_threshold': length_threshold,
            'exclude_over_threshold': dont_count_over_threshold
        },
        'speakers': {
            'total': speaker_count,
            'list': list(speakers_data.keys())
        },
        'recordings': {
            'total_found': total_recordings,
            'unique': unique_recordings,
            'clean': len(clean_recordings),
            'duplicates': duplicate_recordings,
            'over_threshold': over_threshold_recordings,
            'other': len(other_recordings),
            'avg_per_speaker_clean': round(avg_recordings_per_speaker, 2)
        },
        'duration': {
            'clean': {
                'total_seconds': round(clean_duration_secs, 2),
                'total_minutes': round(clean_mins, 2),
                'total_hours': round(clean_hours, 2),
                'avg_per_recording_seconds': round(avg_duration_per_recording, 2)
            },
            'other': {
                'total_seconds': round(other_duration_secs, 2),
                'total_minutes': round(other_mins, 2),
                'total_hours': round(other_hours, 2),
                'duplicates_seconds': round(duplicate_duration_secs, 2),
                'duplicates_minutes': round(duplicate_mins, 2),
                'duplicates_hours': round(duplicate_hours, 2),
                'over_threshold_seconds': round(over_threshold_duration_secs, 2),
                'over_threshold_minutes': round(over_threshold_mins, 2),
                'over_threshold_hours': round(over_threshold_hours, 2)
            },
            'all': {
                'total_seconds': round(all_duration_secs, 2),
                'total_minutes': round(all_mins, 2),
                'total_hours': round(all_hours, 2)
            }
        },
        'speakers_detail': {
            speaker_id: {
                'recordings_clean': data['recordings_clean'],
                'duration_clean_seconds': round(data['duration_clean'], 2),
                'duration_clean_minutes': round(data['duration_clean'] / 60, 2),
                'duration_clean_hours': round(data['duration_clean'] / 3600, 2),
                'recordings_other': data['recordings_other'],
                'duration_other_seconds': round(data['duration_other'], 2),
                'duration_other_minutes': round(data['duration_other'] / 60, 2),
                'duration_other_hours': round(data['duration_other'] / 3600, 2)
            }
            for speaker_id, data in speakers_data.items()
        }
    }
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Data Pull: {data_pull_name}")
    print("")
    print(f"Speakers: {speaker_count}")
    print(f"Total recordings found: {total_recordings}")
    print(f"Unique recordings (after deduplication): {unique_recordings}")
    print(f"  - Clean (within threshold): {len(clean_recordings)}")
    print(f"  - Over threshold: {over_threshold_recordings}")
    print(f"Duplicates removed: {duplicate_recordings}")
    print("")
    print(f"CLEAN Corpus Duration (deduplicated + within threshold):")
    print(f"  {clean_hours:.2f} hours")
    print(f"  {clean_mins:.2f} minutes")
    print(f"  {clean_duration_secs:.2f} seconds")
    print("")
    print(f"OTHER Recordings Duration (duplicates + over threshold):")
    print(f"  {other_hours:.2f} hours")
    print(f"  {other_mins:.2f} minutes")
    print(f"  {other_duration_secs:.2f} seconds")
    print(f"  - Duplicates only: {duplicate_hours:.2f} hours ({duplicate_duration_secs:.2f}s)")
    print(f"  - Over threshold only: {over_threshold_hours:.2f} hours ({over_threshold_duration_secs:.2f}s)")
    print("")
    print(f"ALL Recordings Duration (clean + other):")
    print(f"  {all_hours:.2f} hours")
    print(f"  {all_mins:.2f} minutes")
    print(f"  {all_duration_secs:.2f} seconds")
    print("")
    print(f"Average duration per clean recording: {avg_duration_per_recording:.2f} seconds")
    print(f"Average clean recordings per speaker: {avg_recordings_per_speaker:.2f}")
    print("=" * 70)
    
    return stats, clean_recordings, other_recordings, all_recordings

def save_results(stats, clean_recordings, other_recordings, all_recordings, data_pull_name):
    """Save analysis results to JSON, text, and CSV files."""
    
    results_dir = Path(data_pull_name) / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON stats
    json_file = results_dir / "corpus_stats.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Save human-readable report
    report_file = results_dir / "corpus_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPNIC CORPUS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data Pull: {stats['data_pull_name']}\n")
        f.write(f"\nAnalysis Parameters:\n")
        f.write(f"  Cut back: {stats['parameters']['cut_back_seconds']} seconds\n")
        f.write(f"  Length threshold: {stats['parameters']['length_threshold']} seconds\n")
        f.write(f"  Exclude over threshold: {stats['parameters']['exclude_over_threshold']}\n")
        f.write(f"\nSpeakers: {stats['speakers']['total']}\n")
        f.write(f"Total recordings found: {stats['recordings']['total_found']}\n")
        f.write(f"Unique recordings (after deduplication): {stats['recordings']['unique']}\n")
        f.write(f"  - Clean (within threshold): {stats['recordings']['clean']}\n")
        f.write(f"  - Over threshold: {stats['recordings']['over_threshold']}\n")
        f.write(f"Duplicates removed: {stats['recordings']['duplicates']}\n")
        
        f.write(f"\n{'=' * 70}\n")
        f.write("DURATION STATISTICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("CLEAN Corpus (deduplicated + within threshold):\n")
        f.write(f"  {stats['duration']['clean']['total_hours']:.2f} hours\n")
        f.write(f"  {stats['duration']['clean']['total_minutes']:.2f} minutes\n")
        f.write(f"  {stats['duration']['clean']['total_seconds']:.2f} seconds\n\n")
        
        f.write("OTHER Recordings (duplicates + over threshold):\n")
        f.write(f"  {stats['duration']['other']['total_hours']:.2f} hours\n")
        f.write(f"  {stats['duration']['other']['total_minutes']:.2f} minutes\n")
        f.write(f"  {stats['duration']['other']['total_seconds']:.2f} seconds\n")
        f.write(f"  - Duplicates only: {stats['duration']['other']['duplicates_hours']:.2f} hours\n")
        f.write(f"  - Over threshold only: {stats['duration']['other']['over_threshold_hours']:.2f} hours\n\n")
        
        f.write("ALL Recordings (clean + other):\n")
        f.write(f"  {stats['duration']['all']['total_hours']:.2f} hours\n")
        f.write(f"  {stats['duration']['all']['total_minutes']:.2f} minutes\n")
        f.write(f"  {stats['duration']['all']['total_seconds']:.2f} seconds\n\n")
        
        f.write(f"Averages:\n")
        f.write(f"  Duration per clean recording: {stats['duration']['clean']['avg_per_recording_seconds']:.2f} seconds\n")
        f.write(f"  Clean recordings per speaker: {stats['recordings']['avg_per_speaker_clean']:.2f}\n")
        
        f.write(f"\n{'=' * 70}\n")
        f.write("PER-SPEAKER DETAILS\n")
        f.write("=" * 70 + "\n\n")
        
        # Create a formatted table
        f.write(f"{'Speaker':<12} {'Clean Recs':<12} {'Clean Hrs':<12} {'Other Recs':<12} {'Other Hrs':<12}\n")
        f.write("-" * 70 + "\n")
        
        for speaker_id, speaker_stats in sorted(stats['speakers_detail'].items()):
            f.write(f"{speaker_id:<12} {speaker_stats['recordings_clean']:<12} "
                   f"{speaker_stats['duration_clean_hours']:<12.2f} "
                   f"{speaker_stats['recordings_other']:<12} "
                   f"{speaker_stats['duration_other_hours']:<12.2f}\n")
    
    # Save CSV manifests
    csv_fields = ['speaker_id', 'filename', 'file_path', 'task_id', 'category', 'reason',
                  'original_duration', 'trimmed_duration', 'timestamp', 'prompt',
                  'province', 'city', 'dialect', 'disorder', 'gender', 'age',
                  'accessDevices', 'hasHelper', 'numRecordings', 'numTasks', 'numCompletedTasks']
    
    # Clean recordings manifest
    clean_csv = results_dir / "recordings_clean.csv"
    with open(clean_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(clean_recordings)
    
    # Other recordings manifest
    other_csv = results_dir / "recordings_other.csv"
    with open(other_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(other_recordings)
    
    # All recordings manifest
    all_csv = results_dir / "recordings_all.csv"
    with open(all_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_recordings)
    
    print(f"\n✓ Results saved:")
    print(f"  JSON stats: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Clean recordings CSV: {clean_csv} ({len(clean_recordings)} recordings)")
    print(f"  Other recordings CSV: {other_csv} ({len(other_recordings)} recordings)")
    print(f"  All recordings CSV: {all_csv} ({len(all_recordings)} recordings)")

def main():
    if len(sys.argv) != 2:
        print("ERROR: No data pull name specified")
        print(f"Usage: {sys.argv[0]} <DATA_PULL_NAME>")
        print("Example: python 03_analyze_corpus.py LREC-PAPER")
        sys.exit(1)
    
    data_pull_name = sys.argv[1]
    
    # Run analysis
    stats, clean_recordings, other_recordings, all_recordings = analyze_corpus(data_pull_name, 
                                                                               cut_back_secs=CUT_BACK_SECS, 
                                                                               length_threshold=LENGTH_THRESHOLD, 
                                                                               dont_count_over_threshold=DONT_COUNT_OVER_THRESHOLD)
    


    # Save results
    save_results(stats, clean_recordings, other_recordings, all_recordings, data_pull_name)
    
    print(f"\nNext step:")
    print(f"  python scripts/04_analyze_demographics.py {data_pull_name}")

if __name__ == "__main__":
    main()
