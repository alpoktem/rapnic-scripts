#!/usr/bin/env python3
"""
RAPNIC Corpus Analyzer with Audio Processing
Analyzes audio recordings, detects duplicates, calculates durations, and optionally exports processed audio.

Usage: 
  python 03_analyze_and_pack_corpus.py <DATA_PULL_NAME> [OPTIONS]
  
Examples:
  python 03_analyze_and_pack_corpus.py LREC-PAPER
  python 03_analyze_and_pack_corpus.py LREC-PAPER --export-audio --audio-format wav16k
  python 03_analyze_and_pack_corpus.py PILOT --export-audio

Note: skips open prompts that have taskType label 'response'! 
"""

import os
import json
import wave
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd
import librosa
import soundfile as sf
import numpy as np

#CONSTANTS
LENGTH_THRESHOLD = 25
DONT_COUNT_OVER_THRESHOLD = True
DEFAULT_TRIM_PAD_DURATION = 0.25

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

def trim_audio_librosa(wav_path, pad_duration=DEFAULT_TRIM_PAD_DURATION):
    """
    Adaptively trim audio based on its characteristics.
    Uses dynamic range analysis to determine optimal trimming threshold.
    """
    try:
        y, sr = librosa.load(str(wav_path), sr=None)
        
        # Calculate RMS energy to understand the audio
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        
        # Find the peak RMS and background noise level
        peak_rms = np.max(rms)
        noise_floor = np.percentile(rms, 10)  # Bottom 10% is likely noise
        
        # Calculate dynamic range
        dynamic_range = 20 * np.log10(peak_rms / (noise_floor + 1e-10))
        
        # Adapt top_db based on dynamic range
        if dynamic_range > 40:  # High quality recording with low noise
            top_db = 20
        elif dynamic_range > 30:  # Moderate noise
            top_db = 18
        else:  # Noisy recording
            top_db = 15
        
        # Trim with adaptive threshold
        y_trimmed, indices = librosa.effects.trim(y, top_db=top_db)
        
        # Calculate padding in samples
        pad_samples = int(pad_duration * sr)
        
        # Expand the trim boundaries
        start_idx = max(0, indices[0] - pad_samples)
        end_idx = min(len(y), indices[1] + pad_samples)
        
        y_with_buffer = y[start_idx:end_idx]
        
        duration = len(y_with_buffer) / sr
        return duration, y_with_buffer, sr
    except Exception as e:
        print(f"⚠️  Error trimming {wav_path}: {e}")
        try:
            y, sr = librosa.load(str(wav_path), sr=None)
            return len(y) / sr, y, sr
        except:
            return 0, None, None

def trim_audio_constant(wav_path, seconds=2):
    """
    Trim a constant number of seconds from the beginning.
    
    Args:
        wav_path: Path to WAV file  
        seconds: Number of seconds to trim from beginning
        
    Returns:
        Tuple of (duration, trimmed_audio, sample_rate)
    """
    try:
        y, sr = librosa.load(str(wav_path), sr=None)
        samples_to_trim = int(seconds * sr)
        y_trimmed = y[samples_to_trim:] if len(y) > samples_to_trim else y
        duration = len(y_trimmed) / sr
        return duration, y_trimmed, sr
    except Exception as e:
        print(f"⚠️  Error trimming {wav_path}: {e}")
        return 0, None, None

def export_audio(audio_data, sr, output_path, audio_format='wav16k'):
    """
    Export audio in the specified format.
    
    Args:
        audio_data: Audio samples array
        sr: Sample rate
        output_path: Output file path
        audio_format: Output format ('wav16k', 'wav48k', 'wav')
    """
    if audio_data is None:
        return False
        
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if audio_format == 'wav16k':
            # Resample to 16kHz if needed
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sf.write(output_path, audio_data, 16000)
        elif audio_format == 'wav48k':
            # Resample to 48kHz if needed
            if sr != 48000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)
            sf.write(output_path, audio_data, 48000)
        elif audio_format == 'wav':
            # Keep original sample rate
            sf.write(output_path, audio_data, sr)
        else:
            print(f"⚠️  Unsupported audio format: {audio_format}, using wav16k")
            if sr != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sf.write(output_path, audio_data, 16000)
        
        return True
    except Exception as e:
        print(f"⚠️  Error exporting audio to {output_path}: {e}")
        return False

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
        users_dict = df.set_index('User ID').to_dict('index')
        return users_dict
    except Exception as e:
        print(f"⚠️  Error loading Users.tsv: {e}")
        return {}

def analyze_corpus(data_pull_name, trim_type='librosa', length_threshold=25, 
                   dont_count_over_threshold=True, export_audio_flag=False, audio_format='wav16k'):
    """
    Analyze voice data collection and calculate total time.
    Creates manifests for clean, other, and all recordings.
    Optionally exports processed audio files.
    
    Args:
        data_pull_name: Name of the data pull directory
        trim_type: Type of trimming ('librosa' or 'constant')
        length_threshold: Maximum duration in seconds for clean recordings
        dont_count_over_threshold: If True, exclude recordings over threshold from clean set
        export_audio_flag: If True, export processed audio files
        audio_format: Audio export format (default: 'wav16k')
    
    Returns:
        Tuple of (stats, clean_recordings, other_recordings, all_recordings)
    """
    base_dir = Path(data_pull_name)
    
    # Special handling for COMBINED datasets - only for audio export
    if base_dir.name.startswith('COMBINED_') and export_audio_flag:
        return process_combined_audio_export(data_pull_name, trim_type, audio_format)
    
    recordings_dir = base_dir / "audioapp_recordings"
    
    if not recordings_dir.exists():
        print(f"❌ ERROR: Recordings directory not found: {recordings_dir}")
        print(f"\n💡 If this is a COMBINED dataset, make sure to use --export-audio flag")
        return None, [], [], []
    
    # Setup output directory for processed audio
    if export_audio_flag:
        processed_audio_dir = base_dir / "processed_recordings"
        print(f"📁 Will export processed audio to: {processed_audio_dir}")
    
    # Load user demographics
    users_data = load_users_data(data_pull_name)
    
    # Statistics tracking
    speaker_count = 0
    total_recordings = 0
    clean_duration_secs = 0
    other_duration_secs = 0
    duplicate_duration_secs = 0
    over_threshold_duration_secs = 0
    
    # For deduplication
    task_recordings = defaultdict(list)
    duplicate_recordings = 0
    over_threshold_recordings = 0
    unique_recordings = 0
    
    # Recording lists
    clean_recordings = []
    other_recordings = []
    all_recordings = []
    
    # Per-speaker tracking
    speakers_data = defaultdict(lambda: {
        'recordings_clean': 0,
        'duration_clean': 0,
        'recordings_other': 0,
        'duration_other': 0
    })
    
    # Process each speaker directory
    for speaker_dir in sorted(recordings_dir.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name.startswith('.'):
            continue
        
        speaker_id = speaker_dir.name
        speaker_count += 1
        
        print(f"Processing speaker: {speaker_id}")
        
        # Get user demographics
        user_demo = users_data.get(speaker_id, {})
        
        # Process all WAV files for this speaker
        for wav_file in speaker_dir.glob("*.wav"):
            total_recordings += 1
            
            # Get durations and trimmed audio
            original_duration = get_wav_duration(wav_file)
            
            # Apply trimming based on type
            if trim_type == 'librosa':
                trimmed_duration, audio_data, sr = trim_audio_librosa(wav_file)
            elif trim_type == 'constant':
                trimmed_duration, audio_data, sr = trim_audio_constant(wav_file, seconds=2)
            else:
                y, sr = librosa.load(str(wav_file), sr=None)
                trimmed_duration = len(y) / sr
                audio_data, sr = y, sr
            
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
                'audio_data': audio_data if export_audio_flag else None,
                'sample_rate': sr if export_audio_flag else None,
                # User demographics
                'province': user_demo.get('province', ''),
                'city': user_demo.get('city', ''),
                'dialect': user_demo.get('dialect', ''),
                'disorder': user_demo.get('disorder', ''),
                'gender': user_demo.get('gender', ''),
                'age': user_demo.get('age', ''),
                'hasHelper': user_demo.get('hasHelper', ''),
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
    
    # Export audio if requested
    if export_audio_flag:
        print(f"\n{'=' * 70}")
        print("Exporting processed audio files...")
        print("")
        export_count = 0
        for recording in clean_recordings:
            if recording['audio_data'] is not None:
                speaker_id = recording['speaker_id']
                filename = recording['filename']
                
                output_path = processed_audio_dir / speaker_id / filename
                if export_audio(recording['audio_data'], recording['sample_rate'], output_path, audio_format):
                    export_count += 1
        
        print(f"✓ Exported {export_count} clean recordings to {processed_audio_dir}")
    
    # Clean up audio data from recordings before returning
    for rec in clean_recordings + other_recordings + all_recordings:
        rec.pop('audio_data', None)
        rec.pop('sample_rate', None)
    
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
            'trim_type': trim_type,
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
    print(f"Trimming: {trim_type}")
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

def process_combined_audio_export(data_pull_name, trim_type, audio_format):
    """
    Process audio export for COMBINED datasets and generate analysis.
    Loads existing CSV and exports audio from source directories.
    """
    base_dir = Path(data_pull_name.rstrip('/'))
    
    print(f"📦 Processing COMBINED dataset: {base_dir.name}")
    
    # Load the CSV
    csv_clean = base_dir / "analysis_results" / "recordings_clean.csv"
    csv_other = base_dir / "analysis_results" / "recordings_other.csv"
    csv_all = base_dir / "analysis_results" / "recordings_all.csv"
    
    if not csv_clean.exists():
        print(f"❌ ERROR: No recordings_clean.csv found in {base_dir / 'analysis_results'}")
        print(f"   Run 05_combine_corpus.py first")
        return None, [], [], []
    
    print(f"📄 Loading CSVs...")
    df_clean = pd.read_csv(csv_clean) if csv_clean.exists() else pd.DataFrame()
    df_other = pd.read_csv(csv_other) if csv_other.exists() else pd.DataFrame()
    df_all = pd.read_csv(csv_all) if csv_all.exists() else pd.DataFrame()
    
    print(f"✓ Loaded {len(df_clean)} clean, {len(df_other)} other, {len(df_all)} total recordings")
    
    # Setup output directory
    processed_audio_dir = base_dir / "processed_recordings"
    print(f"\n📁 Exporting processed audio to: {processed_audio_dir}")
    print("")
    
    export_count = 0
    error_count = 0
    
    # Export only clean recordings
    for idx, row in df_clean.iterrows():
        # Get source directory from data_pull column
        if 'data_pull' not in row:
            print(f"❌ ERROR: CSV missing 'data_pull' column")
            return None, [], [], []
        
        source_pull = row['data_pull']
        file_path = row['file_path']
        
        # Construct source audio path
        source_audio_path = Path(source_pull) / file_path
        
        if not source_audio_path.exists():
            print(f"⚠️  Audio not found: {source_audio_path}")
            error_count += 1
            continue
        
        try:
            # Process the audio
            if trim_type == 'librosa':
                _, audio_data, sr = trim_audio_librosa(source_audio_path)
            elif trim_type == 'constant':
                _, audio_data, sr = trim_audio_constant(source_audio_path, seconds=2)
            else:
                y, sr = librosa.load(str(source_audio_path), sr=None)
                audio_data = y
            
            # Export to processed directory
            speaker_id = row['speaker_id']
            filename = row['filename']
            output_path = processed_audio_dir / speaker_id / filename
            
            if export_audio(audio_data, sr, output_path, audio_format):
                export_count += 1
            else:
                error_count += 1
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df_clean)} recordings...")
        
        except Exception as e:
            print(f"⚠️  Error processing {source_audio_path}: {e}")
            error_count += 1
    
    print(f"\n✓ Exported {export_count} recordings to {processed_audio_dir}")
    if error_count > 0:
        print(f"⚠️  {error_count} recordings failed")
    
    # Now generate statistics from the CSVs
    print(f"\n{'=' * 70}")
    print("Generating statistics from combined data...")
    print("")
    
    # Calculate statistics
    speakers = df_all['speaker_id'].unique()
    speaker_count = len(speakers)
    
    # Get unique data pulls
    data_pulls = df_all['data_pull'].unique() if 'data_pull' in df_all.columns else []
    
    # Count recordings
    total_clean = len(df_clean)
    total_other = len(df_other)
    total_all = len(df_all)
    
    # Count by category
    duplicates = len(df_other[df_other['category'] == 'duplicate']) if 'category' in df_other.columns else 0
    over_threshold = len(df_other[df_other['category'] == 'over_threshold']) if 'category' in df_other.columns else 0
    
    # Calculate durations
    clean_duration = df_clean['trimmed_duration'].sum() if 'trimmed_duration' in df_clean.columns else 0
    other_duration = df_other['trimmed_duration'].sum() if 'trimmed_duration' in df_other.columns else 0
    all_duration = df_all['trimmed_duration'].sum() if 'trimmed_duration' in df_all.columns else 0
    
    duplicate_duration = df_other[df_other['category'] == 'duplicate']['trimmed_duration'].sum() if 'category' in df_other.columns else 0
    over_threshold_duration = df_other[df_other['category'] == 'over_threshold']['trimmed_duration'].sum() if 'category' in df_other.columns else 0
    
    # Per-speaker stats
    speakers_data = {}
    for speaker_id in speakers:
        speaker_clean = df_clean[df_clean['speaker_id'] == speaker_id]
        speaker_other = df_other[df_other['speaker_id'] == speaker_id]
        
        speakers_data[speaker_id] = {
            'recordings_clean': len(speaker_clean),
            'duration_clean': speaker_clean['trimmed_duration'].sum() if len(speaker_clean) > 0 else 0,
            'recordings_other': len(speaker_other),
            'duration_other': speaker_other['trimmed_duration'].sum() if len(speaker_other) > 0 else 0
        }
    
    # Calculate averages
    avg_duration_per_recording = clean_duration / total_clean if total_clean > 0 else 0
    avg_recordings_per_speaker = total_clean / speaker_count if speaker_count > 0 else 0
    
    # Build stats dictionary
    stats = {
        'data_pull_name': data_pull_name,
        'is_combined': True,
        'source_data_pulls': list(data_pulls),
        'parameters': {
            'trim_type': trim_type,
            'length_threshold': 25,
            'exclude_over_threshold': True
        },
        'speakers': {
            'total': speaker_count,
            'list': list(speakers)
        },
        'recordings': {
            'total_found': total_all,
            'unique': total_clean + over_threshold,
            'clean': total_clean,
            'duplicates': duplicates,
            'over_threshold': over_threshold,
            'other': total_other,
            'avg_per_speaker_clean': round(avg_recordings_per_speaker, 2)
        },
        'duration': {
            'clean': {
                'total_seconds': round(clean_duration, 2),
                'total_minutes': round(clean_duration / 60, 2),
                'total_hours': round(clean_duration / 3600, 2),
                'avg_per_recording_seconds': round(avg_duration_per_recording, 2)
            },
            'other': {
                'total_seconds': round(other_duration, 2),
                'total_minutes': round(other_duration / 60, 2),
                'total_hours': round(other_duration / 3600, 2),
                'duplicates_seconds': round(duplicate_duration, 2),
                'duplicates_minutes': round(duplicate_duration / 60, 2),
                'duplicates_hours': round(duplicate_duration / 3600, 2),
                'over_threshold_seconds': round(over_threshold_duration, 2),
                'over_threshold_minutes': round(over_threshold_duration / 60, 2),
                'over_threshold_hours': round(over_threshold_duration / 3600, 2)
            },
            'all': {
                'total_seconds': round(all_duration, 2),
                'total_minutes': round(all_duration / 60, 2),
                'total_hours': round(all_duration / 3600, 2)
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
    print(f"{'=' * 70}")
    print("COMBINED ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"Data Pull: {data_pull_name}")
    print(f"Source data pulls: {', '.join(data_pulls)}")
    print("")
    print(f"Trimming: {trim_type}")
    print("")
    print(f"Speakers: {speaker_count}")
    print(f"Total recordings: {total_all}")
    print(f"  - Clean: {total_clean}")
    print(f"  - Duplicates: {duplicates}")
    print(f"  - Over threshold: {over_threshold}")
    print("")
    print(f"CLEAN Corpus Duration:")
    print(f"  {clean_duration / 3600:.2f} hours")
    print(f"  {clean_duration / 60:.2f} minutes")
    print(f"  {clean_duration:.2f} seconds")
    print("")
    print(f"OTHER Recordings Duration:")
    print(f"  {other_duration / 3600:.2f} hours")
    print(f"  {other_duration / 60:.2f} minutes")
    print(f"  {other_duration:.2f} seconds")
    print("")
    print(f"Average duration per clean recording: {avg_duration_per_recording:.2f} seconds")
    print(f"Average clean recordings per speaker: {avg_recordings_per_speaker:.2f}")
    print("=" * 70)
    
    # Return stats but empty recording lists (already in CSV)
    return stats, [], [], []

def save_results(stats, clean_recordings, other_recordings, all_recordings, data_pull_name):
    """Save analysis results to JSON, text, and CSV files."""
    
    if stats is None:
        print("⚠️  No stats to save")
        return
    
    # Check if this is a COMBINED dataset
    is_combined = Path(data_pull_name).name.startswith('COMBINED_')
    
    # Create analysis_results directory
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
        f.write(f"  Trim type: {stats['parameters']['trim_type']}\n")
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
        
        f.write("CLEAN Corpus Duration (deduplicated + within threshold):\n")
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
        f.write(f"{'Speaker':<15} {'Clean Recs':<12} {'Clean Dur (h)':<15} {'Other Recs':<12} {'Other Dur (h)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for speaker_id, data in sorted(stats['speakers_detail'].items()):
            f.write(f"{speaker_id:<15} "
                   f"{data['recordings_clean']:<12} "
                   f"{data['duration_clean_hours']:<15.2f} "
                   f"{data['recordings_other']:<12} "
                   f"{data['duration_other_hours']:<15.2f}\n")
    
    # Only save CSVs for non-COMBINED datasets
    # COMBINED datasets already have CSVs from 05_combine_corpus.py
    if not is_combined:
        # Define CSV fields (only include fields we want in the CSV output)
        csv_fields = [
            'speaker_id', 'filename', 'file_path', 'task_id', 
            'original_duration', 'trimmed_duration',
            'timestamp', 'prompt', 'category', 'reason',
            'province', 'city', 'age', 'gender', 'disorder', 'dialect', 'hasHelper'
        ]
        
        # Filter recordings to only include the fields we want in CSV
        def filter_fields(recording):
            return {k: v for k, v in recording.items() if k in csv_fields}
        
        clean_recordings_filtered = [filter_fields(r) for r in clean_recordings]
        other_recordings_filtered = [filter_fields(r) for r in other_recordings]
        all_recordings_filtered = [filter_fields(r) for r in all_recordings]
        
        # Save clean recordings CSV
        clean_csv = results_dir / "recordings_clean.csv"
        with open(clean_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(clean_recordings_filtered)
        
        # Save other recordings CSV
        other_csv = results_dir / "recordings_other.csv"
        with open(other_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(other_recordings_filtered)
        
        # Save all recordings CSV
        all_csv = results_dir / "recordings_all.csv"
        with open(all_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(all_recordings_filtered)
        
        print(f"\n✓ Results saved:")
        print(f"  JSON stats: {json_file}")
        print(f"  Report: {report_file}")
        print(f"  Clean recordings CSV: {clean_csv} ({len(clean_recordings)} recordings)")
        print(f"  Other recordings CSV: {other_csv} ({len(other_recordings)} recordings)")
        print(f"  All recordings CSV: {all_csv} ({len(all_recordings)} recordings)")
    else:
        print(f"\n✓ Results saved:")
        print(f"  JSON stats: {json_file}")
        print(f"  Report: {report_file}")
        print(f"  CSVs preserved from 05_combine_corpus.py (not overwritten)")

def main():
    parser = argparse.ArgumentParser(
        description='Analyze RAPNIC corpus recordings and optionally export processed audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with adaptive librosa trimming (default)
  python 03_analyze_and_pack_corpus.py LREC-PAPER
  
  # Constant trimming (cut first 2 seconds)
  python 03_analyze_and_pack_corpus.py LREC-PAPER --trim-type constant
  
  # No trimming
  python 03_analyze_and_pack_corpus.py LREC-PAPER --trim-type none
  
  # Export processed audio in 16kHz format
  python 03_analyze_and_pack_corpus.py LREC-PAPER --export-audio --audio-format wav16k
  
  # Process and export in 48kHz
  python 03_analyze_and_pack_corpus.py PILOT --export-audio --audio-format wav48k
        """
    )
    
    parser.add_argument(
        'data_pull_name',
        type=str,
        help='Name of the data pull directory (e.g., LREC-PAPER, PILOT)'
    )
    
    parser.add_argument(
        '--trim-type',
        type=str,
        default='librosa',
        choices=['librosa', 'constant', 'none'],
        help='Type of trimming: librosa (adaptive silence detection), constant (fixed 2 seconds), or none (default: librosa)'
    )
    
    parser.add_argument(
        '--export-audio',
        action='store_true',
        help='Export processed audio files to processed_recordings directory'
    )
    
    parser.add_argument(
        '--audio-format',
        type=str,
        default='wav16k',
        choices=['wav16k', 'wav48k', 'wav'],
        help='Audio export format (default: wav16k). Options: wav16k (16kHz), wav48k (48kHz), wav (original rate)'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    stats, clean_recordings, other_recordings, all_recordings = analyze_corpus(
        args.data_pull_name, 
        trim_type=args.trim_type,
        length_threshold=LENGTH_THRESHOLD, 
        dont_count_over_threshold=DONT_COUNT_OVER_THRESHOLD,
        export_audio_flag=args.export_audio,
        audio_format=args.audio_format
    )
    
    # Save results (will be None for COMBINED datasets)
    if stats is not None:
        save_results(stats, clean_recordings, other_recordings, all_recordings, args.data_pull_name)
        print(f"\nNext step:")
        print(f"  python scripts/04_analyze_demographics.py {args.data_pull_name}")
    else:
        # COMBINED dataset or error
        if Path(args.data_pull_name).name.startswith('COMBINED_'):
            print(f"\n✓ COMBINED dataset processing complete!")
        else:
            print(f"\n⚠️  Processing failed - see errors above")

if __name__ == "__main__":
    main()