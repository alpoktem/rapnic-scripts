#!/usr/bin/env python3
"""
RAPNIC Demographics Analyzer
Analyzes participant demographics from recording manifests.

Usage: python 04_analyze_demographics.py <DATA_PULL_NAME>
Example: python 04_analyze_demographics.py LREC-PAPER
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
import pandas as pd

def load_recordings_csv(data_pull_name, category='all'):
    """Load recordings CSV file."""
    csv_file = Path(data_pull_name) / "analysis_results" / f"recordings_{category}.csv"
    
    if not csv_file.exists():
        print(f"⚠️  Warning: {csv_file} not found")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"⚠️  Error loading {csv_file}: {e}")
        return pd.DataFrame()

def analyze_demographics(data_pull_name):
    """
    Analyze demographics from recordings CSV files.
    Uses the 'all' recordings file to get complete demographics.
    
    Args:
        data_pull_name: Name of the data pull directory
    
    Returns:
        dict: Demographics statistics
    """
    
    # Verify we're in the DATA directory
    if not Path("scripts").exists():
        print("❌ ERROR: This script should be run from the DATA directory")
        print(f"   Current directory: {Path.cwd()}")
        print("   Expected to find: ./scripts/")
        sys.exit(1)
    
    results_dir = Path(data_pull_name) / "analysis_results"
    
    if not results_dir.exists():
        print(f"❌ ERROR: Analysis results directory not found: {results_dir}")
        print(f"   Make sure you've run: python scripts/03_analyze_corpus.py {data_pull_name}")
        sys.exit(1)
    
    print("=" * 70)
    print("RAPNIC Demographics Analyzer")
    print("=" * 70)
    print(f"Data pull: {data_pull_name}")
    print(f"Results directory: {results_dir}")
    print("")
    
    # Load recordings data
    print("Loading recordings data...")
    df = load_recordings_csv(data_pull_name, 'all')
    
    if df.empty:
        print("❌ ERROR: No recordings data found")
        sys.exit(1)
    
    # Get unique speakers only
    print(f"✅ Loaded {len(df)} recordings from {df['speaker_id'].nunique()} speakers")
    print("")
    
    # Create speaker-level demographics (one row per speaker)
    speaker_df = df.groupby('speaker_id').first().reset_index()
    
    total_speakers = len(speaker_df)
    
    # Initialize statistics dictionary
    stats = {
        'data_pull_name': data_pull_name,
        'total_speakers': total_speakers,
        'total_recordings': len(df),
        'demographics': {}
    }
    
    print("=" * 70)
    print("DEMOGRAPHICS ANALYSIS")
    print("=" * 70)
    print(f"Total speakers: {total_speakers}")
    print(f"Total recordings: {len(df)}")
    print("")
    
    # Age distribution
    print("─" * 70)
    print("AGE DISTRIBUTION")
    print("─" * 70)
    age_dist = speaker_df['age'].value_counts().sort_index()
    age_dict = {}
    for age_range, count in age_dist.items():
        pct = (count / total_speakers) * 100
        age_dict[str(age_range)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {age_range}: {count} ({pct:.1f}%)")
    stats['demographics']['age'] = age_dict
    
    # Gender distribution
    print("")
    print("─" * 70)
    print("GENDER DISTRIBUTION")
    print("─" * 70)
    gender_dist = speaker_df['gender'].value_counts()
    gender_dict = {}
    for gender, count in gender_dist.items():
        pct = (count / total_speakers) * 100
        gender_dict[str(gender)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {gender}: {count} ({pct:.1f}%)")
    stats['demographics']['gender'] = gender_dict
    
    # Disorder distribution
    print("")
    print("─" * 70)
    print("DISORDER DISTRIBUTION")
    print("─" * 70)
    disorder_dist = speaker_df['disorder'].value_counts()
    disorder_dict = {}
    for disorder, count in disorder_dist.items():
        pct = (count / total_speakers) * 100
        disorder_dict[str(disorder)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {disorder}: {count} ({pct:.1f}%)")
    stats['demographics']['disorder'] = disorder_dict
    
    # Dialect distribution
    print("")
    print("─" * 70)
    print("DIALECT DISTRIBUTION")
    print("─" * 70)
    dialect_dist = speaker_df['dialect'].value_counts()
    dialect_dict = {}
    for dialect, count in dialect_dist.items():
        pct = (count / total_speakers) * 100
        dialect_dict[str(dialect)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {dialect}: {count} ({pct:.1f}%)")
    stats['demographics']['dialect'] = dialect_dict
    
    # Province distribution
    print("")
    print("─" * 70)
    print("PROVINCE DISTRIBUTION")
    print("─" * 70)
    province_dist = speaker_df['province'].value_counts()
    province_dict = {}
    for province, count in province_dist.items():
        pct = (count / total_speakers) * 100
        province_dict[str(province)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {province}: {count} ({pct:.1f}%)")
    stats['demographics']['province'] = province_dict
    
    # City distribution (top 10)
    print("")
    print("─" * 70)
    print("CITY DISTRIBUTION (Top 10)")
    print("─" * 70)
    city_dist = speaker_df['city'].value_counts().head(10)
    city_dict = {}
    for city, count in city_dist.items():
        pct = (count / total_speakers) * 100
        city_dict[str(city)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {city}: {count} ({pct:.1f}%)")
    stats['demographics']['city_top10'] = city_dict
    
    # Helper support distribution
    print("")
    print("─" * 70)
    print("HELPER SUPPORT DISTRIBUTION")
    print("─" * 70)
    helper_dist = speaker_df['hasHelper'].value_counts()
    helper_dict = {}
    for has_helper, count in helper_dist.items():
        pct = (count / total_speakers) * 100
        helper_dict[str(has_helper)] = {
            'count': int(count),
            'percentage': round(pct, 1)
        }
        print(f"  {has_helper}: {count} ({pct:.1f}%)")
    stats['demographics']['hasHelper'] = helper_dict
    
    # Cross-tabulations
    print("")
    print("=" * 70)
    print("CROSS-TABULATIONS")
    print("=" * 70)
    
    # Disorder by Gender
    print("")
    print("DISORDER BY GENDER")
    print("─" * 70)
    disorder_gender = pd.crosstab(speaker_df['disorder'], speaker_df['gender'])
    print(disorder_gender.to_string())
    stats['cross_tabs'] = {}
    stats['cross_tabs']['disorder_by_gender'] = disorder_gender.to_dict()
    
    # Disorder by Age
    print("")
    print("DISORDER BY AGE")
    print("─" * 70)
    disorder_age = pd.crosstab(speaker_df['disorder'], speaker_df['age'])
    print(disorder_age.to_string())
    stats['cross_tabs']['disorder_by_age'] = disorder_age.to_dict()
    
    # Age by Gender
    print("")
    print("AGE BY GENDER")
    print("─" * 70)
    age_gender = pd.crosstab(speaker_df['age'], speaker_df['gender'])
    print(age_gender.to_string())
    stats['cross_tabs']['age_by_gender'] = age_gender.to_dict()
    
    # Helper support by Disorder
    print("")
    print("HELPER SUPPORT BY DISORDER")
    print("─" * 70)
    helper_disorder = pd.crosstab(speaker_df['disorder'], speaker_df['hasHelper'])
    print(helper_disorder.to_string())
    stats['cross_tabs']['helper_by_disorder'] = helper_disorder.to_dict()
    
    # Helper support by Age
    print("")
    print("HELPER SUPPORT BY AGE")
    print("─" * 70)
    helper_age = pd.crosstab(speaker_df['age'], speaker_df['hasHelper'])
    print(helper_age.to_string())
    stats['cross_tabs']['helper_by_age'] = helper_age.to_dict()
    
    print("")
    print("=" * 70)
    
    return stats, speaker_df

def save_results(stats, speaker_df, data_pull_name):
    """Save demographics analysis results."""
    
    results_dir = Path(data_pull_name) / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save JSON stats
    json_file = results_dir / "demographics_stats.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Save human-readable report
    report_file = results_dir / "demographics_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RAPNIC DEMOGRAPHICS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data Pull: {stats['data_pull_name']}\n")
        f.write(f"\nTotal Speakers: {stats['total_speakers']}\n")
        f.write(f"Total Recordings: {stats['total_recordings']}\n")
        
        f.write(f"\n{'=' * 70}\n")
        f.write("DEMOGRAPHICS DISTRIBUTIONS\n")
        f.write("=" * 70 + "\n\n")
        
        # Age
        f.write("AGE DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for age_range, data in stats['demographics']['age'].items():
            f.write(f"  {age_range}: {data['count']} ({data['percentage']}%)\n")
        
        # Gender
        f.write("\nGENDER DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for gender, data in stats['demographics']['gender'].items():
            f.write(f"  {gender}: {data['count']} ({data['percentage']}%)\n")
        
        # Disorder
        f.write("\nDISORDER DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for disorder, data in stats['demographics']['disorder'].items():
            f.write(f"  {disorder}: {data['count']} ({data['percentage']}%)\n")
        
        # Dialect
        f.write("\nDIALECT DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for dialect, data in stats['demographics']['dialect'].items():
            f.write(f"  {dialect}: {data['count']} ({data['percentage']}%)\n")
        
        # Province
        f.write("\nPROVINCE DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for province, data in stats['demographics']['province'].items():
            f.write(f"  {province}: {data['count']} ({data['percentage']}%)\n")
        
        # Helper
        f.write("\nHELPER SUPPORT DISTRIBUTION\n")
        f.write("─" * 70 + "\n")
        for has_helper, data in stats['demographics']['hasHelper'].items():
            f.write(f"  {has_helper}: {data['count']} ({data['percentage']}%)\n")
        
        # Cross-tabs
        f.write(f"\n{'=' * 70}\n")
        f.write("CROSS-TABULATIONS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DISORDER BY GENDER\n")
        f.write("─" * 70 + "\n")
        disorder_gender_df = pd.DataFrame(stats['cross_tabs']['disorder_by_gender'])
        f.write(disorder_gender_df.to_string())
        
        f.write("\n\nDISORDER BY AGE\n")
        f.write("─" * 70 + "\n")
        disorder_age_df = pd.DataFrame(stats['cross_tabs']['disorder_by_age'])
        f.write(disorder_age_df.to_string())
        
        f.write("\n\nAGE BY GENDER\n")
        f.write("─" * 70 + "\n")
        age_gender_df = pd.DataFrame(stats['cross_tabs']['age_by_gender'])
        f.write(age_gender_df.to_string())
        
        f.write("\n\nHELPER SUPPORT BY DISORDER\n")
        f.write("─" * 70 + "\n")
        helper_disorder_df = pd.DataFrame(stats['cross_tabs']['helper_by_disorder'])
        f.write(helper_disorder_df.to_string())
        
        f.write("\n\nHELPER SUPPORT BY AGE\n")
        f.write("─" * 70 + "\n")
        helper_age_df = pd.DataFrame(stats['cross_tabs']['helper_by_age'])
        f.write(helper_age_df.to_string())
    
    # Save speaker demographics CSV for reference
    speaker_demo_file = results_dir / "speaker_demographics.csv"
    speaker_df[['speaker_id', 'age', 'gender', 'disorder', 'dialect', 
                 'province', 'city', 'hasHelper']].to_csv(speaker_demo_file, index=False)
    
    print(f"\n✅ Results saved:")
    print(f"  JSON stats: {json_file}")
    print(f"  Report: {report_file}")
    print(f"  Speaker demographics: {speaker_demo_file}")

def main():
    if len(sys.argv) != 2:
        print("ERROR: No data pull name specified")
        print(f"Usage: {sys.argv[0]} <DATA_PULL_NAME>")
        print("Example: python 04_analyze_demographics.py LREC-PAPER")
        sys.exit(1)
    
    data_pull_name = sys.argv[1]
    
    # Run analysis
    stats, speaker_df = analyze_demographics(data_pull_name)
    
    # Save results
    save_results(stats, speaker_df, data_pull_name)
    
    print(f"\nNext step:")
    print(f"  python scripts/05_combine_corpus.py PILOT {data_pull_name}")

if __name__ == "__main__":
    main()
