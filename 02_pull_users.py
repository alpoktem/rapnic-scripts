#!/usr/bin/env python3ex
"""
Firebase User Metadata Pull Script
Usage: python 02_pull_users.py [TARGET_DIR]
Example: python 02_pull_users.py LREC-PAPER
"""

import sys
import os
import json
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from pathlib import Path

# Read excluded users from file
SKIP_USERIDS = []
excluded_file = Path(__file__).parent.parent / "excluded_users.txt"
if excluded_file.exists():
    with open(excluded_file, 'r') as f:
        SKIP_USERIDS = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(SKIP_USERIDS)} excluded users from excluded_users.txt")
else:
    print("Warning: excluded_users.txt not found, no users will be excluded")

    
def find_credentials_file():
    """Find Firebase credentials file in scripts directory"""
    script_dir = Path(__file__).parent
    cred_files = list(script_dir.glob("*firebase-adminsdk*.json"))
    
    if not cred_files:
        print("❌ ERROR: Firebase credentials file not found")
        print(f"   Looking in: {script_dir}")
        print("   Expected file pattern: *firebase-adminsdk*.json")
        sys.exit(1)
    
    if len(cred_files) > 1:
        print("⚠️  Warning: Multiple credential files found, using first one:")
        for f in cred_files:
            print(f"   - {f.name}")
    
    return str(cred_files[0])

def pull_users(target_dir):
    """Pull user metadata from Firestore and save to TSV"""
    
    # Verify we're in the DATA directory
    if not Path("scripts").exists():
        print("❌ ERROR: This script should be run from the DATA directory")
        print(f"   Current directory: {os.getcwd()}")
        print("   Expected to find: ./scripts/")
        sys.exit(1)
    
    # Verify target directory exists
    target_path = Path(target_dir)
    if not target_path.exists():
        print(f"⚠️  Target directory '{target_dir}' doesn't exist, creating it...")
        target_path.mkdir(parents=True, exist_ok=True)
    
    output_file = target_path / "Users.tsv"
    
    print("=" * 60)
    print("Firebase User Metadata Pull")
    print("=" * 60)
    print(f"Target directory: {target_dir}")
    print(f"Output file: {output_file}")
    print("")
    
    # Find and load Firebase credentials
    cred_file = find_credentials_file()
    print(f"Using credentials: {Path(cred_file).name}")
    
    try:
        cred = credentials.Certificate(cred_file)
        # Initialize or get existing app
        try:
            app = firebase_admin.get_app()
            print("Using existing Firebase app")
        except ValueError:
            app = firebase_admin.initialize_app(cred)
            print("Initialized new Firebase app")
        
        db = firestore.client()
        print("✓ Connected to Firestore")
        print("")
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to Firebase")
        print(f"   {str(e)}")
        sys.exit(1)
    
    # Fetch users from Firestore
    print("Fetching users from EUsers collection...")
    users_ref = db.collection("EUsers")
    
    try:
        docs = users_ref.stream()
        doc_list = list(docs)
        print(f"✓ Found {len(doc_list)} users in Firestore")
        print("")
    except Exception as e:
        print(f"❌ ERROR: Failed to fetch users")
        print(f"   {str(e)}")
        sys.exit(1)
    
    # Extract user data
    rows = []
    skipped = 0
    
    print("Processing users...")
    for doc in doc_list:
        data = doc.to_dict()
        user_id = data.get("euid", doc.id)
        info_raw = data.get("info", "")
        
        # Skip excluded users
        if user_id in SKIP_USERIDS:
            skipped += 1
            continue
        
        try:
            # Parse the stringified JSON inside the "info" field
            info = json.loads(info_raw)
            demographics = info.get("demographics", {})
        except Exception as e:
            print(f"⚠️  Couldn't parse 'info' for user {user_id}: {e}")
            demographics = {}
        
        rows.append({
            "User ID": user_id,
            "province": demographics.get("province", ""),
            "city": demographics.get("city", ""),
            "dialect": demographics.get("dialect", ""),
            "disorder": demographics.get("disorder", ""),
            "gender": demographics.get("gender", ""),
            "age": demographics.get("age", ""),
            "accessDevices": ", ".join(demographics.get("accessDevices", [])) if isinstance(demographics.get("accessDevices", []), list) else demographics.get("accessDevices", ""),
            "hasHelper": demographics.get("hasHelper", ""),
            "numRecordings": info.get("numRecordings", 0),
            "numTasks": info.get("numTasks", 0),
            "numCompletedTasks": info.get("numCompletedTasks", 0)
        })
    
    print(f"✓ Processed {len(rows)} users")
    print(f"  Skipped {skipped} excluded users")
    print("")
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    # Sort by User ID for consistency
    df = df.sort_values("User ID")
    
    # Write to TSV
    df.to_csv(output_file, sep="\t", index=False)
    
    print("=" * 60)
    print("Export Complete")
    print("=" * 60)
    print(f"✓ Exported to: {output_file}")
    print(f"  Total users: {len(df)}")
    print("")
    
    # Quick summary statistics
    print("Quick Summary:")
    print(f"  Disorders: {df['disorder'].value_counts().to_dict()}")
    print(f"  Gender: {df['gender'].value_counts().to_dict()}")
    print(f"  Has Helper: {df['hasHelper'].value_counts().to_dict()}")
    print("")
    print("Next steps:")
    print(f"  1. Analyze corpus: python scripts/03_analyze_corpus.py {target_dir}")
    print(f"  2. Analyze demographics: python scripts/04_analyze_demographics.py {target_dir}")

def main():
    if len(sys.argv) != 2:
        print("ERROR: No target directory specified")
        print(f"Usage: {sys.argv[0]} <TARGET_DIR>")
        print("Example: python 02_pull_users.py LREC-PAPER")
        print("")
        print("Available targets: PILOT, LREC-PAPER, or any directory name")
        sys.exit(1)
    
    target_dir = sys.argv[1]
    pull_users(target_dir)

if __name__ == "__main__":
    main()
