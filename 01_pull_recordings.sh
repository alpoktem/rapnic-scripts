#!/bin/bash

# Smart Firebase Recording Pull Script
# Usage: ./01_pull_recordings.sh [TARGET_DIR]
# Example: ./01_pull_recordings.sh LREC-PAPER

# Check if target directory argument is provided
if [ $# -eq 0 ]; then
    echo "ERROR: No target directory specified"
    echo "Usage: $0 <TARGET_DIR>"
    echo "Example: $0 LREC-PAPER"
    echo ""
    echo "Available targets: PILOT, LREC-PAPER, or any directory name"
    exit 1
fi

TARGET_DIR="$1"

# Check if we're in the DATA directory
if [ ! -d "scripts" ]; then
    echo "ERROR: This script should be run from the DATA directory"
    echo "Current directory: $(pwd)"
    echo "Expected to find: ./scripts/"
    exit 1
fi

# Configuration
BUCKET="gs://collectivat-euphonia.firebasestorage.app/audioapp_recordings"
OUTPUT_DIR="./${TARGET_DIR}/audioapp_recordings"
MIN_RECORDINGS=100
PARALLEL_DOWNLOADS=3

# State files (stored in target directory)
FILTERED_USERS_FILE="./${TARGET_DIR}/filtered_users.txt"
DOWNLOADED_USERS_FILE="./${TARGET_DIR}/downloaded_users.txt"
EXISTING_USERS_FILE="./${TARGET_DIR}/existing_users.txt"
EXCLUDED_USERS_FILE="./${TARGET_DIR}/excluded_users.txt"

# Read existing users from file (if exists)
EXISTING_USERS=()
if [ -f "$EXISTING_USERS_FILE" ]; then
    while IFS= read -r user; do
        EXISTING_USERS+=("$user")
    done < "$EXISTING_USERS_FILE"
    echo "Loaded ${#EXISTING_USERS[@]} existing users from $EXISTING_USERS_FILE"
fi

# Read excluded users from file (if exists)
EXCLUDED_USERS=()
if [ -f "$EXCLUDED_USERS_FILE" ]; then
    while IFS= read -r user; do
        EXCLUDED_USERS+=("$user")
    done < "$EXCLUDED_USERS_FILE"
    echo "Loaded ${#EXCLUDED_USERS[@]} excluded users from $EXCLUDED_USERS_FILE"
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"
mkdir -p "$OUTPUT_DIR"
touch "$DOWNLOADED_USERS_FILE"

echo "================================================"
echo "Smart Firebase Recording Pull"
echo "================================================"
echo "Target directory: $TARGET_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Minimum recordings threshold: $MIN_RECORDINGS"
echo "Parallel downloads: $PARALLEL_DOWNLOADS users at once"
echo ""

# If targeting PILOT, don't exclude existing users
if [ "$TARGET_DIR" = "PILOT" ]; then
    echo "Note: Targeting PILOT directory - not excluding pilot users"
    EXISTING_USERS=()
else
    echo "Note: Excluding ${#EXISTING_USERS[@]} users already in PILOT"
fi
echo ""

# Phase 1: Filter users (or load from cache)
if [ -f "$FILTERED_USERS_FILE" ]; then
    echo "Phase 1: Loading cached user list from $FILTERED_USERS_FILE"
    echo "(Delete this file to rebuild the list)"
    DOWNLOAD_LIST=($(cat "$FILTERED_USERS_FILE"))
    echo "  Found ${#DOWNLOAD_LIST[@]} users to download"
    echo ""
else
    echo "Phase 1: Filtering users (this will be cached)..."
    echo ""
    
    # Get all user directories
    echo "Fetching user list from Firebase..."
    USER_LIST=$(gsutil ls "$BUCKET/" 2>&1)
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to list bucket contents"
        echo "$USER_LIST"
        exit 1
    fi
    
    # Extract user IDs
    ALL_USERS=$(echo "$USER_LIST" | sed -n 's|.*/audioapp_recordings/\(E[0-9]\{5\}\)/.*|\1|p' | sort -u)
    
    if [ -z "$ALL_USERS" ]; then
        ALL_USERS=$(echo "$USER_LIST" | grep -oE 'E[0-9]{5}' | sort -u)
    fi
    
    if [ -z "$ALL_USERS" ]; then
        echo "ERROR: No users found"
        exit 1
    fi
    
    user_count=$(echo "$ALL_USERS" | wc -l | tr -d ' ')
    echo "Found $user_count total users in Firebase"
    echo ""
    
    DOWNLOAD_LIST=()
    skipped_existing=0
    skipped_excluded=0
    skipped_too_few=0
    
    current=0
    for user in $ALL_USERS; do
        current=$((current + 1))
        
        # Progress bar
        percent=$((current * 100 / user_count))
        bar_length=$((percent / 2))
        bar=$(printf '%*s' "$bar_length" | tr ' ' '█')
        spaces=$(printf '%*s' "$((50 - bar_length))" | tr ' ' '░')
        printf "\r[%s%s] %3d%% (%d/%d) Checking %s...     " "$bar" "$spaces" "$percent" "$current" "$user_count" "$user"
        
        # Check if user already exists (only if not PILOT)
        if [ "$TARGET_DIR" != "PILOT" ] && [[ " ${EXISTING_USERS[@]} " =~ " ${user} " ]]; then
            skipped_existing=$((skipped_existing + 1))
            continue
        fi
        
        # Check if user is excluded
        if [[ " ${EXCLUDED_USERS[@]} " =~ " ${user} " ]]; then
            skipped_excluded=$((skipped_excluded + 1))
            continue
        fi
        
        # Count recordings for this user
        recording_count=$(gsutil ls "$BUCKET/$user/" 2>/dev/null | wc -l | tr -d ' ')
        
        if [ -z "$recording_count" ] || [ "$recording_count" -eq 0 ]; then
            continue
        fi
        
        # Check if user has enough recordings
        if [ "$recording_count" -lt "$MIN_RECORDINGS" ]; then
            skipped_too_few=$((skipped_too_few + 1))
            continue
        fi
        
        # Add to download list
        DOWNLOAD_LIST+=("$user:$recording_count")
    done
    
    echo ""
    echo ""
    echo "Filtering complete:"
    echo "  Skipped (already downloaded): $skipped_existing"
    echo "  Skipped (excluded): $skipped_excluded"
    echo "  Skipped (< $MIN_RECORDINGS recordings): $skipped_too_few"
    echo "  To download: ${#DOWNLOAD_LIST[@]} users"
    echo ""
    
    # Save filtered list
    printf "%s\n" "${DOWNLOAD_LIST[@]}" > "$FILTERED_USERS_FILE"
    echo "Saved filtered list to $FILTERED_USERS_FILE"
    echo ""
fi

if [ ${#DOWNLOAD_LIST[@]} -eq 0 ]; then
    echo "No users to download!"
    exit 0
fi

# Load already downloaded users
ALREADY_DOWNLOADED=($(cat "$DOWNLOADED_USERS_FILE" 2>/dev/null))

# Filter out already downloaded users
REMAINING_LIST=()
for user_info in "${DOWNLOAD_LIST[@]}"; do
    user=$(echo "$user_info" | cut -d: -f1)
    if [[ ! " ${ALREADY_DOWNLOADED[@]} " =~ " ${user} " ]]; then
        REMAINING_LIST+=("$user_info")
    fi
done

already_done=$((${#DOWNLOAD_LIST[@]} - ${#REMAINING_LIST[@]}))
if [ $already_done -gt 0 ]; then
    echo "Resume mode: $already_done users already downloaded"
    echo "Remaining: ${#REMAINING_LIST[@]} users"
    echo ""
fi

if [ ${#REMAINING_LIST[@]} -eq 0 ]; then
    echo "All users already downloaded!"
    exit 0
fi

# Phase 2: Download users
echo "Phase 2: Downloading recordings..."
echo ""

total_to_download=${#REMAINING_LIST[@]}
downloaded=0
failed=0

# Download sequentially to avoid conflicts
index=0
for user_info in "${REMAINING_LIST[@]}"; do
    index=$((index + 1))
    user=$(echo "$user_info" | cut -d: -f1)
    count=$(echo "$user_info" | cut -d: -f2)
    
    echo "[$index/$total_to_download] $user ($count recordings) - Downloading..."
    
    # Download
    gsutil -q cp -r "$BUCKET/$user" "$OUTPUT_DIR/" 2>&1 | head -10 > "/tmp/gsutil_${user}.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "[$index/$total_to_download] $user - ✓ Complete"
        echo "$user" >> "$DOWNLOADED_USERS_FILE"
        downloaded=$((downloaded + 1))
    else
        echo "[$index/$total_to_download] $user - ✗ Failed (see /tmp/gsutil_${user}.log)"
        cat "/tmp/gsutil_${user}.log"
        failed=$((failed + 1))
    fi
done

# Final summary
downloaded_count=$(wc -l < "$DOWNLOADED_USERS_FILE" | tr -d ' ')

echo ""
echo "================================================"
echo "Download Summary"
echo "================================================"
echo "Target directory: $TARGET_DIR"
echo "Total users downloaded so far: $downloaded_count"
echo "Successfully downloaded this run: $downloaded"
echo "Failed this run: $failed"
echo ""
echo "State files:"
echo "  Filtered list: $FILTERED_USERS_FILE"
echo "  Download log: $DOWNLOADED_USERS_FILE"
echo "================================================"
echo ""
echo "Recordings saved to: $OUTPUT_DIR"
echo ""
echo "To restart fresh: rm $FILTERED_USERS_FILE $DOWNLOADED_USERS_FILE"
echo ""
echo "Next steps:"
echo "  1. Pull user metadata: python scripts/02_pull_users.py $TARGET_DIR"
echo "  2. Analyze corpus: python scripts/03_analyze_corpus.py $TARGET_DIR"
