#!/bin/bash
# download_waymo_all.sh

# Create output directory
mkdir -p data
LOGFILE="download_all_progress.log"
BUCKET="gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEST="data"

echo "Starting download of ALL Waymo dataset files" | tee -a "$LOGFILE"

# First, list all the file prefixes in the bucket (train, test, validation)
echo "Listing file prefixes in bucket..." | tee -a "$LOGFILE"
gsutil ls "$BUCKET/" | tee -a "$LOGFILE" > file_prefixes.txt

# Now download everything in the bucket
echo "Downloading ALL files from $BUCKET to $DEST" | tee -a "$LOGFILE"
gsutil -m cp -r "$BUCKET/*" "$DEST/" 2>&1 | tee -a "$LOGFILE"

echo "Download process completed" | tee -a "$LOGFILE"
