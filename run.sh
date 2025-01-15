#!/bin/bash

# Get the hostname
hostname=$(hostname)

# Check if "abacus" is in the hostname
if [[ "$hostname" != *abacus* ]]; then
  echo "Hostname does not contain 'abacus'. Exiting script."
  exit 1
fi

echo "Hostname contains 'abacus', continuing script"
echo "Copying data locally"

# Define source and destination directories
SOURCE="/home/ymahe/IXI-T1/"
DEST="/tmp/ymahe-runtime-dir/"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST"

# Copy the directory with progress
echo "Copying $SOURCE to $DEST..."
cp -r  "$SOURCE" "$DEST"
echo "Directory copied from $SOURCE to $DEST."

# Activate virtual environment and run Python script
source .venv/bin/activate
python main_vae.py --load_checkpoint_training "./vae_checkpoints/Jan_14_2025_69.pth"
