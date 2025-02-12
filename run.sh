#!/bin/bash

# Get the hostname
hostname=$(hostname)

# Check if "abacus" is in the hostname
if [[ "$hostname" != *abacus* ]]; then
  echo "Hostname does not contain 'abacus'. Exiting script."
  exit 1
fi

echo "Hostname contains 'abacus', continuing script"

# Activate virtual environment and run Python script
source .venv/bin/activate
python main_vae.py --load_checkpoint_training vae_checkpoints/Feb_05_2025_319.pth
