#!/bin/bash
# Quick runner for D&D transcription

source ~/venv/bin/activate

if [ -z "$1" ]; then
    echo "Usage: ./run.sh /path/to/recording/folder [output.md]"
    exit 1
fi

FOLDER="$1"
OUTPUT="${2:-transcript.md}"

python "$(dirname "$0")/transcribe.py" "$FOLDER" --output "$OUTPUT"
