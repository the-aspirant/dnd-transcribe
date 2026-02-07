#!/usr/bin/env python3
"""
dnd-transcribe: Multi-track audio → single labeled transcript.

Takes a directory of audio files (one per speaker) and a speaker mapping,
transcribes each track via AssemblyAI, merges by timestamp, and outputs
a single plain text transcript.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import assemblyai as aai


def load_speakers(mapping_path: str) -> dict:
    """Load speaker mapping from JSON file.
    
    Format: {"filename": "Label", ...}
    Example: {"1-kanchosensei.mp3": "Matt (Strahd)", "2-sarah.aac": "Sarah (Ireena)"}
    """
    with open(mapping_path) as f:
        return json.load(f)


def transcribe_track(filepath: str) -> list[dict]:
    """Transcribe a single audio file. Returns list of {start_ms, end_ms, text}."""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(filepath)

    if transcript.status == aai.TranscriptStatus.error:
        print(f"  ERROR: {transcript.error}", file=sys.stderr)
        return []

    sentences = transcript.get_sentences()
    return [
        {"start_ms": s.start, "end_ms": s.end, "text": s.text}
        for s in sentences
    ]


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS."""
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def merge_and_format(tracks: dict[str, list[dict]]) -> str:
    """Merge all tracks by timestamp and format as plain text.
    
    tracks: {speaker_label: [{start_ms, end_ms, text}, ...], ...}
    """
    # Flatten all sentences with speaker labels
    all_lines = []
    for label, sentences in tracks.items():
        for s in sentences:
            all_lines.append((s["start_ms"], label, s["text"]))

    # Sort by timestamp
    all_lines.sort(key=lambda x: x[0])

    # Format output
    output = []
    for start_ms, label, text in all_lines:
        timestamp = ms_to_timestamp(start_ms)
        output.append(f"[{timestamp}] {label}: {text}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe multi-track audio into a single labeled transcript."
    )
    parser.add_argument(
        "audio_dir",
        help="Directory containing audio files (one per speaker)."
    )
    parser.add_argument(
        "speakers",
        help="JSON file mapping filenames to speaker labels."
    )
    parser.add_argument(
        "-o", "--output",
        default="transcript.txt",
        help="Output file path (default: transcript.txt)."
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=3,
        help="Number of parallel transcriptions (default: 3)."
    )
    args = parser.parse_args()

    # Setup
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        print("Error: ASSEMBLYAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)
    aai.settings.api_key = api_key

    # Load speaker mapping
    speakers = load_speakers(args.speakers)
    audio_dir = Path(args.audio_dir)

    # Validate files exist
    for filename in speakers:
        filepath = audio_dir / filename
        if not filepath.exists():
            print(f"Error: {filepath} not found.", file=sys.stderr)
            sys.exit(1)

    # Transcribe all tracks
    print(f"Transcribing {len(speakers)} tracks...")
    tracks = {}

    def process_track(filename, label):
        filepath = str(audio_dir / filename)
        print(f"  [{label}] Uploading and transcribing {filename}...")
        sentences = transcribe_track(filepath)
        print(f"  [{label}] Done — {len(sentences)} sentences.")
        return label, sentences

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {
            pool.submit(process_track, fname, label): label
            for fname, label in speakers.items()
        }
        for future in as_completed(futures):
            label, sentences = future.result()
            tracks[label] = sentences

    # Merge and write
    print("Merging tracks by timestamp...")
    output = merge_and_format(tracks)

    with open(args.output, "w") as f:
        f.write(output + "\n")

    print(f"Transcript written to {args.output}")
    print(f"Total lines: {output.count(chr(10)) + 1}")


if __name__ == "__main__":
    main()
