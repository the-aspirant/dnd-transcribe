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


AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".opus"}


def find_audio_files(audio_dir: Path) -> list[str]:
    """Find all audio files in a directory."""
    return sorted(
        f.name for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_speakers(mapping_path: str | None, audio_dir: Path) -> dict:
    """Load speaker mapping, prompting for any missing labels.
    
    If mapping file exists, loads it and prompts for any audio files
    not yet mapped. If no mapping file, prompts for all audio files.
    Saves the result back to the mapping file.
    """
    existing = {}
    if mapping_path and Path(mapping_path).exists():
        with open(mapping_path) as f:
            existing = json.load(f)

    audio_files = find_audio_files(audio_dir)
    if not audio_files:
        print("Error: No audio files found in directory.", file=sys.stderr)
        sys.exit(1)

    missing = [f for f in audio_files if f not in existing]

    if missing:
        print(f"Found {len(missing)} unmapped audio file(s). Enter speaker labels:")
        print("  (e.g. 'Matt (Strahd)' or 'DM - Kancho')\n")
        for filename in missing:
            label = input(f"  {filename} → ").strip()
            if not label:
                print(f"Error: Label required for {filename}.", file=sys.stderr)
                sys.exit(1)
            existing[filename] = label

        # Save updated mapping
        save_path = mapping_path or str(audio_dir / "speakers.json")
        with open(save_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Mapping saved to {save_path}\n")

    # Show final mapping and confirm
    active = {k: v for k, v in existing.items() if k in audio_files}
    print("Speaker mapping:")
    for filename, label in active.items():
        print(f"  {filename} → {label}")
    confirm = input("\nLook correct? [Y/n] ").strip().lower()
    if confirm and confirm != "y":
        print("Aborted. Edit your speakers.json and re-run.", file=sys.stderr)
        sys.exit(1)
    print()

    return active


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
        nargs="?",
        default=None,
        help="JSON file mapping filenames to speaker labels (optional — will prompt if missing)."
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

    # Load speaker mapping (prompts for any missing labels)
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)
    speakers = load_speakers(args.speakers, audio_dir)

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
