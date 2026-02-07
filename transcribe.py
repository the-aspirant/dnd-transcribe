#!/usr/bin/env python3
"""
dnd-transcribe: Multi-track audio → single labeled transcript.

Takes a directory of audio files (one per speaker), transcribes each
via AssemblyAI, merges by timestamp, outputs a single plain text file.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_BASE = "https://api.assemblyai.com/v2"
AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".ogg", ".m4a", ".wma", ".opus"}


def headers():
    return {"authorization": os.environ["ASSEMBLYAI_API_KEY"]}


def upload_file(filepath: str) -> str:
    """Upload a local file to AssemblyAI. Returns the upload URL."""
    with open(filepath, "rb") as f:
        r = requests.post(f"{API_BASE}/upload", headers=headers(), data=f)
    r.raise_for_status()
    return r.json()["upload_url"]


def transcribe_url(audio_url: str) -> dict:
    """Submit a transcription job and poll until complete."""
    body = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-pro"],
    }
    r = requests.post(f"{API_BASE}/transcript", headers=headers(), json=body)
    r.raise_for_status()
    transcript_id = r.json()["id"]

    # Poll until done
    while True:
        r = requests.get(f"{API_BASE}/transcript/{transcript_id}", headers=headers())
        r.raise_for_status()
        data = r.json()
        if data["status"] == "completed":
            return data
        if data["status"] == "error":
            raise RuntimeError(f"Transcription failed: {data.get('error')}")
        time.sleep(5)


def get_sentences(transcript_id: str) -> list[dict]:
    """Get sentence-level timestamps for a transcript."""
    r = requests.get(f"{API_BASE}/transcript/{transcript_id}/sentences", headers=headers())
    r.raise_for_status()
    return r.json()["sentences"]


def transcribe_track(filepath: str) -> list[dict]:
    """Upload and transcribe a single audio file. Returns [{start, end, text}, ...]."""
    url = upload_file(filepath)
    data = transcribe_url(url)
    sentences = get_sentences(data["id"])
    return [{"start_ms": s["start"], "end_ms": s["end"], "text": s["text"]} for s in sentences]


def ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS."""
    total_seconds = ms // 1000
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_audio_files(audio_dir: Path) -> list[str]:
    """Find all audio files in a directory."""
    return sorted(
        f.name for f in audio_dir.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )


def load_speakers(mapping_path: str | None, audio_dir: Path, skip_confirm: bool = False) -> dict:
    """Load speaker mapping, prompting for any missing labels."""
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

        save_path = mapping_path or str(audio_dir / "speakers.json")
        with open(save_path, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Mapping saved to {save_path}\n")

    # Show final mapping and confirm
    active = {k: v for k, v in existing.items() if k in audio_files}
    print("Speaker mapping:")
    for filename, label in active.items():
        print(f"  {filename} → {label}")
    if not skip_confirm:
        confirm = input("\nLook correct? [Y/n] ").strip().lower()
        if confirm and confirm != "y":
            print("Aborted. Edit your speakers.json and re-run.", file=sys.stderr)
            sys.exit(1)
    print()

    return active


def merge_and_format(tracks: dict[str, list[dict]]) -> str:
    """Merge all tracks by timestamp and format as plain text."""
    all_lines = []
    for label, sentences in tracks.items():
        for s in sentences:
            all_lines.append((s["start_ms"], label, s["text"]))

    all_lines.sort(key=lambda x: x[0])

    output = []
    for start_ms, label, text in all_lines:
        timestamp = ms_to_timestamp(start_ms)
        output.append(f"[{timestamp}] {label}: {text}")

    return "\n".join(output)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe multi-track audio into a single labeled transcript."
    )
    parser.add_argument("audio_dir", help="Directory containing audio files (one per speaker).")
    parser.add_argument("speakers", nargs="?", default=None,
                        help="JSON file mapping filenames to speaker labels (optional).")
    parser.add_argument("-o", "--output", default="transcript.txt",
                        help="Output file path (default: transcript.txt).")
    parser.add_argument("--parallel", type=int, default=3,
                        help="Number of parallel transcriptions (default: 3).")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip confirmation prompt.")
    args = parser.parse_args()

    # Validate
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        print("Error: ASSEMBLYAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_dir():
        print(f"Error: {audio_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    speakers = load_speakers(args.speakers, audio_dir, skip_confirm=args.yes)

    # Transcribe
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
