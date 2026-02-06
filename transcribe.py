#!/usr/bin/env python3
"""
D&D Multi-Track Transcription Tool

Takes a folder of per-speaker AAC files (Discord recording format)
and produces a merged, speaker-labeled transcript.

Usage:
    python transcribe.py /path/to/recording/folder [--output transcript.md]
"""

import argparse
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

# Will use OpenAI Whisper API
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

def get_audio_duration(filepath):
    """Get duration of audio file using ffprobe."""
    import subprocess
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
         '-of', 'json', str(filepath)],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    return 0.0

def transcribe_track(filepath, client):
    """Transcribe a single audio track using Whisper API."""
    print(f"  Transcribing: {filepath.name}...")
    
    with open(filepath, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    return {
        'filename': filepath.name,
        'speaker': extract_speaker_name(filepath.name),
        'text': response.text,
        'segments': response.segments if hasattr(response, 'segments') else [],
        'duration': get_audio_duration(filepath)
    }

def extract_speaker_name(filename):
    """Extract speaker name from filename. Discord format varies."""
    # Common patterns: "Username.aac", "Username-001.aac", "track_Username.aac"
    name = Path(filename).stem
    # Remove common suffixes
    for suffix in ['-001', '-002', '_001', '_002']:
        name = name.replace(suffix, '')
    # Remove 'track_' prefix if present
    if name.startswith('track_'):
        name = name[6:]
    return name

def merge_transcripts(transcriptions):
    """
    Merge transcriptions from multiple speakers into a single timeline.
    Uses segment timestamps to interleave speakers.
    """
    all_segments = []
    
    for t in transcriptions:
        speaker = t['speaker']
        if t['segments']:
            for seg in t['segments']:
                all_segments.append({
                    'speaker': speaker,
                    'start': seg['start'] if isinstance(seg, dict) else seg.start,
                    'end': seg['end'] if isinstance(seg, dict) else seg.end,
                    'text': seg['text'] if isinstance(seg, dict) else seg.text
                })
        elif t['text']:
            # No segments, just full text
            all_segments.append({
                'speaker': speaker,
                'start': 0,
                'end': t['duration'],
                'text': t['text']
            })
    
    # Sort by start time
    all_segments.sort(key=lambda x: x['start'])
    
    return all_segments

def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def format_transcript(segments, output_format='markdown'):
    """Format merged segments into readable transcript."""
    lines = []
    current_speaker = None
    current_block = []
    
    for seg in segments:
        if seg['speaker'] != current_speaker:
            # Flush previous block
            if current_block and current_speaker:
                text = ' '.join(current_block).strip()
                if text:
                    lines.append(f"**{current_speaker}:** {text}\n")
            current_speaker = seg['speaker']
            current_block = [seg['text'].strip()]
        else:
            current_block.append(seg['text'].strip())
    
    # Flush final block
    if current_block and current_speaker:
        text = ' '.join(current_block).strip()
        if text:
            lines.append(f"**{current_speaker}:** {text}\n")
    
    return '\n'.join(lines)

def main():
    parser = argparse.ArgumentParser(description='Transcribe multi-track D&D recordings')
    parser.add_argument('folder', help='Folder containing AAC files')
    parser.add_argument('--output', '-o', default='transcript.md', help='Output file')
    parser.add_argument('--api-key', help='OpenAI API key (or set OPENAI_API_KEY env var)')
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)
    
    # Find audio files
    audio_files = list(folder.glob('*.aac')) + list(folder.glob('*.mp3')) + list(folder.glob('*.wav'))
    if not audio_files:
        print(f"Error: No audio files found in {folder}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Check for API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: No OpenAI API key. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)
    
    if not HAS_OPENAI:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Transcribe all tracks (parallel for speed)
    print("\nTranscribing tracks...")
    transcriptions = []
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(transcribe_track, f, client): f for f in audio_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                transcriptions.append(result)
                print(f"  ✓ {result['speaker']}: {len(result['text'])} chars")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    # Merge and format
    print("\nMerging transcripts...")
    segments = merge_transcripts(transcriptions)
    transcript = format_transcript(segments)
    
    # Write output
    output_path = Path(args.output)
    output_path.write_text(transcript)
    print(f"\n✓ Transcript written to {output_path}")
    print(f"  {len(segments)} segments from {len(transcriptions)} speakers")

if __name__ == '__main__':
    main()
