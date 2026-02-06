#!/usr/bin/env python3
"""
D&D Multi-Track Transcription Tool

Takes a folder of per-speaker AAC files (Discord recording format)
and produces a merged, speaker-labeled transcript.

Usage:
    python transcribe.py /path/to/recording/folder [--output transcript.md]

Supports: Groq (default, fast + cheap), OpenAI, DeepInfra
"""

import argparse
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Provider configurations
PROVIDERS = {
    'groq': {
        'base_url': 'https://api.groq.com/openai/v1',
        'env_key': 'GROQ_API_KEY',
        'model': 'whisper-large-v3-turbo',
        'description': 'Groq (384x realtime, $0.67/1000min)'
    },
    'openai': {
        'base_url': None,  # Uses default
        'env_key': 'OPENAI_API_KEY',
        'model': 'whisper-1',
        'description': 'OpenAI ($6/1000min)'
    },
    'deepinfra': {
        'base_url': 'https://api.deepinfra.com/v1/openai',
        'env_key': 'DEEPINFRA_API_KEY',
        'model': 'openai/whisper-large-v3',
        'description': 'DeepInfra ($0.45/1000min)'
    }
}


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
        return float(data['format'].get('duration', 0))
    return 0.0


def transcribe_track(filepath, client, model):
    """Transcribe a single audio track using Whisper API."""
    print(f"  Transcribing: {filepath.name}...")
    
    with open(filepath, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    # Handle both dict and object responses
    if hasattr(response, 'text'):
        text = response.text
        segments = response.segments if hasattr(response, 'segments') else []
    else:
        text = response.get('text', '')
        segments = response.get('segments', [])
    
    return {
        'filename': filepath.name,
        'speaker': extract_speaker_name(filepath.name),
        'text': text,
        'segments': segments,
        'duration': get_audio_duration(filepath)
    }


def extract_speaker_name(filename):
    """Extract speaker name from filename. Discord format varies."""
    # Common patterns: "Username.aac", "Username-001.aac", "track_Username.aac"
    name = Path(filename).stem
    # Remove common suffixes
    for suffix in ['-001', '-002', '_001', '_002', '-1', '-2']:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
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
                # Handle both dict and object segment formats
                if isinstance(seg, dict):
                    start, end, text = seg['start'], seg['end'], seg['text']
                else:
                    start, end, text = seg.start, seg.end, seg.text
                
                all_segments.append({
                    'speaker': speaker,
                    'start': start,
                    'end': end,
                    'text': text.strip()
                })
        elif t['text']:
            # No segments, just full text
            all_segments.append({
                'speaker': speaker,
                'start': 0,
                'end': t['duration'],
                'text': t['text'].strip()
            })
    
    # Sort by start time
    all_segments.sort(key=lambda x: x['start'])
    
    return all_segments


def format_timestamp(seconds):
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def format_transcript(segments, include_timestamps=False):
    """Format merged segments into readable transcript."""
    lines = []
    current_speaker = None
    current_block = []
    block_start = 0
    
    for seg in segments:
        if not seg['text']:
            continue
            
        if seg['speaker'] != current_speaker:
            # Flush previous block
            if current_block and current_speaker:
                text = ' '.join(current_block).strip()
                if text:
                    if include_timestamps:
                        ts = format_timestamp(block_start)
                        lines.append(f"**[{ts}] {current_speaker}:** {text}\n")
                    else:
                        lines.append(f"**{current_speaker}:** {text}\n")
            
            current_speaker = seg['speaker']
            current_block = [seg['text']]
            block_start = seg['start']
        else:
            current_block.append(seg['text'])
    
    # Flush final block
    if current_block and current_speaker:
        text = ' '.join(current_block).strip()
        if text:
            if include_timestamps:
                ts = format_timestamp(block_start)
                lines.append(f"**[{ts}] {current_speaker}:** {text}\n")
            else:
                lines.append(f"**{current_speaker}:** {text}\n")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe multi-track D&D recordings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Providers:
  groq      - Groq (default): 384x realtime, $0.67/1000min
  openai    - OpenAI: $6/1000min  
  deepinfra - DeepInfra: $0.45/1000min

Set API key via environment variable (GROQ_API_KEY, OPENAI_API_KEY, etc.)
or pass directly with --api-key.
        """
    )
    parser.add_argument('folder', help='Folder containing AAC/MP3/WAV files')
    parser.add_argument('--output', '-o', default='transcript.md', help='Output file')
    parser.add_argument('--provider', '-p', default='groq', choices=PROVIDERS.keys(),
                        help='Transcription provider (default: groq)')
    parser.add_argument('--api-key', help='API key (or set via environment)')
    parser.add_argument('--timestamps', '-t', action='store_true', 
                        help='Include timestamps in output')
    parser.add_argument('--workers', '-w', type=int, default=3,
                        help='Parallel transcription workers (default: 3)')
    args = parser.parse_args()
    
    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)
    
    # Find audio files
    audio_files = []
    for ext in ['*.aac', '*.mp3', '*.wav', '*.m4a', '*.ogg', '*.flac']:
        audio_files.extend(folder.glob(ext))
    
    if not audio_files:
        print(f"Error: No audio files found in {folder}")
        sys.exit(1)
    
    print(f"Found {len(audio_files)} audio files")
    
    # Get provider config
    provider = PROVIDERS[args.provider]
    api_key = args.api_key or os.environ.get(provider['env_key'])
    
    if not api_key:
        print(f"Error: No API key. Set {provider['env_key']} or use --api-key")
        sys.exit(1)
    
    if not HAS_OPENAI:
        print("Error: openai package not installed. Run: pip install openai")
        sys.exit(1)
    
    # Create client
    client_kwargs = {'api_key': api_key}
    if provider['base_url']:
        client_kwargs['base_url'] = provider['base_url']
    client = OpenAI(**client_kwargs)
    
    print(f"\nUsing: {provider['description']}")
    print("Transcribing tracks...")
    
    # Transcribe all tracks (parallel for speed)
    transcriptions = []
    total_duration = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(transcribe_track, f, client, provider['model']): f 
            for f in audio_files
        }
        for future in as_completed(futures):
            try:
                result = future.result()
                transcriptions.append(result)
                total_duration += result['duration']
                print(f"  ✓ {result['speaker']}: {len(result['text'])} chars, "
                      f"{result['duration']:.1f}s")
            except Exception as e:
                filepath = futures[future]
                print(f"  ✗ {filepath.name}: {e}")
    
    if not transcriptions:
        print("Error: No successful transcriptions")
        sys.exit(1)
    
    # Merge and format
    print("\nMerging transcripts...")
    segments = merge_transcripts(transcriptions)
    transcript = format_transcript(segments, include_timestamps=args.timestamps)
    
    # Write output
    output_path = Path(args.output)
    output_path.write_text(transcript)
    
    print(f"\n✓ Transcript written to {output_path}")
    print(f"  {len(segments)} segments from {len(transcriptions)} speakers")
    print(f"  Total audio: {format_timestamp(total_duration)}")


if __name__ == '__main__':
    main()
