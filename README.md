# dnd-transcribe

Transcribe multi-track Discord recordings into a single labeled transcript.

Each audio track = one speaker. Output = one plain text file, sorted by timestamp.

## Setup

```bash
pip install -r requirements.txt
export ASSEMBLYAI_API_KEY=your_key_here
```

## Usage

1. Create a speaker mapping file (`speakers.json`):

```json
{
  "1-kanchosensei.mp3": "Kancho (DM)",
  "3-lupino1956.mp3": "Lupino (Strahd)",
  "4-macaronimatt.mp3": "Matt (Ireena)"
}
```

2. Run:

```bash
python transcribe.py /path/to/audio/files speakers.json -o session2.txt
```

Options:
- `-o, --output` — Output file path (default: `transcript.txt`)
- `--parallel` — Number of parallel transcriptions (default: 3)

## Output format

```
[00:01:23] Kancho (DM): Roll for initiative.
[00:01:25] Matt (Ireena): I got a 17.
[00:01:28] Lupino (Strahd): Natural 20.
```

## How it works

1. Uploads each audio track to AssemblyAI
2. Gets back timestamped sentences
3. Merges all tracks sorted by timestamp
4. Labels each line with the speaker

No chunking, no state files, no cron jobs. One script, one run, one output.

## Limits

- Max file size: 5GB per track
- Max duration: 10 hours per track
- Supported formats: mp3, aac, wav, flac, ogg, m4a, and more

## Cost

~$0.15/hour of audio via AssemblyAI. A typical 3-hour session with 5 players ≈ $2-3.
