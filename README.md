# dnd-transcribe

Transcribe Discord multi-track recordings into speaker-labeled transcripts. Built for D&D session recaps.

## What it does

Discord records voice calls as separate audio files per speaker. This tool:

1. Takes a folder of those AAC/MP3/WAV files
2. Transcribes each using OpenAI's Whisper API (parallel, fast)
3. Merges by timestamp into a single transcript with speaker labels
4. Outputs clean markdown

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/dnd-transcribe.git
cd dnd-transcribe

# Install dependencies
pip install -r requirements.txt

# Also need ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

## Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# Transcribe a session
python transcribe.py /path/to/session-folder

# Custom output file
python transcribe.py /path/to/session-folder --output recap.md
```

### Input

A folder with audio files from Discord's multi-track recording:
```
session-2024-01-15/
├── DungeonMaster.aac
├── Aragorn.aac
├── Gandalf.aac
└── Frodo.aac
```

### Output

```markdown
**DungeonMaster:** You enter the dimly lit tavern. The smell of ale and pipe smoke fills the air.

**Aragorn:** I look around for any suspicious characters.

**DungeonMaster:** Roll perception.

**Aragorn:** That's a 17.

**DungeonMaster:** You notice a hooded figure in the corner, watching your group intently.
```

## Cost

Whisper API: ~$0.006 per minute of audio.
- 3-hour session ≈ $1.08
- 4 players + DM, each track counts separately

## Roadmap

- [ ] Merge tracks to single audio file (for archival)
- [ ] Local Whisper support (slower, but free)
- [ ] Timestamp annotations
- [ ] Auto-cleanup of filler words

## License

MIT
