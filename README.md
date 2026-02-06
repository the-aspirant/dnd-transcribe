# dnd-transcribe

Transcribe Discord multi-track recordings into speaker-labeled transcripts. Built for D&D session recaps.

## What it does

Discord records voice calls as separate audio files per speaker. This tool:

1. Takes a folder of those AAC/MP3/WAV files
2. Transcribes each track in parallel
3. Merges by timestamp into a single speaker-labeled transcript
4. Outputs clean markdown

## Providers

| Provider | Speed | Cost/hr | 3hr session |
|----------|-------|---------|-------------|
| **Groq** (default) | 384x realtime | $0.04 | ~$0.12 |
| DeepInfra | 83x | $0.03 | ~$0.08 |
| OpenAI | 28x | $0.36 | ~$1.08 |

Groq is the default — a 3-hour session transcribes in **~28 seconds** for **12 cents**.

## Installation

```bash
git clone https://github.com/the-aspirant/dnd-transcribe.git
cd dnd-transcribe
pip install -r requirements.txt

# Also need ffmpeg
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg
```

## Usage

```bash
# Set your API key (get free key at console.groq.com)
export GROQ_API_KEY="gsk_..."

# Transcribe a session
python transcribe.py /path/to/session-folder

# With timestamps
python transcribe.py /path/to/session-folder --timestamps

# Custom output file
python transcribe.py /path/to/session-folder -o recap.md

# Use different provider
python transcribe.py /path/to/session-folder --provider openai
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

With `--timestamps`:
```markdown
**[0:00:00] DungeonMaster:** You enter the dimly lit tavern...

**[0:00:15] Aragorn:** I look around for any suspicious characters.
```

## Getting API Keys

- **Groq** (recommended): Free tier at [console.groq.com](https://console.groq.com)
- **DeepInfra**: [deepinfra.com](https://deepinfra.com)
- **OpenAI**: [platform.openai.com](https://platform.openai.com)

## Roadmap

- [ ] Merge tracks to single audio file
- [ ] Local Whisper support (offline, free)
- [ ] Auto-cleanup of filler words
- [ ] Speaker identification from voice

## License

MIT
