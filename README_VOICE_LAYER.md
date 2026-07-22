# Talking Sheep Voice Layer

Complete voice interaction system for the Talking Sheep project with Vietnamese speech recognition, LLM integration, and audio composition with sheep sound effects.

## Architecture

```
Microphone → VAD → STT → LLM → Sentence Splitter → Bleat Selection → Audio Composition → Speaker
```

## Components

### Core Modules

- **`voice_layer.py`**: Sentence splitting, bleat selection, and audio composition
- **`audio_recorder.py`**: Microphone capture with VAD (Voice Activity Detection)
- **`vietnamese_stt.py`**: Vietnamese Speech-to-Text using Whisper
- **`audio_player.py`**: WAV file playback
- **`chat_phogpt.py`**: PhoGPT Q4 LLM integration with application-level thinking status
- **`talking_sheep_voice.py`**: Main orchestration layer

### Features

- **Sentence Splitting**: Splits LLM responses at `.`, `!`, `?` boundaries
- **Bleat Insertion**: 
  - 30% probability of inserting a sheep sound
  - Only between sentences (never inside words)
  - Context-aware selection (happy, confused, or neutral)
- **Audio Composition**: Concatenates a udio segments with optional bleat and pauses
- **VAD Recording**: Automatic recording stops on silence detection

## Installation

```bash
pip install -r requirements.txt
```

### System Dependencies

**Linux (Raspberry Pi):**
```bash
sudo apt-get install portaudio19-dev
sudo apt-get install ffmpeg
sudo apt-get install alsa-utils libsndfile1
```

The launcher does not download PhoGPT. Provision `PhoGPT-4B-Chat.Q4_K_M.gguf`
locally and set `PHOGPT_MODEL_PATH` to its path before starting the assistant.
Kokoro is installed from `Kokoro-Vietnamese[onnx]` and may fetch its runtime
assets on first use.

**Windows:**
- PortAudio is included with pyaudio
- FFmpeg required for pydub audio processing

## Setup

1. **Add sheep sound files** to `assets/bleats/`:
   - `short.wav` - Short neutral bleat
   - `happy.wav` - Happy bleat (for sentences ending with `!`)
   - `confused.wav` - Confused bleat (for sentences ending with `?`)
   - `long.wav` - Long bleat (optional)

2. **Configure environment variables** (optional):
   ```bash
   export PHOGPT_MODEL_PATH=/opt/models/PhoGPT-4B-Chat.Q4_K_M.gguf
   export PHOGPT_CONTEXT=4096
   export PHOGPT_THREADS=4
   ```

## Usage

### Run the voice layer:

```bash
python talking_sheep_voice.py
```

### Run tests:

```bash
pytest test_voice_layer.py -v
```

### Use as a module:

```python
from talking_sheep_voice import TalkingSheepVoice

# Optionally pass an external TTS engine (e.g. Kokoro Vietnamese)
voice = TalkingSheepVoice(tts=my_kokoro_tts)
voice.run_once()  # Single conversation
voice.run_continuous()  # Continuous loop
```

## Pipeline Details

### 1. Recording
- Sample rate: 16kHz
- Channels: 1 (mono)
- VAD threshold: 500 RMS
- Silence duration: 1.0 second

### 2. STT (Whisper)
- Model: `tiny` (fastest for Raspberry Pi)
- Language: Vietnamese
- Output: Plain text

### 3. LLM (PhoGPT)
- Model: local `PhoGPT-4B-Chat.Q4_K_M.gguf` configured by `PHOGPT_MODEL_PATH`
- Context: 4096 tokens
- Temperature: 0.7
- Max tokens: 256

### 4. Audio Composition
- Sample rate: 16kHz
- Bleat fade: 25ms in, 70ms out
- Bleat volume: -3dB
- Pause duration: 100ms

## Testing

Test cases cover:
- Single sentence splitting
- Multiple sentences with different punctuation
- Decimal number handling
- Text without punctuation
- Bleat position selection
- Bleat file selection based on punctuation

## Raspberry Pi Optimization

For better performance on Raspberry Pi:
- Use `tiny` Whisper model
- Reduce `PHOGPT_THREADS` to 2-4
- Ensure adequate cooling

## Voice and sheep character settings

The selected Kokoro voice is controlled by `VOICE_NAME` in `src/voice_layer.py`
and by the `--voice` command-line option; the default is `diem_trinh` (Diễm
Trinh). Bleat gain, fades, and pauses are also centralized in
`src/voice_layer.py` (`BLEAT_VOLUME_DB`, `BLEAT_FADE_IN_MS`,
`BLEAT_FADE_OUT_MS`, `PAUSE_BEFORE_BLEAT_MS`, and `PAUSE_AFTER_BLEAT_MS`).

## Troubleshooting

**No audio input:** Check microphone permissions and PortAudio installation

**Slow STT:** Use smaller Whisper model (`tiny` or `base`)

**Audio playback issues:** Verify FFmpeg installation and audio device configuration

## License

Same as parent Talking Sheep project.
