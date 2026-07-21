# Talking Sheep Voice Layer

Complete voice interaction system for the Talking Sheep project with Vietnamese speech recognition, LLM integration, and TTS with sheep sound effects.

## Architecture

```
Microphone → VAD → STT → LLM → Sentence Splitter → Bleat Selection → TTS → Audio Composition → Speaker
```

## Components

### Core Modules

- **`voice_layer.py`**: Sentence splitting, bleat selection, and audio composition
- **`audio_recorder.py`**: Microphone capture with VAD (Voice Activity Detection)
- **`vietnamese_stt.py`**: Vietnamese Speech-to-Text using Whisper
- **`vietnamese_tts.py`**: Vietnamese Text-to-Speech using edge-tts
- **`audio_player.py`**: WAV file playback
- **`chat_phogpt_q8.py`**: PhoGPT LLM integration (enhanced with wrapper class)
- **`talking_sheep_voice.py`**: Main orchestration layer

### Features

- **Sentence Splitting**: Splits LLM responses at `.`, `!`, `?` boundaries
- **Bleat Insertion**: 
  - 30% probability of inserting a sheep sound
  - Only between sentences (never inside words)
  - Context-aware selection (happy, confused, or neutral)
- **Audio Composition**: Concatenates TTS segments with optional bleat and pauses
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
```

**Windows:**
- PortAudio is included with pyaudio
- FFmpeg required for pydub audio processing

## Setup

1. **Download Vietnamese TTS models** from sherpa-onnx:
   ```bash
   mkdir -p models/tts
   # Download Vietnamese VITS models from:
   # https://github.com/k2-fsa/sherpa-onnx/releases
   # Place model.onnx and tokens.txt in models/tts/
   ```

2. **Add sheep sound files** to `assets/bleats/`:
   - `short.wav` - Short neutral bleat
   - `happy.wav` - Happy bleat (for sentences ending with `!`)
   - `confused.wav` - Confused bleat (for sentences ending with `?`)
   - `long.wav` - Long bleat (optional)

2. **Configure environment variables** (optional):
   ```bash
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

voice = TalkingSheepVoice(tts_model_dir=Path("models/tts"))
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
- Model: vinai/PhoGPT-4B-Chat-Q8_0.gguf
- Context: 4096 tokens
- Temperature: 0.7
- Max tokens: 256

### 4. TTS (sherpa-onnx)
- Model: Vietnamese VITS (local ONNX)
- Provider: CPU
- Threads: 2
- Format: WAV

### 5. Audio Composition
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
- Use smaller TTS voice if available
- Ensure adequate cooling

## Troubleshooting

**No audio input:** Check microphone permissions and PortAudio installation

**Slow STT:** Use smaller Whisper model (`tiny` or `base`)

**TTS errors:** Ensure Vietnamese VITS models are downloaded to `models/tts/`

**Audio playback issues:** Verify FFmpeg installation and audio device configuration

## License

Same as parent Talking Sheep project.
