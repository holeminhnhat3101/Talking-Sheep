# Talking Sheep Architecture

This document describes the architecture and design of Talking Sheep.

## System Overview

Talking Sheep is a voice assistant that runs entirely offline on Raspberry Pi 5. It processes voice input through a streaming pipeline with minimal latency.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Talking Sheep                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Microphone  │───▶│ AudioRecorder│───▶│VietnameseSTT │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                    │                    │                │
│         │                    ▼                    ▼                │
│         │            ┌──────────────┐    ┌──────────────┐       │
│         │            │   VAD +      │    │  Zipformer   │       │
│         │            │  Pre-roll    │    │  Streaming   │       │
│         │            └──────────────┘    └──────────────┘       │
│         │                    │                    │                │
│         │                    └────────────────────┘                │
│         │                                   │                     │
│         │                                   ▼                     │
│         │                          ┌──────────────┐              │
│         │                          │   ChatLLM    │              │
│         │                          │  (Qwen3)     │              │
│         │                          └──────────────┘              │
│         │                                   │                     │
│         │                                   ▼                     │
│         │                          ┌──────────────┐              │
│         │                          │  Streaming    │              │
│         │                          │  Response    │              │
│         │                          └──────────────┘              │
│         │                                   │                     │
│         │                    ┌──────────────┴──────────────┐     │
│         │                    │                             │     │
│         ▼                    ▼                             ▼     │
│  ┌──────────────┐    ┌──────────────┐            ┌──────────────┐│
│  │ AudioPlayer  │    │ VoiceLayer   │            │ Kokoro TTS   ││
│  └──────────────┘    └──────────────┘            └──────────────┘│
│         │                    │                             │     │
│         └────────────────────┴─────────────────────────────┘     │
│                              │                                    │
│                              ▼                                    │
│                       ┌──────────────┐                           │
│                       │   Speakers    │                           │
│                       └──────────────┘                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Audio Recorder (`audio_recorder.py`)

**Responsibilities:**
- Capture audio from microphone
- Voice Activity Detection (VAD)
- Pre-roll recording (capture speech before detection)
- Native format handling
- Channel selection and resampling
- Automatic calibration

**Key Features:**
- Supports multi-channel microphones
- Auto-detects ReSpeaker Mic Array
- Configurable silence threshold
- Noise-adaptive calibration
- Native format preservation for debugging

**Configuration:**
```python
SILENCE_THRESHOLD = 250
SILENCE_DURATION = 1.5
MIN_SPEECH_DURATION = 0.4
MAX_RECORDING_DURATION = 15.0
PRE_ROLL_DURATION = 0.25
```

### 2. Vietnamese STT (`vietnamese_stt.py`)

**Responsibilities:**
- Streaming speech recognition
- Real-time transcription
- Partial result handling

**Technology:**
- Model: Zipformer-30M-RNNT-Streaming-6000h
- Runtime: sherpa-onnx
- Language: Vietnamese

**Configuration:**
```python
STT_MODEL_REPO = "hynt/Zipformer-30M-RNNT-Streaming-6000h"
STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_NUM_THREADS = 2
```

### 3. Chat LLM (`chat_llm.py`)

**Responsibilities:**
- Generate responses using local LLM
- Streaming text generation
- Context management
- System prompt handling

**Technology:**
- Model: Qwen3-1.7B-Q4_K_M.gguf
- Runtime: llama-cpp-python
- Quantization: Q4_K_M

**Configuration:**
```python
LLM_MODEL_REPO = "ggml-org/Qwen3-1.7B-GGUF"
LLM_MAX_TOKENS = 64
LLM_TEMPERATURE = 0.6
LLM_CONTEXT = 1024
LLM_NUM_THREADS = CPU_COUNT
```

### 4. Streaming Response (`streaming_response.py`)

**Responsibilities:**
- Coordinate LLM and TTS streaming
- Queue management between components
- Sentence boundary detection
- Audio segment orchestration
- Performance metrics

**Architecture:**
```
LLM Worker          TTS Worker          Playback Worker
    │                   │                    │
    ▼                   ▼                    ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│ Generate│──────▶  │Synthesize│──────▶│  Play   │
│ Sentences│         │ Sentences│         │ Audio   │
└─────────┘         └─────────┘         └─────────┘
    │                   │                    │
    └───────────────────┴────────────────────┘
                Sentence Queue    Audio Queue
```

**Queue Messages:**
- `SentenceMessage`: Text sentence for TTS
- `SentenceEnd`: End of stream marker
- `SentenceFailure`: Error propagation
- `AudioSpeech`: Synthesized speech segment
- `AudioInterstitial`: Bleat/effect segment
- `AudioEnd`: End of audio marker
- `AudioFailure`: Audio error propagation

### 5. Voice Layer (`voice_layer.py`)

**Responsibilities:**
- Sentence assembly from streaming text
- TTS synthesis using Kokoro Vietnamese
- Audio normalization (48kHz, stereo, 16-bit)
- Bleat insertion between sentences
- Audio segment construction

**Key Functions:**
- `StreamingSentenceAssembler`: Assembles sentences from LLM chunks
- `synthesize_sentence`: Converts text to audio using Kokoro
- `normalize_segment`: Normalizes audio to target format
- `build_inter_sentence_segment`: Creates inter-sentence audio with bleats

### 6. Audio Player (`audio_player.py`)

**Responsibilities:**
- Play audio segments sequentially
- Persistent stream management
- Device selection
- Format conversion

**Key Features:**
- Maintains persistent PyAudio stream
- Supports multiple output devices
- Real-time playback without blocking

### 7. Microphone Menu (`microphone_menu.py`)

**Responsibilities:**
- Interactive microphone selection
- Terminal UI with arrow keys
- Device information display

**Features:**
- Arrow key navigation
- Enter to select
- Escape to cancel
- Single-device auto-selection

### 8. Environment Setup (`env_setup.py`)

**Responsibilities:**
- Configure thread environment
- Set OMP_NUM_THREADS
- Configure BLAS libraries
- Pre-import optimization

**Thread Allocation:**
```python
LLM_NUM_THREADS = CPU_COUNT              # 100% of cores
TTS_INTRA_THREADS = CPU_COUNT * 0.75     # 75% of cores
OMP_NUM_THREADS = CPU_COUNT * 0.75      # 75% of cores
STT_NUM_THREADS = 2                      # Fixed low count
```

## Data Flow

### Recording Phase
```
Microphone → Native Format → VAD → Pre-roll → 
16kHz Mono Float32 → STT Streaming → Text
```

### Processing Phase
```
Text → LLM Streaming → Sentence Assembly → 
Sentence Queue → TTS Worker → Audio Queue
```

### Playback Phase
```
Audio Queue → Playback Worker → Audio Player → Speakers
```

## Streaming Pipeline

### Advantages
- **Low Latency**: First sentence plays while subsequent sentences are synthesized
- **Responsive**: Immediate feedback for user
- **Memory Efficient**: Processes text in chunks rather than full response

### Trade-offs
- **TTS Blocks LLM**: Single producer thread pauses LLM during TTS synthesis
- **No Barge-in**: Cannot interrupt assistant while speaking
- **Sequential**: Recording and playback never overlap

## Thread Model

### Main Thread
- Component initialization
- Microphone selection
- Conversation loop coordination

### LLM Worker Thread
- LLM text generation
- Sentence boundary detection
- Queue management

### TTS Worker Thread
- Sentence synthesis
- Audio normalization
- Bleat insertion
- Audio queue management

### Playback Worker Thread
- Audio segment playback
- Stream management
- Error handling

## Error Handling

### Microphone Errors
- Exponential backoff retry
- Configurable delays
- Graceful degradation

### STT Errors
- Partial result fallback
- Error propagation through queue
- Automatic retry on next utterance

### LLM Errors
- Generator cleanup
- Error propagation
- Context preservation

### TTS Errors
- Sentence skipping on failure
- Error propagation
- Bleat preservation

## Configuration System

### Priority Order
1. Environment variables
2. `config.py` defaults
3. Command-line arguments

### Key Environment Variables
```bash
# Audio
AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=0
SILENCE_THRESHOLD=250

# LLM
LLM_MODEL_PATH=/path/to/model.gguf
LLM_NUM_THREADS=4
LLM_MAX_TOKENS=64

# STT
STT_MODEL_DIR=/path/to/zipformer
STT_NUM_THREADS=2

# TTS
DEFAULT_VOICE=mai_linh
TTS_INTRA_THREADS=3

# Effects
BLEAT_PROBABILITY=1.0
```

## Performance Characteristics

### Latency Breakdown
- VAD Detection: ~100-200ms
- STT Recognition: ~200-400ms
- LLM Generation: ~500-1500ms
- TTS Synthesis: ~300-800ms per sentence
- Total First Sentence: ~1-3 seconds

### Memory Usage
- LLM Model: ~1GB (Q4_K_M quantized)
- STT Model: ~100MB
- TTS Model: ~200MB
- Runtime: ~200-500MB
- Total: ~1.5-2GB

### CPU Usage
- Idle: ~5-10%
- Recording: ~15-25%
- STT: ~20-30%
- LLM: ~60-80%
- TTS: ~40-60%
- Playback: ~10-20%

## Dependencies

### Core Dependencies
- `sherpa-onnx`: STT runtime
- `llama-cpp-python`: LLM runtime
- `kokoro-vietnamese`: TTS engine (submodule)
- `pydub`: Audio processing
- `pyaudio`: Audio I/O
- `numpy`: Numerical operations

### System Dependencies
- PortAudio (for PyAudio)
- SoX (for audio utilities)
- Git (for submodules)

## Security Considerations

- **Offline Operation**: No network calls after initial setup
- **Local Processing**: All data stays on device
- **No Telemetry**: No usage tracking or analytics
- **Model Integrity**: Models loaded from local storage

## Extensibility

### Adding New Voices
1. Add voice files to Kokoro-Vietnamese
2. Update `DEFAULT_VOICE` in config
3. Rebuild Kokoro model if needed

### Changing LLM
1. Download compatible GGUF model
2. Update `LLM_MODEL_PATH`
3. Adjust context and token limits
4. Test prompt compatibility

### Custom Audio Effects
1. Add audio files to `assets/bleats/`
2. Update `load_bleat_segments()` in voice_layer.py
3. Adjust probability and timing in config

## Testing

### Test Structure
```
tests/
├── test_audio_player.py
├── test_audio_recorder_devices.py
├── test_chat_llm.py
├── test_chat_llm_streaming.py
├── test_microphone_menu.py
├── test_smoke.py
├── test_streaming_response.py
├── test_talking_sheep_voice.py
├── test_voice_layer.py
└── test_voice_layer_streaming.py
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_audio_player.py

# Run with coverage
python -m pytest tests/ --cov=src
```

## Future Architecture Considerations

### Potential Improvements
- **Concurrent LLM/TTS**: Separate producer threads for true parallelism
- **Barge-in Support**: Echo cancellation and simultaneous recording
- **Dynamic Threading**: Adaptive thread allocation based on load
- **Model Switching**: Runtime model selection for different use cases
- **Distributed Processing**: Offload processing to external devices

### Scalability
- Multi-room support
- Multi-user sessions
- External API integration (optional)
- Cloud model fallback (optional)