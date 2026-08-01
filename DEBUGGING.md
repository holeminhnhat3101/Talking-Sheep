# Debugging Guide

This guide provides detailed debugging information for Talking Sheep.

## Enable Debug Logging

### Command Line
```bash
./run-chat.sh --log-level DEBUG
```

### Environment Variable
```bash
LOG_LEVEL=DEBUG ./run-chat.sh
```

## Component-Specific Debugging

### Microphone Issues

#### List Available Microphones
```bash
./run-chat.sh --list-mics
```

#### Check System Audio Devices
```bash
# List input devices
arecord -l

# List output devices
aplay -l
```

#### Enable Native Audio Debug
```bash
AUDIO_SAVE_NATIVE_DEBUG=true ./run-chat.sh
```
This saves native format recordings to `runtime/` for inspection.

#### Auto-Calibrate Silence Threshold
```bash
./run-chat.sh --silence-threshold auto
```

#### Manual Threshold Adjustment
```bash
# Increase threshold if background noise triggers recording
./run-chat.sh --silence-threshold 400

# Decrease threshold if speech isn't detected
./run-chat.sh --silence-threshold 150
```

### STT (Speech-to-Text) Issues

#### Enable Partial Results Logging
```bash
STT_LOG_PARTIALS=true ./run-chat.sh
```

#### Check STT Model Files
```bash
ls -la models/zipformer-vi-streaming/
```

Expected files:
- `encoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx`
- `decoder-epoch-31-avg-11-chunk-32-left-128.fp16.onnx`
- `joiner-epoch-31-avg-11-chunk-32-left-128.fp16.onnx`
- `config.json`
- `bpe.model`

#### Change STT Thread Count
```bash
STT_NUM_THREADS=4 ./run-chat.sh
```

#### Change STT Provider
```bash
# Use CPU (default)
STT_PROVIDER=cpu ./run-chat.sh

# Use CUDA if available
STT_PROVIDER=cuda ./run-chat.sh
```

### LLM Issues

#### Check LLM Model
```bash
ls -la models/
```

Expected file: `Qwen3-1.7B-Q4_K_M.gguf`

#### Adjust LLM Thread Count
```bash
LLM_NUM_THREADS=4 ./run-chat.sh
```

#### Adjust LLM Parameters
```bash
# Lower temperature for more deterministic responses
LLM_TEMPERATURE=0.5 ./run-chat.sh

# Increase max tokens for longer responses
LLM_MAX_TOKENS=128 ./run-chat.sh

# Adjust context window
LLM_CONTEXT=2048 ./run-chat.sh
```

#### Custom LLM Model Path
```bash
LLM_MODEL_PATH=/path/to/custom/model.gguf ./run-chat.sh
```

### TTS (Text-to-Speech) Issues

#### Check Kokoro Vietnamese Submodule
```bash
ls -la Kokoro-Vietnamese/
```

If empty, update submodules:
```bash
git submodule update --init --recursive
```

#### Adjust TTS Thread Count
```bash
TTS_INTRA_THREADS=4 ./run-chat.sh
```

#### Change Voice
```bash
DEFAULT_VOICE=mai_linh ./run-chat.sh
```

#### Adjust Speaking Speed
```bash
# Edit src/config.py
SPEAKING_SPEED = 0.8  # Lower = slower
```

### Audio Playback Issues

#### Select Output Device
```bash
AUDIO_OUTPUT_DEVICE=1 ./run-chat.sh
```

#### List Output Devices
```bash
aplay -l
```

### Thread Configuration

The system uses automatic thread allocation based on CPU cores:

- **LLM**: 100% of cores (highest priority)
- **TTS/OMP**: ~75% of cores
- **STT**: Fixed 2 threads

Override defaults:
```bash
LLM_NUM_THREADS=4 TTS_INTRA_THREADS=3 STT_NUM_THREADS=2 ./run-chat.sh
```

## Log Files

Check runtime logs:
```bash
ls -la runtime/
```

## Common Debugging Scenarios

### No Speech Detected
1. Check microphone: `./run-chat.sh --list-mics`
2. Lower threshold: `--silence-threshold 150`
3. Enable debug: `--log-level DEBUG`
4. Check logs for VAD messages

### Poor Recognition Accuracy
1. Enable partials: `STT_LOG_PARTIALS=true`
2. Check audio quality with native debug
3. Verify STT model files are complete
4. Increase STT threads if CPU-bound

### Slow Response Time
1. Check thread allocation: `LLM_NUM_THREADS`, `TTS_INTRA_THREADS`
2. Reduce LLM context: `LLM_CONTEXT=512`
3. Lower max tokens: `LLM_MAX_TOKENS=32`
4. Check CPU usage with `htop`

### Audio Glitches or Distortion
1. Check output device selection
2. Verify sample rate compatibility
3. Check system audio settings
4. Reduce system load

### Microphone Selection Issues
1. Run `--list-mics` to see available devices
2. Use `--input-device N` to select specific device
3. Check device permissions
4. Ensure no other application is using the microphone

## Testing Individual Components

### Test Audio Recording
```bash
python -c "from src.audio_recorder import AudioRecorder; r = AudioRecorder(); print(r.list_selectable_input_devices())"
```

### Test STT
```bash
python -c "from src.vietnamese_stt import VietnameseSTT; stt = VietnameseSTT(); print('STT loaded successfully')"
```

### Test LLM
```bash
python -c "from src.chat_llm import ChatLLM; llm = ChatLLM(); print('LLM loaded successfully')"
```

### Test TTS
```bash
python -c "from src.voice_layer import synthesize_sentence; print('TTS functions available')"
```

## Performance Profiling

Enable performance logging:
```bash
# Add performance logging to your code
import time
start = time.time()
# ... code ...
elapsed = time.time() - start
logger.info(f"Operation took {elapsed:.3f}s")
```

## Network Issues

Although Talking Sheep is designed to work offline, some operations may require network:

- Initial model downloads
- Git submodule updates
- Package installations

Check network connectivity:
```bash
ping -c 3 google.com
```

## Memory Issues

Check memory usage:
```bash
free -h
```

Reduce memory footprint:
```bash
# Use smaller LLM model
LLM_MODEL_PATH=/path/to/smaller/model.gguf ./run-chat.sh

# Reduce context
LLM_CONTEXT=512 ./run-chat.sh
```

## Getting Help

If issues persist:
1. Check existing documentation: `ARCHITECTURE.md`, `TROUBLESHOOTING.md`
2. Review logs in `runtime/`
3. Enable debug logging and reproduce the issue
4. Check GitHub issues for similar problems
5. Create a new issue with:
   - Hardware details (Raspberry Pi model, RAM)
   - OS version
   - Full debug logs
   - Steps to reproduce