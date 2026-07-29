# Talking Sheep Improvement TODO

## User Experience Improvements

### High Priority

- [ ] **Create interactive microphone selection menu**
  - Add arrow key navigation to select microphone
  - Display available microphones with device details
  - Remove need for `--input-device` command-line argument
  - Auto-select if only one microphone available
  - Allow user to cancel selection

- [ ] **Implement sentence-by-sentence streaming from LLM to Kokoro**
  - Stream LLM output as it generates (not wait for complete response)
  - Send completed sentences to Kokoro immediately
  - Start synthesis as soon as first sentence is available
  - Reduce total response time

- [ ] **Implement immediate playback of first synthesized sentence**
  - Play first sentence audio as soon as synthesis completes
  - Continue synthesizing remaining sentences in background
  - Stream playback while synthesis continues
  - Eliminate waiting for all sentences to complete

### Medium Priority

- [ ] **Fix Bluetooth audio output support**
  - PyAudio/PortAudio doesn't automatically route to Bluetooth devices
  - Add `--list-output-devices` flag to show available audio output devices
  - Add interactive output device selection similar to microphone menu
  - Allow users to select Bluetooth devices by name instead of device index
  - Better integration with system audio routing (PulseAudio/PipeWire on Linux)

- [ ] **Add interactive audio output device selection**
  - Similar to microphone selection menu
  - Allow choosing speakers/headphones interactively
  - Remove need for `AUDIO_OUTPUT_DEVICE` environment variable

- [ ] **Add interactive voice selection menu**
  - Display available Kokoro voices
  - Allow runtime voice switching
  - Preview voices before selection

- [ ] **Create configuration file support**
  - Add `config.yaml` or `config.json` for persistent settings
  - Reduce reliance on environment variables
  - Easier for non-technical users

- [ ] **Add first-time setup wizard**
  - Guide users through initial configuration
  - Test microphone and speakers
  - Download models with progress indication
  - Save preferences to config file

- [ ] **Add runtime status indicators**
  - Visual indicator when recording (🎤)
  - Visual indicator when processing (⏳)
  - Visual indicator when speaking (🔊)
  - Clear feedback on system state

- [ ] **Add streaming token display like ChatGPT**
  - Display LLM tokens as they generate in real-time
  - Show token generation speed (tokens/second)
  - Visual streaming output during LLM processing
  - Maintain current streaming implementation for sentence splitting

- [ ] **Add volume control for TTS output**
  - Adjustable volume via command or config
  - Per-voice volume settings
  - Prevent clipping/distortion

- [ ] **Improve error messages and recovery**
  - User-friendly error explanations
  - Suggested fixes for common issues
  - Auto-retry for transient errors
  - Clear troubleshooting steps

- [ ] **Add in-game help system**
  - Display available commands during runtime
  - Show current configuration
  - Provide usage tips

- [ ] **Add model management interface**
  - List available/installed models
  - Easy model download with progress
  - Model version information
  - Space usage display

- [ ] **Add performance monitoring**
  - Display response latency
  - Show resource usage (CPU/memory)
  - Identify bottlenecks
  - Optimization suggestions

### Medium Priority

- [ ] **Test interactive menu and streaming improvements**
  - Verify microphone selection works correctly
  - Test streaming performance improvements
  - Measure latency reduction
  - Ensure audio quality maintained

## Current Status Analysis

### Microphone Selection
- ✅ Currently lists microphones with `--list-mics`
- ✅ Can select via `--input-device` flag
- ❌ Requires manual command-line argument
- ❌ No interactive menu for non-technical users

### LLM to Kokoro Streaming
- ❌ LLM generates complete response first
- ❌ All sentences synthesized sequentially before playback
- ❌ No streaming/sentence-by-sentence synthesis

### Playback Timing
- ❌ All sentences synthesized before any playback
- ❌ Single WAV file created for entire response
- ❌ No immediate playback of first sentence

### Audio Output
- ✅ Supports speakers/headphones via PyAudio
- ✅ `AUDIO_OUTPUT_DEVICE` environment variable
- ✅ Wireless devices work via OS audio layer
