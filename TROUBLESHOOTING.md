# Troubleshooting Guide

This guide provides solutions to common issues encountered when running Talking Sheep.

## Installation Issues

### Submodule Not Loaded

**Problem:** `Kokoro-Vietnamese` directory is empty after cloning.

**Solution:**
```bash
git submodule update --init --recursive
```

### Permission Denied on run-chat.sh

**Problem:** Cannot execute the startup script.

**Solution:**
```bash
chmod +x run-chat.sh
```

### Python Dependencies Fail to Install

**Problem:** `pip install` fails with build errors.

**Solution:**
```bash
# Update pip
python -m pip install --upgrade pip

# Install system dependencies first
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev

# Retry installation
./run-chat.sh
```

### Model Download Fails

**Problem:** Models fail to download during setup.

**Solution:**
```bash
# Check network connectivity
ping -c 3 google.com

# Manually download models
mkdir -p models/zipformer-vi-streaming
cd models/zipformer-vi-streaming
# Download from Hugging Face manually

# Verify files
ls -la
```

## Audio Issues

### No Microphone Detected

**Problem:** "No selectable input devices found" error.

**Solution:**
```bash
# Check system audio devices
arecord -l

# Verify PortAudio installation
python -c "import pyaudio; print(pyaudio.Pa_GetVersionInfo())"

# Install PortAudio if missing
sudo apt-get install portaudio19-dev python3-pyaudio

# Check USB devices
lsusb
```

### Poor Audio Quality

**Problem:** Distorted or low-quality recordings.

**Solution:**
```bash
# Check microphone sample rate
./run-chat.sh --list-mics

# Enable native debug to inspect raw audio
AUDIO_SAVE_NATIVE_DEBUG=true ./run-chat.sh

# Adjust gain/sensitivity in system audio settings
alsamixer
```

### Automatic Recording Triggered

**Problem:** System records background noise as speech.

**Solution:**
```bash
# Increase silence threshold
./run-chat.sh --silence-threshold 400

# Use auto-calibration
./run-chat.sh --silence-threshold auto

# Adjust calibration parameters
AUDIO_CALIBRATION_DURATION=1.0 ./run-chat.sh
AUDIO_THRESHOLD_MULTIPLIER=4.0 ./run-chat.sh
```

### Speech Not Detected

**Problem:** System doesn't detect when you speak.

**Solution:**
```bash
# Decrease silence threshold
./run-chat.sh --silence-threshold 150

# Enable debug logging
./run-chat.sh --log-level DEBUG

# Check microphone input
arecord -f S16_LE -d 5 test.wav
aplay test.wav

# Test with auto-calibration
./run-chat.sh --silence-threshold auto
```

### Audio Playback Issues

**Problem:** No sound or distorted output.

**Solution:**
```bash
# List output devices
aplay -l

# Select specific output device
AUDIO_OUTPUT_DEVICE=1 ./run-chat.sh

# Check system audio
alsamixer

# Test audio output
aplay /usr/share/sounds/alsa/Front_Center.wav
```

## STT (Speech-to-Text) Issues

### Poor Recognition Accuracy

**Problem:** Transcribed text doesn't match speech.

**Solution:**
```bash
# Enable partial results to see recognition process
STT_LOG_PARTIALS=true ./run-chat.sh

# Check audio quality with native debug
AUDIO_SAVE_NATIVE_DEBUG=true ./run-chat.sh

# Verify STT model files
ls -la models/zipformer-vi-streaming/

# Re-download models if corrupted
rm -rf models/zipformer-vi-streaming
./run-chat.sh
```

### STT Not Loading

**Problem:** Error loading STT models.

**Solution:**
```bash
# Check model directory
ls -la models/zipformer-vi-streaming/

# Verify required files exist
ls models/zipformer-vi-streaming/encoder*.onnx
ls models/zipformer-vi-streaming/decoder*.onnx
ls models/zipformer-vi-streaming/joiner*.onnx
ls models/zipformer-vi-streaming/config.json
ls models/zipformer-vi-streaming/bpe.model

# Check file permissions
chmod 644 models/zipformer-vi-streaming/*

# Reduce thread count if memory issues
STT_NUM_THREADS=1 ./run-chat.sh
```

### Slow Recognition

**Problem:** STT takes too long to process speech.

**Solution:**
```bash
# Increase STT threads
STT_NUM_THREADS=4 ./run-chat.sh

# Check CPU usage
htop

# Use greedy search (faster but less accurate)
STT_DECODING_METHOD=greedy_search ./run-chat.sh
```

## LLM Issues

### LLM Not Loading

**Problem:** Error loading LLM model.

**Solution:**
```bash
# Check model file
ls -la models/*.gguf

# Verify model integrity
md5sum models/Qwen3-1.7B-Q4_K_M.gguf

# Re-download model if corrupted
rm models/Qwen3-1.7B-Q4_K_M.gguf
./run-chat.sh

# Check available memory
free -h

# Use smaller model if memory constrained
LLM_MODEL_PATH=/path/to/smaller/model.gguf ./run-chat.sh
```

### Slow Response Generation

**Problem:** LLM takes too long to generate responses.

**Solution:**
```bash
# Increase thread count
LLM_NUM_THREADS=4 ./run-chat.sh

# Reduce context window
LLM_CONTEXT=512 ./run-chat.sh

# Reduce max tokens
LLM_MAX_TOKENS=32 ./run-chat.sh

# Lower temperature (faster sampling)
LLM_TEMPERATURE=0.5 ./run-chat.sh
```

### Poor Response Quality

**Problem:** LLM generates irrelevant or poor responses.

**Solution:**
```bash
# Increase temperature for more variety
LLM_TEMPERATURE=0.7 ./run-chat.sh

# Increase max tokens for longer responses
LLM_MAX_TOKENS=128 ./run-chat.sh

# Adjust system prompt in config.py
# Edit LLM_SYSTEM_PROMPT variable

# Clear conversation history
LLM_HISTORY_MAXLEN=0 ./run-chat.sh
```

### Out of Memory Errors

**Problem:** System runs out of memory during LLM operation.

**Solution:**
```bash
# Check memory usage
free -h

# Reduce context window
LLM_CONTEXT=512 ./run-chat.sh

# Reduce thread count
LLM_NUM_THREADS=2 ./run-chat.sh

# Use smaller model
# Download Qwen3-0.5B or other smaller variant

# Close other applications
# Reboot system
sudo reboot
```

## TTS (Text-to-Speech) Issues

### TTS Not Working

**Problem:** No audio generated from text.

**Solution:**
```bash
# Check Kokoro submodule
ls -la Kokoro-Vietnamese/

# Update submodule if empty
git submodule update --init --recursive

# Check Kokoro installation
cd Kokoro-Vietnamese
pip install -e .

# Test Kokoro directly
python -c "from kokoro_vietnamese import KPipeline; print('OK')"
```

### Poor Voice Quality

**Problem:** TTS audio sounds robotic or distorted.

**Solution:**
```bash
# Try different voice
DEFAULT_VOICE=mai_linh ./run-chat.sh

# Adjust speaking speed
# Edit SPEAKING_SPEED in config.py

# Check Kokoro model files
ls -la Kokoro-Vietnamese/models/

# Reinstall Kokoro
cd Kokoro-Vietnamese
pip uninstall kokoro-vietnamese
pip install -e .
```

### Slow TTS Synthesis

**Problem:** TTS takes too long to synthesize speech.

**Solution:**
```bash
# Increase TTS threads
TTS_INTRA_THREADS=4 ./run-chat.sh

# Check CPU usage
htop

# Reduce speaking speed (faster synthesis)
# Edit SPEAKING_SPEED in config.py (higher = faster)
```

## Performance Issues

### High CPU Usage

**Problem:** System becomes unresponsive due to high CPU usage.

**Solution:**
```bash
# Check which component is using CPU
htop

# Reduce thread counts
LLM_NUM_THREADS=2 TTS_INTRA_THREADS=2 STT_NUM_THREADS=1 ./run-chat.sh

# Reduce LLM context
LLM_CONTEXT=512 ./run-chat.sh

# Use smaller model
LLM_MODEL_PATH=/path/to/smaller/model.gguf ./run-chat.sh
```

### High Memory Usage

**Problem:** System runs out of memory.

**Solution:**
```bash
# Check memory usage
free -h

# Reduce LLM context
LLM_CONTEXT=512 ./run-chat.sh

# Reduce thread counts
LLM_NUM_THREADS=2 TTS_INTRA_THREADS=2 ./run-chat.sh

# Use smaller model
# Download Qwen3-0.5B variant

# Close other applications
# Add swap space if needed
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Slow Overall Response

**Problem:** Entire system feels slow.

**Solution:**
```bash
# Check system resources
htop
free -h

# Optimize thread allocation
LLM_NUM_THREADS=4 TTS_INTRA_THREADS=3 STT_NUM_THREADS=2 ./run-chat.sh

# Reduce LLM parameters
LLM_CONTEXT=512 LLM_MAX_TOKENS=32 ./run-chat.sh

# Check for thermal throttling
vcgencmd measure_temp

# Improve cooling if temperature > 80°C
```

## Hardware Issues

### Raspberry Pi Overheating

**Problem:** System throttles due to high temperature.

**Solution:**
```bash
# Check temperature
vcgencmd measure_temp

# Improve cooling
- Add heatsinks
- Install active cooling fan
- Ensure proper ventilation

# Reduce workload
LLM_NUM_THREADS=2 TTS_INTRA_THREADS=2 ./run-chat.sh

# Underclock if necessary
# Edit /boot/config.txt
# arm_freq=1800
```

### USB Device Issues

**Problem:** Microphone or audio device not recognized.

**Solution:**
```bash
# Check USB devices
lsusb

# Check kernel messages
dmesg | tail -20

# Try different USB port
# Avoid USB 3.0 ports for audio devices

# Increase USB power
# Edit /boot/config.txt
# max_usb_current=1
```

### SD Card Issues

**Problem:** Slow performance or corruption.

**Solution:**
```bash
# Check SD card health
sudo fsck /dev/mmcblk0p2

# Check disk usage
df -h

# Move models to external storage if SD card full
# Edit STT_MODEL_DIR and LLM_MODEL_PATH in config.py

# Use high-quality SD card
# Class 10 or higher recommended
```

## Network Issues

### Git Submodule Fails

**Problem:** Cannot clone or update submodules.

**Solution:**
```bash
# Check network
ping -c 3 github.com

# Try different git protocol
git config --global url."https://github.com/".insteadOf git://github.com/

# Use SSH if available
git submodule update --init --recursive

# Manual clone
git clone https://github.com/holeminhnhat3101/Kokoro-Vietnamese
```

### Model Download Fails

**Problem:** Cannot download models from Hugging Face.

**Solution:**
```bash
# Check network
ping -c 3 huggingface.co

# Use mirror if available
# Set HF_ENDPOINT environment variable

# Manual download
# Download from browser and place in models/ directory

# Use wget
wget https://huggingface.co/.../model.gguf -P models/
```

## System Issues

### Python Version Incompatibility

**Problem:** Errors due to Python version.

**Solution:**
```bash
# Check Python version
python --version

# Requires Python 3.10+
# Update if needed
sudo apt-get update
sudo apt-get install python3.11

# Create virtual environment with correct version
python3.11 -m venv .venv
source .venv/bin/activate
```

### Virtual Environment Issues

**Problem:** Dependencies not found in virtual environment.

**Solution:**
```bash
# Recreate virtual environment
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Permission Issues

**Problem:** Cannot write to directories or files.

**Solution:**
```bash
# Check file permissions
ls -la

# Fix permissions
chmod +x run-chat.sh
chmod -R 755 src/
chmod -R 755 tests/

# Fix ownership if needed
sudo chown -R $USER:$USER .
```

## Getting Help

If issues persist after trying these solutions:

1. **Enable Debug Logging**
   ```bash
   ./run-chat.sh --log-level DEBUG
   ```

2. **Collect System Information**
   ```bash
   # System info
   uname -a
   cat /etc/os-release
   
   # Hardware info
   vcgencmd version
   free -h
   df -h
   
   # Audio devices
   arecord -l
   aplay -l
   
   # Python info
   python --version
   pip list
   ```

3. **Check Logs**
   ```bash
   ls -la runtime/
   cat runtime/*.log
   ```

4. **Review Documentation**
   - `DEBUGGING.md` - Detailed debugging guide
   - `ARCHITECTURE.md` - System architecture
   - `README.md` - General usage

5. **Search Issues**
   - Check GitHub issues for similar problems
   - Search error messages online

6. **Create New Issue**
   When creating an issue, include:
   - Hardware details (Raspberry Pi model, RAM)
   - OS version (`cat /etc/os-release`)
   - Python version (`python --version`)
   - Full error messages
   - Debug logs
   - Steps to reproduce
   - What you've already tried