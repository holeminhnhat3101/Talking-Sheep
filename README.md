# Talking Sheep
 
Một trợ lý giọng nói tiếng Việt chạy offline trên Raspberry Pi 5, có thể nghe người dùng nói, chuyển giọng nói thành văn bản, tạo câu trả lời bằng LLM local, đọc câu trả lời bằng TTS tiếng Việt và phát âm thanh qua loa.
 
## Tính năng
 
- Ghi âm từ microphone với VAD (Voice Activity Detection)
- Thu âm theo sample rate native của thiết bị
- Chuyển đổi audio sang 16 kHz mono cho STT
- Nhận dạng giọng nói tiếng Việt bằng PhoWhisper
- Suy luận bằng LLM local (Qwen3-1.7B-Q4_K_M.gguf)
- Loại bỏ khối `````` và code blocks khỏi output LLM trước khi lưu history
- Chia câu dựa trên dấu câu . ! ?
- Tổng hợp giọng nói bằng Kokoro Vietnamese
- Chèn tiếng cừu tùy chọn từ `assets/bleats`
- Xuất file `runtime/final.wav`
- Phát audio đồng bộ qua loa
- Lưu lịch sử hội thoại ngắn (tối đa 4 cặp)
 
## Kiến trúc
 
```
Microphone
→ AudioRecorder (VAD, native rate → 16 kHz mono)
→ Vietnamese STT (PhoWhisper)
→ Local LLM (Qwen3-1.7B-Q4_K_M.gguf)
→ Làm sạch output (loại bỏ `````` và code blocks)
→ Chia câu
→ Kokoro Vietnamese TTS
→ Chèn tiếng cừu tùy chọn
→ runtime/final.wav
→ AudioPlayer
```
 
**Giải thích các bước:**
- **AudioRecorder**: Ghi âm từ microphone với VAD, tự động điều chỉnh theo sample rate native của thiết bị, chuyển đổi sang 16 kHz mono cho STT
- **Vietnamese STT**: Sử dụng PhoWhisper model để chuyển giọng nói tiếng Việt thành văn bản
- **Local LLM**: Suy luận bằng Qwen3-1.7B-Q4_K_M.gguf qua llama-cpp-python, trả lời bằng tiếng Việt
- **Làm sạch output**: Loại bỏ các khối `````` và code blocks khỏi câu trả lời trước khi lưu vào history và gửi sang TTS
- **Chia câu**: Tách câu dựa trên dấu câu . ! ? để tổng hợp từng câu riêng biệt
- **Kokoro Vietnamese TTS**: Tổng hợp giọng nói tiếng Việt với voice mặc định là `mai_linh`
- **Chèn tiếng cừu**: Chèn ngẫu nhiên file WAV từ `assets/bleats` sau câu đầu tiên (nếu có ít nhất 2 câu)
- **AudioPlayer**: Phát file `runtime/final.wav` đồng bộ qua loa
 
## Yêu cầu phần cứng
 
- Raspberry Pi 5
- RAM: Tối thiểu 8Gb
- Microphone USB hoặc tích hợp
- Loa hoặc thiết bị audio output
- Dung lượng lưu trữ: ~5 GB cho model và môi trường Python
 
## Yêu cầu phần mềm
 
- Hệ điều hành: Raspberry Pi OS (64-bit)
- Python: 3.8+
- Các package hệ thống: python3-venv, python3-dev, build-essential, cmake, pkg-config, portaudio19-dev, libasound2-dev, ffmpeg, libsndfile1, alsa-utils
- Dependencies Python: llama-cpp-python, huggingface-hub, numpy, soundfile, pyaudio, pydub, onnxruntime, requests, PyYAML, transformers, torch
- Model runtime: Qwen3-1.7B-Q4_K_M.gguf (tự động tải từ HuggingFace nếu chưa có)
- STT model: PhoWhisper (tiny hoặc base)
- TTS engine: Kokoro Vietnamese (ONNX runtime)
 
## Cài đặt
 
```bash
git clone <repository-url>
cd Talking-Sheep
chmod +x run-chat.sh
./run-chat.sh
```
 
Script `run-chat.sh` sẽ tự động:
1. Kiểm tra hệ điều hành 64-bit
2. Cài đặt các package hệ thống cần thiết
3. Tạo virtual environment `.venv`
4. Cài đặt Python dependencies từ `requirements-rpi.txt`
5. Cài đặt Kokoro Vietnamese với ONNX support
6. Tải LLM model Qwen3-1.7B-Q4_K_M.gguf vào thư mục `models/`
7. Khởi động ứng dụng
 
## Cấu hình
 
Các biến môi trường tùy chọn(điều chỉnh prompt và các tham số khác trong `src/config.py`):
 
- `LLM_MODEL_PATH`: Đường dẫn đến file GGUF (mặc định: `models/Qwen3-1.7B-Q4_K_M.gguf`)
- `LLM_AUTO_DOWNLOAD`: Tự động tải model (mặc định: `1`)
- `LLM_CONTEXT`: Context window size (mặc định: `2048`)
- `LLM_THREADS`: Số thread LLM (mặc định: số CPU cores)
- `AUDIO_INPUT_DEVICE`: Index thiết bị microphone
- `AUDIO_OUTPUT_DEVICE`: Index thiết bị loa
- `SILENCE_THRESHOLD`: Ngưỡng âm thanh cho VAD (mặc định: `250`)
- `BLEAT_PROBABILITY`: Xác suất chèn tiếng cừu (mặc định: `1.0`)
 
## Sử dụng
 
Sau khi chạy script, ứng dụng sẽ:
1. Hiển thị thông báo "🐑 Talking Sheep sẵn sàng! Nhấn Ctrl+C để thoát."
2. Lắng nghe từ microphone
3. Khi phát hiện giọng nói, ghi âm và chuyển thành văn bản
4. Gửi văn bản đến LLM để tạo câu trả lời
5. Tổng hợp câu trả lời thành giọng nói tiếng Việt
6. Phát câu trả lời qua loa
7. Lặp lại quy trình cho đến khi nhấn Ctrl+C
 
## Giới hạn
 
- Chỉ hoạt động trên Raspberry Pi 5 với Raspberry Pi OS 64-bit
- STT chỉ hỗ trợ tiếng Việt với model PhoWhisper (tiny hoặc base)
- LLM context giới hạn 2048 tokens
- Lịch sử hội thoại chỉ lưu tối đa 4 cặp user-assistant
- TTS chỉ hỗ trợ voice tiếng Việt có sẵn trong Kokoro Vietnamese
- Không hỗ trợ streaming audio (ghi âm và phát diễn ra tuần tự)
 
## Cấu trúc thư mục
 
```
Talking-Sheep/
├── src/
│   ├── audio_recorder.py    # Ghi âm với VAD
│   ├── vietnamese_stt.py     # PhoWhisper STT
│   ├── chat_llm.py           # LLM wrapper
│   ├── voice_layer.py        # TTS và chèn tiếng cừu
│   ├── audio_player.py       # Phát audio
│   ├── config.py             # Cấu hình
│   └── talking_sheep_voice.py # Entry point
├── Kokoro-Vietnamese/        # TTS engine (vendored)
├── assets/
│   └── bleats/               # File WAV tiếng cừu
├── models/                   # LLM model GGUF
├── runtime/                  # File audio tạm thời
├── requirements-rpi.txt      # Dependencies cho Raspberry Pi
└── run-chat.sh               # Script khởi động