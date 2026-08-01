# Talking Sheep

Talking Sheep là một trợ lý giọng nói tiếng Việt chạy offline trên Raspberry Pi 5, có thể nghe qua microphone, nhận dạng giọng nói cục bộ, tạo câu trả lời bằng LLM local và phát lại bằng giọng nói tiếng Việt.

## Tính năng

- Nhận dạng giọng nói tiếng Việt theo thời gian thực bằng Zipformer và `sherpa-onnx`
- Tạo câu trả lời local bằng Qwen3 1.7B qua `llama-cpp-python`
- Tổng hợp giọng nói tiếng Việt bằng Kokoro Vietnamese
- Thu âm theo định dạng native của microphone với VAD, pre-roll, chọn kênh và resampling tự động
- Chèn tiếng cừu tùy chọn giữa các câu nói
- Hoạt động hoàn toàn local sau khi dependencies và model đã được cài đặt

## Yêu cầu

- Raspberry Pi 5, khuyến nghị 8 GB RAM
- Raspberry Pi OS 64-bit
- Python 3.10 trở lên
- Microphone tương thích PortAudio
- Loa hoặc thiết bị audio output
- Khuyến nghị sử dụng tản nhiệt chủ động

## Cài đặt

Clone repository cùng submodule Kokoro-Vietnamese:

```bash
git clone --recurse-submodules https://github.com/holeminhnhat3101/Talking-Sheep.git
cd Talking-Sheep
```

Nếu đã clone repository mà chưa tải submodule:

```bash
git submodule update --init --recursive
```

Cho phép chạy script khởi động:

```bash
chmod +x run-chat.sh
```

Cài dependencies, tải model và khởi động ứng dụng:

```bash
./run-chat.sh
```

Lần chạy đầu có thể mất vài phút vì script sẽ cài package hệ thống, tạo `.venv`, cài Python dependencies và tải các model cần thiết.

## Sử dụng

Khởi động Talking Sheep:

```bash
./run-chat.sh
```

Talking Sheep tự động phát hiện và ưu tiên thiết bị Seeed Studio ReSpeaker Mic Array. Khi sử dụng firmware 6 kênh, hệ thống tự động chọn kênh 0 (processed audio) ở tần số 16 kHz để có chất lượng nhận dạng tốt nhất.

Liệt kê microphone được phát hiện:

```bash
./run-chat.sh --list-mics
```

Chọn microphone:

```bash
./run-chat.sh --input-device 1
```

Dùng hiệu chuẩn VAD tự động:

```bash
./run-chat.sh --silence-threshold auto
```

Bật debug logging:

```bash
./run-chat.sh --log-level DEBUG
```

Hiển thị toàn bộ tùy chọn:

```bash
./run-chat.sh --help
```

Nhấn `Ctrl+C` để dừng ứng dụng.

## Cách hoạt động

```text
Microphone
→ AudioRecorder
→ VAD và pre-roll
→ Audio mono float32 16 kHz
→ Zipformer streaming STT
→ Qwen3 local LLM (streaming)
→ Tách câu theo thời gian thực
→ Kokoro Vietnamese TTS (theo câu)
→ Tiếng cừu tùy chọn giữa các câu
→ AudioPlayer (phát theo dòng)
→ Loa
```

### Pipeline Streaming

Talking Sheep sử dụng pipeline streaming để giảm độ trễ phản hồi:

1. **LLM Streaming**: LLM tạo phản hồi từng phần, lọc bỏ thinking tags và code blocks
2. **Sentence Assembly**: Các câu được tách tại ranh giới . ! ? nhưng giữ nguyên số thập phân (ví dụ: 3.5)
3. **Per-Sentence TTS**: Mỗi câu được tổng hợp giọng nói ngay khi hoàn tất
4. **Immediate Playback**: Câu đầu tiên được phát ngay khi tổng hợp xong, trong khi các câu sau vẫn đang được xử lý
5. **Interstitial Bleats**: Tiếng cừu ngẫu nhiên (theo `BLEAT_PROBABILITY`) được chèn giữa các câu, không trước câu đầu tiên hay sau câu cuối cùng

### Tradeoff One-Producer

- **TTS blocks LLM**: TTS synthesis tạm dừng LLM generation (single producer thread)
- **Playback không blocks**: Phát audio không chặn producer thread
- **Lợi ích**: Độ trễ thấp hơn, phản hồi nhanh hơn cho câu đầu tiên
- **Hạn chế**: Không hỗ trợ barge-in (ngắt lời khi đang phát)

### Microphone Selection

- **One device**: Tự động chọn nếu chỉ có một microphone vật lý
- **Multiple devices**: Hiển thị menu tương tác với phím mũi tên trong SSH terminal
- **Escape**: Cho phép hủy chọn microphone trước khi khởi tạo model
- **Non-interactive**: Chế độ service không chờ input (tự động chọn hoặc fallback)
- **--list-mics**: Liệt kê microphone mà không load STT/LLM/TTS

### Recording & Playback

- Thu âm và phát audio diễn ra tuần tự
- Microphone không ghi âm trong lúc trợ lý đang nói
- Recording chỉ resumes sau khi playback hoàn tất
- Streaming path không tạo file `runtime/final.wav` (khác với batch pipeline cũ)
- AudioPlayer giữ persistent PyAudio stream, không reopen giữa các segment

STT recognizer, LLM và TTS engine được load một lần và tái sử dụng qua nhiều lượt hội thoại.

## Model

### Speech-to-Text

```text
hynt/Zipformer-30M-RNNT-Streaming-6000h
```

Runtime:

```text
sherpa-onnx
```

### Language Model

```text
Qwen3-1.7B-Q4_K_M.gguf
```

Runtime:

```text
llama-cpp-python
```

### Text-to-Speech

```text
Kokoro Vietnamese
```

Giọng mặc định:

```text
mai_linh
```

Kokoro-Vietnamese được giữ dưới dạng Git submodule để các thay đổi riêng của Talking Sheep có thể được quản lý trong một repository độc lập.

## Cấu hình

Cấu hình mặc định nằm trong:

```text
src/config.py
```

Một số biến môi trường thường dùng:

```bash
LLM_MODEL_PATH=/path/to/model.gguf
STT_MODEL_DIR=/path/to/zipformer
STT_NUM_THREADS=2
AUDIO_INPUT_DEVICE=1
AUDIO_OUTPUT_DEVICE=0
SILENCE_THRESHOLD=250
BLEAT_PROBABILITY=1.0  # Xác suất chèn tiếng cừu giữa các câu (0.0-1.0)
```

Ví dụ:

```bash
STT_NUM_THREADS=4 BLEAT_PROBABILITY=0 ./run-chat.sh
```

## Cấu trúc dự án

```text
Talking-Sheep/
├── src/
│   ├── audio_player.py          # Phát audio với persistent stream
│   ├── audio_recorder.py        # Thu âm với VAD, pre-roll, resampling
│   ├── chat_llm.py              # Local LLM integration (Qwen3)
│   ├── config.py                # Cấu hình trung tâm với environment variables
│   ├── env_setup.py             # Thiết lập thread environment
│   ├── microphone_menu.py      # Menu tương tác chọn microphone
│   ├── streaming_response.py   # Pipeline streaming với queue management
│   ├── talking_sheep_voice.py   # Entry point chính
│   ├── vietnamese_stt.py       # Zipformer STT với sherpa-onnx
│   └── voice_layer.py          # TTS, sentence assembly, audio processing
├── tests/                       # Test suite
│   ├── test_audio_player.py
│   ├── test_audio_recorder_devices.py
│   ├── test_chat_llm.py
│   ├── test_chat_llm_streaming.py
│   ├── test_microphone_menu.py
│   ├── test_smoke.py
│   ├── test_streaming_response.py
│   ├── test_talking_sheep_voice.py
│   ├── test_voice_layer.py
│   └── test_voice_layer_streaming.py
├── Kokoro-Vietnamese/          # Git submodule - Vietnamese TTS
├── assets/
│   └── bleats/                 # Tiếng cừu effect files
├── models/                     # Model storage (STT, LLM)
├── runtime/                    # Runtime files (recordings, logs)
├── docs/                       # Tài liệu chi tiết
├── requirements.txt            # Python dependencies
├── requirements-dev.txt        # Development dependencies
├── requirements-rpi.txt        # Raspberry Pi specific dependencies
├── run-chat.sh                 # Script khởi động chính
├── verify_pipeline.py          # Pipeline verification script
├── THIRD_PARTY_LICENSES.md     # Third-party licenses
└── README.md                   # This file
```

## Xử lý lỗi

Kiểm tra microphone:

```bash
./run-chat.sh --list-mics
arecord -l
```

Kiểm tra thiết bị phát:

```bash
aplay -l
```

Nếu VAD không phát hiện giọng nói:

```bash
./run-chat.sh --silence-threshold auto
```

Nếu tiếng ồn nền làm ứng dụng tự ghi âm, tăng threshold:

```bash
./run-chat.sh --silence-threshold 400
```

Nếu cài dependency hoặc tải model thất bại, chạy lại:

```bash
./run-chat.sh
```

Nếu thư mục `Kokoro-Vietnamese` trống sau khi clone:

```bash
git submodule update --init --recursive
```

## Kiến trúc & Tài liệu

### Pipeline Streaming
Talking Sheep sử dụng kiến trúc streaming với 3 worker threads chính:

1. **LLM Worker**: Tạo phản hồi từng phần từ LLM, lọc thinking tags và code blocks
2. **TTS Worker**: Tổng hợp giọng nói cho từng câu hoàn chỉnh
3. **Playback Worker**: Phát audio segments theo thứ tự

Hệ thống sử dụng 2 queue để truyền dữ liệu:
- `SentenceQueue`: Chuyển sentences từ LLM sang TTS
- `AudioQueue`: Chuyển audio segments từ TTS sang playback

### Files Chính
- `streaming_response.py`: Pipeline streaming với queue management và metrics
- `microphone_menu.py`: Menu tương tác chọn microphone với arrow keys
- `env_setup.py`: Thiết lập thread environment trước khi import thư viện nặng
- `voice_layer.py`: Sentence assembly, TTS synthesis, audio normalization

### Debugging Files
Các file debugging Markdown có sẵn trong repository:
- `DEBUGGING.md`: Hướng dẫn debug chi tiết
- `ARCHITECTURE.md`: Tài liệu kiến trúc hệ thống
- `TROUBLESHOOTING.md`: Xử lý lỗi thường gặp
- `TESTING.md`: Hướng dẫn chạy test suite

### Tài liệu Chi Tiết
Các tài liệu chi tiết về kiến trúc, cấu hình, triển khai và xử lý lỗi:
- `implementation-plan.md`: Kế hoạch triển khai chi tiết
- `TODO.md`: Danh sách tính năng cần làm
- `DEBUGGING.md`: Hướng dẫn debug chi tiết (sẽ được tạo)
- `ARCHITECTURE.md`: Tài liệu kiến trúc hệ thống (sẽ được tạo)
- `TROUBLESHOOTING.md`: Xử lý lỗi thường gặp (sẽ được tạo)

## Tính năng Đã Hoãn

Các tính năng sau chưa được triển khai trong phiên bản hiện tại:

- **Barge-in**: Không hỗ trợ ngắt lời khi trợ lý đang phát
- **Echo cancellation**: Không hỗ trợ hoạt động đồng thời microphone và loa
- **Concurrent LLM/TTS**: Chỉ có một producer thread (TTS blocks LLM iteration)
- **Forced sentence flushing**: Không ép buộc tách câu theo giới hạn token hoặc dấu phẩy
- **Dynamic Bluetooth switching**: Không hỗ trợ chuyển đổi sink Bluetooth trong khi chạy

## Giấy phép

Mã nguồn gốc của Talking Sheep được phát hành theo Apache License 2.0.

Các thư viện, model weights và voicepack bên thứ ba vẫn tuân theo giấy phép riêng của từng thành phần. Apache License 2.0 của dự án không thay thế hoặc ghi đè các giấy phép đó.

Xem:

- `LICENSE`
- `THIRD_PARTY_LICENSES.md`
