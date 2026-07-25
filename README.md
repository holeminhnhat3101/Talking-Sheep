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
→ Qwen3 local LLM
→ Kokoro Vietnamese TTS
→ Tiếng cừu tùy chọn
→ AudioPlayer
→ Loa
```

STT recognizer, LLM và TTS engine được load một lần và tái sử dụng qua nhiều lượt hội thoại.

Thu âm và phát audio diễn ra tuần tự. Microphone không ghi âm trong lúc trợ lý đang nói.

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
BLEAT_PROBABILITY=1.0
```

Ví dụ:

```bash
STT_NUM_THREADS=4 BLEAT_PROBABILITY=0 ./run-chat.sh
```

## Cấu trúc dự án

```text
Talking-Sheep/
├── src/
│   ├── audio_player.py
│   ├── audio_recorder.py
│   ├── chat_llm.py
│   ├── config.py
│   ├── talking_sheep_voice.py
│   ├── vietnamese_stt.py
│   └── voice_layer.py
├── Kokoro-Vietnamese/        # Git submodule
├── assets/
│   └── bleats/
├── models/
├── runtime/
├── docs/
├── requirements.txt
├── requirements-rpi.txt
├── run-chat.sh
├── THIRD_PARTY_LICENSES.md
└── README.md
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

## Tài liệu

Các tài liệu chi tiết về kiến trúc, cấu hình, triển khai và xử lý lỗi nên được đặt trong thư mục `docs/` thay vì mở rộng README này.

```text
docs/
├── architecture.md
├── configuration.md
├── deployment.md
└── troubleshooting.md
```

## Giấy phép

Mã nguồn gốc của Talking Sheep được phát hành theo Apache License 2.0.

Các thư viện, model weights và voicepack bên thứ ba vẫn tuân theo giấy phép riêng của từng thành phần. Apache License 2.0 của dự án không thay thế hoặc ghi đè các giấy phép đó.

Xem:

- `LICENSE`
- `THIRD_PARTY_LICENSES.md`
