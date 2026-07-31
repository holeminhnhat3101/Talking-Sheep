# Talking Sheep - Danh sách Cải tiến

## Cải thiện Trải nghiệm Người dùng

### Ưu tiên Cao

- [x] **Tạo menu chọn microphone tương tác**
  - Thêm điều hướng bằng phím mũi tên để chọn microphone ✅
  - Hiển thị các microphone có sẵn với thông tin thiết bị ✅
  - Loại bỏ nhu cầu tham số dòng lệnh `--input-device` ✅
  - Tự động chọn nếu chỉ có một microphone ✅
  - Cho phép người dùng hủy bỏ lựa chọn ✅

- [x] **Triển khai streaming từng câu từ LLM đến Kokoro**
  - Stream đầu ra LLM khi nó tạo ra (không chờ phản hồi hoàn chỉnh) ✅
  - Gửi các câu đã hoàn thành đến Kokoro ngay lập tức ✅
  - Bắt đầu tổng hợp ngay khi câu đầu tiên có sẵn ✅
  - Giảm tổng thời gian phản hồi ✅

- [x] **Triển khai phát lại ngay câu đầu tiên đã tổng hợp**
  - Phát âm thanh câu đầu tiên ngay khi tổng hợp hoàn tất ✅
  - Tiếp tục tổng hợp các câu còn lại trong nền ✅
  - Stream phát lại trong khi tổng hợp tiếp tục ✅
  - Loại bỏ việc chờ tất cả các câu hoàn tất ✅

### Ưu tiên Trung bình

- [ ] **Sửa hỗ trợ đầu ra âm thanh Bluetooth**
  - PyAudio/PortAudio không tự động định tuyến đến thiết bị Bluetooth
  - Thêm cờ `--list-output-devices` để hiển thị thiết bị đầu ra âm thanh có sẵn
  - Thêm lựa chọn thiết bị đầu ra tương tác tương tự menu microphone
  - Cho phép người dùng chọn thiết bị Bluetooth theo tên thay vì chỉ số thiết bị
  - Tích hợp tốt hơn với định tuyến âm thanh hệ thống (PulseAudio/PipeWire trên Linux)

- [ ] **Thêm menu chọn thiết bị đầu ra âm thanh tương tác**
  - Tương tự menu chọn microphone
  - Cho phép chọn loa/headphone tương tác
  - Loại bỏ nhu cầu biến môi trường `AUDIO_OUTPUT_DEVICE`

- [ ] **Thêm menu chọn giọng nói tương tác**
  - Hiển thị các giọng Kokoro có sẵn
  - Cho phép chuyển đổi giọng lúc chạy
  - Xem trước giọng trước khi chọn

- [ ] **Tạo hỗ trợ tệp cấu hình**
  - Thêm `config.yaml` hoặc `config.json` cho cài đặt liên tục
  - Giảm sự phụ thuộc vào biến môi trường
  - Dễ dàng hơn cho người dùng không kỹ thuật

- [ ] **Thêm trình hướng dẫn thiết lập lần đầu**
  - Hướng dẫn người dùng qua cấu hình ban đầu
  - Kiểm tra microphone và loa
  - Tải xuống mô hình với chỉ báo tiến trình
  - Lưu tùy chọn vào tệp cấu hình

- [ ] **Thêm chỉ báo trạng thái thời gian chạy**
  - Chỉ báo hình ảnh khi ghi âm (🎤)
  - Chỉ báo hình ảnh khi xử lý (⏳)
  - Chỉ báo hình ảnh khi nói (🔊)
  - Phản hồi rõ ràng về trạng thái hệ thống

- [ ] **Thêm hiển thị token streaming như ChatGPT**
  - Hiển thị token LLM khi chúng tạo ra theo thời gian thực
  - Hiển thị tốc độ tạo token (token/giây)
  - Đầu ra streaming hình ảnh trong khi xử lý LLM
  - Duy trì triển khai streaming hiện tại để tách câu

- [ ] **Thêm điều khiển âm lượng cho đầu ra TTS**
  - Âm lượng có thể điều chỉnh qua lệnh hoặc cấu hình
  - Cài đặt âm lượng cho mỗi giọng
  - Ngăn chặn clipping/méo âm

- [ ] **Cải thiện thông báo lỗi và khôi phục**
  - Giải thích lỗi thân thiện với người dùng
  - Gợi ý sửa lỗi cho các vấn đề phổ biến
  - Tự động thử lại cho lỗi tạm thời
  - Các bước khắc phục sự cố rõ ràng

- [ ] **Thêm hệ thống trợ giúp trong thời gian chạy**
  - Hiển thị các lệnh có sẵn trong thời gian chạy
  - Hiển thị cấu hình hiện tại
  - Cung cấp mẹo sử dụng

- [ ] **Thêm giao diện quản lý mô hình**
  - Liệt kê các mô hình có sẵn/đã cài đặt
  - Tải xuống mô hình dễ dàng với tiến trình
  - Thông tin phiên bản mô hình
  - Hiển thị sử dụng dung lượng

- [ ] **Thêm giám sát hiệu suất**
  - Hiển thị độ trễ phản hồi
  - Hiển thị sử dụng tài nguyên (CPU/bộ nhớ)
  - Xác định các nút thắt
  - Gợi ý tối ưu hóa

### Ưu tiên Trung bình

- [x] **Kiểm tra cải tiến menu và streaming tương tác**
  - Xác minh lựa chọn microphone hoạt động chính xác ✅
  - Kiểm tra cải tiến hiệu suất streaming ✅
  - Đo lường giảm độ trễ ✅
  - Đảm bảo chất lượng âm thanh được duy trì ✅

## Đã Hoàn thành Gần đây (Triển khai Mới nhất)

### Pipeline Streaming
- ✅ Triển khai streaming các chunk LLM với `_StreamingResponseFilter`
- ✅ Thêm `StreamingSentenceAssembler` để phát hiện ranh giới câu theo thời gian thực
- ✅ Bảo toàn số thập phân (ví dụ: "3.5") trong tách câu
- ✅ Thiết kế thread một producer: LLM → lắp ráp câu → TTS → hàng đợi → phát lại
- ✅ Tiếng cừu xen kẽ với điều khiển `BLEAT_PROBABILITY`
- ✅ Tái sử dụng stream trình phát âm thanh liên tục
- ✅ Hàng đợi an toàn thread với dọn dẹp và xử lý timeout thích hợp
- ✅ Đóng generator và lan truyền ngoại lệ

### Lựa chọn Microphone
- ✅ Menu phím mũi tên tương tác cho terminal SSH
- ✅ Tự động chọn khi có một thiết bị
- ✅ Hỗ trợ hủy bằng phím Escape
- ✅ Fallback không tương tác cho chế độ dịch vụ
- ✅ Điểm số thiết bị với ưu tiên ReSpeaker
- ✅ Lọc thiết bị ảo (default, pulse, pipewire, v.v.)

### Kiểm tra & Tài liệu
- ✅ Phạm vi kiểm tra toàn diện cho các thành phần streaming
- ✅ README cập nhật với tài liệu pipeline streaming
- ✅ Loại bỏ các nhập kiểm tra lỗi thời (choose_bleat, compose_with_bleat)
- ✅ Bộ kiểm tra sẵn sàng cho xác thực runtime Raspberry Pi/CI

## Vấn đề Đã Biết & Cải tiến Cần thiết

### Phát hiện Code Review
- [ ] Thêm giới hạn kích thước hàng đợi để ngăn chặn tăng trưởng bộ nhớ không giới hạn trong streaming
- [ ] Cải thiện xử lý thập phân cho số phức tạp (ví dụ: "3.14159")
- [ ] Thêm xác thực xác suất cho `build_inter_sentence_segment`
- [ ] Cập nhật docstring `create_spoken_response()` để làm rõ trạng thái cũ
- [ ] Thêm kiểm tra hủy phản hồi streaming
- [ ] Thêm kiểm tra trường hợp cạnh cho trình lắp ráp câu

## Phân tích Trạng thái Hiện tại

### Lựa chọn Microphone
- ✅ Hiện tại liệt kê microphone với `--list-mics`
- ✅ Có thể chọn qua cờ `--input-device`
- ✅ Menu tương tác với điều hướng phím mũi tên cho terminal SSH
- ✅ Tự động chọn nếu chỉ có một microphone
- ✅ Phím Escape cho phép hủy trước khi khởi tạo mô hình
- ✅ Chế độ không tương tác cho khởi động dịch vụ

### Streaming LLM đến Kokoro
- ✅ LLM stream các chunk đầu ra khi chúng tạo ra
- ✅ Các câu được lắp ráp và gửi đến TTS ngay lập tức
- ✅ Tổng hợp từng câu streaming đã triển khai
- ✅ Thẻ thinking và khối mã được lọc khỏi đầu vào TTS

### Thời điểm Phát lại
- ✅ Câu đầu tiên phát ngay khi tổng hợp hoàn tất
- ✅ Tổng hợp nền tiếp tục trong khi phát lại
- ✅ Tiếng cừu xen kẽ (được điều khiển bởi BLEAT_PROBABILITY) giữa các câu
- ✅ Không có tiếng cừu đầu/cuối
- ✅ Stream PyAudio liên tục được tái sử dụng giữa các đoạn

### Đầu ra Âm thanh
- ✅ Hỗ trợ loa/headphone qua PyAudio
- ✅ Biến môi trường `AUDIO_OUTPUT_DEVICE`
- ✅ Thiết bị không dây hoạt động qua lớp âm thanh hệ điều hành
- ✅ Hỗ trợ đầu ra mặc định hệ thống Bluetooth
