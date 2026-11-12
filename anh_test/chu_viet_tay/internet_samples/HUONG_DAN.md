# 8 ảnh dòng chữ viết tay tải từ Internet

Nguồn: [Teklia/IAM-line trên Hugging Face](https://huggingface.co/datasets/Teklia/IAM-line), các hàng 0–7 của split test. Đây là ảnh chữ viết tay tiếng Anh thật thuộc IAM, không phải ảnh tạo bằng font. Bản Teklia đã chuẩn hóa chiều cao ảnh về 128 pixel.

Cách dùng: chọn tab **Chữ viết tay → Braille**, upload một JPG trong thư mục, nhấn nhận diện rồi đối chiếu văn bản với `labels.csv`. Có thể thử `iam_test_0006.jpg` trước.

Một số dòng có chữ hoa, số hoặc dấu câu. OCR giữ nguyên kết quả; nhánh chuyển đổi hỗ trợ chữ Latin, dấu cách và các dấu phẩy, chấm, apostrophe (` , . ' `). Số và dấu câu khác vẫn cần sửa trước khi chuyển. Không coi việc sửa thủ công là kết quả OCR ban đầu.

`labels.csv` lưu nguyên văn nhãn từ nguồn; `source.json` ghi truy vấn tải ảnh. Các ảnh này dùng thử chức năng. Do mô hình TrOCR đã fine-tune trên IAM, không dùng vài ảnh này để tuyên bố độ chính xác trên chữ viết tay độc lập hoặc trên ảnh tự chụp.
