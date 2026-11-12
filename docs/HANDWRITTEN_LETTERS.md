# Huấn luyện nhận diện một chữ cái viết tay

## Chọn đúng bài toán

Ứng dụng có hai chế độ trong tab **Chữ viết tay → Braille**:

| Chế độ | Mô hình | Đầu vào |
| --- | --- | --- |
| Một ký tự a–z | CNN nhỏ do dự án huấn luyện | Một chữ riêng lẻ, ví dụ `h` |
| Một từ / dòng | TrOCR Small đã huấn luyện sẵn | Một từ hoặc một dòng, ví dụ `hello world` |

CNN ký tự không tách chữ trong từ viết nối nét. Đưa cả từ `manh` vào chế độ một ký tự sẽ không đọc được bốn chữ. Mô hình ký tự gộp dạng viết hoa/viết thường về cùng một chữ Latin thường; chưa tạo dấu báo viết hoa Braille.

## Cách dùng mô hình đã huấn luyện

Khởi động lại ứng dụng để nạp phiên bản mã/mô hình mới:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Chọn **Chữ viết tay → Braille**.
2. Tại **Nội dung cần đọc**, chọn **Một ký tự a–z**.
3. Chọn **Viết trực tiếp**, viết một chữ cách các mép bảng, hoặc tải ảnh chỉ chứa chữ đó.
4. Nhấn **Nhận diện chữ viết tay**. Xem kết quả và ba chữ gợi ý; đây không phải xác suất nhận diện đúng.
5. Kiểm tra/sửa chữ rồi xác nhận chuyển sang Braille. Ví dụ `h` → `⠓`.

Ứng dụng ưu tiên mô hình học bổ sung đã hoàn tất ở `models/handwriting_letters_personal/`. Mô hình EMNIST ban đầu được giữ riêng tại `models/handwriting_letters/`. Nếu bản học bổ sung không được chọn tốt hơn bằng validation, ứng dụng tiếp tục dùng baseline.

## Dữ liệu và thuật toán

Dùng [EMNIST Letters của NIST](https://www.nist.gov/itl/products-and-services/emnist-dataset): 26 lớp, gộp dạng viết hoa và viết thường. Bốn tệp IDX được tải riêng từ [mirror Royc30ne trên Hugging Face](https://huggingface.co/datasets/Royc30ne/emnist-letters), cố định revision và kiểm tra SHA256. `data/handwriting/emnist_letters/provenance.json` ghi rõ đây là mirror bên thứ ba, nguồn gốc, URL và hash; không gán việc tải này thành tải trực tiếp từ NIST.

- Official train: 124.800 ảnh; tách theo lớp, seed 42 thành **112.320 train + 12.480 validation**.
- Official test: **20.800 ảnh**, dùng sau khi chọn checkpoint bằng validation.
- Sửa hướng ảnh IDX bằng chuyển vị một lần. Mọi ảnh sau đó dùng cùng tiền xử lý với ảnh người dùng.
- Tiền xử lý: mức xám → Otsu → giữ mọi thành phần chấm/nét (kể cả chấm của `i`, `j`) → cắt vùng chữ → giữ tỷ lệ, đưa cạnh dài về 20 pixel → đặt giữa ảnh 28×28. Đầu vào CNN có nét trắng trên nền đen.
- CNN: Conv 1→16, ReLU, MaxPool; Conv 16→32, ReLU, MaxPool; Linear 1568→128, ReLU, Dropout 0,2; Linear 128→26.
- Adam, learning rate 0,001, batch 256, 8 epoch. Chỉ ảnh train được xoay ngẫu nhiên ±10° và dịch vị trí nhẹ. Validation/test không được augmentation.

Mỗi ảnh test là một ký tự thật của bộ EMNIST. Việc chia validation từ official train không có nhãn người viết kèm IDX, nên không tuyên bố validation tách người viết. Độ chính xác EMNIST không đại diện cho mọi kiểu chữ viết chuột, tên riêng hay ảnh điện thoại.

## Kết quả thực tế

| Chỉ số | CNN EMNIST ban đầu | CNN học bổ sung |
| --- | --- | --- |
| Validation accuracy, 12.480 ảnh | 92,47% | 92,92% |
| Official test accuracy, 20.800 ảnh | 92,06% | 92,67% |
| Official test macro-F1 | 0,9205 | 0,9265 |
| Official test top-3 accuracy | 99,00% | 99,03% |
| Mẫu `h` do người dùng xác nhận | Đọc thành `q` trước khi học mẫu | Đọc thành `h` sau khi học mẫu |

**Mẫu `h` trong dòng cuối đã tham gia học bổ sung, nên đó là kiểm tra ảnh đã học, không phải bằng chứng tổng quát hóa.** Số liệu official test là một phép đánh giá riêng trên EMNIST, không trộn ảnh `h` đó vào test. Chưa có tập chữ viết mới của người dùng đủ lớn để đo accuracy riêng theo kiểu chữ cá nhân.

Mẫu được cắt từ screenshot người dùng gửi, không phải ảnh PNG gốc của canvas. Nhãn `h` do người dùng xác nhận. Không có quy tắc ép mọi ảnh thành `h`; chữ dự đoán là lớp có điểm lớn nhất trong 26 lớp CNN.

## Huấn luyện lại từ đầu

Nếu đã cài nhánh TrOCR thì Torch đã có sẵn. Nếu chưa, cài PyTorch CPU trong `.venv` (không cần GPU):

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe scripts\download_emnist_letters.py
.\.venv\Scripts\python.exe scripts\train_handwriting_letters.py --batch-size 256 --epochs 8
```

Thời gian phụ thuộc CPU. Script lưu `model.pt` dạng state_dict, metadata, hash weights, lịch sử epoch và split indices. Checkpoint được chọn bằng accuracy validation rồi loss khi hòa; test chỉ được đánh giá sau lựa chọn. Không chọn tham số bằng kết quả test.

## Bổ sung kiểu chữ của bạn

1. Viết hoặc tải **một** ký tự và chọn chế độ ký tự.
2. Sửa ô văn bản về đúng một chữ, ví dụ `h`.
3. Mở **Bổ sung mẫu chữ của bạn để huấn luyện**, kiểm tra nhãn rồi nhấn **Lưu mẫu chữ đã sửa**.
4. Thu thập nhiều hình dạng cho mỗi chữ. Tốt hơn là có ảnh mới và người viết khác để kiểm tra riêng; không lưu ảnh test EMNIST vào dữ liệu cá nhân dùng học.
5. Dừng ứng dụng, chạy học bổ sung rồi khởi động lại:

```powershell
.\.venv\Scripts\python.exe scripts\adapt_handwriting_letters.py --epochs 3 --batch-size 256
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Nhấn lưu chỉ tạo ảnh và một dòng `path,label` trong `data/handwriting/user_samples/train_manifest.csv`; không tự sửa weights hoặc bắt đầu học trong trình duyệt. Ảnh trùng được kiểm tra, nhãn xung đột được báo lỗi thay vì ghi đè. Dữ liệu lưu cục bộ.

Script học bổ sung bắt đầu từ baseline và kết hợp lại dữ liệu EMNIST train với các mẫu đã xác nhận. Mặc định số lần lấy mỗi mẫu cá nhân được chọn tự động, tối đa 480 lần/mẫu và không quá 4.000 lượt cá nhân tổng cộng. Các lượt lặp/augmentation vẫn thuộc training, không được tính là các ảnh test độc lập.

Lần triển khai đã dùng **một ảnh `h`, 480 lượt lấy mẫu**, cùng 112.320 ảnh EMNIST train trong 3 epoch bổ sung, learning rate 0,0002 và seed 43. Validation giữ nguyên; baseline cũng là ứng viên checkpoint epoch 0 để giảm nguy cơ học bổ sung làm giảm hiệu quả chung. Bản được chọn ở epoch 3. Official test được chạy sau lựa chọn; không dùng nhãn test để huấn luyện.

## Tệp để báo cáo môn AI

- `outputs/handwriting_letters_training/metrics.json`, `history.csv`, `per_class.csv`, `confusion.csv`, `split_indices.npz`: kết quả baseline.
- `outputs/handwriting_letters_training/learning_curve.png`: đường học train/validation. Train có augmentation/Dropout nên không so trực tiếp như cùng điều kiện với validation.
- `outputs/handwriting_letters_personal/metrics.json`, `history.csv`, `per_class.csv`, `confusion.png`: kết quả học bổ sung.
- `outputs/handwriting_letters_personal/enrolled_h_check.json`: chạy CNN thật trên mẫu đã học.
- `outputs/handwriting_letters_personal/ui_h_check.json`: giao diện dùng CNN thật, nhận `h` và tạo `⠓`.
- `models/handwriting_letters*/metadata.json`: cấu hình, nguồn dữ liệu và hash của model đã lưu.

Các test dùng CNN giả trên UI chỉ xác minh luồng chương trình. Accuracy 92,67% được lấy từ đánh giá mô hình thật trên official test, không suy ra từ số pytest vượt qua. Dù đã học thêm mẫu, vẫn cần người dùng kiểm tra ký tự trước khi tạo Braille.

Kiểm tra ngày 12/11/2026: **171 pytest vượt qua**; đã chạy riêng AppTest dùng CNN thật với mẫu `h` đã học và xác nhận đầu ra `h`/`⠓`. Ứng dụng ba tab cũng khởi động thành công. Đây là bằng chứng chức năng cùng với các số liệu đánh giá riêng ở trên, không phải cam kết đúng mọi cách viết.
