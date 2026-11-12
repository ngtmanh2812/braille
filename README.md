# Nhận diện ký tự Braille — Grade 1 Latin

Dự án bài tập lớn môn **Nhập môn Trí tuệ nhân tạo**, nhận diện **một dòng chữ Braille** bằng xử lý ảnh cổ điển và nhận dạng mẫu dựa trên luật. Các bước được tổ chức đơn giản để sinh viên năm nhất có thể đọc, giải thích và thực nghiệm.

Phạm vi MVP là ảnh **chấm đen trên nền trắng**, nhận diện 26 chữ cái Latin thường từ `a` đến `z` theo bảng Braille Grade 1 không viết tắt. Dự án sử dụng Python, OpenCV, NumPy, Matplotlib, Pandas, Streamlit và pytest.

Ảnh tổng hợp phục vụ phát triển và thực nghiệm có kiểm soát. **Kết quả trên ảnh tổng hợp không đại diện cho độ chính xác trên giấy Braille dập nổi thật.** Nhận diện giấy dập nổi là hướng mở rộng cần dữ liệu và quy trình đánh giá riêng.

**Phần mở rộng chữ viết tay:** có thêm nhánh OCR một từ hoặc một dòng Latin bằng mô hình TrOCR Small đã huấn luyện sẵn, sau đó người dùng kiểm tra văn bản và chuyển sang Braille. Nhánh này dùng học sâu, cần thư viện và mô hình cài riêng; nhóm chưa tự huấn luyện mô hình TrOCR. Xem [hướng dẫn chữ viết tay](docs/HANDWRITING.md).

**Nhận diện từng ký tự:** dự án đã tự huấn luyện một CNN riêng trên EMNIST Letters, có thêm bản học bổ sung từ mẫu chữ được người dùng xác nhận. Chọn **Một ký tự a–z** khi viết riêng chữ như `h`; mô hình này không dùng để đọc cả từ. Xem [huấn luyện ký tự viết tay](docs/HANDWRITTEN_LETTERS.md) để biết dữ liệu, kết quả thật và cách thêm mẫu cá nhân.

## Chức năng

- Chuyển ảnh sang mức xám, làm mờ Gaussian, phân ngưỡng Global/Otsu/Adaptive và tùy chọn phép toán hình thái học.
  - Phát hiện chấm bằng thành phần liên thông; lọc theo diện tích, kích thước, tỷ lệ cạnh và độ tròn.
- Phân nhóm chấm thành ô Braille 2×3 bằng lưới cố định hoặc lưới ước lượng, hỗ trợ độ nghiêng nhẹ.
- Giữ đủ sáu vị trí của mỗi ô, kể cả vị trí không có chấm; mã hóa thành chuỗi 6 bit.
- Giải mã chữ cái `a`–`z`, tạo chuỗi Unicode Braille và trả `?` cho mẫu chưa hỗ trợ.
- Chuyển văn bản chữ thường `a`–`z` thành Unicode Braille và ảnh chấm, tải xuống dạng TXT hoặc PNG.
- Đọc một từ/một dòng viết tay bằng TrOCR, cho sửa văn bản trước khi tạo Braille; giữ ô trống giữa các từ ở nhánh này.
- Viết bằng chuột/bút cảm ứng ngay trên bảng: chỉnh nét, hoàn tác, xóa, tải PNG và nhận diện bằng mô hình hiện có.
- Chọn CNN nhận diện một chữ a–z hoặc TrOCR đọc từ/dòng; lưu mẫu chữ đã sửa có nhãn để huấn luyện bổ sung cục bộ.
- Hiển thị ảnh gốc, ảnh nhị phân, ảnh đánh dấu chấm, văn bản dự đoán và thời gian xử lý.
- Sinh ảnh tổng hợp có nhãn; tính CER, độ chính xác theo ô, precision/recall của chấm; xuất CSV và biểu đồ.

## Cấu trúc dự án

```text
braille project/
├── preprocessing.py          # Tiền xử lý và phân ngưỡng ảnh
├── dot_detection.py          # Phát hiện chấm, tính tâm và vẽ ảnh đánh dấu
├── grid_alignment.py         # Căn lưới, phân chấm vào các ô 2×3
├── decoder.py                # Bảng chữ cái Braille và mã hóa Unicode
├── braille_renderer.py       # Vẽ ảnh Braille từ chữ Latin, trả tọa độ chấm
├── pipeline.py               # Điều phối toàn bộ quá trình nhận diện
├── main.py                   # Chạy nhận diện bằng dòng lệnh
├── app.py                    # Giao diện Streamlit
├── handwriting.py            # OCR một dòng bằng TrOCR, nạp mô hình khi cần
├── handwriting_ui.py         # Upload/cắt ảnh, duyệt OCR và tải Braille
├── handwritten_letters.py    # CNN ký tự, tiền xử lý và nạp weights đã huấn luyện
├── letter_samples.py         # Lưu mẫu có nhãn để học bổ sung
├── drawing_canvas.py         # Nhận/kiểm tra ảnh từ bảng viết trực tiếp
├── components/handwriting_canvas/index.html # Canvas cục bộ, không cần CDN
├── latin_line.py             # Chuẩn hóa văn bản OCR đã được kiểm tra
├── evaluation.py             # Tính chỉ số, so sánh cấu hình, xuất kết quả
├── scripts/
│   ├── generate_samples.py   # Sinh ảnh tổng hợp và nhãn
│   ├── download_handwriting_model.py # Tải mô hình OCR từ/dòng
│   ├── download_emnist_letters.py # Tải dữ liệu ký tự có kiểm tra hash
│   ├── train_handwriting_letters.py # Huấn luyện CNN ký tự từ đầu
│   └── adapt_handwriting_letters.py # Học thêm mẫu đã xác nhận
├── requirements-handwriting.txt # Thư viện OCR cài riêng
├── docs/HANDWRITING.md        # Nguồn mô hình, cài đặt và giới hạn chữ viết tay
├── docs/HANDWRITTEN_LETTERS.md # Huấn luyện, đánh giá và dùng CNN ký tự
├── tests/                    # Kiểm thử các module và pipeline
├── requirements.txt          # Danh sách thư viện cần cài
├── README.md                 # Hướng dẫn sử dụng và mô tả thuật toán
└── .gitignore                # Bỏ qua môi trường ảo, cache và dữ liệu sinh ra
```

Các thư mục `data/synthetic/` và `outputs/evaluation/` sẽ được tạo khi chạy lệnh sinh dữ liệu và đánh giá.

## Cài đặt trên Windows

Cần có Python và pip. Các lệnh dưới đây dùng Python 3.11; nếu dùng phiên bản khác đã cài trên máy, thay `py -3.11` bằng lệnh Python tương ứng.

Mở PowerShell trong thư mục dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường ảo, có thể gọi trực tiếp Python trong môi trường đó:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Các lệnh `python` ở phần tiếp theo được chạy sau khi kích hoạt môi trường ảo, hoặc thay bằng `.\.venv\Scripts\python.exe`.

## Chạy ứng dụng

### Giao diện web

```powershell
python -m streamlit run app.py
```

Tải lên ảnh JPG/PNG chứa một dòng Braille, chọn phương pháp phân ngưỡng và tùy chọn làm mờ Gaussian. Với Global Thresholding, có thể điều chỉnh giá trị ngưỡng trực tiếp.

Ứng dụng hiển thị ảnh gốc, ảnh nhị phân, ảnh đánh dấu và văn bản dự đoán. Nút tải TXT lưu văn bản dưới dạng UTF-8. Ảnh hợp lệ nhưng không phát hiện chấm sẽ trả dự đoán rỗng; tệp không đọc được sẽ có thông báo lỗi.

Chọn tab **Chữ → Braille** để chuyển theo chiều ngược lại:

1. Nhập chuỗi chữ thường, ví dụ `abc`, `hello` hoặc `braille` (tối đa 100 ký tự).
2. Xem Unicode Braille và ảnh chấm được tạo tự động. Ví dụ `abc` → `⠁⠃⠉`.
3. Chọn **Tải ảnh PNG** hoặc **Tải Braille TXT**. Có thể upload PNG vừa tải ở tab **Ảnh Braille → Chữ** để kiểm tra nhận diện; chất lượng nhận diện vẫn phụ thuộc bước căn lưới.

Chức năng chuyển chữ hiện chỉ nhận `a`–`z`, chưa hỗ trợ dấu cách, chữ hoa, chữ số, dấu câu hoặc tiếng Việt có dấu. Ký tự ngoài phạm vi sẽ được thông báo lỗi. Ảnh tạo ra là ảnh tổng hợp, không phải ảnh giấy dập nổi.

Tab **Chữ viết tay → Braille** nhận JPG/PNG một từ hoặc một dòng Latin viết tay. Nhấn nhận diện, sửa văn bản nếu cần rồi xác nhận tạo Braille. Riêng nhánh này nhận chữ hoa bằng cách đổi thành chữ thường, giữ dấu cách giữa từ và hỗ trợ dấu phẩy, chấm, apostrophe (` , . ' `); chưa mã hóa dấu báo viết hoa, số, dấu câu khác hay tiếng Việt có dấu. Cần cài thêm theo [docs/HANDWRITING.md](docs/HANDWRITING.md), hai tab cũ không yêu cầu cài OCR.

Để viết bằng chuột, chọn **Viết trực tiếp** tại mục **Cách nhập chữ viết tay** trong tab trên. Viết một dòng trên bảng, chỉnh **Độ dày nét**, **Hoàn tác nét cuối** hoặc **Xóa hết**. Sau khi thả chuột ở nét cuối, nhấn **Nhận diện chữ viết tay**, kiểm tra văn bản rồi xác nhận tạo Braille. Có thể tải PNG của nét vẽ. Tính năng bảng vẽ không cần cài thư viện mới; OCR vẫn dùng mô hình đã cài. Nét vẽ chỉ được giữ theo phiên, nên tải PNG nếu muốn lưu lâu dài.

Nếu chỉ viết một chữ, chọn **Nội dung cần đọc → Một ký tự a–z** trước khi nhận diện. CNN EMNIST ban đầu đạt 92,06% trên 20.800 ảnh test; bản học bổ sung đạt 92,67%. Mẫu `h` người dùng cung cấp hiện được đọc thành `h`, nhưng mẫu này đã tham gia học bổ sung, nên không tính là kết quả trên ảnh mới. Trong phần **Bổ sung mẫu chữ của bạn để huấn luyện**, lưu ảnh sau khi kiểm tra nhãn rồi chạy script học bổ sung theo tài liệu; nhấn lưu không tự cập nhật model.

### Dòng lệnh

Thay `duong_dan\braille.png` bằng đường dẫn ảnh cần nhận diện:

```powershell
python main.py duong_dan\braille.png --threshold otsu --save-debug outputs\annotated.png
```

| Tùy chọn | Ý nghĩa |
| --- | --- |
| `--threshold otsu` | Chọn `global`, `otsu` hoặc `adaptive` |
| `--threshold-value 127` | Giá trị ngưỡng cho phương pháp Global |
| `--no-denoise` | Tắt làm mờ Gaussian |
| `--save-debug outputs\annotated.png` | Lưu ảnh đánh dấu chấm và thông tin ô |

### Sử dụng trong Python

```python
from pipeline import run_pipeline

result = run_pipeline("duong_dan/braille.png")
print(result.text)
print(result.braille)
print(result.processing_time_ms)
```

Đầu vào có thể là đường dẫn ảnh hoặc ảnh NumPy. Ảnh màu NumPy dùng thứ tự kênh BGR/BGRA của OpenCV; ảnh mức xám cũng được chấp nhận.

Kết quả gồm `text`, `braille`, `patterns`, danh sách chấm `dots`, các ô `cells`, ảnh kiểm tra `debug_images` và thời gian `processing_time_ms`.

Dùng `PipelineConfig`, `PreprocessConfig`, `DotDetectionConfig` và `GridConfig` để cấu hình từng bước. Lưới cố định phù hợp với ảnh chuẩn có thông số hiệu chuẩn:

```python
from grid_alignment import GridConfig
from pipeline import PipelineConfig, run_pipeline

config = PipelineConfig(grid=GridConfig(
    mode="fixed",
    dot_spacing=18,
    cell_stride=54,
    origin_x=24,
    origin_y=24,
))
result = run_pipeline("duong_dan/braille.png", config)
```

Các giá trị trên phù hợp với hình học mặc định của bộ sinh ảnh. `dot_spacing` là khoảng cách giữa hai tâm chấm liền kề trong ô, `cell_stride` là khoảng cách ngang giữa hai gốc ô liên tiếp, còn `origin_x` và `origin_y` xác định vị trí chấm số 1 của ô đầu tiên, tính bằng pixel. Chế độ mặc định `estimated` tự ước lượng lưới.

## Mô tả thuật toán

```text
Ảnh đầu vào → Tiền xử lý → Phát hiện chấm → Căn lưới 2×3 → Giải mã → Văn bản
```

1. **Tiền xử lý:** chuyển ảnh về mức xám 8 bit, tùy chọn làm mờ Gaussian rồi phân ngưỡng. Mặt nạ nhị phân được đảo để chấm tối trở thành tiền cảnh trắng. Có thể dùng opening/closing để làm sạch mặt nạ.
2. **Phát hiện chấm:** gán nhãn các thành phần liên thông, loại các thành phần không đạt giới hạn hình học. Tâm thành phần được dùng làm tọa độ chấm. Độ tròn được tính từ diện tích và chu vi đường biên.
3. **Căn lưới:** ước lượng khoảng cách chấm và độ nghiêng nhỏ, đưa tọa độ về ba hàng rồi tìm bố cục hai cột lặp theo từng ô. Mỗi chấm được gán vào vị trí gần nhất trong sáu vị trí; chấm vắng được giữ là `None` và mã hóa bằng `0`.
4. **Giải mã:** tra mẫu 6 bit trong bảng 26 chữ cái Latin Grade 1. Mẫu không hỗ trợ trả `?`. Mỗi ô đồng thời được mã hóa thành ký tự Unicode tại `U+2800 + mặt nạ bit`.

Thứ tự chấm trong ô và chuỗi bit:

```text
Cột trái   Cột phải
    1          4
    2          5
    3          6

Thứ tự bit: 1, 2, 3, 4, 5, 6
Ví dụ: a = 100000; b = 110000; c = 100100
```

Phần giải mã chỉ tra mẫu hình học; chưa có quy tắc ngôn ngữ hoặc sửa lỗi theo ngữ cảnh.

## Sinh dữ liệu và thực nghiệm

```powershell
python scripts\generate_samples.py --output data\synthetic --seed 42 --variations 4
python evaluation.py data\synthetic\labels.csv --output outputs\evaluation --split test
```

Bộ sinh mặc định tạo 20 chuỗi gốc × 4 phiên bản, gồm ảnh chuẩn và các biến thể về độ sáng, nhiễu, góc xoay nhẹ. Tệp `labels.csv` ghi văn bản đúng, nhóm nguồn, cấu hình biến đổi và tọa độ tâm chấm đúng.

Việc chia development/test được thực hiện theo nhóm nguồn: toàn bộ phiên bản của cùng một chuỗi gốc nằm trong cùng tập, tránh để các ảnh biến đổi từ một nguồn xuất hiện ở cả hai tập. Dùng development để điều chỉnh tham số và giữ test để đánh giá cuối cùng.

Lệnh đánh giá được chuẩn bị cho hai so sánh trên cùng tập ảnh:

- Global Thresholding với Adaptive Thresholding, cùng bật Gaussian.
- Otsu không làm mờ với Otsu có làm mờ Gaussian.

Kết quả được xuất thành `evaluation_rows.csv` cho từng ảnh, `evaluation_summary.csv` cho các chỉ số tổng hợp và `comparison_cer.png` cho biểu đồ CER.

| Chỉ số | Cách tính |
| --- | --- |
| CER | Khoảng cách Levenshtein chia số ký tự của văn bản đúng; càng thấp càng tốt |
| Cell accuracy | Tỷ lệ ô có mẫu 6 bit đúng tại cùng vị trí; ô thiếu hoặc thừa tính là sai |
| Dot precision | Số chấm ghép đúng chia tổng số chấm phát hiện |
| Dot recall | Số chấm ghép đúng chia tổng số chấm trong nhãn |
| Thời gian xử lý | Thời gian chạy pipeline cho từng ảnh, đơn vị mili giây |

Chấm dự đoán và chấm trong nhãn được ghép một-một theo khoảng cách tọa độ, với sai số mặc định tối đa 5 pixel. CER tổng hợp dùng tổng số phép sửa chia tổng số ký tự đúng trong tập dữ liệu.

Ngày 11/11/2026 đã chạy bộ so sánh trên 64 ảnh development và 16 ảnh test hiện có: bốn cấu hình đều có CER bằng 0 trên bộ ảnh này. Tuy nhiên, các ca chẩn đoán mới phát hiện lỗi ở lưới tự ước lượng cho ảnh chữ `k`, `o` riêng lẻ và ảnh đổi kích thước. Xem `outputs/review_2026-11-11/REVIEW.md` và CSV đi kèm. Những số liệu này phụ thuộc dữ liệu, seed và cấu hình; không suy rộng sang ảnh giấy dập nổi hay chữ viết tay.

## Kiểm thử

```powershell
python -m pytest -q
```

Bộ test bao phủ bảng 26 chữ cái, Unicode Braille, mẫu không hỗ trợ, phát hiện tâm chấm, lọc nhiễu nhỏ, thứ tự chấm, vị trí chấm vắng, lưới nhiều ô, độ nghiêng nhẹ, pipeline và các trường hợp ảnh rỗng hoặc đầu vào lỗi. Các hàm CER và khoảng cách Levenshtein cũng có test.

Lần kiểm tra ngày 12/11/2026: **171 test vượt qua**, gồm hồi quy các luồng cũ, CNN ký tự, lưu mẫu có nhãn và chuyển chế độ. Test UI chủ yếu dùng mô hình giả để kiểm tra luồng; một phép thử riêng dùng CNN đã học thật trong giao diện nhận mẫu `h` rồi tạo `⠓`. Mẫu này đã tham gia huấn luyện bổ sung, không phải ảnh test độc lập. Accuracy trên 20.800 ảnh EMNIST test được ghi riêng trong `outputs/handwriting_letters_personal/metrics.json`. Xem `docs/HANDWRITTEN_LETTERS.md`; số lượng pytest không thay thế benchmark nhận diện.

## Giới hạn hiện tại

- Chỉ xử lý một dòng chấm tròn tối trên nền sáng; chỉ bù được độ nghiêng nhẹ.
- Khi cả hàng chấm trên cùng hoặc các ô đầu dòng đều vắng, hình học có thể không đủ thông tin để suy ra vị trí. Có thể dùng lưới cố định khi biết bố cục.
- Bộ lọc kích thước chấm và các giả định khoảng cách cần điều chỉnh với ảnh có độ phân giải khác nhiều so với ảnh mẫu.
- Nhánh nhận diện ảnh Braille chưa giải mã khoảng trắng, chữ số, dấu báo viết hoa, dấu câu, chữ viết tắt hoặc tiếng Việt có dấu. Nhánh OCR mới hỗ trợ tạo ô trống giữa các từ sau khi người dùng kiểm tra văn bản.
- Chưa đánh giá trên ảnh giấy Braille dập nổi, ảnh chụp thực tế hoặc dữ liệu lớp học. OCR chữ viết tay mới có smoke test một ảnh IAM crop từ, chưa có benchmark trên bộ ảnh dòng chữ thực tế của nhóm.

