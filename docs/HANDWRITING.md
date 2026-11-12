# Nhận diện chữ viết tay Latin và chuyển sang Braille

## Phạm vi

Nhánh mở rộng nhận ảnh hoặc nét vẽ trực tiếp của **một từ hoặc một dòng chữ Latin viết tay**, dùng TrOCR Small để đọc thành văn bản. Người dùng kiểm tra/sửa văn bản rồi xác nhận tạo Braille. Ảnh nên được cắt sát dòng chữ tối trên nền sáng; giao diện có tùy chọn chọn vùng cắt.

Phần OCR dùng mô hình học sâu **đã huấn luyện sẵn** của Microsoft. Nhóm chưa tự huấn luyện hay fine-tune mô hình này. Nhánh nhận diện chấm Braille ban đầu vẫn dùng xử lý ảnh và luật hình học.

Từ phần mở rộng ngày 12/11/2026, **một ký tự** được đọc bằng CNN do dự án tự huấn luyện, có tùy chọn học bổ sung mẫu cá nhân. Chọn **Nội dung cần đọc → Một ký tự a–z** khi viết riêng chữ như `h`. Xem [tài liệu CNN ký tự](HANDWRITTEN_LETTERS.md) cho nguồn EMNIST, quy trình train/validation/test và số liệu thật. Các mô tả TrOCR bên dưới áp dụng cho chế độ **Một từ / dòng**.

Luồng xử lý:

```text
Ảnh viết tay → Chọn vùng một dòng → TrOCR → Văn bản OCR gốc
                                                ↓
                                      Người dùng kiểm tra/sửa
                                                ↓
                         Chữ Latin thường + dấu cách → Braille/TXT/PNG
```

OCR có thể trả chữ hoa, số hoặc dấu câu. Phần chuyển Braille hỗ trợ a–z, dấu cách và ba dấu cơ bản: phẩy `,` → `⠂` (chấm 2), apostrophe `'` → `⠄` (chấm 3), chấm `.` → `⠲` (chấm 2-5-6). Chữ hoa được đổi thành chữ thường, nhiều dấu cách được gộp thành một; chưa có dấu báo viết hoa. Số, dấu câu khác, xuống dòng và tiếng Việt có dấu được báo lỗi, không bị âm thầm xóa. Dấu cách giữa các từ được giữ bằng ô trống U+2800. Dấu `'` được hiểu là apostrophe; chưa phân tích dấu nháy mở/đóng theo ngữ cảnh, nên đây chưa phải bộ chuyển đổi UEB đầy đủ.

## Cài đặt lần đầu

Chạy từ thư mục dự án trong PowerShell, sau khi đã có `.venv` và thư viện cơ bản:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-handwriting.txt
.\.venv\Scripts\python.exe scripts\download_handwriting_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Cần Internet để cài thư viện và tải mô hình lần đầu. Gói PyTorch CPU trên môi trường Windows đã dùng có kích thước tải khoảng 619 MB; mô hình và tokenizer khoảng 247 MB. Chúng được cài/lưu cho dự án, không cần GPU hoặc dịch vụ OCR trả phí. Nếu mất mạng khi tải mô hình, chạy lại lệnh tải để tiếp tục.

Mô hình lưu tại `models/trocr-small-handwritten/`. Script cố định revision `b4648cfa171985a6745f37ddd637e98c0da958ac` và ghi `provenance.json`. Sau khi cài xong, OCR nạp tệp cục bộ với `local_files_only=True`; ảnh người dùng không được gửi đến dịch vụ suy luận bên ngoài.

Các thư viện OCR là tùy chọn: mở ứng dụng hoặc chạy hai tab Braille cũ không tự tải mô hình, không nạp Torch. Nếu thiếu tệp, tab chữ viết tay hiển thị hướng dẫn cài đặt.

## Cách sử dụng

1. Chọn tab **Chữ viết tay → Braille**.
   Chọn **Một từ / dòng** cho hướng dẫn TrOCR dưới đây; nếu chỉ viết một chữ, dùng **Một ký tự a–z**.
2. Chọn **Tải ảnh** để upload JPG/PNG, hoặc **Viết trực tiếp** để dùng bảng vẽ. Không dùng ảnh cả trang nhiều dòng.
3. Với ảnh tải lên, có thể bật **Cắt vùng chứa dòng chữ**. Với bảng vẽ, viết bằng chuột/bút cảm ứng, chỉnh độ dày nét, dùng **Hoàn tác nét cuối** hoặc **Xóa hết** khi cần. Giữ chữ trong một dòng và khoảng cách giữa các từ.
4. Nhấn **Nhận diện chữ viết tay**. Lần đầu mất thêm thời gian nạp mô hình; lần sau dùng lại mô hình đã nạp.
5. Xem văn bản OCR gốc và sửa ở ô bên dưới, đặc biệt với chữ khó đọc hoặc ký tự chưa hỗ trợ.
6. Nhấn **Xác nhận văn bản và tạo Braille** để xem kết quả, tải PNG hoặc TXT.

Nếu kết quả đọc sai, nhấn **Đọc kỹ / xem gợi ý**. Ứng dụng thử beam search trên ảnh gốc và hai biến thể cắt lề/làm dày nét nhẹ, rồi hiển thị các cách đọc khác nhau. Chọn ở **Các cách đọc gợi ý**, nhấn **Dùng cách đọc này**, kiểm tra lại ô văn bản rồi xác nhận tạo Braille. Nếu không có cách đúng, sửa trực tiếp trong ô văn bản. Các gợi ý chưa phải đáp án chắc chắn hoặc xác suất chính xác; đây không phải thao tác huấn luyện mô hình.

Đổi ảnh hoặc vùng cắt sẽ xóa kết quả OCR cũ. Sửa văn bản sẽ xóa kết quả Braille cũ để tránh tải nhầm dữ liệu. Có thể dùng ảnh mẫu `anh_test/chu_viet_tay/iam_demo.jpeg` nếu tệp này đã được tải trong phiên triển khai.

### Bảng viết trực tiếp

Bảng vẽ có kích thước ảnh 1000×260 pixel, nền trắng và nét đen, co theo chiều rộng màn hình. Thả chuột/kết thúc nét để cập nhật ảnh; ứng dụng không tự chạy mô hình sau mỗi nét. Nhấn **Nhận diện chữ viết tay** khi viết xong. Có nút **Tải ảnh nét vẽ PNG** để lưu riêng đầu vào.

Khi đổi nét, hoàn tác, xóa bảng hoặc đổi cách nhập, kết quả OCR/Braille cũ được vô hiệu hóa. Bảng trắng không được gửi tới mô hình. Khoảng trắng ngoài vùng chữ được cắt trước khi OCR, nhưng khoảng cách giữa các từ vẫn được giữ.

Nếu nét chạm sát viền, giao diện cảnh báo có nguy cơ bị cắt. Hãy viết cách mép; thêm lề ảnh sau đó không phục hồi phần nét đã mất. Mô hình học từ IAM tiếng Anh có thể đọc sai tên riêng hoặc tên tiếng Việt không dấu; chưa có cơ sở khẳng định nhận diện đúng mọi từ Latin.

Nét vẽ được giữ trong phiên hiện tại khi ứng dụng cập nhật. Khi chuyển chế độ rồi quay lại, ảnh bảng vẽ được khôi phục; các nét trước lúc khôi phục được coi là một nhóm để hoàn tác (nút **Hoàn tác phần đã khôi phục**). Tải PNG nếu cần lưu lâu dài; tải lại toàn bộ trang hoặc đóng phiên có thể làm mất nét vẽ chưa lưu.

Component được lưu cùng mã nguồn trong `components/handwriting_canvas/index.html`, dùng HTML Canvas và giao tiếp hai chiều với Streamlit. Không cần thêm npm, CDN hoặc thư viện vẽ mới. Python kiểm tra định dạng, kích thước và dung lượng PNG trước khi chuyển vào luồng OCR cục bộ.

PNG có dấu cách được tạo đúng vị trí ô trống. Nhánh nhận diện ảnh Braille cũ chưa giải mã khoảng trắng, nên không dùng việc đọc lại PNG nhiều từ bằng nhánh cũ làm bằng chứng rằng nó hỗ trợ khoảng trắng.

## API

```python
from handwriting import recognize_handwriting
from latin_line import encode_latin_line
from braille_renderer import render_braille_line

result = recognize_handwriting("anh_viet_tay.png")
print(result.text)  # Giữ nguyên kết quả OCR để kiểm tra trước.

# Văn bản đã được người dùng kiểm tra:
confirmed = "hello world"
braille = encode_latin_line(confirmed)
image, centers = render_braille_line(confirmed)
```

`recognize_handwriting` nhận đường dẫn hoặc NumPy image 8 bit theo BGR/BGRA/grayscale. Ảnh alpha được ghép lên nền trắng. Ảnh rỗng, ảnh đồng màu, vùng quá nhỏ hoặc ảnh trên 20 triệu pixel được báo lỗi trước khi nạp mô hình. Chuỗi dùng chuyển Braille tối đa 200 ký tự, gồm dấu cách.

`processing_time_ms` đo phần xử lý OCR sau khi đã nạp mô hình, không bao gồm thời gian tải/nạp weights. Cờ `truncated` cảnh báo khi kết quả có thể chạm giới hạn 128 token mới. Không xuất điểm tin cậy dạng phần trăm chưa được hiệu chuẩn.

## Kết quả đã chạy và giới hạn

Toàn bộ **138 test vượt qua** sau tích hợp bảng vẽ, gồm kiểm tra ảnh PNG/metadata từ component, khôi phục ảnh, cắt lề giữ khoảng cách từ, bảng trắng, và giao diện với OCR giả lập: không tự OCR, vô hiệu hóa kết quả khi đổi nét/xóa/chuyển chế độ. Các test cũ về Braille, dấu câu và upload tiếp tục vượt qua. Đây là kiểm thử chức năng, không phải accuracy của mô hình.

Đã thử bằng trình duyệt thật: vẽ ba nét thành chữ `A`, mô hình TrOCR trả `A`, sau xác nhận ứng dụng tạo `a`/`⠁`. Hoàn tác và xóa bảng được quan sát hoạt động; xóa bảng xóa kết quả cũ. Đây là smoke test tương tác, chưa đánh giá được nhiều kiểu chữ viết bằng chuột. Ảnh chụp giao diện thử nghiệm nằm trong `outputs/drawing_canvas_review/browser-proof.png`.

Ngày 11/11/2026 đã chạy mô hình thật trên ảnh IAM crop một từ từ bộ fixture của Hugging Face: kết quả đọc được **`industry`**, chuyển thành **`⠊⠝⠙⠥⠎⠞⠗⠽`**. Thời gian suy luận ghi nhận khoảng **716 ms**, không gồm nạp mô hình. Dữ liệu thực tế của lần chạy được lưu trong `outputs/handwriting_smoke_2026-11-11/result.json`.

Đây là **smoke test một ảnh**, không phải độ chính xác trên tập test hay kết quả nhận diện mọi chữ viết tay. Mô hình được công bố cho ảnh một dòng, nhưng phiên triển khai chưa kiểm chứng trên bộ ảnh dòng chữ của nhóm. URL ảnh dòng ví dụ của IAM không tải được trong lần thử này. Cần thu thập ảnh viết tay thực tế có nhãn, gồm nhiều người viết, rồi tính CER/word error rate trên tập chưa dùng điều chỉnh trước khi báo cáo accuracy.

Các hạn chế: chữ viết khó đọc, nét dính, ảnh mờ/bóng, xoay nhiều, ảnh cả trang, tiếng Việt có dấu và các miền ảnh khác dữ liệu IAM có thể gây sai. Người dùng vẫn cần duyệt văn bản trước khi chuyển Braille. Kết quả OCR tốt không chứng minh nhánh nhận diện chấm Braille tốt và ngược lại.

## Ca phát triển được người dùng xác nhận: manh

Người dùng xác nhận ảnh bảng vẽ là `manh`, trong khi OCR ban đầu trả chuỗi khác. Đã thử trên phần nét vẽ cắt từ screenshot (không phải PNG gốc), không huấn luyện lại weights hoặc ép mô hình sinh từ đó. Chế độ đọc kỹ mới trả `manh` trong danh sách gợi ý, cùng các phương án sai khác; chưa bảo đảm tự chọn đúng phương án đầu tiên. Người dùng chọn hoặc sửa rồi mới tạo Braille. Ca này là dữ liệu phát triển đã được quan sát, không phải tập test độc lập.

Các phép thử và kết quả thô nằm trong `outputs/ocr_manh_review/`. Mẫu IAM đối chứng cũ vẫn đọc được `industry`. Cần nhiều từ/người viết khác để đánh giá mức cải thiện chung.

## Nguồn mô hình và ảnh mẫu

- [ICEB Rules of Unified English Braille, mục 7](https://iceb.org/wp-content/uploads/2025/10/Rules-of-Unified-English-Braille-2024.pdf): nguồn ánh xạ ba dấu câu cơ bản.

- [Microsoft TrOCR Small Handwritten — model card](https://huggingface.co/microsoft/trocr-small-handwritten): mô hình đã fine-tune trên IAM, phục vụ OCR ảnh một dòng.
- [Tài liệu TrOCR của Transformers 4.57.1](https://huggingface.co/docs/transformers/v4.57.1/model_doc/trocr): API `TrOCRProcessor` và `VisionEncoderDecoderModel`.
- [TrOCR paper](https://arxiv.org/abs/2109.10282): Li và cộng sự.
- [Hugging Face OCR fixtures](https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr): nguồn ảnh IAM mẫu; nguồn và revision cụ thể nằm trong `anh_test/chu_viet_tay/source.json`.
- [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database): nguồn dữ liệu được nhà phát hành mô hình sử dụng. Một ảnh fixture không có nghĩa nhóm đã tải hay huấn luyện trên toàn bộ IAM.
