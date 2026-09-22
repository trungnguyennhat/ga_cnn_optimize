# Quy tắc làm việc dành cho AI agent

## Bắt buộc trước khi code

1. Đọc toàn bộ `README.md`, `AGENTS.md`, `CHECKPOINTS.md` và `GUIDE.md`.
2. Xác định stage hiện tại trong `CHECKPOINTS.md`.
3. Chỉ code khi người dùng yêu cầu tiếp tục stage đó.

## Quy tắc dữ liệu

- Dataset duy nhất là `data/dermamnist_64.npz`.
- Khi train/search, chỉ đọc file local và không gọi API hoặc tải dữ liệu.
- Không tự tải lại nếu file đã tồn tại.
- Giữ nguyên train/validation/test split chính thức; không tự chia lại.
- Không truy cập test split trước Stage 5.

## Quy trình thực hiện

- Chỉ thực hiện **một stage tại một thời điểm**.
- Chỉ sửa những file cần thiết cho stage hiện tại.
- Không triển khai trước stage sau hoặc thêm dependency/abstraction chưa cần thiết.
- Sau mỗi stage, cập nhật `GUIDE.md` bằng lệnh PowerShell khớp code thực tế.
- Sau khi viết code, dừng và liệt kê ngắn gọn các file đã thay đổi.
- Để người dùng tự chạy và tự kiểm thử; agent **không chạy code, test, benchmark hoặc thí nghiệm**.
- Không đánh dấu stage hoàn thành trước khi người dùng xác nhận.
- Chỉ chuyển stage khi người dùng yêu cầu rõ ràng.

## Khi người dùng báo lỗi

- Chỉ chẩn đoán và sửa lỗi thuộc stage hiện tại.
- Sau khi sửa, dừng để người dùng kiểm thử lại.
- Không triển khai thêm stage khác trong lúc sửa lỗi.

## Mẫu phản hồi khi kết thúc stage

```text
Đã viết xong Stage N: <tên stage>.
Các file đã thay đổi: ...
Đã cập nhật GUIDE.md với lệnh chạy và kiểm thử thủ công.
Chưa chạy hoặc test theo quy ước dự án.
Bạn hãy kiểm thử; khi đạt yêu cầu, hãy xác nhận để tôi chuyển sang Stage N+1.
```
