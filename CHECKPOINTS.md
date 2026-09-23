# Các stage triển khai GA-NAS

Chỉ đánh dấu `[x]` sau khi người dùng tự chạy, kiểm thử và xác nhận. Chỉ thực hiện một stage tại một thời điểm.

## [ ] Stage 1 — Baseline và model builder

- Giữ loader, pipeline huấn luyện và kết quả baseline hiện tại.
- Biểu diễn baseline bằng chromosome kiến trúc.
- Xây dựng CNN từ chromosome 2–4 block và đếm số tham số.
- Kiểm tra gene, kích thước tensor và đầu ra 7 lớp.
- Không đọc test split.

Xác nhận khi baseline vẫn chạy và model builder tạo output `(batch, 7)` cho kiến trúc hợp lệ.

## [ ] Stage 2 — Search space và NSGA-II độc lập

- Sinh, canonicalize, validate và repair chromosome.
- Triển khai dominance, nondominated sorting, crowding distance và tournament selection.
- Triển khai uniform crossover, mutation gene, thêm/xóa block và evaluation budget.
- Dùng fitness callback giả lập, độc lập CNN.

Xác nhận khi toán tử luôn sinh kiến trúc hợp lệ và NSGA-II dừng đúng budget thực tế.

## [ ] Stage 3 — Tích hợp NAS với CNN

- Train mỗi kiến trúc 5 epoch với training hyperparameters cố định.
- Fitness gồm validation macro AUC và số tham số.
- Cache theo canonical architecture và seed; cache hit không train lại.
- Log từng evaluation vào `results/ga_search/`.

Xác nhận khi pipeline `NSGA-II → CNN → Pareto objectives` chạy đúng và không đọc test split.

## [ ] Stage 4 — Random Architecture Search và thực nghiệm

- Thêm Random Architecture Search dùng cùng search space.
- Chạy GA-NAS và Random Search với cùng budget 80, 5 epoch và seed `1, 2, 3`.
- Log cấu hình, metrics, runtime, Pareto rank, crowding và best-so-far.

Xác nhận khi hai phương pháp dừng đúng budget và log đủ để tái lập.

## [ ] Stage 5 — Chọn kiến trúc và đánh giá cuối

- Chọn AUC cao nhất, model nhỏ nhất và knee point từ Pareto front mỗi phương pháp.
- Train lại tối đa 25 epoch với seed `1, 2, 3`, early stopping patience 5.
- Chỉ tại stage này mới đọc test split.
- Lưu kết quả vào `results/final_eval/`.

Xác nhận khi test không tham gia chọn kiến trúc và có đủ kết quả nhiều seed.

## [ ] Stage 6 — Phân tích nghiên cứu

- Tổng hợp mean ± standard deviation, loss, accuracy, macro F1, macro AUC, số tham số và runtime.
- Tạo confusion matrix, đường hội tụ, biểu đồ độ đa dạng và Pareto AUC–model size.
- So sánh baseline, GA-NAS và Random Architecture Search.

Xác nhận khi bảng và biểu đồ phản ánh đúng log thí nghiệm.

## [ ] Stage 7 — Hoàn thiện và bàn giao

- Đồng bộ dependency, tài liệu, đường dẫn, lệnh chạy và output.
- Rà soát tính tái lập và bảo đảm không có tính năng ngoài phạm vi.

Xác nhận khi toàn bộ quy trình có thể tái lập theo `GUIDE.md`.
