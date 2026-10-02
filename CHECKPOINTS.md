# Các stage triển khai GA-NAS

Chỉ đánh dấu `[x]` sau khi người dùng tự chạy, kiểm thử và xác nhận. Chỉ thực hiện một stage tại một thời điểm.

## [x] Stage 1 — Baseline và model builder

- Giữ loader, pipeline huấn luyện và kết quả baseline hiện tại.
- Biểu diễn baseline bằng chromosome kiến trúc.
- Xây dựng CNN từ chromosome 2–4 block và đếm số tham số.
- Kiểm tra gene, kích thước tensor và đầu ra 7 lớp.
- Không đọc test split.

Xác nhận khi baseline vẫn chạy và model builder tạo output `(batch, 7)` cho kiến trúc hợp lệ.

Người dùng đã xác nhận hoàn thành Stage 1 và yêu cầu chuyển sang Stage 2 ngày 2026-10-01.

## [x] Stage 2 — Search space và NSGA-II độc lập

Người dùng đã xác nhận Stage 2 chạy thành công và yêu cầu chuyển sang Stage 3 ngày 2026-10-01.

- Sinh, canonicalize, validate và repair chromosome.
- Triển khai dominance, nondominated sorting, crowding distance và tournament selection.
- Triển khai uniform crossover, mutation gene, thêm/xóa block và evaluation budget.
- Dùng fitness callback giả lập, độc lập CNN.

Xác nhận khi toán tử luôn sinh kiến trúc hợp lệ và NSGA-II dừng đúng budget thực tế.

## [x] Stage 3 — Tích hợp NAS với CNN

Người dùng đã xác nhận Stage 3 hoàn thành và yêu cầu chuyển sang Stage 4 ngày 2026-10-02. Lần chạy seed 42 đã đánh giá đủ 80 kiến trúc, mỗi kiến trúc 25 epoch, và xuất Pareto front từ validation.

- Train mỗi kiến trúc 25 epoch (cùng baseline theo yêu cầu người dùng) với training hyperparameters cố định.
- Fitness gồm validation macro AUC và số tham số.
- Cache theo canonical architecture và seed; cache hit không train lại.
- Log từng evaluation vào `results/ga_search/`.

Xác nhận khi pipeline `NSGA-II → CNN → Pareto objectives` chạy đúng và không đọc test split.

## [ ] Stage 4 — Random Architecture Search và thực nghiệm

Stage hiện tại: đã viết Random Architecture Search và đồng bộ log so sánh với GA-NAS; chờ người dùng chạy seed `42` và xác nhận.

- Thêm Random Architecture Search dùng cùng search space.
- Chạy GA-NAS và Random Search với cùng budget 80, 25 epoch và seed `42`.
- Log cấu hình, metrics, runtime, Pareto rank, crowding, best AUC, model nhỏ nhất và kích thước Pareto front theo từng evaluation.

Xác nhận khi hai phương pháp dừng đúng budget và log đủ để tái lập.

## [ ] Stage 5 — Chọn kiến trúc và đánh giá cuối

- Từ Pareto front validation của mỗi phương pháp, chọn ba đại diện theo quy tắc cố định: AUC cao nhất, knee point trên hai mục tiêu đã chuẩn hóa, và model nhỏ nhất có AUC không thấp hơn baseline.
- Train lại tối đa 25 epoch với seed `1, 2, 3`, early stopping patience 5.
- Chỉ tại stage này mới đọc test split.
- Lưu checkpoint model, cấu hình, lịch sử train, validation/test metrics và prediction dùng cho confusion matrix vào `results/final_eval/`.

Xác nhận khi test không tham gia chọn kiến trúc, ba quy tắc chọn cho kết quả tái lập và có đủ checkpoint/kết quả cho ba seed.

## [ ] Stage 6 — Phân tích nghiên cứu

- Tổng hợp mean ± standard deviation, loss, accuracy, macro F1, macro AUC, số tham số và runtime.
- Tính Hypervolume cuối và Hypervolume theo evaluation cho GA-NAS/Random Search sau khi chuẩn hóa chung; dùng cùng một reference point cố định suy ra từ giới hạn số tham số của search space.
- Tính Coverage hai chiều `C(GA, Random)` và `C(Random, GA)` để đo tỷ lệ điểm Pareto của phương pháp này bị phương pháp kia thống trị.
- So sánh AUC cao nhất, knee point, model nhỏ nhất đạt AUC baseline, mức giảm tham số và chi phí tìm kiếm của baseline, GA-NAS và Random Architecture Search.
- Tạo Pareto AUC–model size có điểm baseline, Hypervolume theo evaluation, best AUC theo evaluation, model nhỏ nhất đạt AUC baseline theo evaluation, confusion matrix và biểu đồ độ ổn định qua các seed retrain.
- Nêu rõ giới hạn: search Stage 4 chỉ dùng seed 42; nhiều seed ở Stage 5 đánh giá độ ổn định của model được chọn, không chứng minh độ ổn định của thuật toán search.
- Không dùng IGD vì không có Pareto front chuẩn đáng tin cậy.

Xác nhận khi Hypervolume/Coverage dùng cùng chuẩn hóa và reference point, bảng/biểu đồ truy ngược đúng log, và kết luận không vượt quá bằng chứng một search seed.

## [ ] Stage 7 — Hoàn thiện và bàn giao

- Đồng bộ dependency, tài liệu, đường dẫn, lệnh chạy và output.
- Rà soát tính tái lập và bảo đảm không có tính năng ngoài phạm vi.

Xác nhận khi toàn bộ quy trình có thể tái lập theo `GUIDE.md`.
