# Các stage triển khai GA-ResNet

Ngày 2026-10-04, người dùng quyết định thay toàn bộ CNN tự thiết kế bằng ResNet-18 và bắt đầu lại từ Stage 1. Các dấu hoàn thành và kết quả CNN cũ không còn là tiến độ chính thức của dự án mới.

Chỉ đánh dấu `[x]` sau khi người dùng tự chạy, kiểm thử và xác nhận. Chỉ thực hiện một stage tại một thời điểm.

## [x] Stage 1 — ResNet-18 baseline và model builder

Người dùng đã chạy, xác nhận kết quả và yêu cầu chuyển sang Stage 2 ngày 2026-10-05.

- Thay baseline CNN cũ bằng ResNet-18 train from scratch.
- Dùng BasicBlock, stage `[2,2,2,2]`, channel `[64,128,256,512]`.
- Dùng stem `3×3`, stride 1, không max-pooling đầu mạng cho ảnh 64×64.
- Biểu diễn baseline bằng chromosome ResNet và đếm số tham số.
- Dùng protocol chung: Adam, learning rate đầu 0.001, batch size 64, cross-entropy không weight, 60 epoch, StepLR mỗi 20 epoch với gamma 0.1, normalize mean/std 0.5/0.5, checkpoint validation macro AUC tốt nhất và seed 42.
- Lưu kết quả tại `results/baseline/seed_42/result.json` và checkpoint tại `results/baseline/seed_42/checkpoint.pt`.
- Không đọc test split.

Xác nhận khi ResNet-18 baseline chạy và model builder tạo output `(batch, 7)` với residual shortcut hợp lệ.

## [x] Stage 2 — ResNet search space và NSGA-II độc lập

Người dùng đã xác nhận và yêu cầu chuyển sang Stage 3 ngày 2026-10-05.

- Chromosome gồm `stage_blocks`, `stage_channels`, `kernel_sizes`, `dropout`.
- Sinh, canonicalize, validate và repair mọi biến thể thành ResNet hợp lệ.
- Triển khai dominance, nondominated sorting, crowding distance và tournament selection.
- Triển khai uniform crossover theo gene, mutation và evaluation budget.
- Dùng fitness callback giả lập, độc lập với việc train ResNet.

Xác nhận khi toán tử luôn sinh kiến trúc hợp lệ và NSGA-II dừng đúng budget thực tế.

## [x] Stage 3 — Tích hợp NAS với ResNet

Người dùng đã chạy GA, yêu cầu phân tích kết quả và yêu cầu chuyển sang Random Search ngày 2026-10-06.

- Train mỗi ResNet candidate bằng proxy protocol 20 epoch, StepLR mỗi 15 epoch; giữ nguyên normalization, unweighted cross-entropy và quy tắc best-validation checkpoint.
- Fitness gồm validation macro AUC và số tham số.
- Tăng phiên bản cache để không thể dùng evaluation CNN cũ.
- Log từng evaluation vào `results/ga_search/`.

Xác nhận khi pipeline `NSGA-II → ResNet → Pareto objectives` đánh giá đủ 60 kiến trúc và không đọc test split.

## [ ] Stage 4 — Random ResNet Search và thực nghiệm

Stage hiện tại: code Random ResNet Search đã được đồng bộ với GA; chờ người dùng chạy và xác nhận.

- Random Architecture Search dùng cùng ResNet search space.
- Chạy GA-NAS và Random Search với cùng budget 60, proxy protocol 20 epoch/StepLR 15 và seed `42`.
- Log cấu hình, metrics, runtime, Pareto rank, crowding, best AUC, model nhỏ nhất và kích thước Pareto front theo từng evaluation.

Xác nhận khi hai phương pháp dừng đúng budget và log đủ để tái lập.

## [ ] Stage 5 — Chọn kiến trúc và đánh giá cuối

- Từ Pareto front validation của mỗi phương pháp, chọn một ResNet theo quy tắc cố định: validation macro AUC cao nhất; thêm ResNet-18 baseline.
- Train lại cả ba mô hình đủ 60 epoch với seed `42`, dùng normalization/loss chung, StepLR mỗi 20 epoch và checkpoint validation macro AUC tốt nhất.
- Chỉ tại stage này mới đọc test split.
- Lưu checkpoint model, cấu hình, lịch sử train, validation/test metrics và prediction dùng cho confusion matrix vào `results/final_eval/`.

Xác nhận khi test không tham gia chọn kiến trúc và có đủ checkpoint/kết quả seed 42 của GA-ResNet, Random-ResNet và ResNet-18.

## [ ] Stage 6 — Phân tích nghiên cứu

- Tổng hợp loss, accuracy, macro precision, macro recall, macro F1, macro AUC, số tham số và runtime.
- Tính Hypervolume cuối và Hypervolume theo evaluation cho GA-NAS/Random Search sau khi chuẩn hóa chung; dùng cùng một reference point cố định suy ra từ giới hạn số tham số của search space.
- Tính Coverage hai chiều `C(GA, Random)` và `C(Random, GA)` để đo tỷ lệ điểm Pareto của phương pháp này bị phương pháp kia thống trị.
- So sánh AUC cao nhất, knee point, model nhỏ nhất đạt AUC baseline, mức giảm tham số và chi phí tìm kiếm của baseline, GA-NAS và Random Architecture Search.
- Tạo Pareto AUC–model size có điểm baseline, Hypervolume theo evaluation, best AUC theo evaluation, model nhỏ nhất đạt AUC baseline theo evaluation và confusion matrix.
- Kết luận tác dụng của GA dựa trên đối chứng Random Search cùng budget, không chỉ dựa trên so sánh với ResNet-18.
- Nêu rõ giới hạn: cả search và đánh giá cuối chỉ dùng seed 42, nên chưa chứng minh độ ổn định qua nhiều seed.
- Không dùng IGD vì không có Pareto front chuẩn đáng tin cậy.

Xác nhận khi Hypervolume/Coverage dùng cùng chuẩn hóa và reference point, bảng/biểu đồ truy ngược đúng log, và kết luận không vượt quá bằng chứng một search seed.

## [ ] Stage 7 — Hoàn thiện và bàn giao

- Đồng bộ dependency, tài liệu, đường dẫn, lệnh chạy và output.
- Rà soát tính tái lập và bảo đảm không có tính năng ngoài phạm vi.
- Bảo đảm tài liệu và output chính thức không còn trộn kết quả CNN cũ với ResNet mới.

Xác nhận khi toàn bộ quy trình có thể tái lập theo `GUIDE.md`.
