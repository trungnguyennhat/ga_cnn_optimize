# Các stage triển khai GA-NAS

Chỉ đánh dấu `[x]` sau khi người dùng tự chạy, kiểm thử và xác nhận. Chỉ thực hiện một stage tại một thời điểm.

## [x] Stage 1 — ResNet-18 baseline và model builder

Người dùng đã chạy baseline và model builder, yêu cầu chuyển sang Stage 2 ngày 2026-10-05.

- Chuyển sang ResNet: stem `Conv(3x3, stride=1) -> BN -> ReLU` (không max pool), 4 stage dùng `BasicBlock`, `AdaptiveAvgPool2d(1)`, dropout và classifier 7 lớp.
- Baseline là ResNet-18 từ scratch: blocks `[2, 2, 2, 2]`, channels `[64, 128, 256, 512]`, kernels `[3, 3, 3, 3]`, dropout `0.0`.
- Biểu diễn kiến trúc bằng chromosome dictionary: `stage_blocks`, `stage_channels`, `kernel_sizes`, `dropout`.
- Chuẩn hóa ảnh DermaMNIST về `[-1, 1]`, huấn luyện 60 epoch với Adam `lr=0.001`, batch size 64, StepLR giảm 10 lần mỗi 20 epoch và lưu checkpoint có validation macro AUC cao nhất.
- Không đọc test split.

Xác nhận khi baseline đạt khoảng 0.94 validation macro AUC, model builder tạo output `(batch, 7)` và `count_parameters` khớp số tham số kỳ vọng (~11.17M).

## [x] Stage 2 — Search space và NSGA-II độc lập

Người dùng đã kiểm thử search space và NSGA-II độc lập, yêu cầu chuyển sang Stage 3 ngày 2026-10-05.

- Search space ResNet: 4 stage, blocks mỗi stage `1..3`, channels theo 4 bộ lựa chọn, kernel size `3` hoặc `5`, dropout `0.0..0.5`.
- Sinh, canonicalize, validate và repair chromosome.
- Triển khai dominance, nondominated sorting, crowding distance và tournament selection.
- Triển khai uniform crossover, mutation từng gene và evaluation budget.
- Dùng fitness callback giả lập, độc lập training CNN.

Xác nhận khi toán tử luôn sinh kiến trúc hợp lệ và NSGA-II dừng đúng budget thực tế.

## [x] Stage 3 — Tích hợp NAS với huấn luyện ResNet

Người dùng đã chạy đủ 60 kiến trúc GA-NAS, yêu cầu phân tích kết quả và chuyển sang Stage 4 ngày 2026-10-06.

- Nối NSGA-II với training ResNet thật bằng proxy evaluation: 20 epoch, Adam `lr=0.001`, StepLR step size 15, checkpoint validation macro AUC tốt nhất.
- Objectives: maximize validation macro AUC và minimize parameter count.
- Population size 10, evaluation budget 60 cá thể được huấn luyện thực sự, seed `42`.
- Cache theo hash của canonical architecture và config; cache hit không tính vào budget.
- Log từng evaluation vào `results/ga_search/`.

Xác nhận khi pipeline `NSGA-II → ResNet → Pareto objectives` đánh giá đủ 60 kiến trúc và không đọc test split.

## [x] Stage 4 — Random ResNet Search và thực nghiệm

Người dùng đã chạy đủ 60 kiến trúc Random Search, yêu cầu phân tích kết quả và chuyển sang Stage 5 ngày 2026-10-06.

- Random Architecture Search dùng cùng ResNet search space.
- Chạy GA-NAS và Random Search với cùng budget 60, proxy protocol 20 epoch/StepLR 15 và seed `42`.
- Log cấu hình, metrics, runtime, Pareto rank, crowding, best AUC, model nhỏ nhất và kích thước Pareto front theo từng evaluation.

Xác nhận khi hai phương pháp dừng đúng budget và log đủ để tái lập.

## [x] Stage 5 — Chọn kiến trúc và đánh giá cuối

Người dùng đã chạy đủ GA, Random và baseline trên test, yêu cầu phân tích kết quả và chuyển sang Stage 6 ngày 2026-10-06.

- Từ Pareto front validation của mỗi phương pháp, chọn một ResNet theo quy tắc cố định: validation macro AUC cao nhất; thêm ResNet-18 baseline.
- Train lại kiến trúc GA và Random đủ 60 epoch với seed `42`, dùng normalization/loss chung, StepLR mỗi 20 epoch và checkpoint validation macro AUC tốt nhất; tái sử dụng checkpoint baseline 60 epoch hiện có.
- Chỉ tại stage này mới đọc test split.
- Lưu checkpoint model, cấu hình, lịch sử train, validation/test metrics và prediction dùng cho confusion matrix vào `results/test_eval/`.

Xác nhận khi test không tham gia chọn kiến trúc và có đủ checkpoint/kết quả seed 42 của GA-ResNet, Random-ResNet và ResNet-18.

## [x] Stage 6 — Phân tích nghiên cứu

Đã hoàn thành và nâng cấp trên nhánh `trang`.

- Tổng hợp loss, accuracy, macro precision, macro recall, macro F1, macro AUC, số tham số và runtime.
- Tính Hypervolume cuối và Hypervolume theo evaluation cho GA-NAS/Random Search sau khi chuẩn hóa chung; dùng cùng một reference point cố định suy ra từ giới hạn số tham số của search space.
- Tính Coverage hai chiều `C(GA, Random)` và `C(Random, GA)`.
- Bổ sung Schott's Spacing Metric ($S$) đo độ phân bố đều của Pareto front.
- Bổ sung kiểm định thống kê giả thuyết Mann-Whitney U test giữa GA và Random.
- Bổ sung phân tích tần suất gen sống sót (Gene survival & frequency) và biểu đồ `gene_distribution.png`.
- Bổ sung biểu đồ tiến triển thế hệ `generational_progression.png`.
- Xuất báo cáo tổng hợp chi tiết cho nhóm: `BAO_CAO_TIEN_HOA_NHOM.md`.

Xác nhận khi bảng/biểu đồ truy ngược đúng log và các chỉ số đo lường tiến hóa nhất quán.

## [x] Stage 7 — Hoàn thiện và bàn giao

Đã hoàn thành kiểm thử đơn vị và bàn giao trên nhánh `trang`.

- Đồng bộ dependency, tài liệu, đường dẫn, lệnh chạy và output.
- Xây dựng bộ Unit Tests toàn diện (`tests/test_search_space.py`, `tests/test_nsga2.py`, `tests/test_model.py`, `tests/test_evolution_stats.py`).
- 100% (14/14) bài kiểm thử đơn vị vượt qua thành công (`Ran 14 tests in 0.255s - OK`).
- Bảo đảm tài liệu và output chính thức hoàn toàn đồng bộ với kiến trúc ResNet.
