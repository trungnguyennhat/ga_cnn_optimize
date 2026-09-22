# Các stage triển khai

Chỉ đánh dấu `[x]` sau khi người dùng tự chạy, kiểm thử và xác nhận. Tất cả stage hiện ở trạng thái `[ ]`.

## [ ] Stage 1 — Loader DermaMNIST và CNN baseline

- Tạo `model.py` với CNN ba convolution block, đầu vào RGB 64×64 và đầu ra 7 lớp.
- Tạo `train.py` đọc `data/dermamnist_64.npz`, kiểm tra keys/shapes/labels và dùng nguyên split chính thức.
- Dùng weighted cross-entropy; hỗ trợ seed, device và hyperparameters baseline.
- Trả về loss, accuracy, macro F1 và macro AUC trên validation.
- Không đọc test split trong baseline.

Xác nhận khi pipeline baseline chạy end-to-end, tensor đúng kích thước, loss giảm và đủ validation metrics.

## [ ] Stage 2 — Genetic Algorithm độc lập

- Tạo `search.py` với chromosome, initialization, tournament selection, uniform crossover, mutation và elitism.
- Learning rate được lấy mẫu/mutation trong log-space.
- Dừng theo evaluation budget thực tế và nhận fitness callback độc lập CNN.

Xác nhận khi GA chạy đúng với fitness function đơn giản và luôn sinh chromosome hợp lệ.

## [ ] Stage 3 — Tích hợp GA với CNN

- Fitness là validation macro AUC one-vs-rest sau 5 epoch.
- Thêm cache theo configuration, deterministic seed và log từng evaluation.
- Cache hit không train lại và không tăng evaluation count.

Xác nhận khi pipeline `GA → CNN → macro AUC` chạy đúng và không truy cập test split.

## [ ] Stage 4 — Random Search và thực nghiệm

- Thêm Random Search trong `search.py`.
- Tạo `run_experiment.py` cho Default CNN, Random Search và GA.
- Dùng cùng search space, split, epoch, budget và independent seeds.
- Lưu configuration, seed, metrics, runtime và best-so-far.

Xác nhận khi ba phương pháp chạy được, GA/Random Search dừng đúng budget và log đủ để tái lập.

## [ ] Stage 5 — Đánh giá cuối và phân tích

- Tạo `analyze.py`, chọn top 3 configurations mỗi phương pháp bằng validation macro AUC.
- Train lại tối đa 25 epoch với seed `1, 2, 3` và early stopping patience 5.
- Chỉ tại stage này mới đánh giá test.
- Xuất mean ± standard deviation, accuracy, macro F1, macro AUC, confusion matrix và biểu đồ search.

Xác nhận khi bảng/biểu đồ được tạo và test set không tham gia chọn cấu hình.

## [ ] Stage 6 — Hoàn thiện và bàn giao

- Hoàn thiện `requirements.txt`, `README.md` và `GUIDE.md`.
- Kiểm tra đường dẫn, output, log và lệnh chạy nhất quán.
- Không thêm tính năng ngoài phạm vi đã thống nhất.

Xác nhận khi người dùng có thể tái lập toàn bộ quy trình theo `GUIDE.md`.
