# Kế hoạch mở rộng bài toán tối ưu kiến trúc mạng bằng Genetic Algorithm (GA-ResNet trên DermaMNIST)

Tài liệu này tổng hợp các hướng mở rộng khả thi dựa trên kết quả thực nghiệm và kiến trúc code hiện tại của dự án (tính đến Stage 6). Mỗi hướng nêu rõ **vấn đề cần giải quyết**, **cách thực hiện cụ thể trên codebase hiện có**, và **kết quả cần thu lại để đánh giá**.

> **Nguyên tắc chung khi mở rộng:**
> - Giữ nguyên quy tắc dữ liệu: chỉ đọc `data/dermamnist_64.npz`, giữ nguyên split chính thức, **không truy cập test split** trong quá trình tìm kiếm hay chọn siêu tham số.
> - Khi thay đổi hàm mục tiêu, không gian tìm kiếm hoặc giao thức train, bắt buộc tăng `training_version` / `architecture_space` trong `src/search_runtime.py` và lưu vào thư mục output riêng để không lẫn với kết quả chính thức của phiên bản ResNet 2 mục tiêu hiện tại.

---

## Tóm tắt hiện trạng làm mốc so sánh (Baseline của kế hoạch mở rộng)

- **Search (Validation, proxy 20 epoch, budget 60, seed 42):**
  - GA-NAS (NSGA-II): Hypervolume `0.9403`, Coverage `C(GA, Random) = 0.50`, Best Val AUC `0.9450`, model nhỏ nhất đạt AUC baseline `537K params`, runtime `~1.68h`.
  - Random Search: Hypervolume `0.9368`, Coverage `C(Random, GA) = 0.00`, Best Val AUC `0.9439`, model nhỏ nhất đạt AUC baseline `666K params`, runtime `~2.39h`.
- **Test cuối (Retrain 60 epoch, seed 42, chọn điểm `max val AUC`):**
  - ResNet-18 Baseline: `11.17M params`, Test AUC `0.9374`, Acc `77.71%`, Macro F1 `0.5519`.
  - GA-ResNet: `2.77M params` (-75.2%), Test AUC `0.9364`, Acc `78.80%`, Macro F1 `0.5919`.
  - Random-ResNet: `5.82M params` (-47.9%), Test AUC `0.9401`, Acc `79.10%`, Macro F1 `0.5866`.
  - *Nhánh thử nghiệm Surrogate-NSGA-II (budget 40)*: `0.44M params` (-96.0%), Test AUC `0.9263`, Acc `77.31%`, Macro F1 `0.5414`.

---

## Hướng 1: Hardware-Aware Multi-Objective NAS — Bổ sung FLOPs (MACs) và/hoặc Inference Latency

### 1.1. Động lực
Hiện tại bài toán chỉ tối ưu 2 mục tiêu (`Max Val AUC`, `Min parameter_count`). Với ảnh `64×64` và stem không có max-pool (`src/model.py`), các stage đầu (`64×64`, `32×32`) ít tham số nhưng tốn rất nhiều FLOPs, còn stage 4 (`8×8`) nhiều tham số nhưng ít FLOPs. Do đó GA có xu hướng "lách" mục tiêu bằng cách cắt channel ở stage cuối nhưng đặt kernel `5×5` ở các stage đầu (ví dụ kiến trúc GA chọn có `kernel_sizes = [5, 5, 5, 3]`), khiến mạng ít params nhưng chi phí tính toán vẫn cao.

### 1.2. Cách thực hiện
1. **Đo FLOPs (MACs) và Latency trong `src/model.py`:**
   - Viết hàm `count_flops(model, input_shape=(1, 3, 64, 64))` sử dụng forward hook cho `nn.Conv2d` và `nn.Linear` (không cần thêm dependency nặng, tính chính xác tuyệt đối và tất định).
   - *(Tùy chọn)* Viết hàm `measure_latency_ms(model, device, input_shape=(1, 3, 64, 64), warmup=20, runs=100)` đo trung vị thời gian suy luận trên GPU/CPU mục tiêu (cần chạy ở điều kiện máy ổn định).
2. **Tổng quát hóa NSGA-II sang $M$ mục tiêu trong `src/nsga2.py`:**
   - Chuyển `Individual.fitness` từ tuple 2 chiều `(auc, params)` sang tuple $M$ chiều, ví dụ 3 mục tiêu `(max val_auc, min params, min flops)` kèm vector hướng tối ưu `directions = ("max", "min", "min")`.
   - Cập nhật `dominates()` và `crowding_distance()` lặp theo `range(len(directions))`.
   - *(Nâng cao nếu lên 3–4 mục tiêu)* Triển khai **NSGA-III** (dùng reference points thay cho crowding distance) để giữ phân bố đồng đều trên mặt Pareto 3D.
3. **Cập nhật runtime, cache và phân tích:**
   - Ghi thêm `flops` (và `latency_ms` nếu có) vào từng file `evaluations/<cache_key>.json` và `summary.json` trong `src/search_runtime.py`.
   - Cập nhật `src/analyze.py` tính **3D Hypervolume** (chuẩn hóa chung từ cận min/max của search space cho cả Params và FLOPs), Coverage hai chiều trên Pareto 3D, và vẽ biểu đồ Pareto 2D chiếu theo từng cặp (`AUC–FLOPs`, `AUC–Params`, `Params–FLOPs`) hoặc 3D scatter.

### 1.3. Kết quả cần thu lại
- **So sánh định lượng:** 3D Hypervolume, Coverage hai chiều `C(GA, Random)` / `C(Random, GA)` trên không gian `(AUC, Params, FLOPs)`.
- **Phân tích cấu trúc kiến trúc tìm được:** Xác nhận GA 3 mục tiêu không còn dồn `kernel 5×5` vào stage 1–2 một cách lãng phí; so sánh mức giảm **cả Params lẫn FLOPs** (và tốc độ inference thực tế) so với ResNet-18 baseline và bản GA 2 mục tiêu cũ.
- **Biểu đồ:** Pareto front `AUC vs FLOPs`, `AUC vs Params`, và bảng tổng hợp thêm cột `GFLOPs` / `Latency (ms)` ở `test_metrics.csv`.

---

## Hướng 2: Nâng cấp Training Protocol — Chống Overfitting & Xử lý Mất cân bằng lớp (Class Imbalance)

### 2.1. Động lực
- Khi retrain 60 epoch, cả 3 mô hình đều đạt `best_epoch` ở **epoch 21–22** rồi overfit mạnh do `src/train.py` chưa có **Data Augmentation** và **Weight Decay**.
- Trên DermaMNIST (lệch lớp nặng), dùng `unweighted_cross_entropy` khiến **Macro Recall chỉ ~0.53–0.57** và **Macro F1 chỉ ~0.55–0.59** dù AUC đạt ~0.94.
- Ngoài ra, tập validation 1,003 ảnh dễ bị overfit trong quá trình NAS, làm lệch thứ hạng giữa proxy 20 epoch và test cuối.

### 2.2. Cách thực hiện
1. **Thêm Data Augmentation cho ảnh da liễu (`src/train.py`):**
   - Áp dụng trên tập train (không áp dụng trên val/test): `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation` (bội số 90° hoặc ±15°), và nhẹ `ColorJitter` (brightness/contrast).
   - Bảo đảm tính tái lập bằng generator gắn với `seed`.
2. **Cải thiện Optimizer & Scheduler (`src/train.py`):**
   - Thay `Adam` mặc định bằng `AdamW` với `weight_decay` (ví dụ `1e-4` hoặc `1e-3`).
   - Thay `StepLR` cứng bằng `CosineAnnealingLR(T_max=epochs)` để hội tụ mượt hơn, giảm hiện tượng đạt đỉnh đột ngột ngay sau bước nhảy LR.
3. **Xử lý mất cân bằng lớp:**
   - Thử nghiệm (theo dạng ablation study): `Weighted Cross-Entropy` (trọng số nghịch đảo tần suất lớp tính **chỉ trên tập train**) hoặc `Focal Loss`.
   - Cân nhắc thêm **Macro F1** hoặc **Balanced Accuracy** vào theo dõi/mục tiêu phụ (vì bài toán sàng lọc y tế cần giảm bỏ sót các lớp hiếm).
4. **Đánh giá độ tương quan Proxy vs Full Training:**
   - Lấy mẫu 10–15 kiến trúc từ kết quả search, train đủ 60 epoch và tính **Spearman / Kendall rank correlation** giữa `val_auc (20 ep)` và `val_auc (60 ep)` để kiểm chứng proxy protocol mới có xếp hạng đáng tin cậy hơn không.

### 2.3. Kết quả cần thu lại
- **Ablation table** so sánh trực tiếp protocol cũ vs protocol mới trên cùng ResNet-18 baseline và GA-ResNet: `Val/Test AUC`, `Macro F1`, `Macro Recall`, `best_epoch`, khoảng cách `train_loss - val_loss`.
- **Chỉ số tương quan thứ hạng** (Spearman $\rho$, Kendall $\tau$) giữa proxy training và full training.
- **Confusion matrix** mới chứng minh tỷ lệ nhận diện đúng ở các lớp thiểu số tăng lên mà không làm tụt AUC tổng thể.

---

## Hướng 3: Mở rộng và Chuẩn hóa Không gian kiến trúc (Search Space)

### 3.1. Động lực
- Không gian hiện tại (`src/search_space.py`) chỉ xoay quanh `BasicBlock` cố định với 4 gene (`stage_blocks`, `stage_channels`, `kernel_sizes`, `dropout`), giới hạn khả năng nén FLOPs sâu hơn.
- `stage_channels` hiện cho phép giảm ngược giữa các stage (ví dụ `[64, 32, 64, 128]` hoặc `[64, 64, 256, 128]`), tạo nút thắt cổ chai bất thường khi độ phân giải giảm một nửa.

### 3.2. Cách thực hiện
1. **Ràng buộc hình học hợp lý (`src/search_space.py`):**
   - Bổ sung quy tắc sửa/ràng buộc đơn điệu không giảm cho channel (`c1 <= c2 <= c3 <= c4`) trong `validate_architecture()` và `repair_architecture()`, loại bỏ các cấu hình vô nghĩa để GA tập trung vào vùng kiến trúc tốt.
2. **Bổ sung các building block hiện đại (`src/model.py` & `src/search_space.py`):**
   - **Gene `block_types` theo từng stage:** cho phép chọn giữa `BasicBlock`, `DepthwiseSeparableBlock` (hoặc `MBConv / InvertedResidual` kiểu MobileNetV2 với expansion ratio `2` hoặc `4`). Điều này giúp giảm đột biến cả FLOPs lẫn Params.
   - **Gene `use_se` (Squeeze-and-Excitation):** mảng 4 bit `[0/1, 0/1, 0/1, 0/1]` bật/tắt cơ chế channel attention ở từng stage (rất hữu ích cho ảnh y tế, tăng ít FLOPs nhưng cải thiện phân biệt đặc trưng).
   - **Gene `activation`:** chọn giữa `ReLU` và `SiLU` (Swish).
3. **Cập nhật toán tử di truyền:**
   - Cập nhật `uniform_crossover()`, `mutate_architecture()`, `canonical_architecture()` và `repair_architecture()` tương thích với bộ gene mới.

### 3.3. Kết quả cần thu lại
- **So sánh Pareto front giữa Search Space v1 (chỉ BasicBlock) và Search Space v2 (Hybrid BasicBlock/MBConv + SE):**
  - Sự dịch chuyển của đường Pareto về phía góc trên-trái (AUC cao hơn tại cùng mức Params/FLOPs).
  - Phân tích tần suất gene xuất hiện trên Pareto front (ví dụ: SE có thường xuyên được GA chọn ở stage 3–4 không? MBConv có thay thế BasicBlock ở stage 1–2 để tiết kiệm FLOPs không?).

---

## Hướng 4: Nâng cao Hiệu quả Tìm kiếm — Multi-Fidelity NAS & Nâng cấp Surrogate Search

### 4.1. Động lực
- GA hiện tại hội tụ `best_auc` ngay ở evaluation 13 và chỉ chạy được ~6 thế hệ (`population = 10`, `budget = 60`), tốn ~1.68 giờ cho 60 lần train đủ 20 epoch.
- Nhánh `src/surrogate_search/` hiện dùng 1 model `XGBoostSurrogate` dự đoán điểm (chưa ước lượng độ bất định) và mới thử budget 40; kết quả Test AUC (0.9263) đang thấp hơn GA thuần (0.9364) do surrogate chọn kiến trúc quá nhỏ (`445K`) bị overfit vào validation proxy.

### 4.2. Cách thực hiện
1. **Multi-Fidelity / Early Stopping trong quá trình đánh giá (`src/train.py`, `src/search_runtime.py`):**
   - Áp dụng **Successive Halving / Median Stopping Rule**: dừng sớm ở epoch 5 hoặc epoch 10 các kiến trúc có `val_auc` thấp hơn trung vị (hoặc ngưỡng phân vị) của các kiến trúc đã đánh giá ở cùng epoch.
   - Nhờ tiết kiệm 40–60% thời gian trên các kiến trúc yếu, có thể nâng `population_size` lên `16–20` và `evaluation_budget` lên `120–150` trong cùng quỹ thời gian ~2 giờ.
2. **Nâng cấp `src/surrogate_search/` thành Uncertainty-Aware Surrogate NAS:**
   - Thay 1 model XGBoost bằng **Ensemble (ví dụ 5–10 XGBoost với bootstrap/seed khác nhau) hoặc Random Forest / Gaussian Process** để dự đoán cả kỳ vọng $\mu(x)$ lẫn độ lệch chuẩn $\sigma(x)$.
   - Dùng hàm acquisition đa mục tiêu như **UCB** ($\mu + \beta\sigma$) hoặc **Expected Hypervolume Improvement (EHVI)** thay vì chỉ lọc theo point prediction.
   - Thêm bước đánh giá chất lượng surrogate trên tập hold-out / cross-validation (MAE, RMSE, **Spearman rank correlation**) sau mỗi vòng lặp online để theo dõi surrogate có thực sự học được quy luật kiến trúc không.
   - Đưa về cùng budget 60 và cùng seed để so sánh công bằng trực tiếp với GA-NAS thuần và Random Search.

### 4.3. Kết quả cần thu lại
- **Đường hội tụ theo thời gian thực (wall-clock time) và theo số evaluation:** so sánh `Random Search` vs `NSGA-II` vs `Multi-fidelity NSGA-II` vs `Surrogate-Assisted NSGA-II`.
- **Chỉ số chất lượng surrogate:** Spearman $\rho$ giữa AUC dự đoán và AUC thật qua các vòng lặp online.
- **Hiệu quả chi phí:** cùng một mức Hypervolume hoặc Best AUC, phương pháp mới giảm được bao nhiêu % thời gian GPU (GPU-hours).

---

## Hướng 5: Nâng cao Độ chặt chẽ Thực nghiệm & Đánh giá Khoa học

### 5.1. Động lực
- Hiện mới chạy **1 seed (`42`)** nên chênh lệch `0.001–0.003` AUC trên tập Test chưa đủ ý nghĩa thống kê.
- Stage 5 hiện chỉ đánh giá Test cho 1 điểm duy nhất (`max val AUC`), bỏ qua **knee point** (`604K params`) và **smallest model reaching baseline** (`537K params`) — những điểm thể hiện giá trị cốt lõi của tối ưu đa mục tiêu.

### 5.2. Cách thực hiện
1. **Đánh giá đa đại diện trên Pareto front (`src/test_eval.py`):**
   - Thay vì chỉ chọn `max_val_auc`, mở rộng `src/test_eval.py` tự động chọn và retrain 60 epoch **3 đại diện** từ Pareto front validation của mỗi phương pháp:
     1. `max_auc`: Kiến trúc có validation AUC cao nhất.
     2. `knee_point`: Điểm cân bằng tốt nhất giữa AUC và chi phí (Params/FLOPs).
     3. `smallest_baseline_equivalent`: Kiến trúc nhỏ nhất có validation AUC $\ge$ baseline.
2. **Thực nghiệm đa seed (Multi-seed) & Kiểm định thống kê (`src/analyze.py`):**
   - Chạy độc lập ít nhất **3–5 seeds** (ví dụ `42, 43, 44` cho search và/hoặc retrain).
   - Báo cáo `mean ± std` cho Hypervolume, Coverage, Test AUC, Macro F1, Params, FLOPs.
   - Chạy kiểm định phi tham số (Wilcoxon signed-rank test hoặc Mann-Whitney U test) nếu đủ số lần lặp.
3. **Kiểm tra khả năng tổng quát hóa sang dataset y tế khác (Cross-dataset validation):**
   - Thử áp dụng cùng pipeline (không sửa code lõi) lên 1–2 tập khác cùng họ MedMNIST 64×64 như `PathMNIST` hoặc `BloodMNIST` để kiểm chứng GA-NAS không chỉ hoạt động tốt trên riêng DermaMNIST.

### 5.3. Kết quả cần thu lại
- **Bảng kết quả Test đầy đủ cho 3 điểm đại diện Pareto** (chứng minh ngay cả `knee_point` và `smallest_baseline_equivalent` của GA cũng giữ được hiệu năng ngang ngửa baseline trong khi nhẹ hơn 90–95%).
- **Bảng `mean ± std`** trên nhiều seeds kèm đường hội tụ có dải sai số (shaded error band).
- *(Nếu mở rộng dataset)* Bảng so sánh độ ổn định của GA-NAS vs Random Search trên 2–3 tập MedMNIST.

---

## Đề xuất lộ trình ưu tiên triển khai

Tùy theo mục tiêu (đồ án môn học, khóa luận hay bài báo nghiên cứu), có thể chọn theo thứ tự ưu tiên sau:

| Mức ưu tiên | Hướng mở rộng | Độ khó triển khai | Chi phí GPU | Giá trị mang lại |
|---|---|---:|---:|---|
| **Ưu tiên 1** | **Hướng 1 (Thêm FLOPs -> NSGA-II 3 mục tiêu)** + **Hướng 5.1 (Test cả 3 điểm Pareto)** | Thấp – Trung bình | Thấp (1 lần search) | Giải quyết trực tiếp điểm yếu "chỉ tối ưu params", thể hiện rõ giá trị đa mục tiêu |
| **Ưu tiên 2** | **Hướng 2 (Augmentation + AdamW + CosineLR + Weighted/Focal Loss)** | Thấp | Thấp – Trung bình | Khắc phục overfit sau epoch 21, cải thiện rõ Macro F1/Recall và ổn định xếp hạng proxy–test |
| **Ưu tiên 3** | **Hướng 3 (Search Space v2: MBConv + SE + ràng buộc channel)** | Trung bình | Trung bình | Đẩy đường Pareto lên mức nén cực cao cả Params lẫn FLOPs |
| **Ưu tiên 4** | **Hướng 4 (Multi-fidelity + Ensemble Surrogate UCB/EHVI)** | Trung bình – Cao | Trung bình | Tăng quy mô tìm kiếm, hoàn thiện nhánh `src/surrogate_search/` đang có |
| **Ưu tiên 5** | **Hướng 5.2–5.3 (Multi-seed 3–5 lần + Cross-dataset MedMNIST)** | Thấp | Cao | Nâng độ tin cậy thống kê lên chuẩn nghiên cứu / paper |
