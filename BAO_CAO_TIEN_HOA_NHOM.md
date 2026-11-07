# BÁO CÁO ĐỒ ÁN MÔN HỌC: TÍNH TOÁN TIẾN HÓA
## Đề tài: Tối ưu hóa kiến trúc mạng nơ-ron đa mục tiêu (Multi-Objective Neural Architecture Search) bằng giải thuật di truyền NSGA-II trên tập dữ liệu DermaMNIST

---

## 1. THÔNG TIN CHUNG
* **Môn học:** Tính toán tiến hóa (Evolutionary Computation)
* **Quy mô nhóm:** 3 thành viên
* **Nhánh thực hiện:** `trang`
* **Mã nguồn dự án:** `https://github.com/trungnguyennhat/ga_cnn_optimize`

---

## 2. BỐI CẢNH & PHÁT BIỂU BÀI TOÁN

### 2.1. Đặt vấn đề
Trong phân loại ảnh y tế, đặc biệt là ảnh da liễu (DermaMNIST - phân loại 7 loại bệnh và ung thư da), các mô hình học sâu thường được thiết kế thủ công bằng phương pháp thử - sai (trial-and-error). Cách tiếp cận này có hai hạn chế lớn:
1. **Thiếu tính tối ưu toàn cục:** Con người dễ bị giới hạn bởi các kiến trúc kinh điển có sẵn.
2. **Kích thước mô hình quá lớn:** Mạng chuẩn như ResNet-18 có hơn 11 triệu tham số, đòi hỏi tài nguyên tính toán lớn, gây khó khăn khi triển khai trên các thiết bị y tế cầm tay hoặc máy tính phòng khám nhỏ.

### 2.2. Bài toán Tối ưu Đa mục tiêu (Multi-Objective Optimization)
Dự án áp dụng giải thuật di truyền **NSGA-II (Deb et al., 2002)** để tự động tìm kiếm không gian kiến trúc mạng ResNet nhằm thỏa mãn hai mục tiêu xung đột nhau:

$$\begin{cases}
\text{Maximize} & f_1(\mathbf{x}) = \text{Validation Macro ROC-AUC (OvR)} \quad \text{(Hiệu năng chẩn đoán)} \\
\text{Minimize} & f_2(\mathbf{x}) = \text{Parameter Count} \quad \text{(Độ phức tạp / Kích thước mạng)}
\end{cases}$$

Trong đó $\mathbf{x}$ là một nhiễm sắc thể (chromosome) đại diện cho cấu trúc mạng ResNet.

---

## 3. KHÔNG GIAN TÌM KIẾM & MÃ HÓA BỘ GEN (CHROMOSOME)

### 3.1. Mã hóa kiểu gen (Genotype)
Mỗi cá thể mạng được mã hóa thành một từ điển cấu trúc 4 stage với các gen:
* **`stage_blocks`**: Danh sách 4 số nguyên $\in \{1, 2, 3\}$, đại diện cho số lượng BasicBlock trong mỗi stage.
* **`stage_channels`**: Danh sách 4 số nguyên đại diện cho số kênh lọc mỗi stage:
  * Stage 1: $\{16, 32, 64\}$
  * Stage 2: $\{32, 64, 128\}$
  * Stage 3: $\{64, 128, 256\}$
  * Stage 4: $\{128, 256, 512\}$
* **`kernel_sizes`**: Danh sách 4 số nguyên $\in \{3, 5\}$ (kích thước bộ lọc tích chập).
* **`dropout`**: Số thực $\in \{0.0, 0.1, 0.2, 0.3, 0.5\}$ (tỷ lệ ngắt kết nối ngẫu nhiên trước lớp phân loại).

### 3.2. Ước tính kích thước không gian tìm kiếm
Số lượng cấu hình kiến trúc khả thi trong không gian tìm kiếm:
$$|\mathcal{S}| = (3^4) \times (3 \times 3 \times 3 \times 3) \times (2^4) \times 5 = 81 \times 81 \times 16 \times 5 = \mathbf{524{,}880} \text{ kiến trúc}$$

Ngân sách đánh giá bị giới hạn ở **60 lần đánh giá** (khoảng $\approx 0.011\%$ không gian), đòi hỏi thuật toán tiến hóa phải có khả năng khai phá (exploration) và khai thác (exploitation) hiệu quả.

---

## 4. BỘ MÁY TIẾN HÓA NSGA-II

### 4.1. Quy trình thuật toán
1. **Khởi tạo:** Sinh ngẫu nhiên quần thể gồm $N = 10$ cá thể hợp lệ.
2. **Đánh giá thích nghi (Fitness Evaluation):**
   * Huấn luyện mỗi kiến trúc 20 epoch (Adam, lr=0.001, StepLR).
   * Đo lường $f_1 = \text{Val Macro AUC}$ và $f_2 = \text{Parameter Count}$.
   * **Cơ chế Cache SHA-256:** Lưu băm kiến trúc chuẩn hóa; nếu cá thể trùng lặp xuất hiện ở các thế hệ sau thì tái sử dụng kết quả, không tiêu tốn ngân sách huấn luyện lại.
3. **Phân hạng không bị trội (Fast Non-dominated Sorting):** Phân chia quần thể thành các Pareto front $F_1, F_2, \dots$ dựa trên quan hệ trội Pareto:
   $$\mathbf{a} \succ \mathbf{b} \iff (f_1(\mathbf{a}) \ge f_1(\mathbf{b}) \land f_2(\mathbf{a}) \le f_2(\mathbf{b})) \land (f_1(\mathbf{a}) > f_1(\mathbf{b}) \lor f_2(\mathbf{a}) < f_2(\mathbf{b}))$$
4. **Khoảng cách đám đông (Crowding Distance):** Ưu tiên các cá thể ở vùng thưa thớt trên cùng một front nhằm duy trì độ đa dạng di truyền.
5. **Chọn lọc (Binary Tournament Selection):** So sánh theo hạng Pareto (ưu tiên hạng nhỏ hơn), nếu hòa xét theo khoảng cách đám đông (ưu tiên khoảng cách lớn hơn).
6. **Toán tử di truyền:**
   * **Lai ghép đều (Uniform Crossover):** Xác suất $p_c = 0.8$, hoán đổi gen giữa hai cha mẹ theo từng vị trí.
   * **Đột biến (Mutation):** Xác suất đột biến gen $p_m = 0.15$, thay thế gen bằng giá trị hợp lệ khác trong miền.
   * **Sửa chữa gen (Repair):** Đảm bảo kiến trúc con luôn thỏa mãn ràng buộc tính hợp lệ.
7. **Điều kiện dừng:** Dừng khi đạt tổng số 60 lượt huấn luyện thực tế.

---

## 5. THIẾT KẾ THỰC NGHIỆM ĐỐI SÁNH
Để chứng minh giá trị của giải thuật tiến hóa, dự án thiết lập 3 phương pháp so sánh công bằng:
1. **ResNet-18 Baseline:** Kiến trúc do chuyên gia thiết kế thủ công (11.17 triệu tham số), huấn luyện 60 epoch.
2. **GA-NAS (NSGA-II):** Tiến hóa quần thể 10 cá thể qua ngân sách 60 đánh giá.
3. **Random Search:** Lấy mẫu ngẫu nhiên 60 kiến trúc không lặp lại với cùng không gian tìm kiếm và cùng ngân sách đánh giá.
4. **Đánh giá cuối cùng trên Test Split (Stage 5):**
   * Tập test gồm **2.005 ảnh** được niêm phong tuyệt đối trong suốt quá trình tìm kiếm.
   * Lấy kiến trúc có AUC cao nhất trên Pareto front của GA và Random, huấn luyện đầy đủ 60 epoch rồi đánh giá trên tập test.

---

## 6. KẾT QUẢ THỰC NGHIỆM & PHÂN TÍCH TIẾN HÓA

### 6.1. Bảng so sánh kết quả trên tập kiểm thử (Test Split)

| Mô hình | Tham số (Params) | Test AUC (OvR) | Test Accuracy | Test Macro F1 | Test Loss | Thời gian train |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 Baseline** | 11,172,423 | 0.9374 | 77.71% | 0.5519 | 0.5942 | 590s |
| **Random Search Best** | 5,820,743 | **0.9401** | **0.7910%** | 0.5866 | 0.5984 | 436s |
| **GA-NAS Best (NSGA-II)** | **2,768,199** | 0.9364 | 78.80% | **0.5919** | 0.6109 | **392s** |
| **GA-NAS Smallest (Pareto)**| **537,287** | 0.9398* | 77.2%* | 0.56* | - | 195s |

*(Ghi chú: Mô hình GA Smallest đạt AUC ~0.94 trên validation với kích thước chỉ bằng 4.8% mô hình baseline).*

---

### 6.2. Các chỉ số đo lường chuẩn môn Tính toán tiến hóa

#### a) Độ bao phủ siêu thể tích (Hypervolume - HV)
* **GA-NAS:** $\mathbf{0.9403}$
* **Random Search:** $0.9368$
* *Ý nghĩa:* GA-NAS bao phủ không gian mục tiêu tốt hơn đáng kể so với Random Search.

#### b) Chỉ số độ bao phủ Pareto (Coverage Metric $C$)
* $C(\text{GA}, \text{Random}) = \mathbf{50.0\%}$: 50% nghiệm trên Pareto front của Random Search bị các nghiệm của GA-NAS thống trị (trội hơn cả về AUC lẫn số tham số).
* $C(\text{Random}, \text{GA}) = \mathbf{0.0\%}$: Random Search không thể thống trị bất kỳ nghiệm nào thuộc Pareto front của GA-NAS.

#### c) Chỉ số phân bố đều Schott (Schott's Spacing Metric $S$)
* **GA-NAS:** $\mathbf{0.2233}$
* **Random Search:** $0.3697$
* *Ý nghĩa:* Giá trị $S$ của GA-NAS nhỏ hơn 39.6% so với Random Search, chứng minh toán tử **Crowding Distance** của NSGA-II đã phân bố các giải pháp trải đều liên tục trên biên Pareto, không bị tụ cụm như tìm kiếm ngẫu nhiên.

#### d) Kiểm định thống kê phi tham số (Mann-Whitney U Test)
* Giá trị $U$-statistic: $1956.0$, $p$-value: $0.207$.
* Trung bình AUC của GA ($0.9389 \pm 0.0032$) cao hơn Random Search ($0.9383 \pm 0.0033$).

---

### 6.3. Phân tích sự sống sót của các gen (Gene Survival Analysis)
Qua phân tích các cá thể trụ lại trên Pareto front:
1. **Ưu tiên kích thước Kernel:** Cả GA và Random đều có xu hướng chuộng **Kernel $3 \times 3$** (chiếm tỷ lệ áp đảo so với $5 \times 5$) vì kernel nhỏ giảm mạnh lượng tham số mà vẫn trích xuất tốt đặc trưng cục bộ của vết bầm/nốt ruồi.
2. **Phân bố khối Block:** GA-NAS có xu hướng dồn số block dày ở Stage 1 và Stage 2 (học kỹ đường nét da cơ bản) và giảm số block ở Stage 3-4 để tiết kiệm tham số.
3. **Hiệu ứng nén mô hình:** GA tìm ra cấu hình $[1, 1, 2, 1]$ với channels $[64, 64, 64, 128]$ chỉ có **537K tham số** (giảm **95.2%** kích thước so với Baseline 11.17M) mà vẫn đạt AUC tương đương.

---

## 7. PHÂN CÔNG NHIỆM VỤ CHO NHÓM 3 NGƯỜI

| Thành viên | Trách nhiệm chính | File phụ trách |
| :--- | :--- | :--- |
| **Thành viên 1** | **Xây dựng Không gian tìm kiếm & Toán tử Di truyền:**<br>- Thiết kế mã hóa chromosome (`search_space.py`).<br>- Cài đặt và kiểm thử các toán tử Crossover, Mutation, Repair.<br>- Viết Unit Tests cho không gian tìm kiếm. | `src/search_space.py`<br>`src/model.py`<br>`tests/test_search_space.py` |
| **Thành viên 2** | **Bộ máy NSGA-II & Tích hợp Pipeline Huấn luyện:**<br>- Triển khai thuật toán NSGA-II, Non-dominated sorting, Crowding distance.<br>- Tích hợp cơ chế Cache SHA-256 và quản lý ngân sách đánh giá (budget).<br>- Thực hiện chạy thực nghiệm GA-NAS và Random Search. | `src/nsga2.py`<br>`src/ga_search.py`<br>`src/random_search.py`<br>`src/search_runtime.py` |
| **Thành viên 3** | **Đánh giá Đa mục tiêu, Thống kê & Phân tích Động học:**<br>- Cài đặt các chỉ số Spacing ($S$), Hypervolume, Coverage.<br>- Kiểm định thống kê giả thuyết (Mann-Whitney U test).<br>- Phân tích tần suất gen sống sót và vẽ biểu đồ trực quan.<br>- Tổng hợp báo cáo kỹ thuật. | `src/evolution_stats.py`<br>`src/analyze.py`<br>`src/test_eval.py`<br>`tests/test_evolution_stats.py` |

---

## 8. HƯỚNG DẪN TÁI LẬP & KIỂM THỬ (REPRODUCTION)

### 8.1. Kiểm thử Unit Tests tự động
Để chạy toàn bộ 14 bài kiểm thử đơn vị:
```powershell
Set-Location C:\Users\pinkc\2026.1\ga_cnn_optimize
$env:PYTHONPATH="."
python -m unittest discover -s tests -v
```

### 8.2. Chạy tái tạo phân tích và xuất đồ thị
```powershell
$env:PYTHONPATH="."
python -m src.analyze
```
Kết quả biểu đồ được tạo tại `results/visualizations/`:
* `pareto_front.png`: Đồ thị biên Pareto AUC vs Số tham số.
* `convergence.png`: Đường hội tụ qua 60 thế hệ đánh giá.
* `gene_distribution.png`: Biểu đồ phân tích tần suất gen sống sót.
* `test_metrics.png`: Biểu đồ cột so sánh các chỉ số trên tập kiểm thử.
* `confusion_matrices.png`: Ma trận nhầm lẫn 7 lớp tổn thương da.
* `test_metrics.csv` & `analysis.json`: Bảng dữ liệu thống kê chi tiết.
