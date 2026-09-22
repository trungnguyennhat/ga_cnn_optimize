# Tối ưu siêu tham số CNN bằng Genetic Algorithm trên DermaMNIST

## Mục tiêu

Xây dựng Genetic Algorithm (GA) để tối ưu siêu tham số CNN phân loại tổn thương da trên DermaMNIST 64×64, sau đó so sánh với CNN mặc định và Random Search trong cùng ngân sách tính toán.

GA không bắt buộc phải thắng Random Search; kết luận phải dựa trên kết quả nhiều seed và chi phí tìm kiếm.

## Dataset

- Nguồn: DermaMNIST thuộc MedMNIST, dựa trên HAM10000.
- File local: `data/dermamnist_64.npz`.
- Ảnh RGB `64×64`, bài toán multi-class gồm 7 lớp.
- Split chính thức: 7.007 train, 1.003 validation, 2.005 test.
- Giấy phép: CC BY-NC 4.0.
- Code chỉ đọc file local; không tự tải dữ liệu khi train hoặc search.

| Nhãn | Lớp |
|---:|---|
| 0 | Actinic keratoses and intraepithelial carcinoma |
| 1 | Basal cell carcinoma |
| 2 | Benign keratosis-like lesions |
| 3 | Dermatofibroma |
| 4 | Melanoma |
| 5 | Melanocytic nevi |
| 6 | Vascular lesions |

## CNN baseline

CNN gồm ba khối `Conv2D → ReLU → MaxPool`, dropout và classifier 7 lớp. Loss là weighted cross-entropy với trọng số tính từ train labels.

```yaml
learning_rate: 0.001
batch_size: 64
dropout: 0.3
optimizer: Adam
filters: 32
epochs: 15
seed: 42
device: auto
```

## Không gian tìm kiếm

| Siêu tham số | Giá trị |
|---|---|
| Learning rate | `1e-4` đến `1e-2`, log-scale |
| Batch size | `32, 64, 128` |
| Dropout | `0.1, 0.2, 0.3, 0.4, 0.5` |
| Optimizer | `Adam, AdamW` |
| Filters lớp đầu | `16, 32, 64` |

Chromosome: `(learning_rate, batch_size, dropout, optimizer, filters)`.

## Genetic Algorithm

```yaml
population_size: 10
evaluation_budget: 80
tournament_size: 3
crossover: uniform
crossover_rate: 0.8
mutation_probability_per_gene: 0.15
elitism: 1
fitness_epochs: 5
fitness: validation_macro_auc_ovr
```

- Dừng theo số cấu hình thực sự được train; cache hit không tăng budget.
- GA và Random Search dùng cùng search space, split, epoch và evaluation budget.
- Mỗi phương pháp chạy tối thiểu 3 independent seeds.
- Test set không được dùng trong baseline, search hoặc chọn cấu hình.

## Metrics và đánh giá cuối

- Fitness chính: macro ROC AUC one-vs-rest trên validation.
- Metrics bổ sung: accuracy, macro F1, loss, runtime và confusion matrix.
- Lấy top 3 cấu hình của mỗi phương pháp theo validation macro AUC.
- Train lại tối đa 25 epoch với seed `1, 2, 3` và early stopping patience 5.
- Chỉ sau khi chọn xong cấu hình mới đánh giá test và báo cáo mean ± standard deviation.

## Cấu trúc dự kiến

```text
project/
├── data/dermamnist_64.npz
├── model.py
├── train.py
├── search.py
├── run_experiment.py
├── analyze.py
├── requirements.txt
└── README.md
```

Tiến độ và quy trình làm việc được quy định trong `CHECKPOINTS.md`, `AGENTS.md` và `GUIDE.md`.
