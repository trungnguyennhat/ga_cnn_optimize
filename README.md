# Tìm kiếm kiến trúc CNN đa mục tiêu bằng Genetic Algorithm trên DermaMNIST

## Mục tiêu

Xây dựng Neural Architecture Search dùng NSGA-II để đồng thời tối đa hóa validation macro ROC AUC và tối thiểu hóa số tham số của CNN phân loại tổn thương da trên DermaMNIST 64×64.

So sánh CNN baseline thiết kế thủ công, Random Architecture Search và GA-NAS trong cùng không gian kiến trúc, split, số epoch, seed và ngân sách đánh giá. GA không bắt buộc phải thắng Random Search; kết luận dựa trên Pareto front, chi phí tìm kiếm và đánh giá cuối nhiều seed.

## Dataset

- File duy nhất: `data/dermamnist_64.npz`.
- Ảnh RGB `64×64`, 7 lớp.
- Split chính thức: 7.007 train, 1.003 validation, 2.005 test.
- Code chỉ đọc file local, không tự tải dữ liệu khi train hoặc search.
- Test split không được truy cập trước đánh giá cuối.

## CNN baseline

Baseline gồm ba block `Conv2D → ReLU → MaxPool`, filters `32, 64, 128`, dropout `0.3` và classifier 7 lớp. Huấn luyện bằng Adam, learning rate `0.001`, batch size `64`, weighted cross-entropy, 25 epoch và seed `42`.

Baseline và mỗi CNN trong GA mặc định train 25 epoch, dùng chung `DEFAULT_EPOCHS` trong `src/train.py`; các lệnh baseline, GA và Random Search đều hỗ trợ `--epochs`.

Kết quả baseline hiện tại được giữ tại `results/baseline/seed_42.json`.

Code Python nằm trong `src/`: `train.py` huấn luyện baseline và cung cấp luồng train dùng chung, `model.py` xây dựng CNN, `search_space.py` chứa chromosome và toán tử, `nsga2.py` triển khai NSGA-II, `search_runtime.py` dùng chung việc train/cache/log, `ga_search.py` chạy GA-NAS và `random_search.py` chạy Random Architecture Search. Lệnh PowerShell đầy đủ nằm trong `GUIDE.md`.

## Không gian kiến trúc

Chromosome có 2–4 convolution block và một dropout toàn mạng:

```json
{
  "blocks": [
    {"filters": 32, "kernel_size": 3, "pooling": "max", "batch_norm": false},
    {"filters": 64, "kernel_size": 5, "pooling": "avg", "batch_norm": true}
  ],
  "dropout": 0.3
}
```

| Gene | Giá trị |
|---|---|
| Số block | `2, 3, 4` |
| Filters mỗi block | `16, 32, 64, 128` |
| Kernel size | `3, 5` |
| Pooling | `max, avg` |
| Batch normalization | `true, false` |
| Dropout | `0.1, 0.2, 0.3, 0.4, 0.5` |

Activation cố định là ReLU. Mỗi block có pooling kích thước 2; classifier là `Flatten → Dropout → Linear(7)`.

## NSGA-II và ngân sách

```yaml
population_size: 10
evaluation_budget: 80
tournament_size: 3
crossover: uniform_by_block
crossover_rate: 0.8
mutation_probability: 0.15
fitness_epochs: 25
objectives:
  - maximize: validation_macro_auc_ovr
  - minimize: parameter_count
```

- NSGA-II dùng nondominated sorting và crowding distance, không gộp hai mục tiêu bằng hệ số phạt.
- Cache dùng canonical architecture và seed; cache hit không tăng evaluation budget.
- GA-NAS và Random Architecture Search dùng cùng search space và ngân sách 80 kiến trúc thực sự được train.
- GA-NAS và Random Architecture Search cùng chạy một lần với seed `42` để so sánh trực tiếp.
- Mỗi evaluation được log cùng best AUC, số tham số nhỏ nhất và kích thước Pareto front tính đến thời điểm đó; summary cuối chứa Pareto rank và crowding distance.

## Đánh giá cuối

Từ Pareto front validation của mỗi phương pháp, chọn kiến trúc AUC cao nhất, knee point và model nhỏ nhất có AUC không thấp hơn baseline. Train lại tối đa 25 epoch với seed `1, 2, 3`, early stopping patience 5, lưu checkpoint, sau đó mới đánh giá test.

Báo cáo mean ± standard deviation của loss, accuracy, macro F1, macro AUC, số tham số và runtime. So sánh GA-NAS với Random Search bằng Hypervolume chung, Coverage hai chiều, đường hội tụ và Pareto AUC–model size có điểm baseline; kèm confusion matrix và độ ổn định của các model được chọn. Không dùng IGD vì không có Pareto front chuẩn đáng tin cậy.

Tiến độ và lệnh chạy được quy định trong `CHECKPOINTS.md`, `AGENTS.md` và `GUIDE.md`.
