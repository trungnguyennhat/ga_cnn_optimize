# Tìm kiếm kiến trúc CNN đa mục tiêu bằng Genetic Algorithm trên DermaMNIST

## Mục tiêu

Xây dựng Neural Architecture Search dùng NSGA-II để đồng thời tối đa hóa validation macro ROC AUC và tối thiểu hóa số tham số của CNN phân loại tổn thương da trên DermaMNIST 64×64.

So sánh CNN baseline thiết kế thủ công, Random Architecture Search và GA-NAS trong cùng không gian kiến trúc, split, số epoch và ngân sách đánh giá. GA không bắt buộc phải thắng Random Search; kết luận dựa trên nhiều seed, Pareto front và chi phí tìm kiếm.

## Dataset

- File duy nhất: `data/dermamnist_64.npz`.
- Ảnh RGB `64×64`, 7 lớp.
- Split chính thức: 7.007 train, 1.003 validation, 2.005 test.
- Code chỉ đọc file local, không tự tải dữ liệu khi train hoặc search.
- Test split không được truy cập trước đánh giá cuối.

## CNN baseline

Baseline gồm ba block `Conv2D → ReLU → MaxPool`, filters `32, 64, 128`, dropout `0.3` và classifier 7 lớp. Huấn luyện bằng Adam, learning rate `0.001`, batch size `64`, weighted cross-entropy, 15 epoch và seed `42`.

Kết quả baseline hiện tại được giữ tại `results/baseline/seed_42.json`.

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
fitness_epochs: 5
objectives:
  - maximize: validation_macro_auc_ovr
  - minimize: parameter_count
```

- NSGA-II dùng nondominated sorting và crowding distance, không gộp hai mục tiêu bằng hệ số phạt.
- Cache dùng canonical architecture và seed; cache hit không tăng evaluation budget.
- GA-NAS và Random Architecture Search dùng cùng search space và ngân sách 80 kiến trúc thực sự được train.
- Mỗi phương pháp chạy với seed `1, 2, 3`.

## Đánh giá cuối

Từ Pareto front của mỗi phương pháp, chọn kiến trúc AUC cao nhất, nhỏ nhất và knee point. Train lại tối đa 25 epoch với seed `1, 2, 3`, early stopping patience 5, sau đó mới đánh giá test.

Báo cáo mean ± standard deviation của loss, accuracy, macro F1, macro AUC, số tham số và runtime; kèm confusion matrix, đường hội tụ, độ đa dạng và Pareto AUC–model size.

Tiến độ và lệnh chạy được quy định trong `CHECKPOINTS.md`, `AGENTS.md` và `GUIDE.md`.
