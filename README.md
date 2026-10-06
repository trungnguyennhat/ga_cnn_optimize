# Tối ưu kiến trúc ResNet đa mục tiêu bằng Genetic Algorithm trên DermaMNIST

## Mục tiêu

Xây dựng Neural Architecture Search dùng NSGA-II để đồng thời tối đa hóa validation macro ROC AUC và tối thiểu hóa số tham số của các biến thể ResNet phân loại tổn thương da trên DermaMNIST 64×64.

Baseline mới là ResNet-18 train from scratch. GA-NAS và Random Architecture Search cùng tìm trong một không gian ResNet, dùng chung split, số epoch, seed và ngân sách. Random Search được giữ để xác định cải thiện đến từ cơ chế tiến hóa của GA hay chỉ từ việc thử nhiều kiến trúc.

## Dataset

- File duy nhất: `data/dermamnist_64.npz`.
- Ảnh RGB `64×64`, 7 lớp.
- Split chính thức: 7.007 train, 1.003 validation, 2.005 test.
- Code chỉ đọc file local, không tự tải dữ liệu khi train hoặc search.
- Test split không được truy cập trước đánh giá cuối.

## ResNet-18 baseline

Baseline dùng BasicBlock và cấu hình stage `[2, 2, 2, 2]`, channel `[64, 128, 256, 512]`. Cho ảnh 64×64, stem dùng convolution `3×3`, stride 1 và không có max-pooling đầu mạng. Sau bốn residual stage là global average pooling và classifier 7 lớp.

Mọi mô hình train from scratch, không dùng pretrained weights. Baseline và final retrain dùng 60 epoch với `StepLR(step_size=20, gamma=0.1)`. GA và Random Search dùng proxy training 20 epoch với `StepLR(step_size=15, gamma=0.1)`. Các protocol cùng dùng Adam, learning rate đầu `0.001`, batch size `64`, cross-entropy không class weight, normalize từng kênh với mean/std `0.5/0.5`, chọn checkpoint bằng validation macro AUC và seed `42`.

Kết quả ResNet-18 baseline được ghi tại `results/baseline/seed_42/result.json` và checkpoint tại `results/baseline/seed_42/checkpoint.pt`.

`src/model.py` và `src/train.py` triển khai ResNet-18 baseline. `src/search_space.py` và `src/nsga2.py` triển khai search space cùng NSGA-II; `src/ga_search.py` nối NSGA-II với validation fitness thật của ResNet. `src/random_search.py` dùng cùng bộ sinh kiến trúc và validation fitness để làm đối chứng cùng budget.

## Không gian kiến trúc

Chromosome biểu diễn một ResNet bốn stage:

```json
{
  "stage_blocks": [2, 1, 3, 2],
  "stage_channels": [32, 64, 128, 256],
  "kernel_sizes": [3, 3, 5, 3],
  "dropout": 0.2
}
```

| Gene | Giá trị |
|---|---|
| Số BasicBlock mỗi stage | `1, 2, 3` |
| Channel stage 1 | `16, 32, 64` |
| Channel stage 2 | `32, 64, 128` |
| Channel stage 3 | `64, 128, 256` |
| Channel stage 4 | `128, 256, 512` |
| Kernel size mỗi stage | `3, 5` |
| Dropout trước classifier | `0.0, 0.1, 0.2, 0.3, 0.5` |

BasicBlock, BatchNorm, ReLU, bốn stage và residual shortcut được giữ cố định. ResNet-18 baseline tương ứng với block `[2,2,2,2]`, channel `[64,128,256,512]`, kernel `[3,3,3,3]`, dropout 0. GA chỉ tối ưu kiến trúc, không tối ưu optimizer hoặc learning rate.

## NSGA-II và ngân sách

```yaml
population_size: 10
evaluation_budget: 60
tournament_size: 3
crossover: uniform_by_gene
crossover_rate: 0.8
mutation_probability: 0.15
fitness_epochs: 20
fitness_scheduler_step: 15
objectives:
  - maximize: validation_macro_auc_ovr
  - minimize: parameter_count
```

- NSGA-II dùng nondominated sorting và crowding distance, không gộp hai mục tiêu bằng hệ số phạt.
- Cache dùng canonical architecture, seed, cấu hình train và phiên bản kiến trúc; cache CNN cũ không được tái sử dụng.
- GA-NAS và Random Architecture Search dùng cùng ResNet search space và ngân sách 60 kiến trúc thực sự được train.
- GA-NAS và Random Architecture Search cùng chạy một lần với seed `42` để so sánh trực tiếp.
- Mỗi evaluation được log cùng best AUC, số tham số nhỏ nhất và kích thước Pareto front tính đến thời điểm đó; summary cuối chứa Pareto rank và crowding distance.

## Đánh giá cuối

Từ Pareto front validation của mỗi phương pháp, chọn một ResNet có AUC cao nhất; đồng thời dùng ResNet-18 baseline. Train lại ba mô hình đủ 60 epoch với seed `42`, giữ normalization/loss nhưng dùng final scheduler step 20 và checkpoint validation tốt nhất, sau đó mới đánh giá test.

Báo cáo loss, accuracy, macro precision, macro recall, macro F1, macro AUC, số tham số và runtime. So sánh GA-NAS với Random Search bằng Hypervolume chung, Coverage hai chiều, đường hội tụ và Pareto AUC–model size có điểm baseline; kèm confusion matrix. Không dùng IGD vì không có Pareto front chuẩn đáng tin cậy.

Tiến độ và lệnh chạy được quy định trong `CHECKPOINTS.md`, `AGENTS.md` và `GUIDE.md`.

## Chính sách output khi chuyển sang ResNet

Mỗi phương pháp lưu kết quả theo thư mục seed. Baseline dùng `results/baseline/seed_42/`, tương ứng với cấu trúc của `results/ga_search/seed_42/` và `results/random_search/seed_42/`. Không để cache của CNN cũ và ResNet mới cùng tồn tại.
