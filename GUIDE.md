# Hướng dẫn chạy dự án

AI agent cập nhật phần tương ứng sau mỗi stage nhưng không tự chạy huấn luyện hoặc kiểm thử.

> **Thiết kế hiện tại từ 2026-10-04:** dự án bắt đầu lại với ResNet-18. Stage 1–6 đã có code và lệnh ResNet thực tế.

## Kế hoạch GA-ResNet mới

- Protocol baseline/final: seed `42`, 60 epoch và `StepLR(step_size=20, gamma=0.1)`. Protocol proxy bắt buộc cho GA/Random Search: seed `42`, 20 epoch và `StepLR(step_size=15, gamma=0.1)`. Cả hai dùng Adam, learning rate đầu `0.001`, batch size `64`, unweighted cross-entropy, normalize mean/std `0.5/0.5` và checkpoint validation macro AUC tốt nhất.
- Baseline mới: ResNet-18 BasicBlock train from scratch, stem `3×3` stride 1, không max-pooling đầu mạng, output 7 lớp.
- GA/Random tìm kiếm `stage_blocks`, `stage_channels`, `kernel_sizes`, `dropout`; các chi tiết search space nằm trong `README.md`.
- Population 10, budget 60, tournament 3, crossover 0.8 và mutation 0.15.
- Output path giữ nguyên, nhưng không được dùng chung cache CNN cũ và ResNet mới.

Trước khi chạy ResNet lần đầu, người dùng sẽ xóa chính xác các output CNN cũ sau. Chỉ chạy các lệnh này sau khi code ResNet của stage tương ứng đã sẵn sàng:

```powershell
Remove-Item -LiteralPath "results\baseline\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "results\ga_search\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "results\random_search\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath "results\test_eval" -Recurse -Force -ErrorAction SilentlyContinue
```

Các file/thư mục ResNet mới sẽ được tạo lại đúng tại `results/baseline/seed_42/`, `results/ga_search/seed_42/`, `results/random_search/seed_42/` và `results/test_eval/`.

Toàn bộ code Python nằm trong `src/`. Chạy các lệnh bên dưới từ thư mục gốc dự án bằng `python -m src.<module>`. Đường dẫn dataset và thư mục kết quả mặc định được xác định theo vị trí code: `data/dermamnist_64.npz` và `results/baseline/` ở thư mục gốc. Đường dẫn tương đối truyền qua `--data-path` hoặc `--output-dir` được tính từ thư mục đang chạy lệnh.

## 1. Chuẩn bị môi trường

```powershell
cd D:\Downloads\code_D_disk\ga_cnn_optimize
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Không phát hiện')"
nvidia-smi
```

## 2. Tải DermaMNIST 64×64

Nguồn chính thức là Zenodo. Lệnh này chỉ cần chạy một lần:

```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1" -OutFile "data\dermamnist_64.npz"
(Get-FileHash -Algorithm MD5 "data\dermamnist_64.npz").Hash.ToLower()
```

MD5 mong đợi: `b70a2f5635c6199aeaa28c31d7202e1f`.

Phương án dự phòng bằng MedMNIST API:

```powershell
.\.venv\Scripts\python.exe -c "from medmnist import DermaMNIST; DermaMNIST(split='train', root='data', size=64, download=True)"
```

Kiểm tra arrays và kích thước:

```powershell
.\.venv\Scripts\python.exe -c "import numpy as np; d=np.load(r'data\dermamnist_64.npz'); print({k:d[k].shape for k in d.files})"
```

Kết quả phải có 6 arrays với kích thước ảnh train/validation/test lần lượt là `(7007, 64, 64, 3)`, `(1003, 64, 64, 3)` và `(2005, 64, 64, 3)`.

## 3. Stage 1 — ResNet-18 baseline và model builder

Xóa riêng kết quả baseline CNN cũ trước lần chạy ResNet-18 đầu tiên:

```powershell
Remove-Item -LiteralPath "results\baseline\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
```

Chạy ResNet-18 baseline với cấu hình chính thức:

```powershell
.\.venv\Scripts\python.exe -m src.train --learning-rate 0.001 --batch-size 64 --optimizer Adam --epochs 60 --seed 42 --device auto --output-dir "results\baseline"
```

`--device` nhận `auto`, `cpu`, `cuda` hoặc GPU cụ thể như `cuda:0`. Ảnh được normalize từ `[0,1]` sang `[-1,1]`; loss là cross-entropy không class weight. Learning rate là `0.001` ở epoch 1–20, `0.0001` ở 21–40 và `0.00001` ở 41–60. Validation được tính mỗi epoch và mô hình khôi phục checkpoint có macro AUC tốt nhất. Loader chỉ đọc train/validation; Stage 1 không đọc test.

Output gồm `results/baseline/seed_42/result.json` và `results/baseline/seed_42/checkpoint.pt`. JSON chứa chromosome, parameter count, lịch sử train/validation/lr từng epoch, best epoch, metrics và runtime; file `.pt` chứa trọng số checkpoint tốt nhất.

Kiểm tra model builder bằng một ResNet candidate:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from src.model import build_model, count_parameters; a={'stage_blocks':[1,2,1,2],'stage_channels':[16,32,64,128],'kernel_sizes':[3,5,3,5],'dropout':0.2}; m=build_model(a); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape)); print('Parameters:', count_parameters(m))"
```

Kiểm tra ResNet-18 baseline:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from src.model import BASELINE_ARCHITECTURE, build_model, count_parameters; m=build_model(BASELINE_ARCHITECTURE); print(BASELINE_ARCHITECTURE); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape)); print('Parameters:', count_parameters(m))"
```

Kiểm tra JSON sau khi train:

```powershell
$baseline = Get-Content -Raw "results\baseline\seed_42\result.json" | ConvertFrom-Json
$baseline | Select-Object experiment, architecture, parameter_count, config, metrics, runtime_seconds
```

Stage 1 đạt yêu cầu khi hai model builder đều tạo output `(2, 7)`, parameter count dương, baseline train đủ 60 epoch, learning rate đổi đúng ở epoch 21 và 41, có `seed_42/checkpoint.pt`, và JSON ghi đúng architecture cùng `best_epoch`. Chờ người dùng xác nhận trước khi chuyển Stage 2.

## 4. Stage 2 — ResNet search space và NSGA-II

Chromosome gồm `stage_blocks`, `stage_channels`, `kernel_sizes`, `dropout`. Mỗi gene danh sách có đúng bốn phần tử tương ứng bốn stage; crossover chọn từng phần tử từ một trong hai bố mẹ và mutation thay từng phần tử độc lập với xác suất `0.15`. NSGA-II ở stage này chỉ dùng fitness giả lập, không đọc dataset và không train model.

Kiểm tra thủ công search space và NSGA-II với budget nhỏ:

```powershell
.\.venv\Scripts\python.exe -c "import random; from src.search_space import random_architecture, validate_architecture, uniform_crossover, mutate_architecture; r=random.Random(42); a=random_architecture(r); b=random_architecture(r); c=mutate_architecture(uniform_crossover(a,b,r),r); validate_architecture(c); print(a); print(b); print(c)"
.\.venv\Scripts\python.exe -c "from src.nsga2 import run_nsga2; from src.search_space import canonical_architecture; f=lambda a,s:(sum(a['stage_blocks'])/12, sum(a['stage_channels'])); x=run_nsga2(f,evaluation_budget=20,seed=42); print('evaluations:',x['evaluations']); print('generations:',x['generations']); print('pareto:',len(x['pareto_front'])); print('unique:',len({canonical_architecture(i.architecture) for i in x['evaluated']}))"
```

Kết quả hợp lệ khi mọi chromosome có đúng bốn gene đã chốt, không có kiến trúc trùng trong `evaluated`, và cả `evaluations` lẫn `unique` đều bằng `20`. Chờ người dùng xác nhận trước khi chuyển Stage 3.

## 5. Stage 3 — Tích hợp NAS với ResNet

GA dùng proxy training 20 epoch, seed 42, input chuẩn hóa về `[-1, 1]`, unweighted cross-entropy, Adam với learning rate ban đầu `0.001`, `StepLR(step_size=15, gamma=0.1)`, và lấy fitness tại checkpoint có validation macro AUC tốt nhất. Xóa toàn bộ cache GA của cấu hình cũ trước khi chạy:

```powershell
Remove-Item -LiteralPath "results\ga_search\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
```

Chạy GA-ResNet:

```powershell
.\.venv\Scripts\python.exe -m src.ga_search --epochs 20 --seed 42 --budget 60 --population-size 10 --device auto --output-dir "results\ga_search"
```

Lệnh dùng NSGA-II với population 10, budget 60 kiến trúc khác nhau, tournament 3, crossover 0.8 và mutation 0.15. Mỗi ResNet train 20 epoch theo proxy protocol; đây là điểm dùng để xếp hạng kiến trúc, không phải kết quả cuối để so trực tiếp với baseline 60 epoch. Chương trình chỉ đọc train/validation chính thức từ `data/dermamnist_64.npz`, không tải dữ liệu và không đọc test arrays.

Console in training loss mỗi epoch, AUC validation và số tham số thực tế mỗi evaluation; cuối cùng in Pareto front và đường dẫn kết quả. Với lệnh trên, output baseline nằm tại `results/baseline/seed_42/result.json`; output GA nằm trong `results/ga_search/seed_42/`:

- `evaluations/<cache_key>.json`: chromosome, cấu hình, metrics ở best validation epoch, lịch sử train/validation/lr đủ 20 epoch, số tham số và runtime của mỗi ResNet; ghi ngay sau khi train thành công.
- `progress.json`: thứ tự evaluation, đường dẫn kết quả, số lần train mới và cache hit, cập nhật sau từng evaluation.
- `summary.json`: toàn bộ fitness đã đánh giá, Pareto front, số generation và runtime của lần chạy.

Số epoch và toàn bộ giao thức train được ghi vào config và khóa cache. Cache Stage 3 dùng `architecture_space: resnet_stage_channels_v1` và `training_version: 3`, nên không thể dùng evaluation từ protocol 60/20 cũ. Cache còn kiểm tra canonical architecture, seed, thiết bị và SHA-256 tensor train/validation. Không thay đổi dataset hoặc code train giữa các lần tiếp tục; nếu thay đổi luồng train phải tăng `training_version` trong `src/search_runtime.py`.

Chạy lại đúng lệnh sẽ dựng lại tiến trình search từ seed và dùng các evaluation đã train; kiến trúc trùng không train lại. Budget 60 là tổng số kiến trúc khác nhau có kết quả train trong search đó, bao gồm kết quả đã train ở lần trước; cache hit không thêm một lần train hay tiêu thêm budget. `new_trainings` chỉ đếm train mới trong lần gọi hiện tại; lần đầu không có cache phải có `evaluations: 60`, `new_trainings: 60`, `cache_hits: 0`. Khi chạy lại search đã hoàn tất: `new_trainings: 0`, `cache_hits: 60`. Khi chạy bị ngắt, chạy lại cùng lệnh để dùng các kết quả đã lưu; ResNet đang train dở sẽ train lại từ đầu.

Có thể chọn `--device cpu`, `--device cuda`, `--device cuda:0`, thay `--output-dir`, hoặc dùng `--budget` và `--population-size` cho một lần kiểm tra thủ công nhỏ. Kết quả search chính thức phải dùng đúng lệnh budget 60 và 20 epoch ở trên. Stage 3 chưa hoàn thành cho đến khi người dùng chạy và xác nhận.

## 6. Stage 4 — Random ResNet Search

Random Architecture Search phải dùng đúng bộ sinh `random_architecture` của GA-NAS và không đánh giá lặp kiến trúc trong cùng lần chạy. Mỗi kiến trúc phải dùng cùng dataset, seed 42 và proxy protocol 20 epoch/StepLR 15 giống GA. Hai phương pháp dùng cùng budget 60:

```powershell
Remove-Item -LiteralPath "results\random_search\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m src.random_search --epochs 20 --seed 42 --budget 60 --device auto --output-dir "results\random_search"
```

Kết quả GA đã có trong `results/ga_search/seed_42/` được giữ nguyên; không cần chạy lại GA. Lệnh đầu chỉ xóa output Random Search cũ, sau đó Random Search lưu kết quả mới vào `results/random_search/seed_42/`.

- `evaluations/<cache_key>.json`: kiến trúc, toàn bộ validation metrics, lịch sử train loss, số tham số và runtime ResNet.
- `progress.json`: trạng thái train/cache cùng `best_auc_so_far`, `smallest_parameters_so_far` và `pareto_size_so_far` sau từng evaluation.
- `summary.json`: cấu hình tái lập, runtime toàn search, toàn bộ kiến trúc kèm Pareto rank/crowding distance và Pareto front cuối. `crowding_distance: null` biểu thị cá thể biên có crowding vô hạn.

Random Search bỏ kiến trúc trùng và tiếp tục lấy mẫu đến đủ 60 kiến trúc duy nhất. Chạy lại đúng lệnh sẽ tái dựng đúng thứ tự từ seed và đọc các evaluation đã lưu; kết quả train dở chưa có JSON hoàn chỉnh sẽ được train lại. Hai phương pháp không đọc test split.

Kiểm tra thủ công sau khi mỗi lệnh kết thúc:

```powershell
$ga = Get-Content -Raw "results\ga_search\seed_42\summary.json" | ConvertFrom-Json
$random = Get-Content -Raw "results\random_search\seed_42\summary.json" | ConvertFrom-Json
$ga | Select-Object experiment, evaluations, new_trainings, cache_hits, runtime_seconds
$random | Select-Object experiment, evaluations, new_trainings, cache_hits, runtime_seconds
$ga.pareto_front
$random.pareto_front
```

Mỗi summary phải có `evaluations: 60`. GA và Random Search seed 42 đã hoàn tất cùng budget/cấu hình, log đủ để tái lập và Pareto front chỉ được tạo từ validation. Người dùng đã xác nhận chuyển sang Stage 5 ngày 2026-10-06.

## 7. Stage 5 — Chọn ResNet và đánh giá cuối

Stage 5 chọn kiến trúc có validation macro AUC cao nhất từ Pareto front của GA và Random Search. Hai kiến trúc này được train lại từ đầu với seed `42`, đủ 60 epoch theo đúng giao thức Stage 1, rồi khôi phục checkpoint có validation macro AUC tốt nhất. ResNet-18 baseline không train lại mà dùng `results/baseline/seed_42/checkpoint.pt`, vì checkpoint này đã được train bằng cùng seed và protocol. Kiến trúc được chọn hoàn toàn bằng validation; test chỉ được đọc sau khi lựa chọn đã cố định.

Nếu lần chạy bị dừng sau khi một mô hình đã có đủ `checkpoint.pt` và `result.json`, chạy lại cùng lệnh sẽ tái sử dụng kết quả đó thay vì train lại. Chỉ dùng lệnh xóa dưới đây khi muốn chủ động train lại cả GA và Random từ đầu.

Xóa output cũ chỉ khi muốn chủ động train lại cả hai mô hình từ đầu:

```powershell
Remove-Item -LiteralPath "results\test_eval" -Recurse -Force -ErrorAction SilentlyContinue
```

Chạy hoặc tiếp tục đánh giá cuối:

```powershell
.\.venv\Scripts\python.exe -m src.test_eval --seed 42 --epochs 60 --device auto --output-dir "results\test_eval"
```

Chương trình phải yêu cầu hai file `results/ga_search/seed_42/summary.json` và `results/random_search/seed_42/summary.json` đều có đủ 60 evaluation. Output gồm:

- `results/test_eval/<ga|random>/seed_42/checkpoint.pt`: trọng số sau khi train lại, ở epoch có validation macro AUC tốt nhất.
- `results/test_eval/baseline/seed_42/checkpoint.pt`: bản sao checkpoint baseline hiện có, không train lại.
- `results/test_eval/<ga|random|baseline>/seed_42/result.json`: kiến trúc, quy tắc chọn, lịch sử train, validation/test metrics, nhãn thật và dự đoán test.
- `results/test_eval/summary.json`: bảng kết quả gọn của cả ba mô hình.

Kiểm tra thủ công sau khi lệnh hoàn tất:

```powershell
$final = Get-Content -Raw "results\test_eval\summary.json" | ConvertFrom-Json
$final | Select-Object experiment, seed, selection_uses, test_uses
$final.results | Select-Object model, parameter_count, best_epoch, validation_metrics, test_metrics
Get-ChildItem "results\test_eval" -Recurse -File
```

Kết quả hợp lệ khi có đúng ba model `ga`, `random`, `baseline`; mỗi model có `checkpoint.pt` và `result.json`; summary ghi `selection_uses: validation_only`, `test_uses: test_evaluation_only`. Stage 5 chưa được coi là hoàn thành cho đến khi bạn chạy và xác nhận.

## 8. Stage 6 — Phân tích GA-ResNet

Stage 6 chỉ đọc kết quả đã có, không train hoặc chạy inference lại. Chạy:

```powershell
.\.venv\Scripts\python.exe -m src.analyze --output-dir "results\visualizations"
```

Output gồm:

- `analysis.json`: Hypervolume, Coverage hai chiều, knee point, model nhỏ nhất đạt AUC baseline, runtime và dữ liệu hội tụ.
- `test_metrics.csv`: loss, accuracy, macro precision/recall/F1/AUC, số tham số và runtime của GA, Random, baseline.
- `pareto_front.png`: toàn bộ điểm search, Pareto front và baseline.
- `convergence.png`: best AUC, Hypervolume và model nhỏ nhất đạt AUC baseline theo evaluation.
- `test_metrics.png`: so sánh các metrics test.
- `confusion_matrices.png`: confusion matrix test của ba mô hình.

Kiểm tra thủ công:

```powershell
$analysis = Get-Content -Raw "results\visualizations\analysis.json" | ConvertFrom-Json
$analysis.normalization
$analysis.coverage
$analysis.search | Format-List
Import-Csv "results\visualizations\test_metrics.csv" | Format-Table
Get-ChildItem "results\visualizations" -File
```

Stage 6 đạt yêu cầu khi có đủ sáu file trên, Hypervolume của hai phương pháp dùng cùng normalization/reference point và Coverage được tính trên Pareto front cuối.

## 9. Stage 7 — Hoàn thiện và bàn giao

Stage 7 sẽ rà soát toàn bộ dependency, lệnh, input, output và tính tái lập.
