# Hướng dẫn chạy dự án

AI agent cập nhật phần tương ứng sau mỗi stage nhưng không tự chạy huấn luyện hoặc kiểm thử.

Quy trình mới nằm ở [mục 10](#10-hướng-1--nas-ba-mục-tiêu-và-thực-nghiệm-đa-seed). Kết quả của quy trình cũ đã được lưu ở `results/archive_before_hardware_aware_2026-11-25/`. Các đường dẫn `results/baseline/`, `results/ga_search/`, `results/random_search/` và `results/test_eval/` ở các mục trước mô tả layout của lần chạy cũ.

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
.\.venv\Scripts\python.exe -c "from src.ga_search.nsga2 import run_nsga2; from src.search_space import canonical_architecture; f=lambda a,s:(sum(a['stage_blocks'])/12, sum(a['stage_channels'])); x=run_nsga2(f,evaluation_budget=20,seed=42); print('evaluations:',x['evaluations']); print('generations:',x['generations']); print('pareto:',len(x['pareto_front'])); print('unique:',len({canonical_architecture(i.architecture) for i in x['evaluated']}))"
```

Kết quả hợp lệ khi mọi chromosome có đúng bốn gene đã chốt, không có kiến trúc trùng trong `evaluated`, và cả `evaluations` lẫn `unique` đều bằng `20`. Chờ người dùng xác nhận trước khi chuyển Stage 3.

## 5. Stage 3 — Tích hợp NAS với ResNet

GA dùng proxy training 20 epoch, seed 42, input chuẩn hóa về `[-1, 1]`, unweighted cross-entropy, Adam với learning rate ban đầu `0.001`, `StepLR(step_size=15, gamma=0.1)`, và lấy fitness tại checkpoint có validation macro AUC tốt nhất. Xóa toàn bộ cache GA của cấu hình cũ trước khi chạy:

```powershell
Remove-Item -LiteralPath "results\ga_search\seed_42" -Recurse -Force -ErrorAction SilentlyContinue
```

Chạy GA-ResNet:

```powershell
.\.venv\Scripts\python.exe -m src.ga_search.run_search --epochs 20 --seed 42 --budget 60 --population-size 10 --device auto --output-dir "results\ga_search"
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
.\.venv\Scripts\python.exe -m src.random_search.run_search --epochs 20 --seed 42 --budget 60 --device auto --output-dir "results\random_search"
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

Stage 6 chỉ đọc kết quả đã có, không train hoặc chạy inference lại. Để phân tích lại run hai mục tiêu đã lưu, chạy:

```powershell
$archive = "results\archive_before_hardware_aware_2026-11-25"
.\.venv\Scripts\python.exe -m src.analyze --ga-summary "$archive\ga_search\seed_42\summary.json" --random-summary "$archive\random_search\seed_42\summary.json" --baseline-result "$archive\baseline\seed_42\result.json" --test-dir "$archive\test_eval" --output-dir "results\visualizations"
```

Output gồm:

- `analysis.json`: Hypervolume, Coverage hai chiều, knee point, model nhỏ nhất đạt AUC baseline, runtime và dữ liệu hội tụ.
- `test_metrics.csv`: loss, accuracy, macro precision/recall/F1/AUC, số tham số và runtime của GA, Random, baseline.
- `pareto_front.png`: toàn bộ điểm search, Pareto front và baseline.
- `convergence.png`: best AUC, Hypervolume và model nhỏ nhất đạt AUC baseline theo evaluation.
- `test_metrics.png`: so sánh các metrics test.
- `confusion_matrices.png`: confusion matrix test của ba mô hình.
- `generational_progression.png` hiển thị AUC và số tham số theo thế hệ, khoảng cách Hamming trung bình và bản đồ nhiệt độ đa dạng của 13 vị trí gene. Mục `ga_generations` trong `analysis.json` lưu ID cá thể sống sót và tần suất gene. Với 60 lượt và quần thể 10, kết quả có G0 đến G5.

Kiểm tra thủ công:

```powershell
$analysis = Get-Content -Raw "results\visualizations\analysis.json" | ConvertFrom-Json
$analysis.normalization
$analysis.coverage
$analysis.search | Format-List
Import-Csv "results\visualizations\test_metrics.csv" | Format-Table
Get-ChildItem "results\visualizations" -File
```

Stage 6 đạt yêu cầu khi có đủ các file trên, Hypervolume của hai phương pháp dùng cùng normalization/reference point và Coverage được tính trên Pareto front cuối.

## 9. Stage 7 — Hoàn thiện và bàn giao

Stage 7 sẽ rà soát toàn bộ dependency, lệnh, input, output và tính tái lập.

## 10. Hướng 1 — NAS ba mục tiêu và thực nghiệm đa seed

NSGA-II tối đa hóa validation macro AUC, tối thiểu hóa số tham số và MACs. Random Search lấy mẫu từ cùng search space và xếp hạng trên cùng ba mục tiêu. MACs được đếm cho một ảnh RGB `64×64`, gồm Conv2d (kể cả shortcut) và Linear. Cột `gflops_conv_linear` quy đổi một MAC thành hai FLOPs. Quy ước này loại trừ bias, BatchNorm, activation, pooling và các phép cộng residual. MACs đo số phép tính theo kiến trúc. Latency cần một phép đo riêng trên thiết bị đích.

Các lệnh sau chạy từ thư mục gốc dự án với `data/dermamnist_64.npz` đã có sẵn. Search dùng split train/validation chính thức và giữ protocol proxy 20 epoch, Adam, StepLR 15 của thực nghiệm trước.

Kiểm tra thủ công với hai seed, ba kiến trúc mỗi phương pháp và một epoch. Kết quả smoke test có thư mục riêng:

```powershell
.\.venv\Scripts\python.exe -m src.run_multiseed --seeds 42 43 --budget 3 --epochs 1 --population-size 2 --device auto --output-dir "results\hardware_aware_smoke"
Import-Csv "results\hardware_aware_smoke\analysis\search_metrics.csv" | Format-Table
```

Kiểm tra bảng có bốn dòng, mỗi summary ghi `evaluations: 3`, mỗi evaluation có `macs` dương và Hypervolume nằm trong `[0, 1]`. Trong `analysis.json`, MACs của ResNet-18 theo quy ước này phải bằng `2221673984`.

Chạy thực nghiệm chính với ba seed. Mỗi seed dùng budget 60 cho GA và Random, tổng cộng 360 lượt đánh giá kiến trúc:

```powershell
.\.venv\Scripts\python.exe -m src.run_multiseed --seeds 42 43 44 --budget 60 --epochs 20 --population-size 10 --device auto
```

Chạy lại cùng lệnh với cùng thiết bị để tiếp tục từ các evaluation đã lưu. Kết quả chuẩn ở `results/hardware_aware/`, với `ga_search/seed_<seed>/` và `random_search/seed_<seed>/` chứa cache, progress và summary. Cache ba mục tiêu dùng `training_version: 4`. Dùng `--output-dir` riêng cho budget hoặc protocol khác.

Nếu cần chạy riêng một phương pháp hoặc bổ sung một seed:

```powershell
.\.venv\Scripts\python.exe -m src.ga_search.run_search --hardware-aware --seed 44 --budget 60 --epochs 20 --population-size 10 --device auto
.\.venv\Scripts\python.exe -m src.random_search.run_search --hardware-aware --seed 44 --budget 60 --epochs 20 --device auto
```

Tổng hợp lại từ log validation. Có thể đối chiếu GA hai mục tiêu đã lưu bằng cách đếm lại MACs của toàn bộ kiến trúc cũ và chấm trên cùng không gian ba mục tiêu. Lệnh thứ hai yêu cầu protocol và thiết bị trong log seed 42 khớp với bản lưu trữ:

```powershell
.\.venv\Scripts\python.exe -m src.run_multiseed --phase analysis --seeds 42 43 44
.\.venv\Scripts\python.exe -m src.run_multiseed --phase analysis --seeds 42 43 44 --legacy-ga-summary "results\archive_before_hardware_aware_2026-11-25\ga_search\seed_42\summary.json"
```

Thư mục `results/hardware_aware/analysis/` có `search_metrics.csv`, `analysis.json`, `pareto_projections.png` và `hypervolume_convergence.png`. Báo cáo ghi Hypervolume 3D với cận chi phí cố định từ search space và điểm tham chiếu `(0, 0, 0)` sau chuẩn hóa. Bảng có Coverage hai chiều, AUC tốt nhất và chi phí của kiến trúc đó. JSON có tần suất kernel `5×5` ở từng stage trên Pareto front, thống kê `mean ± sample std` qua các seed và chênh lệch GA trừ Random theo cặp seed. Dải trên biểu đồ hội tụ là một độ lệch chuẩn mẫu mỗi phía. Đối chiếu với GA cũ chỉ có seed 42.

Mỗi seed GA có thêm `generational_progression_seed_<seed>.png` và mục `by_seed.<seed>.ga_search.ga_generations` trong JSON. Quần thể sống sót được tái dựng từ thứ tự đánh giá và đủ ba mục tiêu, kể cả MACs. Độ đa dạng tại mỗi vị trí gene là tỷ lệ cặp cá thể mang giá trị khác nhau. Trung bình 13 tỷ lệ là khoảng cách Hamming chuẩn hóa của quần thể.

Sau khi đã chốt seed, giao thức và tiêu chí chọn bằng validation, chạy đánh giá cuối bằng lệnh riêng. Mỗi phương pháp chọn kiến trúc theo validation AUC cao nhất, ưu tiên ít tham số khi hòa, rồi train lại 60 epoch. ResNet-18 được train theo cùng seed nếu chưa có checkpoint trong thư mục mới. Test split chỉ được đọc ở pha đánh giá cuối:

```powershell
.\.venv\Scripts\python.exe -m src.run_multiseed --phase test --seeds 42 43 44 --device auto
Import-Csv "results\hardware_aware\analysis\test_metrics.csv" | Format-Table
Get-Content -Raw "results\hardware_aware\analysis\test_analysis.json" | ConvertFrom-Json
```

Các file `test_metrics.csv` và `test_analysis.json` ghi kết quả test theo seed và thống kê gộp cho GA, Random và baseline, gồm test AUC, macro F1, Params, MACs và GFLOPs theo quy ước trên. Đây là thống kê mô tả cho các seed trên cùng split DermaMNIST.
