# Hướng dẫn chạy dự án

AI agent cập nhật phần tương ứng sau mỗi stage nhưng không tự chạy huấn luyện hoặc kiểm thử.

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

## 3. Stage 1 — Baseline và model builder

Chạy baseline với cấu hình mặc định trong `README.md`:

```powershell
.\.venv\Scripts\python.exe -m src.train
```

Có thể thay đổi các siêu tham số và thiết bị từ dòng lệnh, ví dụ:

```powershell
.\.venv\Scripts\python.exe -m src.train --learning-rate 0.001 --batch-size 64 --dropout 0.3 --optimizer Adam --filters 32 --epochs 25 --seed 42 --device auto
```

`--device` nhận `auto`, `cpu`, `cuda` hoặc một GPU cụ thể như `cuda:0`. Có thể dùng file dữ liệu ở vị trí khác với `--data-path`, hoặc đổi thư mục kết quả bằng `--output-dir`.

Loader chỉ đọc `train_images`, `train_labels`, `val_images`, `val_labels`; kiểm tra kích thước chính thức và nhãn từ 0 đến 6. Baseline không đọc test split. Cuối mỗi epoch, chương trình in training loss; khi kết thúc, chương trình in JSON gồm `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`, `val_macro_auc_ovr`, `epochs`, `seed` và `device`.

Kết quả được lưu tại `results/baseline/seed_<seed>.json`, ví dụ `results/baseline/seed_42.json`. File chứa cấu hình, lịch sử training loss, metrics cuối và runtime. Chạy lại cùng seed sẽ cập nhật file đó. Kết quả các bước sau sẽ được tách theo nội dung vào các thư mục `ga_search`, `random_search` và `final_eval`.

Kiểm tra model builder bằng một chromosome hai block:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from src.model import build_model, count_parameters; a={'blocks':[{'filters':16,'kernel_size':3,'pooling':'max','batch_norm':False},{'filters':32,'kernel_size':5,'pooling':'avg','batch_norm':True}],'dropout':0.2}; m=build_model(a); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape)); print('Parameters:', count_parameters(m))"
```

Kết quả mong đợi có `Output: (2, 7)` và số tham số dương.

Kiểm tra chromosome baseline vẫn tạo đúng CNN ba block:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from src.model import BASELINE_ARCHITECTURE, build_model; m=build_model(BASELINE_ARCHITECTURE); print(m); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape))"
```

Kiểm thử thủ công thành công khi pipeline chạy hết 25 epoch, training loss nhìn chung giảm, không có lỗi về tensor/dataset và JSON cuối cùng có đủ bốn validation/training metrics nêu trên.

## 4. Stage 2 — Search space và NSGA-II độc lập

Đã được người dùng xác nhận thành công ngày 2026-10-01. Đã xóa `src/check_nsga2.py`; không cần chạy fitness giả lập nữa. `src/search_space.py` và `src/nsga2.py` được dùng trực tiếp trong pipeline chính.

## 5. Stage 3 — Tích hợp NAS với CNN

Baseline và GA mặc định cùng 25 epoch và seed 42. Xóa kết quả GA seed 1 cũ nếu không còn dùng:

```powershell
Remove-Item -LiteralPath "results\ga_search\seed_1" -Recurse -Force -ErrorAction SilentlyContinue
```

Chạy lại GA và baseline:

```powershell
.\.venv\Scripts\python.exe -m src.ga_search --epochs 25 --seed 42 --device auto --output-dir "results\ga_search"
.\.venv\Scripts\python.exe -m src.train --epochs 25 --seed 42 --device auto --output-dir "results\baseline"
```

Lệnh này dùng NSGA-II với population 10, budget 80 kiến trúc khác nhau, tournament 3, crossover 0.8 và mutation 0.15. Mỗi CNN train đúng 25 epoch, Adam, learning rate 0.001, batch size 64, weighted cross-entropy. Chỉ đọc train/validation chính thức từ `data/dermamnist_64.npz`, không tải dữ liệu, không đọc test arrays.

Console in training loss mỗi epoch, AUC validation và số tham số thực tế mỗi evaluation; cuối cùng in Pareto front và đường dẫn kết quả. Với lệnh trên, output baseline nằm tại `results/baseline/seed_42.json`; output GA nằm trong `results/ga_search/seed_42/`:

- `evaluations/<cache_key>.json`: chromosome, cấu hình, metrics thật (loss, accuracy, macro F1, macro AUC), lịch sử train loss và runtime của mỗi CNN; ghi ngay sau khi train thành công.
- `progress.json`: thứ tự evaluation, đường dẫn kết quả, số lần train mới và cache hit, cập nhật sau từng evaluation.
- `summary.json`: toàn bộ fitness đã đánh giá, Pareto front, số generation và runtime của lần chạy.

Số epoch được ghi vào config, metrics và khóa cache: kết quả 5 epoch không được dùng cho search 25 epoch. GA train lại từng CNN từ đầu với 25 epoch. Cache dùng canonical architecture và seed, đồng thời kiểm tra cấu hình, phiên bản luồng train, thiết bị và SHA-256 tensor train/validation (không truy cập test). Cache được ghi qua file tạm rồi thay thế để tránh kết quả JSON dang dở. Không thay đổi dataset hoặc code train giữa các lần tiếp tục; nếu thay đổi luồng train phải tăng `training_version` trong `src/ga_search.py` để tránh dùng kết quả cũ.

Chạy lại đúng lệnh sẽ dựng lại tiến trình search từ seed và dùng các evaluation đã train; kiến trúc trùng không train lại. Budget 80 là tổng số kiến trúc khác nhau có kết quả train trong search đó, bao gồm kết quả đã train ở lần trước; cache hit không thêm một lần train hay tiêu thêm budget. `new_trainings` chỉ đếm train mới trong lần gọi hiện tại; lần đầu không có cache phải có `evaluations: 80`, `new_trainings: 80`, `cache_hits: 0`. Khi chạy lại search đã hoàn tất: `new_trainings: 0`, `cache_hits: 80`. Khi chạy bị ngắt, chạy lại cùng lệnh để dùng các kết quả đã lưu; CNN đang train dở sẽ train lại từ đầu.

Có thể chọn `--device cpu`, `--device cuda`, `--device cuda:0`, thay `--output-dir`, hoặc dùng `--budget` và `--population-size` để chạy search thực tế với quy mô khác. Cả baseline và GA hỗ trợ `--epochs`; khi đổi số epoch hãy truyền cùng giá trị cho hai lệnh. Không có lệnh check riêng. Stage 3 đạt yêu cầu khi lệnh chính kết thúc đủ budget và xuất Pareto front từ AUC validation cùng số tham số CNN thực tế. Chờ bạn xác nhận trước khi chuyển Stage 4.

## 6. Stage 4 — Random Architecture Search và thực nghiệm

Chưa có code. Stage 4 sẽ bổ sung lệnh chạy GA-NAS và Random Architecture Search với cùng budget.

## 7. Stage 5 — Chọn kiến trúc và đánh giá cuối

Chưa có code. Stage 5 sẽ bổ sung lệnh chọn ba đại diện Pareto, retrain và test.

## 8. Stage 6 — Phân tích nghiên cứu

Chưa có code. Stage 6 sẽ bổ sung lệnh tạo bảng, confusion matrix, đường hội tụ, độ đa dạng và Pareto front.

## 9. Stage 7 — Hoàn thiện và bàn giao

Stage 7 sẽ rà soát toàn bộ dependency, lệnh, input, output và tính tái lập.
