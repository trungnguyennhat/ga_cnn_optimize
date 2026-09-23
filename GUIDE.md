# Hướng dẫn chạy dự án

AI agent cập nhật phần tương ứng sau mỗi stage nhưng không tự chạy huấn luyện hoặc kiểm thử.

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
.\.venv\Scripts\python.exe train.py
```

Có thể thay đổi các siêu tham số và thiết bị từ dòng lệnh, ví dụ:

```powershell
.\.venv\Scripts\python.exe train.py --learning-rate 0.001 --batch-size 64 --dropout 0.3 --optimizer Adam --filters 32 --epochs 15 --seed 42 --device auto
```

`--device` nhận `auto`, `cpu`, `cuda` hoặc một GPU cụ thể như `cuda:0`. Có thể dùng file dữ liệu ở vị trí khác với `--data-path`, hoặc đổi thư mục kết quả bằng `--output-dir`.

Loader chỉ đọc `train_images`, `train_labels`, `val_images`, `val_labels`; kiểm tra kích thước chính thức và nhãn từ 0 đến 6. Baseline không đọc test split. Cuối mỗi epoch, chương trình in training loss; khi kết thúc, chương trình in JSON gồm `train_loss`, `val_loss`, `val_accuracy`, `val_macro_f1`, `val_macro_auc_ovr`, `epochs`, `seed` và `device`.

Kết quả được lưu tại `results/baseline/seed_<seed>.json`, ví dụ `results/baseline/seed_42.json`. File chứa cấu hình, lịch sử training loss, metrics cuối và runtime. Chạy lại cùng seed sẽ cập nhật file đó. Kết quả các bước sau sẽ được tách theo nội dung vào các thư mục `ga_search`, `random_search` và `final_eval`.

Kiểm tra model builder bằng một chromosome hai block:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from model import build_model, count_parameters; a={'blocks':[{'filters':16,'kernel_size':3,'pooling':'max','batch_norm':False},{'filters':32,'kernel_size':5,'pooling':'avg','batch_norm':True}],'dropout':0.2}; m=build_model(a); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape)); print('Parameters:', count_parameters(m))"
```

Kết quả mong đợi có `Output: (2, 7)` và số tham số dương.

Kiểm tra chromosome baseline vẫn tạo đúng CNN ba block:

```powershell
.\.venv\Scripts\python.exe -c "import torch; from model import BASELINE_ARCHITECTURE, build_model; m=build_model(BASELINE_ARCHITECTURE); print(m); print('Output:', tuple(m(torch.zeros(2,3,64,64)).shape))"
```

Kiểm thử thủ công thành công khi pipeline chạy hết 15 epoch, training loss nhìn chung giảm, không có lỗi về tensor/dataset và JSON cuối cùng có đủ bốn validation/training metrics nêu trên.

## 4. Stage 2 — Search space và NSGA-II độc lập

Chưa có code. Stage 2 sẽ bổ sung lệnh kiểm tra chromosome, Pareto sorting, crossover, mutation và evaluation budget bằng fitness giả lập.

## 5. Stage 3 — Tích hợp NAS với CNN

Chưa có code. Stage 3 sẽ bổ sung lệnh chạy GA-NAS, cache và log vào `results/ga_search/`.

## 6. Stage 4 — Random Architecture Search và thực nghiệm

Chưa có code. Stage 4 sẽ bổ sung lệnh chạy GA-NAS và Random Architecture Search với cùng budget.

## 7. Stage 5 — Chọn kiến trúc và đánh giá cuối

Chưa có code. Stage 5 sẽ bổ sung lệnh chọn ba đại diện Pareto, retrain và test.

## 8. Stage 6 — Phân tích nghiên cứu

Chưa có code. Stage 6 sẽ bổ sung lệnh tạo bảng, confusion matrix, đường hội tụ, độ đa dạng và Pareto front.

## 9. Stage 7 — Hoàn thiện và bàn giao

Stage 7 sẽ rà soát toàn bộ dependency, lệnh, input, output và tính tái lập.
