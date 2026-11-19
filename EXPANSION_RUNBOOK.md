# Chạy các thí nghiệm mở rộng GA-NAS

Các lệnh bên dưới dùng PowerShell từ thư mục gốc của repo. Mỗi khối nêu các file dự kiến được tạo. Số liệu của các lượt train và search mới sẽ có sau khi bạn chạy.

## Chuẩn bị

- Dùng môi trường Python 3.11 hoặc 3.12 cho bộ dependency PyTorch CUDA trong `requirements.txt`. Thay `--device cuda` bằng `--device cpu` nếu chạy trên CPU.
- Đặt file chính thức `dermamnist_64.npz` tại `data/dermamnist_64.npz`. Hướng 5.3 cần thêm `data/bloodmnist_64.npz`. Cả hai dùng ảnh RGB 64×64 và split chính thức. Có thể chỉ định đường dẫn khác với `--data-path`.
- Các ví dụ chạy theo preset v2 mặc định sau khi ablation. Nếu chọn loss, AMP hoặc preload khác, hãy truyền cùng các cờ cho baseline, proxy, GA, Random, surrogate và final eval. Dùng thư mục riêng cho từng cấu hình ablation.
- `train`, `proxy_correlation` và các lệnh search chỉ đọc train/validation. Chốt cấu hình bằng validation. Chạy final eval hoặc `protocol_report --evaluate-test` sau khi chốt quy tắc đánh giá.

```powershell
python -m pip install -r requirements.txt
```

Kiểm tra tùy chọn trên CPU trước khi dùng GPU với `python -m unittest src.test_expansion src.surrogate_search.test_surrogate`.

## Bước 0 · Báo cáo lại kết quả v1 và đánh giá các điểm Pareto

Chạy khối đầu để lấy Hypervolume và MACs từ artifacts đã lưu. Khối thứ hai tái dùng checkpoint max-AUC, khôi phục xác suất bằng inference và train thêm các kiến trúc đại diện.

```powershell
python -m src.analyze --test-dir results/test_eval --n-bootstrap 0 --output-dir results/expansion/step0_offline

python -m src.test_eval --protocol v1 --device cuda --pareto-representatives --knee-scale linear `
  --reuse-test-dir results/test_eval --output-dir results/expansion/step0_test
python -m src.analyze --test-dir results/expansion/step0_test --n-bootstrap 2000 `
  --output-dir results/expansion/step0_analysis
```

**Kết quả dự kiến.** Trên artifacts v1 hiện có, GA knee có 603.911 params, GA nhỏ nhất đạt ngưỡng có 537.287 params, Random dùng cùng kiến trúc 666.007 params cho cả hai vai trò. Tối đa ba kiến trúc mới cần retrain khi hai checkpoint max-AUC có sẵn. Log-Hypervolume theo Params dự kiến là khoảng 0,8402 và 0,7995 lần lượt cho GA và Random. `analysis.json` chứa CI 95% bootstrap theo lớp và CI chênh lệch ghép cặp. `per_class_metrics.csv` chứa support, precision, recall, F1 và OvR AUC theo lớp. `test_metrics.csv`, `architecture_costs.csv` và các biểu đồ nằm trong thư mục phân tích.

`--knee-scale log` có thể chọn đại diện khác. Hypervolume log dùng AUC gốc, chi phí log10 chuẩn hóa theo cận chung và điểm tham chiếu gốc tọa độ. Khi thiếu `--baseline-proxy-result`, ngưỡng chọn đại diện là validation AUC của baseline 60 epoch. Tên `smallest_baseline` chỉ mô tả quy tắc chọn trên validation. CI bootstrap ở bước này đo biến động theo ảnh test với checkpoint cố định. Hướng 5.2 đo thêm biến động qua các seed.

## Hướng 2 · Chốt training protocol và kiểm chứng proxy

Giữ kiến trúc ResNet-18 và GA max-AUC v1 cố định để so sánh. Các run dưới đây tách ảnh hưởng của loss trong v2. Preset v2 mặc định dùng weighted CE theo căn nghịch tần suất.

```powershell
$fixedGa = 'results/test_eval/ga/seed_42/result.json'
$ablation = 'results/expansion/ablation'

python -m src.train --protocol v2 --device cuda
python -m src.train --protocol v2 --device cuda --architecture-json $fixedGa --output-dir "$ablation/ga_weighted"
python -m src.train --protocol v2 --device cuda --loss unweighted_ce --output-dir "$ablation/baseline_unweighted"
python -m src.train --protocol v2 --device cuda --loss unweighted_ce `
  --architecture-json $fixedGa --output-dir "$ablation/ga_unweighted"

$runs = @(
  '--run', 'baseline_v1=results/baseline/seed_42/result.json',
  '--run', "ga_v1=$fixedGa",
  '--run', 'baseline_v2=results/v2_protocol/baseline/seed_42/result.json',
  '--run', "ga_v2=$ablation/ga_weighted/seed_42/result.json",
  '--run', "baseline_v2_unweighted=$ablation/baseline_unweighted/seed_42/result.json",
  '--run', "ga_v2_unweighted=$ablation/ga_unweighted/seed_42/result.json"
)
python -m src.protocol_report @runs --output-dir "$ablation/validation"

python -m src.proxy_correlation --protocol v1 --device cuda --num-models 12 --top-k 3
python -m src.proxy_correlation --protocol v2 --device cuda --num-models 12 --top-k 3
```

Để thử focal loss, thay `--loss unweighted_ce` bằng `--loss focal` và đổi thư mục output. Để đo tác động của preload hoặc AMP, chạy lại trên cùng kiến trúc với `--no-preload-gpu` hoặc `--amp` và thư mục output riêng. Dùng cùng phần cứng khi so thời gian. Nếu muốn lập bảng test của các checkpoint cố định sau khi khóa giao thức, dùng `python -m src.protocol_report @runs --evaluate-test --device cuda --output-dir "$ablation/test"`.

**Kết quả dự kiến.** `protocol_ablation.csv` và `.json` chứa validation AUC, macro F1, macro recall nếu lịch sử có lưu, best epoch, loss gap và thời gian. Báo cáo test tùy chọn thêm test metrics và confusion matrices. Không so trực tiếp độ lớn loss gap khi hai run dùng hàm loss khác nhau. `results/proxy_correlation/seed_42/correlation_summary.json` và `results/v2_protocol/proxy_correlation/seed_42/correlation_summary.json` chứa Spearman, Kendall, top-k recall 20→60 và xếp hạng sớm 5/10→20. Mẫu kiến trúc mặc định là cùng một nhóm từ GA và Random v1. Proxy v1 tái sử dụng record nếu seed, lịch LR, thiết bị và dữ liệu khớp. Cosine v2 dùng lịch riêng cho 20 và 60 epoch.

## Hướng 1 · Tối ưu AUC và MACs

Đo mốc ResNet-18 theo đúng proxy 20 epoch, rồi chạy GA và Random trên cùng search space và ngân sách. Baseline 60 epoch của Hướng 2 được tái sử dụng cho final eval.

```powershell
$mac = @('--protocol', 'v2', '--space', 'v1', '--objectives', 'macs')
$macRoot = 'results/v2_protocol/space_v1_macs'
$proxyDir = 'results/v2_protocol/baseline_proxy'
$proxy = "$proxyDir/seed_42/result.json"

python -m src.train --protocol v2 --device cuda --epochs 20 --scheduler-step-size 15 --output-dir $proxyDir
python -m src.ga_search @mac --device cuda --budget 60 --tournament-size 2
python -m src.random_search @mac --device cuda --budget 60
python -m src.test_eval @mac --device cuda --pareto-representatives --knee-scale log --baseline-proxy-result $proxy --latency
python -m src.analyze @mac --baseline-proxy-result $proxy --output-dir "$macRoot/visualizations/seed_42"
```

Nếu cần đánh giá ba mục tiêu, dùng biến thể sau. Chi phí của kiến trúc dưới ngưỡng baseline được xét theo Params trong cấu hình ba mục tiêu và MACs trong cấu hình hai mục tiêu `(AUC, MACs)`.

```powershell
$tri = @('--protocol', 'v2', '--space', 'v1', '--objectives', 'params-macs')
python -m src.ga_search @tri --device cuda --budget 120 --population-size 16 --tournament-size 2
python -m src.random_search @tri --device cuda --budget 120
python -m src.test_eval @tri --device cuda --expected-budget 120 --pareto-representatives --knee-scale log --baseline-proxy-result $proxy
python -m src.analyze @tri --expected-budget 120 --baseline-proxy-result $proxy
```

**Kết quả dự kiến.** Mỗi phương pháp ghi `summary.json`, `progress.json` và cache theo kiến trúc. `search_comparison.json` báo cáo Hypervolume 2D hoặc 3D, coverage hai chiều và cận chuẩn hóa. `pareto_projections.png` chiếu Pareto lên ba cặp trục. Bảng test có MACs, GFLOPs theo quy ước 2 FLOPs/MAC và latency khi bật `--latency`. Latency là trung vị batch 1 sau warm-up trên thiết bị đã chọn. GFLOPs tính theo Conv2d và Linear.

## Hướng 4 · MO-ASHA và surrogate có độ bất định

Chọn mốc dừng dựa trên `early_proxy_rankings` của pilot v2. Các mốc 5/10 dưới đây là cấu hình thử nghiệm với trần 20 epoch. MO-ASHA chạy tuần tự trên một worker. Mỗi trial được tiếp tục trực tiếp từ trạng thái tại mốc, và shortlist cuối chỉ dùng trial hoàn thành đủ 20 epoch.

```powershell
python -m src.multi_fidelity @mac --device cuda --budget 60 --sampler ga --rungs 5 10 --reduction-factor 2 --grace-trials 6
python -m src.surrogate_search.run_search @mac --device cuda --budget 60 --backend rf --acquisition ucb --run-id rf_ucb
python -m src.surrogate_search.run_search @mac --device cuda --budget 60 --backend gp --acquisition ehvi --run-id gp_ehvi

$asha = "$macRoot/mo_asha_ga/seed_42/summary.json"
$rf = "$macRoot/surrogate_search/rf_ucb/seed_42/summary.json"
$gp = "$macRoot/surrogate_search/gp_ehvi/seed_42/summary.json"
python -m src.compare_searches --objectives macs --output-dir results/expansion/efficiency `
  --run "ga=$macRoot/ga_search/seed_42/summary.json" --run "random=$macRoot/random_search/seed_42/summary.json" `
  --run "mo_asha=$asha" --run "rf_ucb=$rf" --run "gp_ehvi=$gp"

python -m src.test_eval @mac --device cuda --baseline-proxy-result $proxy --pareto-representatives --knee-scale log `
  --extra-summary "mo_asha=$asha" --extra-summary "rf_ucb=$rf" --extra-summary "gp_ehvi=$gp" `
  --reuse-test-dir "$macRoot/test_eval" --output-dir results/expansion/efficiency_test
python -m src.analyze @mac --baseline-proxy-result $proxy --test-dir results/expansion/efficiency_test `
  --output-dir results/expansion/efficiency_analysis
```

**Kết quả dự kiến.** MO-ASHA lưu `trials_per_rung`, `full_evaluations`, `total_epochs` và Pareto ở fidelity đầy đủ. Surrogate lưu MAE, RMSE, Spearman trước khi cập nhật bằng batch mới và qua CV ba fold. So sánh chung có hội tụ theo lượt đánh giá và wall-clock, thời gian train tích lũy, coverage và số epoch thực tế. Dùng lượt chạy sạch với `cache_hits=0` khi so wall-clock. Trên một GPU, chia `training_seconds` cho 3.600 để tính giờ train theo thời gian ghi nhận. Có thể thay `--sampler ga` bằng `--sampler random` để đo riêng tác động của sàng lọc.

## Hướng 3 · Search space v2

Biến thể v2 chọn BasicBlock hoặc MBConv expansion 2 theo stage, bật SE theo stage và chọn ReLU/SiLU cho toàn mạng. Channel không giảm giữa các stage. Dùng cùng protocol, seed, budget và mục tiêu với Hướng 1 để đo tác động của không gian.

```powershell
$space2 = @('--protocol', 'v2', '--space', 'v2', '--objectives', 'macs')
$space2Root = 'results/v2_protocol/space_v2_macs'

python -m src.ga_search @space2 --device cuda --budget 60 --tournament-size 2
python -m src.random_search @space2 --device cuda --budget 60
python -m src.test_eval @space2 --device cuda --pareto-representatives --knee-scale log --baseline-proxy-result $proxy
python -m src.analyze @space2 --baseline-proxy-result $proxy
python -m src.compare_searches --objectives macs --output-dir results/expansion/search_spaces `
  --run "ga_v1=$macRoot/ga_search/seed_42/summary.json" --run "ga_v2=$space2Root/ga_search/seed_42/summary.json" `
  --run "random_v1=$macRoot/random_search/seed_42/summary.json" --run "random_v2=$space2Root/random_search/seed_42/summary.json"
```

**Kết quả dự kiến.** `search_comparison.json` dùng cận chung cho cả hai search space và chứa `pareto_gene_counts` theo stage. Xem Pareto và tần suất gene để đánh giá thay đổi về AUC, MACs và Params. Có thể truyền `@space2` cho các lệnh MO-ASHA và surrogate nếu cần thử phối hợp.

## Hướng 5.2 · Lặp lại với nhiều seed

Ví dụ này bổ sung seed 43 và 44 cho cấu hình hai mục tiêu ở Hướng 1. Seed 42 được đọc từ báo cáo đã tạo ở Hướng 1. Có thể thêm 45 và 46 trước khi tổng hợp.

```powershell
foreach ($seed in 43, 44) {
  $seedProxy = "$proxyDir/seed_$seed/result.json"
  python -m src.train --protocol v2 --device cuda --seed $seed
  python -m src.train --protocol v2 --device cuda --seed $seed --epochs 20 --scheduler-step-size 15 --output-dir $proxyDir
  python -m src.ga_search @mac --device cuda --seed $seed --budget 60 --tournament-size 2
  python -m src.random_search @mac --device cuda --seed $seed --budget 60
  python -m src.test_eval @mac --device cuda --seed $seed --pareto-representatives --knee-scale log --baseline-proxy-result $seedProxy
  python -m src.analyze @mac --seed $seed --baseline-proxy-result $seedProxy --output-dir "$macRoot/visualizations/seed_$seed"
}
$dermaAnalyses = 42, 43, 44 | ForEach-Object { "$macRoot/visualizations/seed_$_/analysis.json" }
python -m src.aggregate_results --analysis @dermaAnalyses --output-dir results/expansion/multi_seed
```

**Kết quả dự kiến.** `multi_seed.json` và `dataset_summary.csv` có `n`, mean, std mẫu cho Hypervolume, coverage, AUC, macro F1, Params và MACs. Report tính Wilcoxon ghép cặp cho GA và Random khi có ít nhất ba cặp seed. Số seed nhỏ cho độ phân giải thống kê thấp. `convergence_dermamnist.png` vẽ mean ± std theo ngân sách. Với đại diện không đạt ngưỡng baseline ở một số seed, `n` phản ánh số seed thực sự có đại diện.

## Hướng 5.3 · BloodMNIST

Dùng cấu hình đã chốt trên DermaMNIST, rồi chạy lại search và train từ đầu trên split chính thức của BloodMNIST. Khối dưới chạy một seed để kiểm tra khả năng áp dụng. Lặp cùng các seed ở cả hai tập nếu muốn so độ ổn định.

```powershell
$blood = @('--dataset', 'bloodmnist', '--protocol', 'v2', '--space', 'v1', '--objectives', 'macs')
$bloodRoot = 'results/bloodmnist/v2_protocol/space_v1_macs'
$bloodProxyDir = 'results/bloodmnist/v2_protocol/baseline_proxy'
$bloodProxy = "$bloodProxyDir/seed_42/result.json"

python -m src.train --dataset bloodmnist --protocol v2 --device cuda
python -m src.train --dataset bloodmnist --protocol v2 --device cuda --epochs 20 --scheduler-step-size 15 --output-dir $bloodProxyDir
python -m src.ga_search @blood --device cuda --budget 60 --tournament-size 2
python -m src.random_search @blood --device cuda --budget 60
python -m src.test_eval @blood --device cuda --pareto-representatives --knee-scale log --baseline-proxy-result $bloodProxy
python -m src.analyze @blood --baseline-proxy-result $bloodProxy --output-dir "$bloodRoot/visualizations/seed_42"
$dataAnalyses = @($dermaAnalyses) + @("$bloodRoot/visualizations/seed_42/analysis.json")
python -m src.aggregate_results --analysis @dataAnalyses --output-dir results/expansion/cross_dataset
```

**Kết quả dự kiến.** Checkpoint BloodMNIST có classifier 8 lớp. Báo cáo per-class và confusion matrix dùng 8 lớp. `dataset_summary.csv` xuất kết quả GA, Random và baseline theo từng dataset với số seed tương ứng. Báo cáo đa dataset giữ riêng các số liệu vì AUC, chi phí classifier và cận Hypervolume phụ thuộc dataset.

