# Quick Start

Commands below use PowerShell and assume execution from the repository root.

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python -m pytest -q
```

## 2. Prepare Helmet Data

```powershell
python -m kaggle datasets download -d andrewmvd/hard-hat-detection `
  -p data/source/helmet --unzip

python scripts/dataset/prepare_helmet_dataset.py `
  --source-dir data/source/helmet `
  --output-dir data/helmet-yolo

python scripts/dataset/analyze_yolo_dataset.py `
  --data data/helmet-yolo/helmet.yaml `
  --output-dir outputs/dataset_analysis/helmet
```

## 3. Train YOLOv8n

```powershell
python scripts/train/train_yolov8.py `
  --data data/helmet-yolo/helmet.yaml `
  --model yolov8n.pt `
  --project outputs/baseline_training `
  --name helmet_yolov8n `
  --epochs 50 `
  --imgsz 640 `
  --batch 32 `
  --device 0 `
  --seed 2026
```

## 4. Generate Attacks

```powershell
python scripts/attacks/run_all_attacks.py `
  --dataset-root data/helmet-yolo `
  --model outputs/baseline_training/helmet_yolov8n/weights/best.pt `
  --split test `
  --output-root outputs/attacks
```

## 5. Benchmark Robustness

```powershell
python scripts/benchmark/benchmark_robustness.py `
  --model outputs/baseline_training/helmet_yolov8n/weights/best.pt `
  --clean-data data/helmet-yolo/helmet.yaml `
  --attacks-root outputs/attacks `
  --split test `
  --output-dir outputs/results
```

## 6. Run TCVM on a Sequence

The sequence root must contain ordered images, YOLO labels, and frame-level attack labels.

```powershell
python scripts/benchmark/benchmark_tcvm.py `
  --sequence-root outputs/tcvm_analysis/example_sequence `
  --model yolov8n.pt `
  --output-dir outputs/tcvm_analysis/example_sequence/tcvm `
  --device 0 `
  --conf 0.15 `
  --anomaly-threshold 0.45 `
  --frame-labels outputs/tcvm_analysis/example_sequence/frame_labels.csv `
  --flow-scale 0.25
```

## 7. Run Ablations

```powershell
python scripts/benchmark/run_ablation.py `
  --sequence-root outputs/tcvm_analysis/example_sequence `
  --model yolov8n.pt `
  --output-dir outputs/ablations/example_sequence `
  --device 0 `
  --conf 0.15 `
  --anomaly-threshold 0.45 `
  --frame-labels outputs/tcvm_analysis/example_sequence/frame_labels.csv `
  --flow-scale 0.25
```

## 8. Generate Plots

```powershell
python scripts/visualization/plot_robustness.py `
  --robustness-csv outputs/results/robustness_summary.csv `
  --output-dir outputs/figures
```

## 9. Quality Gate

```powershell
python -m compileall src scripts
python -m pytest -q
```

Keep datasets, weights, videos, Kaggle credentials, and raw runs outside Git history.
