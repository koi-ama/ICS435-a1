# ICS435 Assignment 1

This repository contains the implementation and report artifacts for ICS435 Assignment 1:
comparison of KNN, Decision Tree, and Random Forest on the `sklearn` Breast Cancer dataset.

## Repository URL
- https://github.com/koi-ama/ICS435-a1

## Project Structure
- `src/main.py`: end-to-end experiment runner
- `requirements.txt`: Python dependencies
- `outputs/tables/`: generated metrics tables
- `outputs/figures/`: confusion matrices and ablation plots
- `outputs/metadata/`: dataset metadata and split information
- `report/`: report source and generated PDF
- `docs/planAndTasks/20260211_ics435_a1/`: planning and task management docs

## Setup
```bash
python3 -m pip install -r requirements.txt
```

## Run Experiments
```bash
python3 src/main.py
```

Optional output directory:
```bash
python3 src/main.py --output-dir outputs
```

## Assignment Requirements Covered
- Dataset loading via `load_breast_cancer`
- 80/20 train-test split
- StandardScaler applied for KNN
- Baseline models:
  - KNN (`n_neighbors=5`)
  - Decision Tree (default)
  - Random Forest (`n_estimators=100`)
- Metrics for all models: Accuracy, Precision, Recall, F1
- Confusion matrix per model
- Hyperparameter ablation:
  - KNN: `n_neighbors` variations
  - Decision Tree: `max_depth` variations
  - Random Forest: `max_depth` and `min_samples_split` variations

## Key Output Files
- Baseline metrics: `outputs/tables/metrics_baseline.csv`
- Ablation metrics: `outputs/tables/metrics_ablation.csv`
- Best configs summary: `outputs/tables/best_configs.csv`
- Confusion matrices:
  - `outputs/figures/confusion_matrix_knn.png`
  - `outputs/figures/confusion_matrix_decisiontree.png`
  - `outputs/figures/confusion_matrix_randomforest.png`

## Reproducibility Notes
- Random state is fixed at `42` for data split and tree-based models.
- Dataset metadata and split sizes are saved in `outputs/metadata/dataset_metadata.json`.
