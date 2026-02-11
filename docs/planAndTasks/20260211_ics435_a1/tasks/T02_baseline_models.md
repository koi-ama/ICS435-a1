# T02: Baseline Model Training Pipeline

## Purpose
Implement training and inference flow for KNN, Decision Tree, and Random Forest with assignment-specified defaults.

## Scope
- Load breast cancer dataset.
- 80/20 train-test split.
- Apply StandardScaler to KNN pipeline.
- Train baseline models and produce predictions.

## Dependencies
- T01

## Work Steps
1. Implement dataset loading and split.
2. Implement model factory for baseline configs:
   - KNN: `n_neighbors=5`
   - Decision Tree: default
   - Random Forest: `n_estimators=100`
3. Save baseline predictions for evaluation.

## Acceptance Criteria
- All three models train successfully.
- Predictions are generated on test set.
- Code remains reproducible via fixed random state where applicable.

## Verification
- `python3 src/main.py`
- Check generated baseline outputs under `outputs/`.

## Status
- `Done`

## Completion Notes
- Added baseline model implementations for KNN, Decision Tree, and Random Forest.
- Applied StandardScaler within the KNN pipeline only.
- Confirmed baseline output generation in `outputs/tables/metrics_baseline.csv`.
