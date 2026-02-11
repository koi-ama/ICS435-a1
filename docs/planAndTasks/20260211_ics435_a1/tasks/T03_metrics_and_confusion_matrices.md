# T03: Metrics and Confusion Matrices

## Purpose
Evaluate each baseline model and generate required metrics and confusion matrices.

## Scope
- Compute Accuracy, Precision, Recall, F1 for each model.
- Generate one confusion matrix per model.
- Save tabular comparison data for report insertion.

## Dependencies
- T02

## Work Steps
1. Implement evaluation helper.
2. Compute and save metrics table to CSV/JSON.
3. Plot and save confusion matrix images.

## Acceptance Criteria
- Metrics exist for all three models.
- Three confusion matrix files are created.
- Comparison table is report-ready.

## Verification
- Inspect `outputs/tables/metrics_baseline.csv`.
- Confirm confusion matrix image files exist in `outputs/figures/`.

## Status
- `Done`

## Completion Notes
- Saved baseline metrics table to `outputs/tables/metrics_baseline.csv`.
- Generated confusion matrices:
  - `outputs/figures/confusion_matrix_knn.png`
  - `outputs/figures/confusion_matrix_decisiontree.png`
  - `outputs/figures/confusion_matrix_randomforest.png`
