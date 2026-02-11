# T04: Hyperparameter Ablation Study

## Purpose
Measure model behavior under key hyperparameter changes and capture performance differences.

## Scope
- KNN: vary `n_neighbors`.
- Decision Tree: vary `max_depth`.
- Random Forest: vary `max_depth` and/or `min_samples_split`.
- Save result comparison artifacts.

## Dependencies
- T03

## Work Steps
1. Define ablation grids for each model.
2. Execute evaluations for each configuration.
3. Save ablation metrics to table and optional figures.

## Acceptance Criteria
- Each model has at least one tuned variation compared to baseline.
- Ablation table clearly shows metric differences.

## Verification
- Inspect `outputs/metrics_ablation.csv`.
- Confirm at least 2 settings per model are evaluated.

## Status
- `Done`

## Completion Notes
- Evaluated KNN with `n_neighbors` in `[3, 5, 7, 9, 11]`.
- Evaluated Decision Tree with `max_depth` in `[None, 3, 5, 7, 10]`.
- Evaluated Random Forest with variations in `max_depth` and `min_samples_split`.
- Saved ablation metrics and plots under `outputs/tables/` and `outputs/figures/`.
