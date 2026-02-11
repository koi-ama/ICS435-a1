from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ICS435 Assignment 1: model comparison on breast cancer dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save generated tables and figures.",
    )
    return parser.parse_args()


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    metadata_dir = output_dir / "metadata"
    for directory in (output_dir, figures_dir, tables_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "root": output_dir,
        "figures": figures_dir,
        "tables": tables_dir,
        "metadata": metadata_dir,
    }


def evaluate_model(
    model_name: str,
    setting: str,
    model: Any,
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple[dict[str, Any], Any]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "model": model_name,
        "setting": setting,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm


def save_confusion_matrix(cm, labels: list[str], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def run_baseline_experiments(X_train, y_train, X_test, y_test):
    baseline_models = {
        "KNN": (
            "n_neighbors=5",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("classifier", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        ),
        "DecisionTree": (
            "default",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
        ),
        "RandomForest": (
            "n_estimators=100",
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        ),
    }

    records: list[dict[str, Any]] = []
    confusion_matrices: dict[str, Any] = {}
    for model_name, (setting, model) in baseline_models.items():
        metrics, cm = evaluate_model(
            model_name, setting, model, X_train, y_train, X_test, y_test
        )
        records.append(metrics)
        confusion_matrices[model_name] = cm
    return records, confusion_matrices


def run_ablation_experiments(X_train, y_train, X_test, y_test):
    records: list[dict[str, Any]] = []

    for n_neighbors in [3, 5, 7, 9, 11]:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ]
        )
        setting = f"n_neighbors={n_neighbors}"
        metrics, _ = evaluate_model(
            "KNN", setting, model, X_train, y_train, X_test, y_test
        )
        records.append(metrics)

    for max_depth in [None, 3, 5, 7, 10]:
        model = DecisionTreeClassifier(
            max_depth=max_depth, random_state=RANDOM_STATE
        )
        setting = f"max_depth={max_depth}"
        metrics, _ = evaluate_model(
            "DecisionTree", setting, model, X_train, y_train, X_test, y_test
        )
        records.append(metrics)

    rf_configs = [
        {"max_depth": None, "min_samples_split": 2},
        {"max_depth": 5, "min_samples_split": 2},
        {"max_depth": 10, "min_samples_split": 2},
        {"max_depth": None, "min_samples_split": 4},
        {"max_depth": 10, "min_samples_split": 4},
    ]
    for config in rf_configs:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            random_state=RANDOM_STATE,
        )
        setting = (
            f"n_estimators=100|max_depth={config['max_depth']}|"
            f"min_samples_split={config['min_samples_split']}"
        )
        metrics, _ = evaluate_model(
            "RandomForest", setting, model, X_train, y_train, X_test, y_test
        )
        records.append(metrics)

    return records


def save_ablation_plots(ablation_df: pd.DataFrame, figures_dir: Path) -> None:
    for model_name, model_df in ablation_df.groupby("model"):
        sorted_df = model_df.sort_values("f1", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(sorted_df["setting"], sorted_df["f1"])
        ax.set_title(f"{model_name} Ablation (sorted by F1)")
        ax.set_ylabel("F1 Score")
        ax.set_xlabel("Hyperparameter Setting")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(figures_dir / f"ablation_{model_name.lower()}.png", dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    dirs = ensure_output_dirs(args.output_dir)

    dataset = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset.target,
    )

    baseline_records, baseline_cms = run_baseline_experiments(
        X_train, y_train, X_test, y_test
    )
    baseline_df = pd.DataFrame(baseline_records).sort_values("model")
    baseline_df.to_csv(dirs["tables"] / "metrics_baseline.csv", index=False)

    label_names = [dataset.target_names[0], dataset.target_names[1]]
    for model_name, cm in baseline_cms.items():
        save_confusion_matrix(
            cm,
            labels=label_names,
            title=f"{model_name} Confusion Matrix",
            output_path=dirs["figures"] / f"confusion_matrix_{model_name.lower()}.png",
        )

    ablation_records = run_ablation_experiments(X_train, y_train, X_test, y_test)
    ablation_df = pd.DataFrame(ablation_records).sort_values(["model", "f1"], ascending=[True, False])
    ablation_df.to_csv(dirs["tables"] / "metrics_ablation.csv", index=False)
    save_ablation_plots(ablation_df, dirs["figures"])

    best_configs_df = (
        ablation_df.sort_values("f1", ascending=False)
        .groupby("model", as_index=False)
        .first()
        .sort_values("model")
    )
    best_configs_df.to_csv(dirs["tables"] / "best_configs.csv", index=False)

    all_metrics_df = pd.concat(
        [baseline_df.assign(experiment_type="baseline"), ablation_df.assign(experiment_type="ablation")],
        ignore_index=True,
    )
    all_metrics_df.to_csv(dirs["tables"] / "metrics_all.csv", index=False)

    metadata = {
        "dataset_name": "sklearn.datasets.load_breast_cancer",
        "n_samples": int(dataset.data.shape[0]),
        "n_features": int(dataset.data.shape[1]),
        "target_names": dataset.target_names.tolist(),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
    }
    write_json(dirs["metadata"] / "dataset_metadata.json", metadata)

    print("Experiment completed.")
    print(f"Baseline metrics: {dirs['tables'] / 'metrics_baseline.csv'}")
    print(f"Ablation metrics: {dirs['tables'] / 'metrics_ablation.csv'}")
    print(f"Figures directory: {dirs['figures']}")


if __name__ == "__main__":
    main()
