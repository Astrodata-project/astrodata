"""Example 3.2: Compare serial and parallel grid search selectors."""

import argparse
import math
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from astrodata.ml.metrics import SklearnMetric
from astrodata.ml.model_selection import GridSearchSelector
from astrodata.ml.model_selection.GridSearchSelector_parallel import (
    GridSearchSelectorParallel,
)
from astrodata.ml.models import SklearnModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the standard and parallel GridSearch selectors using the same "
            "random seed and an adjustable training set size."
        )
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.75,
        help="Fraction of the dataset reserved for training (0 < train_size < 1).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducible data splits and selectors.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Number of parallel jobs for the parallel selector. "
            "Use a positive integer, or <= 0 to use (CPU count + n_jobs)."
        ),
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help=(
            "Fraction of the training data used as validation inside the selectors. "
            "Must be in (0, 1)."
        ),
    )
    parser.add_argument(
        "--toy-size",
        type=int,
        default=None,
        help=(
            "If set, generate a synthetic classification dataset with the given number "
            "of samples instead of using the breast cancer dataset."
        ),
    )
    parser.add_argument(
        "--toy-features",
        type=int,
        default=30,
        help=(
            "Number of features for the synthetic dataset (only used with --toy-size). "
            "Must be >= 4 to ensure informative and redundant features can be created."
        ),
    )
    parser.add_argument(
        "--toy-classes",
        type=int,
        default=2,
        help="Number of target classes for the synthetic dataset (only used with --toy-size).",
    )

    args = parser.parse_args()
    if not 0 < args.train_size < 1:
        raise ValueError("train_size must be between 0 and 1 (exclusive).")
    if not 0 < args.val_size < 1:
        raise ValueError("val_size must be between 0 and 1 (exclusive).")
    if args.toy_size is not None and args.toy_size <= 0:
        raise ValueError("toy_size must be a positive integer when provided.")
    if args.toy_size is not None and args.toy_features < 4:
        raise ValueError("toy_features must be at least 4 when generating synthetic data.")
    if args.toy_size is not None and args.toy_classes < 2:
        raise ValueError("toy_classes must be at least 2 when generating synthetic data.")
    return args


def run_selector(selector, X_train, y_train, X_test, y_test) -> Dict[str, float]:
    start = time.perf_counter()
    selector.fit(X_train, y_train, X_test=X_test, y_test=y_test)
    duration = time.perf_counter() - start
    metrics = selector.get_best_metrics() or {}
    return {
        "selector": selector,
        "best_params": selector.get_best_params(),
        "metrics": metrics,
        "duration": duration,
    }


def prepare_dataset(args: argparse.Namespace) -> Tuple[pd.DataFrame, pd.Series, str]:
    if args.toy_size is None:
        dataset = load_breast_cancer()
        X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        y = pd.Series(dataset.target)
        return X, y, "breast_cancer"

    informative = max(2, int(math.ceil(0.6 * args.toy_features)))
    informative = min(informative, args.toy_features - 2)
    redundant = max(0, args.toy_features - informative)

    X, y = make_classification(
        n_samples=args.toy_size,
        n_features=args.toy_features,
        n_informative=informative,
        n_redundant=redundant,
        n_repeated=0,
        n_classes=args.toy_classes,
        random_state=args.random_state,
    )

    feature_names = [f"feature_{i}" for i in range(args.toy_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series, f"toy(size={args.toy_size}, features={args.toy_features})"


def main() -> None:
    args = parse_args()

    np.random.seed(args.random_state)

    X, y, dataset_label = prepare_dataset(args)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=args.train_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = SklearnModel(model_class=LinearSVC, penalty="l2", loss="squared_hinge")
    accuracy = SklearnMetric(accuracy_score, greater_is_better=True)
    param_grid = {
        "C": [0.1, 1, 10],
        "max_iter": [1000, 2000],
        "tol": [1e-3, 1e-4],
    }

    serial_selector = GridSearchSelector(
        model=model,
        param_grid=param_grid,
        scorer=accuracy,
        val_size=args.val_size,
        random_state=args.random_state,
        metrics=[accuracy],
    )

    parallel_selector = GridSearchSelectorParallel(
        model=model,
        param_grid=param_grid,
        n_jobs=args.n_jobs,
        scorer=accuracy,
        val_size=args.val_size,
        random_state=args.random_state,
        metrics=[accuracy],
    )

    serial_result = run_selector(serial_selector, X_train, y_train, X_test, y_test)
    parallel_result = run_selector(parallel_selector, X_train, y_train, X_test, y_test)

    durations = {
        "serial": serial_result["duration"],
        "parallel": parallel_result["duration"],
    }
    print("\n=== Comparison Summary ===")
    print(f"Dataset: {dataset_label}")
    print(f"Random state: {args.random_state}")
    print(f"Training size fraction: {args.train_size}")
    print(f"Validation size fraction: {args.val_size}")
    print(f"Parallel selector n_jobs: {parallel_selector.n_jobs}")
    print("")

    print("Best parameters (serial):", serial_result["best_params"])
    print("Best parameters (parallel):", parallel_result["best_params"])
    print("")

    for label, metrics in ("serial", serial_result["metrics"]), (
        "parallel",
        parallel_result["metrics"],
    ):
        print(f"Validation metrics ({label}):")
        if not metrics:
            print("  No metrics recorded.")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print("")

    params_match = serial_result["best_params"] == parallel_result["best_params"]
    metrics_match = np.allclose(  # type: ignore[arg-type]
        list(serial_result["metrics"].values()),
        list(parallel_result["metrics"].values()),
        atol=1e-8,
    )

    print(f"Best parameters identical: {params_match}")
    print(f"Validation metrics identical (within tolerance): {metrics_match}")
    print("")
    print("Fit duration (seconds):")
    for label, duration in durations.items():
        print(f"  {label}: {duration:.3f}s")

    if not params_match or not metrics_match:
        raise AssertionError(
            "Serial and parallel grid searches did not converge to the same result."
        )


if __name__ == "__main__":
    main()
