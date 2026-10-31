"""Assignment 1, Question 3: CART decision trees on the Raisin dataset.

Evaluates four ``DecisionTreeClassifier`` configurations (gini/entropy, with and
without cost-complexity pruning) on one shared stratified 70/30 hold-out split
and with stratified 10-fold cross-validation on the complete dataset.

Run from anywhere with the project virtual environment:

    .venv/bin/python src/cart_question3.py

Outputs (under ``results/cart/``):
    question3_results.csv   machine-readable metrics, one row per configuration
    question3_summary.txt   human-readable summary
    figures/*.png           plots of the trees trained on the 70% training set
"""

from __future__ import annotations

import argparse
import platform
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are only written to disk.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Raisin_Dataset.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "results" / "cart"

SHEET_NAME = "Raisin_Grains_Dataset"
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = (
    "Area",
    "MajorAxisLength",
    "MinorAxisLength",
    "Eccentricity",
    "ConvexArea",
    "Extent",
    "Perimeter",
)

RANDOM_STATE = 1
TEST_SIZE = 0.30
N_SPLITS = 10
# Fixed by the assignment rather than tuned, so no data is used to choose it.
PRUNING_ALPHA = 0.1

# Trees deeper than this also get a zoomed-in figure of their top levels.
OVERVIEW_DEPTH = 3
OVERVIEW_FIGSIZE = (30.0, 9.0)


@dataclass(frozen=True)
class CartConfig:
    """One CART variant from Question 3."""

    name: str
    criterion: str
    ccp_alpha: float

    @property
    def pruned(self) -> bool:
        return self.ccp_alpha > 0.0


CONFIGURATIONS = (
    CartConfig("gini_unpruned", "gini", 0.0),
    CartConfig("entropy_unpruned", "entropy", 0.0),
    CartConfig("gini_pruned", "gini", PRUNING_ALPHA),
    CartConfig("entropy_pruned", "entropy", PRUNING_ALPHA),
)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the Raisin data sheet and check it has the expected columns."""
    data = pd.read_excel(path, sheet_name=SHEET_NAME)
    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]
    if list(data.columns) != expected:
        raise ValueError(f"Expected columns {expected}, found {list(data.columns)}")
    return data


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return data[list(FEATURE_COLUMNS)], data[TARGET_COLUMN]


def make_holdout_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return the stratified 70/30 split as X_train, X_test, y_train, y_test."""
    # Stratifying keeps the 50/50 class balance in both parts, so the test
    # accuracy does not depend on a chance imbalance between the classes.
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def make_cv_splitter() -> StratifiedKFold:
    # Every configuration gets a splitter with the same seed, and therefore the
    # same folds, so their CV scores are compared on identical partitions.
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def build_classifier(config: CartConfig) -> DecisionTreeClassifier:
    """Build an unfitted tree; growth limits stay at their unlimited defaults.

    ``random_state`` is fixed as well because scikit-learn permutes the features
    at every split, which can otherwise change the choice between tied splits.
    """
    # Cost-complexity pruning grows the full tree, then repeatedly removes the
    # weakest subtree while its effective alpha (impurity reduction per extra
    # leaf) stays within ccp_alpha; ccp_alpha=0.0 therefore means no pruning.
    # Gini peaks at 0.5 and entropy at 1.0 here, so the same alpha is not an
    # equally strong penalty for the two criteria.
    return DecisionTreeClassifier(
        criterion=config.criterion,
        ccp_alpha=config.ccp_alpha,
        random_state=RANDOM_STATE,
    )


def run_experiments(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, DecisionTreeClassifier]]:
    """Evaluate every configuration.

    Returns one metrics row per configuration, plus the trees fitted on the
    training split.
    """
    X, y = split_features_target(data)
    # Split once, outside the loop: all configurations see the same examples, so
    # differences in test accuracy come from the configuration, not the split.
    X_train, X_test, y_train, y_test = make_holdout_split(X, y)

    rows = []
    models = {}
    for config in CONFIGURATIONS:
        # No leakage: the test set is only used to score the finished tree, and
        # no preprocessing is fitted on the data (trees need no feature scaling).
        model = build_classifier(config).fit(X_train, y_train)
        models[config.name] = model

        # A separate estimate that uses the complete dataset, including the
        # hold-out test examples. It stays free of leakage because each fold
        # trains a fresh clone on the other 9 folds and scores the unseen fold.
        fold_scores = cross_val_score(
            build_classifier(config), X, y, cv=make_cv_splitter(), scoring="accuracy"
        )
        rows.append(
            {
                "configuration": config.name,
                "criterion": config.criterion,
                "pruned": config.pruned,
                "ccp_alpha": config.ccp_alpha,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_accuracy": accuracy_score(y_test, model.predict(X_test)),
                "n_leaves": int(model.get_n_leaves()),
                "tree_depth": int(model.get_depth()),
                "n_nodes": int(model.tree_.node_count),
                "cv_mean_accuracy": float(fold_scores.mean()),
                # Population standard deviation (ddof=0), NumPy's default.
                "cv_std_accuracy": float(fold_scores.std()),
                **{
                    f"cv_fold_{fold}_accuracy": float(score)
                    for fold, score in enumerate(fold_scores, start=1)
                },
            }
        )
    return pd.DataFrame(rows), models


def format_summary(data: pd.DataFrame, results: pd.DataFrame) -> str:
    """Render the results as plain text; contains no timestamps, so it is stable."""
    class_counts = data[TARGET_COLUMN].value_counts().sort_index()
    classes = ", ".join(f"{name}={count}" for name, count in class_counts.items())
    first = results.iloc[0]

    header = (
        f"{'Configuration':<18}{'Criterion':<11}{'ccp_alpha':>9}{'Test acc':>10}"
        f"{'Leaves':>8}{'Depth':>7}{'CV mean':>10}{'CV std':>9}"
    )
    table = [header, "-" * len(header)]
    for row in results.itertuples():
        table.append(
            f"{row.configuration:<18}{row.criterion:<11}{row.ccp_alpha:>9.1f}"
            f"{row.test_accuracy:>10.4f}{row.n_leaves:>8d}{row.tree_depth:>7d}"
            f"{row.cv_mean_accuracy:>10.4f}{row.cv_std_accuracy:>9.4f}"
        )

    lines = [
        "Assignment 1 - Question 3: CART (scikit-learn DecisionTreeClassifier)",
        "=" * 69,
        "",
        f"Dataset: {DATA_PATH.name}, {len(data)} examples, "
        f"{len(FEATURE_COLUMNS)} numeric features, classes: {classes}",
        "",
        "Hold-out evaluation (test accuracy, leaves, depth):",
        f"  Stratified split, random_state={RANDOM_STATE}, "
        f"{first.n_train} training / {first.n_test} test examples.",
        "  The same split is used for all four configurations; each tree is",
        "  trained on the training set only.",
        "",
        "Cross-validation (CV mean, CV std):",
        f"  StratifiedKFold(n_splits={N_SPLITS}, shuffle=True, "
        f"random_state={RANDOM_STATE}) on the complete dataset.",
        "  CV std is the population standard deviation (ddof=0) over the folds.",
        "",
        "Classifier settings:",
        f"  random_state={RANDOM_STATE}; all growth limits at their defaults "
        "(max_depth=None, etc.).",
        "",
        *table,
        "",
        "Leaf count and depth describe the tree trained on the 70% training set.",
        "",
        f"Software: Python {platform.python_version()}, "
        f"scikit-learn {sklearn.__version__}, pandas {pd.__version__}, "
        f"NumPy {np.__version__}",
        "",
    ]
    return "\n".join(lines)


def full_tree_figsize(model: DecisionTreeClassifier) -> tuple[float, float]:
    """Size a canvas by leaf count and depth so node text stays legible."""
    return (
        max(9.0, 1.5 * model.get_n_leaves()),
        max(5.0, 1.5 * (model.get_depth() + 1)),
    )


def render_tree(
    model: DecisionTreeClassifier,
    figsize: tuple[float, float],
    max_depth: int | None = None,
) -> tuple[plt.Figure, list]:
    """Draw a tree on a new figure; return the figure and its node annotations."""
    fig, ax = plt.subplots(figsize=figsize)
    annotations = plot_tree(
        model,
        ax=ax,
        feature_names=list(FEATURE_COLUMNS),
        class_names=[str(name) for name in model.classes_],
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=10,
    )
    return fig, annotations


def _save_tree(
    model: DecisionTreeClassifier,
    title: str,
    path: Path,
    figsize: tuple[float, float],
    max_depth: int | None = None,
) -> None:
    fig, _ = render_tree(model, figsize, max_depth)
    fig.axes[0].set_title(title, fontsize=14, pad=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_tree_figures(
    results: pd.DataFrame,
    models: dict[str, DecisionTreeClassifier],
    figures_dir: Path,
) -> list[Path]:
    """Draw each complete tree, plus a top-levels view for the deep ones.

    The depth limit of the top-levels view only affects drawing: it is passed to
    ``plot_tree``, never to the classifier.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for row in results.itertuples():
        model = models[row.configuration]
        leaves = model.get_n_leaves()
        depth = model.get_depth()
        description = (
            f"CART {row.criterion}, ccp_alpha={row.ccp_alpha:g} "
            f"({leaves} leaves, depth {depth}, test accuracy {row.test_accuracy:.4f})"
        )

        full_path = figures_dir / f"{row.configuration}_full_tree.png"
        _save_tree(
            model,
            f"Complete tree: {description}",
            full_path,
            figsize=full_tree_figsize(model),
        )
        written.append(full_path)

        if depth > OVERVIEW_DEPTH:
            overview_path = figures_dir / f"{row.configuration}_top_levels.png"
            _save_tree(
                model,
                f"Top {OVERVIEW_DEPTH} levels only (drawing limit; '(...)' marks "
                f"hidden subtrees)\n{description}",
                overview_path,
                figsize=OVERVIEW_FIGSIZE,
                max_depth=OVERVIEW_DEPTH,
            )
            written.append(overview_path)
    return written


def run(output_dir: Path = OUTPUT_DIR, data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Run the whole experiment and write every output under ``output_dir``."""
    data = load_dataset(data_path)
    results, models = run_experiments(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "question3_results.csv", index=False)
    summary = format_summary(data, results)
    (output_dir / "question3_summary.txt").write_text(summary, encoding="utf-8")
    save_tree_figures(results, models, output_dir / "figures")

    print(summary)
    print(f"Results written to {output_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for the outputs (default: results/cart)",
    )
    run(output_dir=parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
