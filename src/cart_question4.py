"""Assignment 1, Question 4: choosing CART's ccp_alpha for the Raisin dataset.

Estimates the best cost-complexity pruning parameter for a Gini tree in two ways:
    Part A: a separate validation set (60% train / 10% validation / 30% test)
    Part B: 10-fold cross-validation on the training set (70% train / 30% test)

Both parts use the 30% test set of Question 3, so test accuracies are comparable.

Run from anywhere with the project virtual environment:

    .venv/bin/python src/cart_question4.py

Outputs (under ``results/cart/question4/``):
    question4_results.csv        selected alpha and final tree of each part
    question4_q3_comparison.csv  Parts A and B next to Question 3's Gini trees
    part_a_candidates.csv        validation accuracy of every candidate alpha
    part_b_candidates.csv        cross validation accuracy of every candidate alpha
    question4_timings.csv        execution times (the only machine-dependent CSV)
    question4_summary.txt        human readable summary
    figures/*.png                accuracy against alpha, and the final trees
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: figures are only written to disk.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

import cart_question3 as q3

OUTPUT_DIR = q3.OUTPUT_DIR / "question4"

VALIDATION_FRACTION = 0.10
TIMING_REPEATS = 5

# Scores closer than this count as tied. Validation accuracies are exact ratios,
# but CV means of equal correct counts can differ by float rounding. The tolerance
# is far below the smallest real difference (one example in 630, about 0.0016).
TIE_TOLERANCE = 1e-12


def build_gini_tree(ccp_alpha: float = 0.0) -> DecisionTreeClassifier:
    # As in Question 3, the fixed seed makes the choice between tied splits
    # deterministic. Growth limits stay at their defaults: only alpha prunes.
    return DecisionTreeClassifier(
        criterion="gini", ccp_alpha=ccp_alpha, random_state=q3.RANDOM_STATE
    )


def split_train_validation(
    X_outer_train: pd.DataFrame, y_outer_train: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split Question 3's 70% training set into 60% and 10% of the full dataset.

    Splitting only the outer training set keeps Question 3's test set untouched.
    The validation size is an example count (10% of the full dataset, so 90)
    because a fraction of 1/7 would depend on float rounding to get exactly 90.
    """
    n_total = round(len(X_outer_train) / (1 - q3.TEST_SIZE))
    return train_test_split(
        X_outer_train,
        y_outer_train,
        test_size=round(VALIDATION_FRACTION * n_total),
        random_state=q3.RANDOM_STATE,
        stratify=y_outer_train,
    )


def candidate_alphas(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Return the distinct alphas on the pruning path of the training set's tree.

    Exact duplicates occur when several subtrees are equally weak; they produce
    the same pruned tree, so they would not be separate candidates. The largest
    alpha stays in, because the assignment evaluates every value. On this training
    set it gives a single-node tree, which scores about 50% on a balanced dataset.
    In cross-validation the same alpha may retain splits in individual folds, whose
    trees differ, so its mean CV accuracy can be higher than that.
    """
    path = build_gini_tree().cost_complexity_pruning_path(X_train, y_train)
    return np.unique(path.ccp_alphas)


def select_alpha(table: pd.DataFrame, score_column: str) -> tuple[float, np.ndarray]:
    """Return the alpha with the best score, and every alpha tied for that score.

    Tie rule: the largest tied alpha wins, because it gives the smallest tree among
    the candidates that score the same. No other selection criterion is used.
    """
    scores = table[score_column].to_numpy()
    is_tied = np.isclose(scores, scores.max(), rtol=0.0, atol=TIE_TOLERANCE)
    tied_alphas = table["ccp_alpha"].to_numpy()[is_tied]
    return float(tied_alphas.max()), tied_alphas


def search_with_validation_set(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Part A: score each candidate on the validation set, then select alpha."""
    rows = []
    for alpha in candidate_alphas(X_train, y_train):
        model = build_gini_tree(alpha).fit(X_train, y_train)
        rows.append(
            {
                "ccp_alpha": float(alpha),
                "validation_accuracy": accuracy_score(
                    y_validation, model.predict(X_validation)
                ),
            }
        )
    table = pd.DataFrame(rows)
    return table, *select_alpha(table, "validation_accuracy")


def search_with_cross_validation(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, float, np.ndarray]:
    """Part B: score each candidate by 10-fold CV, then select alpha.

    The folds partition the training set only, and every candidate is scored on
    the same folds. Each fold grows its own tree, and alpha prunes that tree; the
    alphas themselves come from the tree grown on the whole training set.
    """
    rows = []
    for alpha in candidate_alphas(X_train, y_train):
        fold_scores = cross_val_score(
            build_gini_tree(alpha),
            X_train,
            y_train,
            cv=q3.make_cv_splitter(),
            scoring="accuracy",
        )
        rows.append(
            {
                "ccp_alpha": float(alpha),
                "cv_mean_accuracy": float(fold_scores.mean()),
                # Population standard deviation (ddof=0), as in Question 3.
                "cv_std_accuracy": float(fold_scores.std()),
                **{
                    f"cv_fold_{fold}_accuracy": float(score)
                    for fold, score in enumerate(fold_scores, start=1)
                },
            }
        )
    table = pd.DataFrame(rows)
    return table, *select_alpha(table, "cv_mean_accuracy")


def timed(function, repeats: int) -> tuple[object, list[float]]:
    """Call ``function`` repeatedly; return its result and each run's seconds.

    Repeating reduces timing noise. The work is deterministic, so every call
    returns the same result.
    """
    seconds = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        seconds.append(time.perf_counter() - start)
    return result, seconds


def evaluate_approach(
    approach: str,
    search,
    score_column: str,
    n_search_train: int,
    n_validation: int,
    X_final_train: pd.DataFrame,
    y_final_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timing_repeats: int,
) -> dict:
    """Run the shared steps: search, refit with the chosen alpha, then test once.

    ``search`` never receives the test set. Test data is only used in the final
    scoring below, after alpha and the final tree have been fixed.
    """
    (table, alpha, tied_alphas), search_seconds = timed(search, timing_repeats)
    model, refit_seconds = timed(
        lambda: build_gini_tree(alpha).fit(X_final_train, y_final_train),
        timing_repeats,
    )
    predictions = model.predict(X_test)
    return {
        "approach": approach,
        "candidates": table,
        "model": model,
        "search_seconds": search_seconds,
        "refit_seconds": refit_seconds,
        "row": {
            "approach": approach,
            "criterion": model.criterion,
            "selected_ccp_alpha": alpha,
            "n_candidates": len(table),
            "selection_metric": score_column,
            "selection_score": float(table[score_column].max()),
            "n_tied_candidates": len(tied_alphas),
            "tied_alpha_min": float(tied_alphas.min()),
            "tied_alpha_max": float(tied_alphas.max()),
            "n_search_train": n_search_train,
            "n_validation": n_validation,
            "n_final_train": len(X_final_train),
            "n_test": len(X_test),
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_correct": int((predictions == y_test.to_numpy()).sum()),
            "n_leaves": int(model.get_n_leaves()),
            "tree_depth": int(model.get_depth()),
            "n_nodes": int(model.tree_.node_count),
        },
    }


def run_part_a(
    X_outer_train: pd.DataFrame,
    y_outer_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timing_repeats: int = TIMING_REPEATS,
) -> dict:
    """Part A: select on the validation set, then refit on train + validation."""
    X_train, X_validation, y_train, y_validation = split_train_validation(
        X_outer_train, y_outer_train
    )
    result = evaluate_approach(
        approach="A_validation_set",
        search=lambda: search_with_validation_set(
            X_train, y_train, X_validation, y_validation
        ),
        score_column="validation_accuracy",
        n_search_train=len(X_train),
        n_validation=len(X_validation),
        # The final tree learns from the union of train and validation, as the
        # assignment asks; alpha stays the value chosen on the 540-example tree.
        X_final_train=pd.concat([X_train, X_validation]),
        y_final_train=pd.concat([y_train, y_validation]),
        X_test=X_test,
        y_test=y_test,
        timing_repeats=timing_repeats,
    )
    add_tree_sizes(result["candidates"], X_train, y_train)
    return result


def run_part_b(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    timing_repeats: int = TIMING_REPEATS,
) -> dict:
    """Part B: select by CV on the training set, then refit on all of it."""
    result = evaluate_approach(
        approach="B_cross_validation",
        search=lambda: search_with_cross_validation(X_train, y_train),
        score_column="cv_mean_accuracy",
        n_search_train=len(X_train),
        n_validation=0,
        X_final_train=X_train,
        y_final_train=y_train,
        X_test=X_test,
        y_test=y_test,
        timing_repeats=timing_repeats,
    )
    add_tree_sizes(result["candidates"], X_train, y_train)
    return result


def add_tree_sizes(
    table: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series
) -> None:
    """Record each candidate's tree size on its training set, for reporting only.

    This runs after the timed search, so it does not count towards the timings.
    """
    models = [
        build_gini_tree(alpha).fit(X_train, y_train) for alpha in table["ccp_alpha"]
    ]
    table["n_leaves"] = [int(model.get_n_leaves()) for model in models]
    table["tree_depth"] = [int(model.get_depth()) for model in models]


def question3_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """Refit Question 3's Gini trees on the shared split, for comparison only."""
    rows = []
    for config in q3.CONFIGURATIONS:
        if config.criterion != "gini":
            continue
        model = q3.build_classifier(config).fit(X_train, y_train)
        predictions = model.predict(X_test)
        rows.append(
            {
                "method": f"Q3_{config.name}",
                "ccp_alpha": config.ccp_alpha,
                "n_final_train": len(X_train),
                "n_test": len(X_test),
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_correct": int((predictions == y_test.to_numpy()).sum()),
                "n_leaves": int(model.get_n_leaves()),
                "tree_depth": int(model.get_depth()),
            }
        )
    return rows


def run_experiments(data: pd.DataFrame, timing_repeats: int = TIMING_REPEATS) -> dict:
    """Run both parts and build the comparison with Question 3.

    Comparing test accuracies happens only here, after both parts are complete:
    it reports the outcome and plays no part in choosing alpha or an approach.
    """
    X, y = q3.split_features_target(data)
    # The same call as in Question 3, so all methods share one 270-example test set.
    X_train, X_test, y_train, y_test = q3.make_holdout_split(X, y)

    part_a = run_part_a(X_train, y_train, X_test, y_test, timing_repeats)
    part_b = run_part_b(X_train, y_train, X_test, y_test, timing_repeats)

    comparison = question3_baselines(X_train, y_train, X_test, y_test)
    for part in (part_a, part_b):
        row = part["row"]
        comparison.append(
            {
                "method": f"Q4_{part['approach']}",
                "ccp_alpha": row["selected_ccp_alpha"],
                "n_final_train": row["n_final_train"],
                "n_test": row["n_test"],
                "test_accuracy": row["test_accuracy"],
                "test_correct": row["test_correct"],
                "n_leaves": row["n_leaves"],
                "tree_depth": row["tree_depth"],
            }
        )

    timings = pd.DataFrame(
        [
            {
                "approach": part["approach"],
                "repeats": len(part["search_seconds"]),
                "search_median_seconds": statistics.median(part["search_seconds"]),
                "search_min_seconds": min(part["search_seconds"]),
                "search_max_seconds": max(part["search_seconds"]),
                "refit_median_seconds": statistics.median(part["refit_seconds"]),
                # Search and refit are timed separately, so this adds their medians;
                # it is not the median of a directly measured end-to-end duration.
                "sum_of_medians_seconds": statistics.median(part["search_seconds"])
                + statistics.median(part["refit_seconds"]),
            }
            for part in (part_a, part_b)
        ]
    )
    return {
        "part_a": part_a,
        "part_b": part_b,
        "results": pd.DataFrame([part_a["row"], part_b["row"]]),
        "comparison": pd.DataFrame(comparison),
        "timings": timings,
    }


def format_summary(experiments: dict) -> str:
    """Render the results as text; all numbers come from the current run."""
    a = experiments["part_a"]["row"]
    b = experiments["part_b"]["row"]
    a_time, b_time = experiments["timings"].to_dict("records")

    def part_lines(row: dict, score_name: str) -> list[str]:
        return [
            f"  Candidate alphas:     {row['n_candidates']}",
            f"  Best {score_name}:".ljust(24) + f"{row['selection_score']:.4f}",
            f"  Candidates tied:      {row['n_tied_candidates']} (alpha "
            f"{row['tied_alpha_min']:.6f} to {row['tied_alpha_max']:.6f})",
            f"  Selected alpha:       {row['selected_ccp_alpha']:.6f}",
            f"  Final tree:           {row['n_leaves']} leaves, depth "
            f"{row['tree_depth']}, trained on {row['n_final_train']} examples",
            f"  Test accuracy:        {row['test_accuracy']:.4f} "
            f"({row['test_correct']}/{row['n_test']})",
        ]

    header = (
        f"{'Method':<26}{'ccp_alpha':>10}{'Test acc':>10}{'Correct':>9}"
        f"{'Leaves':>8}{'Depth':>7}"
    )
    table = [header, "-" * len(header)]
    for row in experiments["comparison"].itertuples():
        table.append(
            f"{row.method:<26}{row.ccp_alpha:>10.6f}{row.test_accuracy:>10.4f}"
            f"{row.test_correct:>5d}/{row.n_test:<3d}{row.n_leaves:>8d}"
            f"{row.tree_depth:>7d}"
        )

    difference = a["test_correct"] - b["test_correct"]
    if difference == 0:
        accuracy_outcome = "Parts A and B classify the same number of test examples."
    else:
        leader = "A" if difference > 0 else "B"
        accuracy_outcome = (
            f"Part {leader} classifies {abs(difference)} more of the "
            f"{a['n_test']} test examples correctly."
        )

    lines = [
        "Assignment 1 - Question 4: selecting ccp_alpha for CART (Gini)",
        "=" * 62,
        "",
        "Method:",
        f"  Outer split: Question 3's stratified 70/30 split (random_state="
        f"{q3.RANDOM_STATE}).",
        "  Candidates: distinct alphas of cost_complexity_pruning_path on the",
        "  applicable training set only.",
        "  Tie rule: the largest alpha among candidates with the best score",
        f"  (scores within {TIE_TOLERANCE:g} count as tied).",
        "  The test set is scored once per part, after alpha and the tree are fixed.",
        "",
        f"Part A: validation set ({a['n_search_train']} train / {a['n_validation']} "
        f"validation / {a['n_test']} test)",
        *part_lines(a, "validation acc"),
        "",
        f"Part B: 10-fold CV on the training set ({b['n_search_train']} train / "
        f"{b['n_test']} test)",
        *part_lines(b, "mean CV acc"),
        "",
        "Comparison on the same test set (including Question 3's Gini trees):",
        *table,
        "",
        accuracy_outcome,
        "This compares one split and one test set, so it reports an outcome rather",
        "than a general ranking of the methods.",
        "",
        f"Execution time (median of {a_time['repeats']} runs; machine-dependent):",
        f"  {'':<22}{'Search':>10}{'Refit':>10}{'Sum':>10}",
        f"  {'Part A':<22}{a_time['search_median_seconds']:>9.3f}s"
        f"{a_time['refit_median_seconds']:>9.4f}s"
        f"{a_time['sum_of_medians_seconds']:>9.3f}s",
        f"  {'Part B':<22}{b_time['search_median_seconds']:>9.3f}s"
        f"{b_time['refit_median_seconds']:>9.4f}s"
        f"{b_time['sum_of_medians_seconds']:>9.3f}s",
        f"  Part B / Part A sum: "
        f"{b_time['sum_of_medians_seconds'] / a_time['sum_of_medians_seconds']:.1f} "
        "times",
        "  Search: pruning path, fitting and scoring every candidate (one fit each in",
        "  Part A, ten in Part B, whose path also has more candidates), and selection.",
        "  Refit: fitting the final tree on the 630 examples.",
        "  Sum: median search time + median refit time, not the median of a directly",
        "  measured end-to-end duration (search and refit are timed separately).",
        "  Excluded: data loading, splitting, test scoring, tree-size reporting,",
        "  figures and file writing. Runs use one process.",
        "",
    ]
    return "\n".join(lines)


def _plot_accuracy(
    table: pd.DataFrame,
    score_column: str,
    selected_alpha: float,
    y_label: str,
    title: str,
    path: Path,
    std_column: str | None = None,
) -> None:
    alphas = table["ccp_alpha"].to_numpy()
    scores = table[score_column].to_numpy()
    selected_score = scores[np.flatnonzero(alphas == selected_alpha)[0]]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    # A pruned tree stays the same until the next candidate alpha, hence the steps.
    ax.plot(
        alphas,
        scores,
        marker="o",
        markersize=5,
        linewidth=2,
        drawstyle="steps-post",
        color="#2563a8",
        label=y_label,
    )
    if std_column is not None:
        std = table[std_column].to_numpy()
        ax.fill_between(
            alphas,
            scores - std,
            scores + std,
            step="post",
            color="#2563a8",
            alpha=0.15,
            linewidth=0,
            label="± 1 standard deviation across folds",
        )
    ax.axvline(selected_alpha, color="#555555", linewidth=1, linestyle="--")
    ax.plot(
        [selected_alpha],
        [selected_score],
        marker="o",
        markersize=11,
        markerfacecolor="none",
        markeredgewidth=2,
        color="#b5401f",
        linestyle="none",
        label=f"Selected alpha = {selected_alpha:.6f} (accuracy {selected_score:.4f})",
    )
    # Most candidates sit close to zero with one far larger, so a linear axis would
    # squash them together; symlog also shows alpha = 0, which a log axis cannot.
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlabel("ccp_alpha (linear up to 0.001, logarithmic above)")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_final_tree(row: dict, model: DecisionTreeClassifier, path: Path) -> None:
    fig, _ = q3.render_tree(model, q3.full_tree_figsize(model))
    fig.axes[0].set_title(
        f"Final tree, {row['approach']}: alpha={row['selected_ccp_alpha']:.6f} "
        f"({row['n_leaves']} leaves, depth {row['tree_depth']}, "
        f"test accuracy {row['test_accuracy']:.4f})",
        fontsize=14,
        pad=12,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_figures(experiments: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    part_a = experiments["part_a"]
    part_b = experiments["part_b"]
    _plot_accuracy(
        part_a["candidates"],
        "validation_accuracy",
        part_a["row"]["selected_ccp_alpha"],
        "Validation accuracy",
        "Part A: validation accuracy against ccp_alpha (Gini)",
        figures_dir / "part_a_validation_accuracy_vs_alpha.png",
    )
    _plot_accuracy(
        part_b["candidates"],
        "cv_mean_accuracy",
        part_b["row"]["selected_ccp_alpha"],
        "Mean 10-fold CV accuracy",
        "Part B: mean cross-validation accuracy against ccp_alpha (Gini)",
        figures_dir / "part_b_cv_accuracy_vs_alpha.png",
        std_column="cv_std_accuracy",
    )
    for name, part in (("part_a", part_a), ("part_b", part_b)):
        _plot_final_tree(part["row"], part["model"], figures_dir / f"{name}_final_tree.png")


def run(
    output_dir: Path = OUTPUT_DIR,
    data_path: Path = q3.DATA_PATH,
    timing_repeats: int = TIMING_REPEATS,
) -> dict:
    """Run Question 4 and write every output under ``output_dir``."""
    experiments = run_experiments(q3.load_dataset(data_path), timing_repeats)

    output_dir.mkdir(parents=True, exist_ok=True)
    experiments["results"].to_csv(output_dir / "question4_results.csv", index=False)
    experiments["comparison"].to_csv(
        output_dir / "question4_q3_comparison.csv", index=False
    )
    experiments["part_a"]["candidates"].to_csv(
        output_dir / "part_a_candidates.csv", index=False
    )
    experiments["part_b"]["candidates"].to_csv(
        output_dir / "part_b_candidates.csv", index=False
    )
    experiments["timings"].to_csv(output_dir / "question4_timings.csv", index=False)

    summary = format_summary(experiments)
    (output_dir / "question4_summary.txt").write_text(summary, encoding="utf-8")
    save_figures(experiments, output_dir / "figures")

    print(summary)
    print(f"Results written to {output_dir}")
    return experiments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="directory for the outputs (default: results/cart/question4)",
    )
    run(output_dir=parser.parse_args().output_dir)


if __name__ == "__main__":
    main()
