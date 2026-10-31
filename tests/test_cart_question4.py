"""Tests for the Question 4 alpha selection.

Everything written by these tests goes to pytest's temporary directories, never
to the project's ``results/`` folder.
"""

import functools
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cart_question3 as q3  # noqa: E402
import cart_question4 as q4  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return q3.load_dataset()


@pytest.fixture(scope="module")
def outer_split(data):
    """X_train, X_test, y_train, y_test of the shared 70/30 split."""
    return q3.make_holdout_split(*q3.split_features_target(data))


@pytest.fixture(scope="module")
def inner_split(outer_split):
    """X_train, X_validation, y_train, y_validation of Part A."""
    X_train, _, y_train, _ = outer_split
    return q4.split_train_validation(X_train, y_train)


@pytest.fixture(scope="module")
def experiments(data):
    return q4.run_experiments(data, timing_repeats=1)


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("question4")
    q4.run(output_dir=directory, timing_repeats=1)
    return directory


@pytest.fixture
def recorded_calls(monkeypatch):
    """Record the data rows of every tree fit and predict, and each fit's criterion."""
    calls = {"fit": [], "predict": [], "fit_criterion": []}

    def spy_on(method_name):
        original = getattr(DecisionTreeClassifier, method_name)

        @functools.wraps(original)
        def wrapper(self, X, *args, **kwargs):
            calls[method_name].append(set(X.index))
            if method_name == "fit":
                calls["fit_criterion"].append(self.criterion)
            return original(self, X, *args, **kwargs)

        monkeypatch.setattr(DecisionTreeClassifier, method_name, wrapper)

    spy_on("fit")
    spy_on("predict")
    return calls


def run_part(part, outer_split, X_test=None, y_test=None):
    X_train, default_X_test, y_train, default_y_test = outer_split
    run = q4.run_part_a if part == "A" else q4.run_part_b
    return run(
        X_train,
        y_train,
        default_X_test if X_test is None else X_test,
        default_y_test if y_test is None else y_test,
        timing_repeats=1,
    )


# --- Sample counts -------------------------------------------------------------


def test_part_a_sample_counts_and_stratification(outer_split, inner_split):
    X_train, X_validation, y_train, y_validation = inner_split
    assert (len(X_train), len(X_validation), len(outer_split[1])) == (540, 90, 270)
    assert y_train.value_counts().to_dict() == {"Kecimen": 270, "Besni": 270}
    assert y_validation.value_counts().to_dict() == {"Kecimen": 45, "Besni": 45}


def test_part_b_sample_counts(outer_split):
    assert (len(outer_split[0]), len(outer_split[1])) == (630, 270)


def test_part_a_sets_partition_the_dataset(data, outer_split, inner_split):
    train, validation, test = (
        set(inner_split[0].index),
        set(inner_split[1].index),
        set(outer_split[1].index),
    )
    assert train.isdisjoint(validation)
    assert test.isdisjoint(train | validation)
    assert train | validation | test == set(data.index)
    assert train | validation == set(outer_split[0].index)


def test_reported_counts_match_the_design(experiments):
    a, b = experiments["part_a"]["row"], experiments["part_b"]["row"]
    assert (a["n_search_train"], a["n_validation"]) == (540, 90)
    assert (b["n_search_train"], b["n_validation"]) == (630, 0)
    for row in (a, b):
        assert (row["n_final_train"], row["n_test"]) == (630, 270)


def test_inner_split_is_reproducible(outer_split, inner_split):
    X_train, _, y_train, _ = outer_split
    repeated_validation = q4.split_train_validation(X_train, y_train)[1]
    assert repeated_validation.index.equals(inner_split[1].index)


# --- Gini and the final models -------------------------------------------------


@pytest.mark.parametrize("part", ["A", "B"])
def test_every_tree_fitted_during_a_part_uses_gini(outer_split, recorded_calls, part):
    # Covers the pruning path, every candidate (and CV fold) and the final refit.
    result = run_part(part, outer_split)
    assert len(recorded_calls["fit_criterion"]) > result["row"]["n_candidates"]
    assert set(recorded_calls["fit_criterion"]) == {"gini"}


def test_build_gini_tree_settings():
    params = q4.build_gini_tree(0.01).get_params()
    assert params["criterion"] == "gini"
    assert params["ccp_alpha"] == 0.01
    assert params["random_state"] == 1
    assert params["max_depth"] is None
    assert params["max_leaf_nodes"] is None


@pytest.mark.parametrize("part", ["part_a", "part_b"])
def test_final_model_uses_gini_and_chosen_alpha(experiments, part):
    result = experiments[part]
    assert result["model"].criterion == "gini"
    assert result["row"]["criterion"] == "gini"
    assert result["model"].ccp_alpha == result["row"]["selected_ccp_alpha"]


def test_final_models_train_on_all_630_examples(experiments):
    for part in ("part_a", "part_b"):
        assert experiments[part]["model"].tree_.n_node_samples[0] == 630


# --- Candidate alphas ----------------------------------------------------------


def test_part_a_candidates_come_from_540_training_examples(experiments, inner_split):
    X_train, _, y_train, _ = inner_split
    path = DecisionTreeClassifier(
        criterion="gini", random_state=1
    ).cost_complexity_pruning_path(X_train, y_train)
    assert experiments["part_a"]["candidates"]["ccp_alpha"].tolist() == (
        np.unique(path.ccp_alphas).tolist()
    )


def test_part_b_candidates_come_from_630_training_examples(experiments, outer_split):
    X_train, _, y_train, _ = outer_split
    path = DecisionTreeClassifier(
        criterion="gini", random_state=1
    ).cost_complexity_pruning_path(X_train, y_train)
    assert experiments["part_b"]["candidates"]["ccp_alpha"].tolist() == (
        np.unique(path.ccp_alphas).tolist()
    )


def test_candidates_are_distinct_and_cover_both_ends_of_the_path(experiments):
    for part in ("part_a", "part_b"):
        table = experiments[part]["candidates"]
        assert table["ccp_alpha"].is_unique
        assert table["ccp_alpha"].is_monotonic_increasing
        assert table["ccp_alpha"].iloc[0] == 0.0
        assert table["n_leaves"].iloc[-1] == 1
        assert table["n_leaves"].is_monotonic_decreasing


# --- Which data each step touches ----------------------------------------------


def test_part_a_fits_only_on_allowed_data_and_tests_once(
    outer_split, inner_split, recorded_calls
):
    test_rows = set(outer_split[1].index)
    train_rows = set(inner_split[0].index)
    validation_rows = set(inner_split[1].index)

    run_part("A", outer_split)

    # Every fit uses the 540 examples, except the one final refit on all 630.
    final_rows = train_rows | validation_rows
    assert recorded_calls["fit"].count(final_rows) == 1
    assert all(rows in (train_rows, final_rows) for rows in recorded_calls["fit"])
    predict_on_test = [rows for rows in recorded_calls["predict"] if rows & test_rows]
    assert predict_on_test == [test_rows]
    assert all(
        rows == validation_rows
        for rows in recorded_calls["predict"]
        if rows != test_rows
    )


def test_part_b_cv_uses_only_the_training_set(outer_split, monkeypatch, recorded_calls):
    X_train, X_test, _, _ = outer_split
    train_rows = set(X_train.index)
    test_rows = set(X_test.index)

    cv_inputs = []
    real_cross_val_score = q4.cross_val_score

    def spy(estimator, X, y, **kwargs):
        cv_inputs.append(set(X.index))
        return real_cross_val_score(estimator, X, y, **kwargs)

    monkeypatch.setattr(q4, "cross_val_score", spy)
    result = run_part("B", outer_split)

    assert len(cv_inputs) == result["row"]["n_candidates"]
    assert all(rows == train_rows for rows in cv_inputs)
    assert all(rows <= train_rows for rows in recorded_calls["fit"])
    assert all(rows.isdisjoint(test_rows) for rows in recorded_calls["fit"])
    predict_on_test = [rows for rows in recorded_calls["predict"] if rows & test_rows]
    assert predict_on_test == [test_rows]
    # Ten folds per candidate: each fold trains on 567 and scores the other 63.
    fold_fits = [rows for rows in recorded_calls["fit"] if len(rows) == 567]
    assert len(fold_fits) == 10 * result["row"]["n_candidates"]


@pytest.mark.parametrize("part", ["A", "B"])
def test_changing_the_test_set_changes_nothing_but_the_test_score(outer_split, part):
    _, X_test, _, y_test = outer_split
    baseline = run_part(part, outer_split)

    corrupted_X = X_test.sample(frac=1.0, random_state=0).set_axis(X_test.index) * 1.5
    flipped_y = y_test.map({"Kecimen": "Besni", "Besni": "Kecimen"})
    changed = run_part(part, outer_split, corrupted_X, flipped_y)

    assert_frame_equal(baseline["candidates"], changed["candidates"], check_exact=True)
    assert baseline["row"]["selected_ccp_alpha"] == changed["row"]["selected_ccp_alpha"]
    assert export_text(baseline["model"]) == export_text(changed["model"])
    assert baseline["row"]["test_accuracy"] != changed["row"]["test_accuracy"]


def test_search_functions_do_not_accept_test_data():
    for function in (q4.search_with_validation_set, q4.search_with_cross_validation):
        assert not any("test" in name for name in inspect.signature(function).parameters)


# --- Selection rule ------------------------------------------------------------


def test_selection_maximises_the_score(experiments):
    for part, column in (
        ("part_a", "validation_accuracy"),
        ("part_b", "cv_mean_accuracy"),
    ):
        result = experiments[part]
        table = result["candidates"].set_index("ccp_alpha")
        selected = result["row"]["selected_ccp_alpha"]
        best = table[column].max()
        assert table.loc[selected, column] == pytest.approx(best, abs=1e-12)


def test_tie_rule_picks_the_largest_tied_alpha():
    table = pd.DataFrame(
        {"ccp_alpha": [0.0, 0.01, 0.02, 0.03], "score": [0.8, 0.9, 0.9, 0.7]}
    )
    alpha, tied = q4.select_alpha(table, "score")
    assert alpha == 0.02
    assert tied.tolist() == [0.01, 0.02]


def test_float_rounding_does_not_break_a_tie():
    table = pd.DataFrame(
        {"ccp_alpha": [0.01, 0.02], "score": [0.8, 0.8 - 1e-15]}
    )
    assert q4.select_alpha(table, "score")[0] == 0.02


def test_one_example_difference_is_not_a_tie():
    table = pd.DataFrame(
        {"ccp_alpha": [0.01, 0.02], "score": [500 / 630, 499 / 630]}
    )
    alpha, tied = q4.select_alpha(table, "score")
    assert alpha == 0.01
    assert tied.tolist() == [0.01]


# --- Reported metrics and Question 3 comparison --------------------------------


def test_test_accuracy_matches_recomputation(experiments, outer_split):
    _, X_test, _, y_test = outer_split
    for part in ("part_a", "part_b"):
        result = experiments[part]
        predictions = result["model"].predict(X_test)
        correct = (predictions == y_test.to_numpy()).sum()
        assert result["row"]["test_correct"] == correct
        assert result["row"]["test_accuracy"] == pytest.approx(correct / 270)


def test_question3_baselines_match_question3(data, experiments):
    q3_results, _ = q3.run_experiments(data)
    q3_results = q3_results.set_index("configuration")
    comparison = experiments["comparison"].set_index("method")
    for name, alpha in (("gini_unpruned", 0.0), ("gini_pruned", 0.1)):
        baseline = comparison.loc[f"Q3_{name}"]
        assert baseline["ccp_alpha"] == alpha
        assert baseline["test_accuracy"] == q3_results.loc[name, "test_accuracy"]
        assert baseline["n_leaves"] == q3_results.loc[name, "n_leaves"]
        assert baseline["tree_depth"] == q3_results.loc[name, "tree_depth"]


def test_comparison_lists_all_four_methods(experiments):
    assert experiments["comparison"]["method"].tolist() == [
        "Q3_gini_unpruned",
        "Q3_gini_pruned",
        "Q4_A_validation_set",
        "Q4_B_cross_validation",
    ]


# --- Reproducibility, timing and output files ----------------------------------


def test_experiment_is_reproducible(data, experiments):
    second = q4.run_experiments(data, timing_repeats=1)
    assert_frame_equal(experiments["results"], second["results"], check_exact=True)
    assert_frame_equal(
        experiments["comparison"], second["comparison"], check_exact=True
    )
    for part in ("part_a", "part_b"):
        assert_frame_equal(
            experiments[part]["candidates"],
            second[part]["candidates"],
            check_exact=True,
        )


def test_timed_repeats_the_call_and_returns_its_result():
    calls = []
    result, seconds = q4.timed(lambda: calls.append(1) or "done", 3)
    assert result == "done"
    assert len(calls) == len(seconds) == 3
    assert all(value >= 0 for value in seconds)


def test_timings_are_reported_for_both_parts(experiments):
    timings = experiments["timings"]
    assert timings["approach"].tolist() == ["A_validation_set", "B_cross_validation"]
    assert (timings["search_median_seconds"] > 0).all()
    assert (timings["refit_median_seconds"] > 0).all()


def test_expected_output_files_exist(output_dir):
    for name in (
        "question4_results.csv",
        "question4_q3_comparison.csv",
        "part_a_candidates.csv",
        "part_b_candidates.csv",
        "question4_timings.csv",
        "question4_summary.txt",
        "figures/part_a_validation_accuracy_vs_alpha.png",
        "figures/part_b_cv_accuracy_vs_alpha.png",
        "figures/part_a_final_tree.png",
        "figures/part_b_final_tree.png",
    ):
        assert (output_dir / name).stat().st_size > 0


def test_candidate_csv_round_trips(output_dir, experiments):
    for part, name in (
        ("part_a", "part_a_candidates.csv"),
        ("part_b", "part_b_candidates.csv"),
    ):
        written = pd.read_csv(output_dir / name, float_precision="round_trip")
        assert_frame_equal(written, experiments[part]["candidates"], check_exact=True)


def test_summary_reports_selected_alpha_and_test_accuracy(output_dir, experiments):
    summary = (output_dir / "question4_summary.txt").read_text(encoding="utf-8")
    for part in ("part_a", "part_b"):
        row = experiments[part]["row"]
        assert f"{row['selected_ccp_alpha']:.6f}" in summary
        assert f"{row['test_accuracy']:.4f}" in summary


def test_rerun_writes_identical_deterministic_outputs(output_dir, tmp_path):
    q4.run(output_dir=tmp_path, timing_repeats=1)
    for name in (
        "question4_results.csv",
        "question4_q3_comparison.csv",
        "part_a_candidates.csv",
        "part_b_candidates.csv",
    ):
        assert (tmp_path / name).read_bytes() == (output_dir / name).read_bytes()
