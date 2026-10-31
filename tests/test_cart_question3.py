"""Tests for the Question 3 CART experiment.

Everything written by these tests goes to pytest's temporary directories, never
to the project's ``results/`` folder.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_numeric_dtype
from pandas.testing import assert_frame_equal
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cart_question3 as q3  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return q3.load_dataset()


@pytest.fixture(scope="module")
def experiment(data):
    return q3.run_experiments(data)


@pytest.fixture(scope="module")
def output_dir(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cart")
    q3.run(output_dir=directory)
    return directory


# --- Dataset loading and structure ---------------------------------------------


def test_data_path_is_project_relative():
    assert q3.DATA_PATH == q3.PROJECT_ROOT / "data" / "Raisin_Dataset.xlsx"
    assert q3.DATA_PATH.is_file()


def test_dataset_shape_and_columns(data):
    assert data.shape == (900, 8)
    assert list(data.columns) == [*q3.FEATURE_COLUMNS, "Class"]


def test_dataset_features_are_numeric_and_complete(data):
    assert all(is_numeric_dtype(data[column]) for column in q3.FEATURE_COLUMNS)
    assert data.notna().all(axis=None)


def test_dataset_classes_are_balanced(data):
    assert data["Class"].value_counts().to_dict() == {"Kecimen": 450, "Besni": 450}


def test_load_dataset_rejects_unexpected_columns(tmp_path, data):
    path = tmp_path / "bad.xlsx"
    data.drop(columns="Area").to_excel(path, sheet_name=q3.SHEET_NAME, index=False)
    with pytest.raises(ValueError, match="Expected columns"):
        q3.load_dataset(path)


def test_split_features_target_excludes_target_from_features(data):
    X, y = q3.split_features_target(data)
    assert "Class" not in X.columns
    assert y.name == "Class"


# --- Split and cross-validation setup ------------------------------------------


def test_holdout_split_is_70_30_stratified_and_disjoint(data):
    X_train, X_test, y_train, y_test = q3.make_holdout_split(
        *q3.split_features_target(data)
    )
    assert (len(X_train), len(X_test)) == (630, 270)
    assert y_train.value_counts().to_dict() == {"Kecimen": 315, "Besni": 315}
    assert y_test.value_counts().to_dict() == {"Kecimen": 135, "Besni": 135}
    assert set(X_train.index).isdisjoint(X_test.index)


def test_holdout_split_is_reproducible(data):
    X, y = q3.split_features_target(data)
    assert q3.make_holdout_split(X, y)[1].index.equals(
        q3.make_holdout_split(X, y)[1].index
    )


def test_cv_splitter_settings():
    splitter = q3.make_cv_splitter()
    assert isinstance(splitter, StratifiedKFold)
    assert splitter.n_splits == 10
    assert splitter.shuffle is True
    assert splitter.random_state == 1


# --- Model configurations ------------------------------------------------------


def test_exactly_the_four_required_configurations_exist():
    assert {(c.criterion, c.ccp_alpha) for c in q3.CONFIGURATIONS} == {
        ("gini", 0.0),
        ("entropy", 0.0),
        ("gini", 0.1),
        ("entropy", 0.1),
    }
    assert len(q3.CONFIGURATIONS) == 4


@pytest.mark.parametrize("config", q3.CONFIGURATIONS, ids=lambda c: c.name)
def test_classifier_parameters_match_configuration(config):
    model = q3.build_classifier(config)
    assert isinstance(model, DecisionTreeClassifier)
    params = model.get_params()
    assert params["criterion"] == config.criterion
    assert params["ccp_alpha"] == config.ccp_alpha
    assert params["random_state"] == 1
    # Growth must not be limited: pruning comes from ccp_alpha alone.
    assert params["max_depth"] is None
    assert params["max_leaf_nodes"] is None
    assert params["min_samples_split"] == 2
    assert params["min_samples_leaf"] == 1


# --- Experiment behaviour ------------------------------------------------------


def test_split_is_computed_once_for_all_configurations(data, monkeypatch):
    calls = []
    real_split = q3.make_holdout_split

    def counting_split(X, y):
        calls.append(1)
        return real_split(X, y)

    monkeypatch.setattr(q3, "make_holdout_split", counting_split)
    q3.run_experiments(data)
    assert len(calls) == 1


def test_models_are_trained_on_the_training_set_only(experiment):
    _, models = experiment
    assert set(models) == {c.name for c in q3.CONFIGURATIONS}
    for model in models.values():
        assert model.tree_.n_node_samples[0] == 630


def test_reported_holdout_metrics_match_recomputation(data, experiment):
    results, models = experiment
    _, X_test, _, y_test = q3.make_holdout_split(*q3.split_features_target(data))
    for row in results.itertuples():
        model = models[row.configuration]
        expected = np.mean(model.predict(X_test) == y_test.to_numpy())
        assert row.test_accuracy == pytest.approx(expected)
        assert row.n_leaves == model.get_n_leaves()


def test_reported_cv_metrics_match_manual_folds(data, experiment):
    results, _ = experiment
    X, y = q3.split_features_target(data)
    for config, row in zip(q3.CONFIGURATIONS, results.itertuples()):
        scores = []
        for train_index, test_index in q3.make_cv_splitter().split(X, y):
            model = q3.build_classifier(config)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            scores.append(model.score(X.iloc[test_index], y.iloc[test_index]))
        assert row.cv_mean_accuracy == pytest.approx(np.mean(scores))
        assert row.cv_std_accuracy == pytest.approx(np.std(scores))


def test_pruning_never_grows_the_tree(experiment):
    results, _ = experiment
    leaves = results.set_index("configuration")["n_leaves"]
    assert leaves["gini_pruned"] <= leaves["gini_unpruned"]
    assert leaves["entropy_pruned"] <= leaves["entropy_unpruned"]


def test_unpruned_trees_are_grown_until_pure(experiment):
    # The dataset has no conflicting duplicates, so every leaf must end up pure.
    _, models = experiment
    for name in ("gini_unpruned", "entropy_unpruned"):
        tree = models[name].tree_
        leaf_impurities = tree.impurity[tree.children_left == -1]
        assert np.all(leaf_impurities == 0.0)


def test_experiment_is_reproducible(data, experiment):
    first, _ = experiment
    second, _ = q3.run_experiments(data)
    assert_frame_equal(first, second, check_exact=True)


# --- Output files --------------------------------------------------------------


def test_csv_round_trips_the_results(output_dir, experiment):
    expected, _ = experiment
    written = pd.read_csv(
        output_dir / "question3_results.csv", float_precision="round_trip"
    )
    assert_frame_equal(written, expected, check_exact=True)


def test_summary_reports_every_configuration(output_dir, experiment):
    results, _ = experiment
    summary = (output_dir / "question3_summary.txt").read_text(encoding="utf-8")
    for row in results.itertuples():
        assert row.configuration in summary
        assert f"{row.test_accuracy:.4f}" in summary
        assert f"{row.cv_mean_accuracy:.4f}" in summary
        assert f"{row.cv_std_accuracy:.4f}" in summary


def test_rerun_writes_identical_tabular_outputs(output_dir, tmp_path):
    q3.run(output_dir=tmp_path)
    for name in ("question3_results.csv", "question3_summary.txt"):
        assert (tmp_path / name).read_bytes() == (output_dir / name).read_bytes()


def test_a_complete_tree_figure_exists_per_configuration(output_dir):
    for config in q3.CONFIGURATIONS:
        figure = output_dir / "figures" / f"{config.name}_full_tree.png"
        assert figure.stat().st_size > 0


def count_overlapping_nodes(model, figsize, max_depth=None):
    fig, annotations = q3.render_tree(model, figsize, max_depth)
    fig.canvas.draw()
    boxes = [
        annotation.get_bbox_patch().get_window_extent()
        for annotation in annotations
        if annotation.get_bbox_patch() is not None
    ]
    plt.close(fig)
    return sum(
        first.overlaps(second)
        for index, first in enumerate(boxes)
        for second in boxes[index + 1 :]
    )


@pytest.mark.parametrize("config", q3.CONFIGURATIONS, ids=lambda c: c.name)
def test_complete_tree_figure_has_no_overlapping_nodes(experiment, config):
    model = experiment[1][config.name]
    assert count_overlapping_nodes(model, q3.full_tree_figsize(model)) == 0


@pytest.mark.parametrize("name", ["gini_unpruned", "entropy_unpruned"])
def test_top_levels_figure_has_no_overlapping_nodes(experiment, name):
    model = experiment[1][name]
    overlaps = count_overlapping_nodes(model, q3.OVERVIEW_FIGSIZE, q3.OVERVIEW_DEPTH)
    assert overlaps == 0


def test_top_levels_figure_exists_for_deep_trees_only(output_dir):
    names = {path.name for path in (output_dir / "figures").glob("*_top_levels.png")}
    assert names == {"gini_unpruned_top_levels.png", "entropy_unpruned_top_levels.png"}
