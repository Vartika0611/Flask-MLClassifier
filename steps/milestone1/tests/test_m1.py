"""Tests for milestone 1: Query and Validate Features."""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, "/app")


class TestMilestone1:
    """Tests for milestone 1: feature loading and validation from SQLite."""

    def test_validation_report_exists(self) -> None:
        """The /app/validation_report.json file must exist after milestone 1."""
        assert Path("/app/validation_report.json").exists(), (
            "/app/validation_report.json does not exist"
        )

    def test_validation_report_structure(self) -> None:
        """The report must have keys: row_count, feature_columns, label_counts, validation_passed."""
        report = json.loads(Path("/app/validation_report.json").read_text())
        for key in ("row_count", "feature_columns", "label_counts", "validation_passed"):
            assert key in report, f"Missing key '{key}' in validation_report.json"

    def test_validation_report_row_count(self) -> None:
        """The row_count must equal the actual number of rows in the features table."""
        report = json.loads(Path("/app/validation_report.json").read_text())
        conn = sqlite3.connect("/app/iris.db")
        actual = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        conn.close()
        assert report["row_count"] == actual, (
            f"row_count {report['row_count']} != actual {actual}"
        )

    def test_validation_report_feature_columns(self) -> None:
        """feature_columns must be the four Iris column names in order."""
        report = json.loads(Path("/app/validation_report.json").read_text())
        expected = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        assert report["feature_columns"] == expected

    def test_validation_report_label_counts_exact(self) -> None:
        """label_counts must match the actual per-label counts from iris.db."""
        report = json.loads(Path("/app/validation_report.json").read_text())
        lc = report["label_counts"]
        conn = sqlite3.connect("/app/iris.db")
        for lbl in ("0", "1", "2"):
            expected = conn.execute(
                "SELECT COUNT(*) FROM features WHERE label=?", (int(lbl),)
            ).fetchone()[0]
            assert lbl in lc, f"Label '{lbl}' missing from label_counts"
            assert lc[lbl] == expected, (
                f"label_counts['{lbl}'] = {lc[lbl]}, expected {expected}"
            )
        conn.close()

    def test_validation_passed_true(self) -> None:
        """validation_passed must be true."""
        report = json.loads(Path("/app/validation_report.json").read_text())
        assert report["validation_passed"] is True

    def test_load_features_returns_correct_shapes(self) -> None:
        """load_features() must return X with shape (n,4) and y with shape (n,)."""
        from app import load_features
        X, y = load_features()
        assert X.shape[1] == 4, f"X should have 4 columns, got {X.shape[1]}"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

    def test_validate_features_passes_valid_data(self) -> None:
        """validate_features(X, y) must not raise for valid Iris data."""
        from app import load_features, validate_features
        X, y = load_features()
        validate_features(X, y)  # should not raise

    def test_validate_features_mismatched_dimensions(self) -> None:
        """validate_features must raise ValueError when X and y have different lengths."""
        from app import validate_features
        with pytest.raises(ValueError):
            validate_features(np.ones((10, 4)), np.ones((5,), dtype=int))

    def test_validate_features_nan_values(self) -> None:
        """validate_features must raise ValueError when X contains NaN."""
        from app import validate_features
        X = np.ones((5, 4))
        X[0, 0] = np.nan
        with pytest.raises(ValueError):
            validate_features(X, np.array([0, 1, 2, 0, 1]))

    def test_validate_features_inf_values(self) -> None:
        """validate_features must raise ValueError when X contains Inf."""
        from app import validate_features
        X = np.ones((3, 4))
        X[1, 2] = np.inf
        with pytest.raises(ValueError):
            validate_features(X, np.array([0, 1, 2]))

    def test_validate_features_invalid_label(self) -> None:
        """validate_features must raise ValueError when a label is outside {0,1,2}."""
        from app import validate_features
        with pytest.raises(ValueError):
            validate_features(np.ones((3, 4)), np.array([0, 1, 5]))

    def test_validate_features_non_positive_values(self) -> None:
        """validate_features must raise ValueError when a feature value is <= 0."""
        from app import validate_features
        X = np.ones((3, 4))
        X[0, 0] = -1.0
        with pytest.raises(ValueError):
            validate_features(X, np.array([0, 1, 2]))

    def test_load_features_raises_on_empty_table(self) -> None:
        """load_features must raise ValueError when the features table is empty."""
        from app import load_features
        conn = sqlite3.connect("/app/iris.db")
        # Back up rows, clear table, test, restore
        rows = conn.execute(
            "SELECT sepal_length, sepal_width, petal_length, petal_width, label FROM features"
        ).fetchall()
        conn.execute("DELETE FROM features")
        conn.commit()
        try:
            with pytest.raises(ValueError):
                load_features()
        finally:
            conn.executemany(
                "INSERT INTO features (sepal_length, sepal_width, petal_length, petal_width, label)"
                " VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
            conn.close()
