"""Tests for milestone 2: Train and Persist Model."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, "/app")


class TestMilestone2:
    """Tests for milestone 2: model training and metric persistence."""

    def test_milestone_1_artifacts_persist(self) -> None:
        """The /app/validation_report.json from milestone 1 must still exist."""
        assert Path("/app/validation_report.json").exists()

    def test_model_file_exists(self) -> None:
        """The trained model file /app/model.joblib must exist."""
        assert Path("/app/model.joblib").exists(), (
            "/app/model.joblib does not exist"
        )

    def test_model_is_random_forest(self) -> None:
        """The saved model must be a RandomForestClassifier instance."""
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        clf = joblib.load("/app/model.joblib")
        assert isinstance(clf, RandomForestClassifier), (
            f"Expected RandomForestClassifier, got {type(clf).__name__}"
        )

    def test_model_hyperparameters(self) -> None:
        """The model must have n_estimators=100 and random_state=42."""
        import joblib
        clf = joblib.load("/app/model.joblib")
        assert clf.n_estimators == 100, (
            f"Expected n_estimators=100, got {clf.n_estimators}"
        )
        assert clf.random_state == 42, (
            f"Expected random_state=42, got {clf.random_state}"
        )

    def test_model_predicts_valid_classes(self) -> None:
        """The model must predict labels in {0, 1, 2} for Iris samples."""
        import joblib
        import numpy as np
        clf = joblib.load("/app/model.joblib")
        samples = np.array([
            [5.1, 3.5, 1.4, 0.2],
            [6.7, 3.0, 5.2, 2.3],
            [5.9, 3.0, 4.2, 1.5],
        ], dtype=np.float64)
        preds = clf.predict(samples)
        assert set(preds.tolist()).issubset({0, 1, 2})

    def test_model_accuracy_reasonable(self) -> None:
        """RandomForest on Iris training data must achieve >= 0.90 accuracy."""
        import joblib
        from sklearn.metrics import accuracy_score
        from app import load_features
        clf = joblib.load("/app/model.joblib")
        X, y = load_features()
        acc = accuracy_score(y, clf.predict(X))
        assert acc >= 0.90, f"Model accuracy {acc:.3f} is below 0.90"

    def test_metrics_db_exists(self) -> None:
        """The metrics database /app/metrics.db must exist."""
        assert Path("/app/metrics.db").exists()

    def test_metrics_table_has_rows(self) -> None:
        """The metrics table must have at least one row."""
        conn = sqlite3.connect("/app/metrics.db")
        count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        conn.close()
        assert count >= 1, "metrics table has no rows"

    def test_metrics_run_id_non_empty(self) -> None:
        """run_id in metrics must be a non-empty string."""
        conn = sqlite3.connect("/app/metrics.db")
        row = conn.execute("SELECT run_id FROM metrics LIMIT 1").fetchone()
        conn.close()
        assert row is not None and len(row[0]) > 0, "run_id must be non-empty"

    def test_metrics_values_match_model(self) -> None:
        """Persisted accuracy, precision, f1 must match sklearn recomputation."""
        import joblib
        from sklearn.metrics import accuracy_score, f1_score, precision_score
        from app import load_features

        clf = joblib.load("/app/model.joblib")
        X, y = load_features()
        preds = clf.predict(X)
        expected_acc = accuracy_score(y, preds)
        expected_prec = precision_score(y, preds, average="macro")
        expected_f1 = f1_score(y, preds, average="macro")

        conn = sqlite3.connect("/app/metrics.db")
        row = conn.execute(
            "SELECT accuracy, precision, f1 FROM metrics ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        conn.close()
        acc, prec, f1 = row
        assert abs(acc - expected_acc) < 1e-6, f"accuracy mismatch: {acc} vs {expected_acc}"
        assert abs(prec - expected_prec) < 1e-6, f"precision mismatch: {prec} vs {expected_prec}"
        assert abs(f1 - expected_f1) < 1e-6, f"f1 mismatch: {f1} vs {expected_f1}"

    def test_train_and_save_calls_load_and_validate(self) -> None:
        """train_and_save_model() must call load_features() and validate_features()."""
        from unittest.mock import patch
        from app import train_and_save_model, load_features, validate_features
        with patch("app.load_features", wraps=load_features) as mock_lf, \
             patch("app.validate_features", wraps=validate_features) as mock_vf:
            train_and_save_model()
            mock_lf.assert_called_once()
            mock_vf.assert_called_once()
