"""Tests for milestone 3: Serve Predictions Through Flask."""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, "/app")

BASE_URL = "http://127.0.0.1:5000"
_server_proc = None


def _start_server():
    global _server_proc
    if _server_proc is None or _server_proc.poll() is not None:
        _server_proc = subprocess.Popen(
            ["python", "/app/app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
    return _server_proc


def _post(path, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


def _get(path):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


class TestMilestone3:
    """Tests for milestone 3: Flask API serving predictions."""

    def test_milestone_artifacts_persist(self) -> None:
        """Artifacts from milestones 1 and 2 must still exist."""
        assert Path("/app/validation_report.json").exists()
        assert Path("/app/model.joblib").exists()

    def test_api_verified_json_exists(self) -> None:
        """The /app/api_verified.json file must exist."""
        assert Path("/app/api_verified.json").exists()

    def test_api_verified_structure(self) -> None:
        """api_verified.json must contain train, predict, and metrics keys."""
        data = json.loads(Path("/app/api_verified.json").read_text())
        assert "train" in data and "predict" in data and "metrics" in data

    # --- Live /train endpoint ---

    def test_flask_train_live_returns_200(self) -> None:
        """Live POST /train must return HTTP 200."""
        _start_server()
        status, _ = _post("/train")
        assert status == 200, f"Expected 200 from POST /train, got {status}"

    def test_flask_train_live_returns_metrics_keys(self) -> None:
        """Live POST /train response must contain accuracy, precision, f1 as floats."""
        _start_server()
        status, body = _post("/train")
        assert status == 200
        for key in ("accuracy", "precision", "f1"):
            assert key in body, f"Key '{key}' missing from /train response"
            assert isinstance(body[key], float), f"'{key}' must be a float"
            assert 0.0 <= body[key] <= 1.0, f"'{key}' value {body[key]} out of range"

    # --- Live /predict endpoint ---

    def test_flask_predict_live_valid_input(self) -> None:
        """Live POST /predict with valid Iris input must return HTTP 200 and a label."""
        _start_server()
        status, body = _post("/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
        assert status == 200, f"Expected 200, got {status}"
        assert "prediction" in body
        assert body["prediction"] in (0, 1, 2)

    def test_flask_predict_live_missing_key(self) -> None:
        """Live POST /predict without 'features' key must return HTTP 400."""
        _start_server()
        status, _ = _post("/predict", {"data": [1, 2, 3, 4]})
        assert status == 400, f"Expected 400, got {status}"

    def test_flask_predict_live_wrong_length(self) -> None:
        """Live POST /predict with wrong number of features must return HTTP 400."""
        _start_server()
        status, _ = _post("/predict", {"features": [5.1, 3.5]})
        assert status == 400, f"Expected 400, got {status}"

    def test_flask_predict_no_model_returns_503(self) -> None:
        """Live POST /predict must return HTTP 503 when model.joblib does not exist."""
        _start_server()
        model_path = "/app/model.joblib"
        backup = model_path + ".bak"
        os.rename(model_path, backup)
        try:
            status, _ = _post("/predict", {"features": [5.1, 3.5, 1.4, 0.2]})
            assert status == 503, f"Expected 503 when model missing, got {status}"
        finally:
            os.rename(backup, model_path)

    # --- Live /metrics endpoint ---

    def test_flask_metrics_live_returns_200(self) -> None:
        """Live GET /metrics must return HTTP 200 and a non-empty list."""
        _start_server()
        status, body = _get("/metrics")
        assert status == 200, f"Expected 200, got {status}"
        assert isinstance(body, list) and len(body) >= 1

    def test_flask_metrics_live_row_keys(self) -> None:
        """Each metrics row must have run_id, accuracy, precision, f1 keys."""
        _start_server()
        _, body = _get("/metrics")
        row = body[0]
        for key in ("run_id", "accuracy", "precision", "f1"):
            assert key in row, f"Key '{key}' missing from metrics row"

    def test_flask_metrics_no_db_returns_503(self) -> None:
        """Live GET /metrics must return HTTP 503 when metrics.db does not exist."""
        _start_server()
        db_path = "/app/metrics.db"
        backup = db_path + ".bak"
        os.rename(db_path, backup)
        try:
            status, _ = _get("/metrics")
            assert status == 503, f"Expected 503 when metrics.db missing, got {status}"
        finally:
            os.rename(backup, db_path)
