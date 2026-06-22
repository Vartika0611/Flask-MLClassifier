#!/bin/bash
set -euo pipefail

# Write the complete final app.py with all three milestones implemented
cat > /app/app.py << 'PYEOF'
"""Flask inference API — fully implemented."""
import os
import sqlite3

from flask import Flask, jsonify, request

app = Flask(__name__)

DB_PATH = "/app/iris.db"
MODEL_PATH = "/app/model.joblib"
METRICS_DB = "/app/metrics.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_features():
    """Load all rows from the features table; return (X, y) as numpy arrays."""
    import numpy as np
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT sepal_length, sepal_width, petal_length, petal_width, label FROM features"
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        raise ValueError("features table is empty")
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    try:
        X = np.array([[r[c] for c in feature_cols] for r in rows], dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"Missing required feature column: {exc}") from exc
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    return X, y


def validate_features(X, y):  # noqa: N803
    """Validate feature array X and label array y."""
    import numpy as np
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y length mismatch")
    if X.shape[1] != 4:
        raise ValueError("Expected 4 feature columns")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or Inf values")
    if not set(y.tolist()).issubset({0, 1, 2}):
        raise ValueError("Labels must be in {0, 1, 2}")
    if not (X > 0).all():
        raise ValueError("All feature values must be positive")


def train_and_save_model():
    """Train a RandomForestClassifier and persist model + metrics."""
    import uuid
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score

    X, y = load_features()
    validate_features(X, y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, average="macro"))
    f1 = float(f1_score(y, preds, average="macro"))
    joblib.dump(clf, MODEL_PATH)
    conn2 = sqlite3.connect(METRICS_DB)
    conn2.execute(
        "CREATE TABLE IF NOT EXISTS metrics "
        "(run_id TEXT, accuracy REAL, precision REAL, f1 REAL)"
    )
    conn2.execute(
        "INSERT INTO metrics VALUES (?,?,?,?)",
        (str(uuid.uuid4()), acc, prec, f1),
    )
    conn2.commit()
    conn2.close()
    return {"accuracy": acc, "precision": prec, "f1": f1}


@app.route("/train", methods=["POST"])
def train():
    """Trigger model training. Returns JSON with accuracy, precision, f1."""
    try:
        result = train_and_save_model()
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON body {"features": [sl, sw, pl, pw]}, return {"prediction": int}."""
    import numpy as np
    import joblib

    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 503
    data = request.get_json(silent=True)
    if not data or "features" not in data:
        return jsonify({"error": "Missing 'features' key"}), 400
    feats = data["features"]
    if not isinstance(feats, list) or len(feats) != 4:
        return jsonify({"error": "'features' must be a list of 4 numbers"}), 400
    try:
        arr = np.array(feats, dtype=np.float64).reshape(1, -1)
    except Exception:
        return jsonify({"error": "Invalid feature values"}), 400
    clf = joblib.load(MODEL_PATH)
    pred = int(clf.predict(arr)[0])
    return jsonify({"prediction": pred}), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return all training runs as a JSON list."""
    if not os.path.exists(METRICS_DB):
        return jsonify({"error": "No metrics available"}), 503
    conn2 = sqlite3.connect(METRICS_DB)
    rows = conn2.execute(
        "SELECT run_id, accuracy, precision, f1 FROM metrics"
    ).fetchall()
    conn2.close()
    if not rows:
        return jsonify({"error": "No metrics rows"}), 503
    result = [
        {"run_id": r[0], "accuracy": r[1], "precision": r[2], "f1": r[3]}
        for r in rows
    ]
    return jsonify(result), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
PYEOF

# Start Flask, call all endpoints, write api_verified.json
python - << 'INNEREOF'
import json
import subprocess
import time
import urllib.error
import urllib.request

proc = subprocess.Popen(
    ["python", "/app/app.py"],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
time.sleep(4)

BASE = "http://127.0.0.1:5000"

def post_json(url, body=None):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}

def get_json(url):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}

train_status, train_body = post_json(f"{BASE}/train")
predict_status, predict_body = post_json(
    f"{BASE}/predict", {"features": [5.1, 3.5, 1.4, 0.2]}
)
metrics_status, metrics_body = get_json(f"{BASE}/metrics")

proc.terminate()
proc.wait()

report = {
    "train": {"status": train_status},
    "predict": {"status": predict_status,
                "prediction": predict_body.get("prediction", -1)},
    "metrics": {"status": metrics_status,
                "count": len(metrics_body) if isinstance(metrics_body, list) else 0},
}
with open("/app/api_verified.json", "w") as f:
    json.dump(report, f)

print("Milestone 3 complete:", report)
assert train_status == 200, f"POST /train returned {train_status}"
assert predict_status == 200, f"POST /predict returned {predict_status}"
assert metrics_status == 200, f"GET /metrics returned {metrics_status}"
INNEREOF
