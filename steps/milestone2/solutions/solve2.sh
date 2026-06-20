#!/bin/bash
set -euo pipefail

python - << 'PYEOF'
# Write complete app.py with milestones 1+2 implemented
app_src = '''"""Flask inference API. Milestones 1-3."""
import os
import sqlite3

from flask import Flask, jsonify, request  # noqa: F401

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
    """Trigger model training."""
    # TODO: implement in milestone 3
    raise NotImplementedError


@app.route("/predict", methods=["POST"])
def predict():
    """Return prediction for input features."""
    # TODO: implement in milestone 3
    raise NotImplementedError


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return all training run metrics."""
    # TODO: implement in milestone 3
    raise NotImplementedError


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
'''

with open("/app/app.py", "w") as f:
    f.write(app_src)

import sys
sys.path.insert(0, "/app")
import importlib
import app as app_module
importlib.reload(app_module)
result = app_module.train_and_save_model()
print("Milestone 2 complete:", result)
PYEOF
