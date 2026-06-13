#!/bin/bash
set -euo pipefail

python - << 'PYEOF'
import json
import sqlite3
from collections import Counter

import numpy as np

DB_PATH = "/app/iris.db"

# --- Implement load_features ---
conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cur = conn.cursor()
cur.execute("SELECT sepal_length, sepal_width, petal_length, petal_width, label FROM features")
rows = cur.fetchall()
conn.close()

if not rows:
    raise ValueError("features table is empty")

X = np.array([[r["sepal_length"], r["sepal_width"], r["petal_length"], r["petal_width"]]
               for r in rows], dtype=np.float64)
y = np.array([r["label"] for r in rows], dtype=np.int64)

# --- validate_features ---
if X.shape[0] != y.shape[0]:
    raise ValueError("X and y length mismatch")
if X.shape[1] != 4:
    raise ValueError("Expected 4 feature columns")
if not np.isfinite(X).all():
    raise ValueError("X contains NaN or Inf")
if not set(y.tolist()).issubset({0, 1, 2}):
    raise ValueError("Labels must be in {0,1,2}")
if not (X > 0).all():
    raise ValueError("All feature values must be positive")

# --- Write complete app.py with load_features and validate_features implemented ---
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
    """Load all rows from the features table; return (X, y) as numpy arrays.

    Returns:
        tuple: (X, y) where X is shape (n, 4) float64 and y is shape (n,) int.
    Raises:
        ValueError: if the features table is empty or missing required columns.
    """
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
        X = np.array(
            [[r[c] for c in feature_cols] for r in rows], dtype=np.float64
        )
    except Exception as exc:
        raise ValueError(f"Missing required feature column: {exc}") from exc
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    return X, y


def validate_features(X, y):  # noqa: N803
    """Validate feature array X and label array y.

    Raises:
        ValueError: on any validation failure.
    """
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
    """Train a RandomForestClassifier on database features and persist it.

    Returns:
        dict: {"accuracy": float, "precision": float, "f1": float}
    """
    # TODO: implement in milestone 2
    raise NotImplementedError


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

# --- Write validation_report.json ---
label_counts = Counter(int(v) for v in y)
report = {
    "row_count": int(X.shape[0]),
    "feature_columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
    "validation_passed": True,
}
with open("/app/validation_report.json", "w") as f:
    json.dump(report, f)

print("Milestone 1 complete:", report)
PYEOF
