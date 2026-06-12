You are working in /app, which contains:
- iris.db       — SQLite database with a `features` table (columns: id, sepal_length, sepal_width, petal_length, petal_width, label)
- app.py        — A Flask application skeleton with stubbed helper functions

The `features` table holds 150 rows of Iris data. Labels are integers: 0, 1, or 2.

Milestone 1: Complete the two feature-loading helper functions in /app/app.py.

1. `load_features()` — Query all rows from iris.db's `features` table. Raise `ValueError` if the table is empty or any of the four feature columns is missing. Return `(X, y)` as numpy float64 arrays where X has shape (n, 4) and y has shape (n,).

2. `validate_features(X, y)` — Validate that X and y have matching first dimension, X has exactly 4 columns, no NaN or Inf values exist in X, all labels in y are in {0, 1, 2}, and all feature values are positive (> 0). Raise `ValueError` for any violation.

Write the results of a validation run to /app/validation_report.json with keys:
- "row_count": int
- "feature_columns": list of the four feature column names in order
- "label_counts": object mapping label string ("0","1","2") to count
- "validation_passed": true
