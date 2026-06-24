FROM python:3.11-slim-bookworm@sha256:ad5daef4139b65d81a9ba35176e5cd1f53e32c2faa793cfcde73a5cfd8e70f37

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        sqlite3 \
        tmux \
        asciinema \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    flask==3.0.3 \
    scikit-learn==1.4.2 \
    numpy==1.26.4 \
    joblib==1.4.2

WORKDIR /app

COPY app/ /app/

RUN python /app/seed_db.py
