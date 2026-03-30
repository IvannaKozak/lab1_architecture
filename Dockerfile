# ---------- Stage 1: build dependencies ----------
FROM apache/airflow:2.9.3-python3.11 AS builder

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

COPY requirements.txt /opt/airflow/requirements.txt

RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt && \
    pip install --no-cache-dir dvc mlflow

# ---------- Stage 2: runtime ----------
FROM apache/airflow:2.9.3-python3.11

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
WORKDIR /opt/airflow

# Копіюємо саме Python environment Airflow
COPY --from=builder /home/airflow/.local /home/airflow/.local

ENV PATH="/home/airflow/.local/bin:${PATH}"

COPY . /opt/airflow/project

WORKDIR /opt/airflow/project
