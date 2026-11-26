# Image for both the twin (`rdt twin`) and the replay (`rdt replay`); see compose.yaml.
# The dataset is not in the image: compose mounts it read-only at /app/data/raw/dataset/Full.
FROM python:3.14-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[mqtt]"
COPY config ./config

ENTRYPOINT ["rdt", "--config", "config/docker.yaml"]
