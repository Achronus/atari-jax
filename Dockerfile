FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    git curl wget build-essential cmake \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install \
      "jax[cpu]" \
      flax \
      optax \
      chex \
      numpy \
      ale-py \
      gymnasium \
      opencv-python-headless \
      pytest \
      pytest-benchmark \
      matplotlib

WORKDIR /workspace
COPY . /workspace
RUN pip install -e .
