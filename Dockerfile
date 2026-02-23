FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git build-essential cmake libz-dev \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && python3.13 -m pip install --upgrade pip

RUN ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

RUN python -m pip install \
    "jax[cuda12]" \
    flax \
    optax \
    chex \
    numpy \
    ale-py \
    "gymnasium[other]" \
    opencv-python-headless \
    pytest \
    pytest-xdist \
    pytest-benchmark \
    matplotlib

WORKDIR /workspace
COPY . /workspace
RUN python -m pip install -e .
