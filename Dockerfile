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
    && python3.13 -m pip install --upgrade pip uv

RUN ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python3

# CUDA-capable JAX must be installed before the editable install so pip does
# not downgrade to the CPU-only build declared in pyproject.toml.
RUN python -m pip install "jax[cuda12]"

WORKDIR /workspace
COPY . /workspace

# pip install -e . satisfies all pyproject.toml [dependencies] (chex, ale-py,
# gymnasium, tqdm) and registers atarax as editable at /workspace.
# jax[cuda12] is already present so the >=0.9.0.1 constraint is met without a
# downgrade.  Test-group deps are not in [dependencies] and are installed
# explicitly.
RUN python -m pip install -e . \
    && python -m pip install pytest pytest-xdist pygame
