FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/setup
COPY requirements-docker.txt ./requirements-docker.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Torch CPU wheels
RUN pip install \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cpu

# PyG CPU wheels matched to torch 2.5.0
RUN pip install \
    pyg_lib \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

RUN pip install torch-geometric==2.6.1
RUN pip install -r requirements-docker.txt

WORKDIR /workspace
CMD ["bash"]
