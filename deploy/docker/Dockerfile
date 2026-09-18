# YUE2 // GROOVE — Linux + NVIDIA CUDA image.
#
# Same environment docs/LINUX_CUDA.md validates (torch 2.10.0+cu128, yue2-infer 0.1.6, the
# SheetSage2 Cover environment), baked once so a pod or a desktop container starts in minutes
# instead of installing for half an hour. Contains code only: the YuE2 / SheetSage2 / MERT
# weights (CC BY-NC 4.0) are downloaded by the user at first start into /data (mount it).
#
#   docker build -f deploy/docker/Dockerfile -t yue2-groove .
#   docker run --gpus all -p 7860:7860 -v groove-data:/data ghcr.io/deadjoe/yue2_groove:latest
#
# Base: RunPod's PyTorch image (Ubuntu 24.04, CUDA 12.8.1, Python 3.12, uv, ffmpeg, nginx,
# sshd). Its /start.sh stays the entrypoint; it runs /pre_start.sh, which is ours.
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HOME=/data/hf \
    YUE2_GROOVE_RUNS=/data/runs \
    YUE2_GROOVE_MODELS=/data/models \
    YUE2_GROOVE_SHEETSAGE_PYTHON=/app/.venv-sheetsage2/bin/python \
    YUE2_GROOVE_PORT=7860 \
    PYTHONUTF8=1

WORKDIR /app

# 1. The app's dependencies and upstream YuE2, on the cu128 torch wheel — torch first so PyPI's
#    torch is never pulled (docs/LINUX_CUDA.md §6). Only pyproject.toml and the override file
#    are copied here, so this layer (and the SheetSage one below) stays cached across code
#    changes; the app itself is installed last.
COPY pyproject.toml /app/
COPY overrides/ /app/overrides/
RUN uv venv env --python 3.12 \
 && uv pip install --python env/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 \
 && python3 -c "import tomllib; p = tomllib.load(open('pyproject.toml', 'rb'))['project']; print('\\n'.join(p['dependencies'] + p['optional-dependencies']['yue2']))" > /tmp/requirements.txt \
 && uv pip install --python env/bin/python -r /tmp/requirements.txt --overrides overrides/linux.txt hf_transfer \
 && rm /tmp/requirements.txt \
 && env/bin/python -c "import yue2, gradio; print('deps ok')"

# 2. The SheetSage2 (Cover) environment: its own Python 3.11 venv and torch 2.8 / cu126, as in
#    the README. Only the model's requirements.txt is fetched at build time; the model itself is
#    downloaded at first start.
RUN uv python install 3.11 \
 && uv venv .venv-sheetsage2 --python 3.11 \
 && uv pip install --python .venv-sheetsage2/bin/python huggingface-hub==0.36.0 hf_transfer \
 && uv pip install --python .venv-sheetsage2/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 \
 && .venv-sheetsage2/bin/hf download m-a-p/SheetSage2 requirements.txt --local-dir /tmp/sheetsage2 \
 && uv pip install --python .venv-sheetsage2/bin/python -r /tmp/sheetsage2/requirements.txt \
 && rm -rf /tmp/sheetsage2 \
 && .venv-sheetsage2/bin/python -c "import torch, transformers; print('sheetsage env ok', torch.__version__)"

# 3. The app itself (code only; see .dockerignore).
COPY . /app
RUN uv pip install --python env/bin/python -e . --no-deps \
 && env/bin/python -c "import yue2, yue2_groove, gradio; print('groove ok')"

COPY deploy/docker/groove-start.sh /usr/local/bin/groove-start
COPY deploy/docker/gpu_check.py /usr/local/bin/groove-gpu-check
COPY deploy/docker/pre_start.sh /pre_start.sh
RUN chmod +x /usr/local/bin/groove-start /pre_start.sh && mkdir -p /data

EXPOSE 7860
VOLUME ["/data"]
# CMD is inherited from the base image: /start.sh (nginx, then /pre_start.sh -> groove-start,
# then optional sshd/jupyter, then sleep infinity).
