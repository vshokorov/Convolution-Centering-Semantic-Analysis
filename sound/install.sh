#!/usr/bin/env bash

ENV_NAME="sound"
PYTHON_VERSION="3.12"
BRANCH="main"

conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

conda install -c conda-forge -y \
    sox \
    libsndfile \
    ffmpeg

python -m pip install --upgrade pip

python -m pip install \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
    ipykernel \
    wget \
    text-unidecode

python -m pip install \
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${BRANCH}"

python -m ipykernel install --user \
    --name "${ENV_NAME}" \
    --display-name "${ENV_NAME}"
