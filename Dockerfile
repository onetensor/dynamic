FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
# Use stable cu126 index, not nightly
ARG TORCH_INDEX_URL="https://download.pytorch.org/whl/cu126"
ARG TORCH_VERSION="2.9.1"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python-is-python3 python3-dev\
    git curl ca-certificates \
    build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /dynamic

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY requirements.txt /dynamic/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" --upgrade

ARG FLASH_ATTN_WHL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu126torch2.9-cp312-cp312-linux_x86_64.whl"
RUN pip install --no-deps "${FLASH_ATTN_WHL}"

RUN grep -vE '^torch($|[<=>])' requirements.txt > /tmp/requirements.txt && \
    pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

COPY . /dynamic

CMD ["bash"]
ENTRYPOINT []