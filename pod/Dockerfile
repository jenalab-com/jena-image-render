# RunPod image for jena-image-render: ComfyUI behind a basic-auth proxy, plus the
# jena_fetch extension (model downloads, chunked uploads, idle self-termination).
# No keys, passwords or model weights are baked in.
FROM pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime

ARG COMFY_REF=v0.38.0
ARG CADDY_VERSION=2.11.4
# The base image uses the Ubuntu system Python (PEP 668); this container is single-purpose.
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PIP_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://github.com/caddyserver/caddy/releases/download/v${CADDY_VERSION}/caddy_${CADDY_VERSION}_linux_amd64.tar.gz" \
    | tar -xz -C /usr/local/bin caddy && caddy version
# ComfyUI is GPL-3.0 and is cloned unmodified from its official repository.
RUN git clone --depth 1 --branch "${COMFY_REF}" https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && python -c "import torch, torchvision; print(f'torch=={torch.__version__}'); print(f'torchvision=={torchvision.__version__}')" > /tmp/constraints.txt \
    && pip install -r /opt/ComfyUI/requirements.txt -c /tmp/constraints.txt \
    && pip install requests

COPY jena_fetch /opt/ComfyUI/custom_nodes/jena_fetch
COPY Caddyfile /opt/Caddyfile
COPY entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 3000
ENTRYPOINT ["/opt/entrypoint.sh"]
