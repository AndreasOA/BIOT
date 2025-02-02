FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash vscode
USER vscode

# Set up Python environment
ENV PATH="${PATH}:/home/vscode/.local/bin"
RUN python3 -m pip install --user --upgrade pip

# Set working directory
WORKDIR /workspace

# Install PyTorch with CUDA support (we'll let requirements.txt handle the specific version)
RUN pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Note: requirements.txt will be installed by postCreateCommand in devcontainer.json
