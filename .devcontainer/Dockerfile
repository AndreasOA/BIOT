FROM nvidia/cuda:12.6.0-devel-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    vim \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /home/vscode

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && bash ~/miniconda.sh -b -p /home/vscode/miniconda \
    && rm ~/miniconda.sh

# Add conda to PATH
ENV PATH="/home/vscode/miniconda/bin:${PATH}"

# Set up conda environment from environment.yml
COPY --chown=vscode:vscode environment.yml /home/vscode/environment.yml
RUN conda env create -f /home/vscode/environment.yml

# Initialize conda in shell
RUN conda init bash

# Set default command to activate the environment
RUN echo "conda activate xlstm_biot" >> ~/.bashrc

# Set working directory for mounted code
WORKDIR /workspace