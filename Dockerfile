# Use the official nvidia cuda image as base
# https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install necessary dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget curl bzip2 ca-certificates curl git ffmpeg build-essential \
    libopenblas-dev liblapack-dev && \
    apt-get clean
    
# Ensure git is in the PATH and that it is available
ENV PATH /usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/git/bin:$PATH
RUN echo "Checking git version:" && git --version

# Install Miniconda based on the platform
RUN echo "Architecture: $(uname -m)"
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        curl -s -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        curl -s -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh; \
    else \
        echo "Unsupported architecture: $(uname -m)"; exit 1; \
    fi
RUN /bin/bash miniconda.sh -b -p /opt/conda && rm miniconda.sh

# Initialize conda
RUN /opt/conda/bin/conda init bash

# Update conda
RUN conda update -n base -c defaults conda && conda clean -ya

# Create an environment called madness and install packages
RUN /bin/bash -c "source /opt/conda/bin/activate && \
    conda create -n madness python=3.10 -y && \
    conda activate madness && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    # pip install pytorch-lightning==1.9.0 && \
    pip install pytorch-lightning==2.1.0 && \
    pip install python-dotenv numpy scipy matplotlib tqdm networkx fastdtw cvxpy nflows torchinfo wandb && \
    conda install -c conda-forge cupy -y && \
    conda clean -ya"
    
# Ensure the madness environment is activated by default in bash
RUN echo "source /opt/conda/bin/activate madness" >> /root/.bashrc

# Set the default command to run bash
CMD ["bash"]