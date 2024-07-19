# Use the official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install necessary dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git ffmpeg && \
    apt-get clean

# Install Miniconda based on the platform
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh; \
    elif [ "$(uname -m)" = "aarch64" ]; then \
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda.sh; \
    else \
        echo "Unsupported architecture: $(uname -m)"; exit 1; \
    fi && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Initialize conda
RUN /opt/conda/bin/conda init bash

# Update conda
RUN conda update -n base -c defaults conda && conda clean -ya

# Create an environment called madness and install packages
RUN /opt/conda/bin/conda create -n madness python=3.10 -y && \
    /bin/bash -c "source activate madness && \
    pip install scipy numpy matplotlib tqdm networkx fastdtw cvxpy numba nflows lightning && \
    conda install -c conda-forge cupy -y"
    
# Ensure the madness environment is activated by default in bash
RUN echo "source activate madness" >> /root/.bashrc

# Set the default command to run bash
CMD ["bash"]