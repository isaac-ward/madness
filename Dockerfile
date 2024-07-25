# Use the official Ubuntu base image
FROM ubuntu:20.04

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
RUN /opt/conda/bin/conda create -n madness python=3.10 -y
RUN /bin/bash -c "source activate madness && pip install scipy"
RUN /bin/bash -c "source activate madness && pip install numpy"
RUN /bin/bash -c "source activate madness && pip install matplotlib"
RUN /bin/bash -c "source activate madness && pip install tqdm"
RUN /bin/bash -c "source activate madness && pip install networkx"
RUN /bin/bash -c "source activate madness && pip install fastdtw"
RUN /bin/bash -c "source activate madness && pip install cvxpy"
RUN /bin/bash -c "source activate madness && pip install numba"
RUN /bin/bash -c "source activate madness && pip install nflows"
RUN /bin/bash -c "source activate madness && pip install lightning"
RUN /bin/bash -c "source activate madness && pip install wandb"
RUN /bin/bash -c "source activate madness && conda install -c conda-forge cupy -y"
    
# Ensure the madness environment is activated by default in bash
RUN echo "source activate madness" >> /root/.bashrc

# Set the default command to run bash
CMD ["bash"]