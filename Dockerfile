# Use the official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH

# Install necessary dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git ffmpeg && \
    apt-get clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Initialize conda
RUN /opt/conda/bin/conda init bash

# Update conda
RUN conda update -n base -c defaults conda && conda clean -ya

# Create an environment called madness and install packages
RUN /opt/conda/bin/conda create -n madness python=3.10 -y && \
    /bin/bash -c "source activate madness && \
    pip install scipy numpy matplotlib tqdm networkx fastdtw cvxpy numba && \
    conda install -c conda-forge cupy -y"
    
# Ensure the madness environment is activated by default in bash
RUN echo "source activate madness" >> /root/.bashrc

# Set the default command to run bash
CMD ["bash"]
