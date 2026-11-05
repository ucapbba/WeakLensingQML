FROM mcr.microsoft.com/devcontainers/python:3-3.10-bookworm

# Native build dependencies for the cpp/ QML library and the pymaster/healpy stack.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        cmake \
        gdb \
        pkg-config \
        libeigen3-dev \
        libgsl-dev \
        libcfitsio-dev \
        libhealpix-cxx-dev \
        libfftw3-dev \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-fonts-recommended \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/pip-tmp/
RUN pip3 install --no-cache-dir -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
