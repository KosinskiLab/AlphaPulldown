###############################################################################
# Dockerfile for AlphaFold 3 + AlphaPulldown
# w/ fix for UnicodeDecodeError in build_data
###############################################################################
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# -----------------------------------------------------------------------------
# 1) Basic Setup & System Packages
# -----------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --quiet && \
    apt-get upgrade --yes --quiet && \
    apt-get install --yes --quiet --no-install-recommends \
      software-properties-common \
      git \
      wget \
      gcc \
      g++ \
      gfortran \
      make \
      zlib1g-dev \
      zstd \
      libblas-dev \
      liblapack-dev \
      cmake \
      ninja-build \
      libeigen3-dev \
      locales && \
    rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# -----------------------------------------------------------------------------
# 2) Install Miniforge & Mamba
# -----------------------------------------------------------------------------
RUN wget -q -P /tmp \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

RUN conda install -y -c conda-forge mamba

# -----------------------------------------------------------------------------
# 3) Create "af3" Environment (Core Packages)
# -----------------------------------------------------------------------------
RUN mamba create -n af3 -y \
    -c conda-forge -c bioconda -c omnia \
    python=3.11 \
    tqdm \
    openmm=8.0 \
    pdbfixer=1.9 \
    kalign2 \
    modelcif \
    hmmer \
    hhsuite \
    rdkit=2024.3.5 \
    tensorflow-cpu=2.13.* \
    scipy=1.10.* \
    biopython=1.* \
    appdirs=1.4.* \
    numpy=1.26.* \
    ml-collections \
    && conda clean --all -f -y

# -----------------------------------------------------------------------------
# 4) Extra Python Packages
# -----------------------------------------------------------------------------
RUN conda run -n af3 pip install --upgrade pip && \
    conda run -n af3 pip install --force-reinstall --no-cache-dir \
      dm-haiku==0.0.13 \
      chex==0.1.87 \
      dm-tree==0.1.8 \
      jaxtyping==0.2.34 \
      jmp==0.0.4 \
      ml-dtypes==0.5.0 \
      "jax[cuda12]"==0.5.3 \
      triton==3.1.0 \
      jax-triton==0.2.0 && \
    conda run -n af3 pip cache purge

# -----------------------------------------------------------------------------
# 5) Clone + Install AlphaPulldown (No Deps)
# -----------------------------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git /AlphaPulldown
#COPY . /AlphaPulldown
WORKDIR AlphaPulldown
RUN conda run -n af3 pip install . --no-deps && ls

# -----------------------------------------------------------------------------
# 6) ENV Vars to Force UTF-8 Reading + Build AlphaFold 3
# -----------------------------------------------------------------------------
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUTF8=1
ENV SKBUILD_CONFIGURE_OPTIONS="-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF"
ENV SKBUILD_BUILD_OPTIONS="-j1"
WORKDIR /AlphaPulldown/alphafold3
ENV CMAKE_CXX_STANDARD=17
#RUN apt-get update && apt-get install -y g++-12
ENV CXXFLAGS="-O2 -fno-lto -std=gnu++20 -fno-inline"

RUN conda run -n af3 pip install --upgrade pip scikit_build_core pybind11 "cmake>=3.28" ninja && \
    conda run -n af3 pip install --no-build-isolation --no-deps . && \
    conda run -n af3 build_data

# -----------------------------------------------------------------------------
# 7) Optional XLA Vars
# -----------------------------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95

# -----------------------------------------------------------------------------
# 8) Link run_structure_prediction.py
# -----------------------------------------------------------------------------
RUN ls -la /AlphaPulldown && \
    ls -la /AlphaPulldown/alphapulldown/scripts && \
    ls -la /AlphaPulldown/alphafold3/ && \
    if [ -f /AlphaPulldown/alphapulldown/scripts/run_structure_prediction.py ]; then \
        ln -s /AlphaPulldown/alphapulldown/scripts/run_structure_prediction.py /usr/local/bin/run_structure_prediction.py && \
        chmod +x /usr/local/bin/run_structure_prediction.py; \
    else \
        echo "Error: run_structure_prediction.py not found in expected locations." && exit 1; \
    fi

# -----------------------------------------------------------------------------
# 9) Provide Shell Inside "af3"
# -----------------------------------------------------------------------------
ENV PATH="/opt/conda/envs/af3/bin:$PATH"
ENTRYPOINT ["/bin/bash"]

# -----------------------------------------------------------------------------
# 10) Validate imports automatically (Optional)
# -----------------------------------------------------------------------------
RUN conda run -n af3 python -c "\
import csv; import dataclasses; import datetime; import functools; import logging; import os; \
import pathlib; import string; import textwrap; import time; import typing; \
from collections.abc import Sequence; from typing import List, Dict, Union; \
import haiku as hk; import jax; import numpy as np; from jax import numpy as jnp; \
from alphafold.common import residue_constants; \
from alphafold.common.protein import Protein, to_mmcif; \
from alphafold3.common import base_config, folding_input; \
from alphafold3.constants import chemical_components; \
import alphafold3.cpp; \
from alphafold3.data import featurisation, pipeline; \
from alphafold3.jax.attention import attention; \
from alphafold3.model import features, params, post_processing; \
from alphafold3.model.components import utils; \
from alphafold3.model import model; \
from alphapulldown.folding_backend.folding_backend import FoldingBackend; \
from alphapulldown.objects import MultimericObject, MonomericObject, ChoppedObject; \
print('All imports succeeded!')"
