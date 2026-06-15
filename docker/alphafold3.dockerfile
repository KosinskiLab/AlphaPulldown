FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=${VIRTUAL_ENV}
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV PATH="/hmmer/bin:${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        gcc-12 g++-12 \
        build-essential \
        libc6-dev \
        wget \
        ca-certificates \
        git \
        patch \
        xz-utils \
        bzip2 \
        make \
        pkg-config \
        zlib1g-dev \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# Force gcc-12 (avoid gcc-13 ICE on 24.04)
ENV CC=gcc-12
ENV CXX=g++-12

# ---------------------------------------------------------------------
# Python venv and package installer
# ---------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.9.24 /uv /uvx /bin/
RUN uv venv ${VIRTUAL_ENV}

# ---------------------------------------------------------------------
# HMMER (with seq_limit patch)
# ---------------------------------------------------------------------
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -O /hmmer_build/hmmer-3.4.tar.gz && \
    echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  /hmmer_build/hmmer-3.4.tar.gz" | sha256sum -c - && \
    tar -xzf /hmmer_build/hmmer-3.4.tar.gz -C /hmmer_build

COPY alphafold3/docker/jackhmmer_seq_limit.patch /hmmer_build/

RUN cd /hmmer_build && \
    patch -p0 < jackhmmer_seq_limit.patch && \
    cd /hmmer_build/hmmer-3.4 && \
    ./configure --prefix=/hmmer && \
    make -j$(nproc) && \
    make install && \
    cd easel && make install && \
    rm -rf /hmmer_build

# ---------------------------------------------------------------------
# Copy AlphaPulldown with its checked-out submodules. Building from the local
# checkout is important for PR/SIF validation: it exercises this branch's
# alphafold3 submodule pointer instead of cloning the repository default branch.
# ---------------------------------------------------------------------
COPY . /app/AlphaPulldown

# ---------------------------------------------------------------------
# Install AlphaFold3 from its locked v3.0.3/Tokamax environment
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown/alphafold3

ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ALPHAFOLD3=3.0.3

ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV CFLAGS="-O2 -pipe"
ENV CXXFLAGS="-O2 -pipe"

RUN uv sync --frozen --all-groups --no-editable && \
    uv pip install --no-cache git+https://github.com/openmm/pdbfixer.git

# ---------------------------------------------------------------------
# Build CCD database
# ---------------------------------------------------------------------
RUN uv run --no-sync build_data

# ---------------------------------------------------------------------
# Install AlphaPulldown
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown
RUN uv pip install --no-cache .

# ---------------------------------------------------------------------
# Runtime env
# ---------------------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95
ENV TF_FORCE_UNIFIED_MEMORY='1'
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------
RUN python - << 'EOF'
from alphafold3.constants import chemical_components
from alphapulldown.folding_backend.folding_backend import FoldingBackend
print("AF3 + AlphaPulldown import OK, CCD present")
EOF
