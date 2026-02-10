FROM nvidia/cuda:12.6.3-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/hmmer/bin:${VIRTUAL_ENV}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-dev \
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
    && rm -rf /var/lib/apt/lists/*

# Force gcc-12 (avoid gcc-13 ICE on 24.04)
ENV CC=gcc-12
ENV CXX=g++-12

# ---------------------------------------------------------------------
# Python venv  (PIN pip so pip-tools works)
# ---------------------------------------------------------------------
RUN python3 -m venv ${VIRTUAL_ENV} && \
    pip install --no-cache-dir --upgrade "pip<25.3" setuptools wheel && \
    pip install --no-cache-dir "pip-tools==7.5.2"

# ---------------------------------------------------------------------
# HMMER (with seq_limit patch)
# ---------------------------------------------------------------------
RUN mkdir /hmmer_build /hmmer && \
    wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -O /hmmer_build/hmmer-3.4.tar.gz && \
    echo "ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  /hmmer_build/hmmer-3.4.tar.gz" | sha256sum -c - && \
    tar -xzf /hmmer_build/hmmer-3.4.tar.gz -C /hmmer_build

RUN wget -O /hmmer_build/jackhmmer_seq_limit.patch \
    https://raw.githubusercontent.com/google-deepmind/alphafold3/main/docker/jackhmmer_seq_limit.patch

RUN cd /hmmer_build && \
    patch -p0 < jackhmmer_seq_limit.patch && \
    cd /hmmer_build/hmmer-3.4 && \
    ./configure --prefix=/hmmer && \
    make -j$(nproc) && \
    make install && \
    cd easel && make install && \
    rm -rf /hmmer_build

# ---------------------------------------------------------------------
# Clone AlphaPulldown with submodules
# ---------------------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git /app/AlphaPulldown

# ---------------------------------------------------------------------
# Install AlphaFold3 (regenerate dev-requirements inside image)
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown/alphafold3

# force real PyPI, not a mirror
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=600 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV PIP_CACHE_DIR=/tmp/pip-cache
ENV CMAKE_BUILD_PARALLEL_LEVEL=1
ENV CFLAGS="-O2 -pipe"
ENV CXXFLAGS="-O2 -pipe"
RUN apt-get update && apt-get install -y clang lld
ENV CC=clang
ENV CXX=clang++
ENV CMAKE_BUILD_PARALLEL_LEVEL=2


RUN rm -rf "$PIP_CACHE_DIR" && mkdir -p "$PIP_CACHE_DIR" && \
    pip-compile \
        --no-reuse-hashes \
        --extra=dev \
        --generate-hashes \
        --resolver=backtracking \
        --output-file=dev-requirements.txt \
        pyproject.toml && \
    pip install --require-hashes -r dev-requirements.txt && \
    pip install git+https://github.com/openmm/pdbfixer.git && \
    pip install --no-deps . && \
    rm -rf "$PIP_CACHE_DIR"

# ---------------------------------------------------------------------
# Build CCD database
# ---------------------------------------------------------------------
RUN build_data

# ---------------------------------------------------------------------
# Install AlphaPulldown
# ---------------------------------------------------------------------
WORKDIR /app/AlphaPulldown
RUN pip install --no-cache-dir .

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

