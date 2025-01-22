# ---------------------------------------------------------
# Base CUDA image
# ---------------------------------------------------------
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# ---------------------------------------------------------
# 1) System packages
#    (Add cmake, ninja, gfortran, dev libs so AlphaFold 3 can build)
# ---------------------------------------------------------
RUN apt update --quiet && \
    DEBIAN_FRONTEND=noninteractive apt install --yes --quiet --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 2) Install Miniforge (Conda at /opt/conda)
# ---------------------------------------------------------
RUN wget -q -P /tmp \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
  bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
  rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# ---------------------------------------------------------
# 3) Install Mamba in the base environment
# ---------------------------------------------------------
RUN conda install -y -c conda-forge mamba

# ---------------------------------------------------------
# 4) Create/populate environment "af3" with big compiled libs
#    (RDKit, TensorFlow, SciPy, JAX, etc. from Conda channels)
# ---------------------------------------------------------
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
    jaxlib=0.4.14 \
    numpy=1.26.* \
    && conda clean --all -f -y

# ---------------------------------------------------------
# 5) pip install "pure-Python" packages not easily found in Conda
# ---------------------------------------------------------
RUN conda run -n af3 pip install --no-cache-dir \
    dm-haiku==0.0.13 \
    chex==0.1.87 \
    dm-tree==0.1.8 \
    jaxtyping==0.2.34 \
    jmp==0.0.4 \
    ml-dtypes==0.5.0 \
    && conda run -n af3 pip cache purge

# ---------------------------------------------------------
# 6) (Optional) Clone & install AlphaPulldown
# ---------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git
WORKDIR /AlphaPulldown
RUN conda run -n af3 pip install . --no-deps

# ---------------------------------------------------------
# 7) Copy & Build AlphaFold 3 Source
#    (Now that we have cmake, ninja, dev libs)
# ---------------------------------------------------------
COPY alphafold3/ /app/alphafold3
WORKDIR /app/alphafold3
RUN conda run -n af3 pip install --upgrade pip && \
    conda run -n af3 pip install --no-deps . && \
    conda run -n af3 build_data

# ---------------------------------------------------------
# 8) Environment variables for XLA, optional
# ---------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

# ---------------------------------------------------------
# 9) Provide default shell inside "af3"
# ---------------------------------------------------------
SHELL ["conda", "run", "-n", "af3", "/bin/bash", "-c"]

ENTRYPOINT ["/bin/bash"]

