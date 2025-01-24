# ---------------------------------------------------------
# Base CUDA Image
# ---------------------------------------------------------
FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# ---------------------------------------------------------
# 1) Install System Packages
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
        locales \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 2) Set Locale to en_US.UTF-8
# ---------------------------------------------------------
RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# ---------------------------------------------------------
# 3) Install Miniforge (Conda) at /opt/conda
# ---------------------------------------------------------
RUN wget -q -P /tmp \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniforge3-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# ---------------------------------------------------------
# 4) Install Mamba in the Base Environment
# ---------------------------------------------------------
RUN conda install -y -c conda-forge mamba

# ---------------------------------------------------------
# 5) Create and Populate the "af3" Conda Environment
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
    biopython=1.* \
    appdirs=1.4.* \
    numpy=1.26.* \
    ml-collections \
    && conda clean --all -f -y

# ---------------------------------------------------------
# 6) Pip Install "Pure-Python" Packages in the "af3" Environment
# ---------------------------------------------------------
RUN conda run -n af3 pip install --upgrade pip && \
    conda run -n af3 pip install --no-cache-dir \
        dm-haiku==0.0.13 \
        chex==0.1.87 \
        dm-tree==0.1.8 \
        jaxtyping==0.2.34 \
        jmp==0.0.4 \
        ml-dtypes==0.5.0 \
        "jax[cuda12]" && \
    conda run -n af3 pip cache purge

# ---------------------------------------------------------
# 7) Clone and Install AlphaPulldown
# ---------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git /AlphaPulldown

WORKDIR /AlphaPulldown

# Install AlphaPulldown without dependencies
RUN conda run -n af3 pip install . --no-deps

# ---------------------------------------------------------
# 8) Build AlphaFold 3 Source
# ---------------------------------------------------------
WORKDIR /AlphaPulldown/alphafold3

RUN conda run -n af3 pip install --upgrade pip && \
    conda run -n af3 pip install --no-deps . && \
    conda run -n af3 build_data

# ---------------------------------------------------------
# 9) Set Environment Variables for XLA (Optional)
# ---------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

# ---------------------------------------------------------
# 10) Ensure run_structure_prediction.py is Accessible
# ---------------------------------------------------------
RUN ls -la /AlphaPulldown && \
    ls -la /AlphaPulldown/alphapulldown/scripts && \
    ls -la /AlphaPulldown/alphafold3/ && \
    if [ -f /AlphaPulldown/alphapulldown/scripts/run_structure_prediction.py ]; then \
        ln -s /AlphaPulldown/alphapulldown/scripts/run_structure_prediction.py /usr/local/bin/run_structure_prediction.py && \
        chmod +x /usr/local/bin/run_structure_prediction.py; \
    else \
        echo "Error: run_structure_prediction.py not found in expected locations." && exit 1; \
    fi

# ---------------------------------------------------------
# 11) Provide Default Shell Inside "af3" Environment
# ---------------------------------------------------------
ENV PATH="/opt/conda/envs/af3/bin:$PATH"

# ---------------------------------------------------------
# 12) Set ENTRYPOINT to Bash
# ---------------------------------------------------------
ENTRYPOINT ["/bin/bash"]
