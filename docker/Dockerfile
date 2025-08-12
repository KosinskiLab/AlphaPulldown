FROM nvidia/cuda:12.6.0-base-ubuntu22.04

# ---------------------------------------------------------
# 1. System Packages Installation
# ---------------------------------------------------------
RUN apt-get update -q && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends \
      software-properties-common \
      git \
      wget \
      gcc \
      g++ \
      make \
      zlib1g-dev \
      zstd \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------
# 2. Miniforge Installation
# ---------------------------------------------------------
RUN wget -q -P /tmp https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

# ---------------------------------------------------------
# 3. Conda + Mamba Environment
# ---------------------------------------------------------
RUN conda install --solver=classic -y \
    conda-forge::conda-libmamba-solver \
    conda-forge::libmamba \
    conda-forge::libmambapy \
    conda-forge::libarchive \
    conda-forge::git && \
    mamba install -y -c conda-forge -c bioconda -c omnia --solver classic \
      openmm==8.0 \
      pdbfixer==1.9 \
      kalign2 \
      modelcif \
      pip \
      hmmer \
      hhsuite \
      python=3.11 && \
    conda clean --all --force-pkgs-dirs --yes

# ---------------------------------------------------------
# 4. Clone AlphaPulldown & Install
# ---------------------------------------------------------
RUN git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git
WORKDIR /AlphaPulldown
RUN pip3 install .

# ---------------------------------------------------------
# 5. Additional Python Tools
# ---------------------------------------------------------
RUN pip3 install --upgrade pip --no-cache-dir && \
    pip3 install --upgrade --no-cache-dir \
      pytest \
      "jax[cuda12]" && \
    chmod u+s /sbin/ldconfig.real

# ---------------------------------------------------------
# 6. AlphaFold 3 Source & Dependencies
# ---------------------------------------------------------
COPY alphafold3/ /app/alphafold3
WORKDIR /app/alphafold3
RUN pip3 install -r dev-requirements.txt && \
    pip3 install --no-deps . && \
    build_data

# ---------------------------------------------------------
# 7. Environment for XLA
# ---------------------------------------------------------
ENV XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
ENV XLA_PYTHON_CLIENT_PREALLOCATE=true
ENV XLA_CLIENT_MEM_FRACTION=0.95

ENTRYPOINT ["bash"]

