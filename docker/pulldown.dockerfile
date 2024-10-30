# syntax = docker/dockerfile:1.4

ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
ARG CUDA

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt update -y && apt upgrade -y
RUN apt install -y --reinstall libp11-kit0 libffi7
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        tzdata \
        wget \
        bc \
        openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

RUN wget -q -P /tmp https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniforge3-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"

RUN conda install --solver=classic -y \
    conda-forge::conda-libmamba-solver \
    conda-forge::libmamba \
    conda-forge::libmambapy \
    conda-forge::libarchive \
    conda-forge::git

RUN mamba install -y -c conda-forge -c bioconda -c omnia --solver classic \
      openmm==8.0 \
      pdbfixer==1.9 \
      kalign2 \
      modelcif \
      pip \
      hmmer \
      hhsuite \
      python=3.10 \
      && conda clean --all --force-pkgs-dirs --yes

RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts

RUN --mount=type=ssh git clone --recurse-submodules git@github.com:KosinskiLab/AlphaPulldown.git
WORKDIR AlphaPulldown
RUN pip3 install .

RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      pytest \
      "jax[cuda12]"
RUN chmod u+s /sbin/ldconfig.real

ENTRYPOINT ["bash"]
