# syntax=docker/dockerfile:1.4
ARG CUDA=12.2.2
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
ARG CUDA

SHELL ["/bin/bash","-o","pipefail","-c"]

# Trim docs/locales to shrink APT-installed footprint
RUN set -eux; \
  echo 'path-exclude=/usr/share/man/*'      >  /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-exclude=/usr/share/doc/*'      >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-exclude=/usr/share/locale/*'   >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-include=/usr/share/locale/en*' >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  printf 'Acquire::Languages "none";\n'     >  /etc/apt/apt.conf.d/99no-languages

# Only minimal runtime deps to bootstrap micromamba; no upgrade, no dev toolchain
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      ca-certificates curl bzip2 tzdata openssh-client; \
    rm -rf /var/lib/apt/lists/*

# Micromamba bootstrap (smaller than Miniforge)
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    mkdir -p "$MAMBA_ROOT_PREFIX"; \
    curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV PATH="/opt/conda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/lib:${LD_LIBRARY_PATH}"

RUN set -eux; \
    micromamba install -y -r "/opt/conda" -n base \
      -c conda-forge -c bioconda \
      python=3.11 \
      kalign2 \
      modelcif \
      hmmer \
      hhsuite \
      pdbfixer=1.9 \
      pip \
      git \
    && micromamba clean -a -y

#RUN micromamba run -n base python -m pip install --no-cache-dir "openmm==8.1.1"

# Clone from repo
RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN  git clone --recurse-submodules https://github.com/KosinskiLab/AlphaPulldown.git
WORKDIR AlphaPulldown

#DEBUG
#WORKDIR /AlphaPulldown
#COPY . /AlphaPulldown
RUN pip install --no-build-isolation .
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      "jax[cuda12]"==0.5.3

# Strip Python caches to reduce layer size
RUN find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /opt/conda -type f -name "*.pyc" -delete \
 && find /AlphaPulldown -type d -name "__pycache__" -prune -exec rm -rf {} + \
 && find /AlphaPulldown -type f -name "*.pyc" -delete

# Clean out APT bootstrap tools (curl/bzip2) to shave a bit more
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get purge -y curl bzip2 || true; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.cache

ENTRYPOINT ["bash"]

