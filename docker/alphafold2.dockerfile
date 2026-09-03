# syntax=docker/dockerfile:1.4

# Global build args must be declared before the first FROM so that every stage's
# FROM line can interpolate them.
ARG CUDA=12.2.2

# ---------------------------------------------------------------------------
# Patched HHblits, built in a throwaway stage so the runtime image never gains
# a toolchain.
#
# Stock hhblits keys its realignment bookkeeping by database entry NAME, but
# ffindex names are unique only within one database. AlphaFold searches two at
# once (-d bfd -d uniref30), so identically named entries collide and the wrong
# HMM is reused during MAC realignment, aborting long queries with
#   MergeMasterSlave: did not find N match states in sequence 1 of <hit>
# soedinglab/hh-suite#389 keys those maps by the concrete HHEntry* instead. It is
# unmerged and in no release (3.3.0 is newest and master still has the bug), so it
# is built here from master + the vendored patch. VERSION_PATCH is bumped to 3.3.1
# so features generated with it are distinguishable in recorded metadata.
#
# Base must match the runtime image (ubuntu 20.04) so the binary links cleanly.
# ---------------------------------------------------------------------------
FROM ubuntu:20.04 AS hhblits-builder
RUN set -eux; \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates; \
    rm -rf /var/lib/apt/lists/*
COPY docker/patches/hhsuite-pr389.diff /tmp/hhsuite-pr389.diff
RUN set -eux; \
    git clone --recursive https://github.com/soedinglab/hh-suite.git /tmp/hh-suite; \
    cd /tmp/hh-suite; \
    git apply --exclude=data/test.sh /tmp/hhsuite-pr389.diff; \
    sed -i 's/set(HHSUITE_VERSION_PATCH 0)/set(HHSUITE_VERSION_PATCH 1)/' CMakeLists.txt; \
    cmake -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/opt/hhsuite; \
    cmake --build build -j"$(nproc)"; \
    cmake --install build

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
      ca-certificates curl bzip2 tzdata; \
    rm -rf /var/lib/apt/lists/*

# Micromamba bootstrap (smaller than Miniforge)
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    mkdir -p "$MAMBA_ROOT_PREFIX"; \
    curl -k -L https://micro.mamba.pm/api/micromamba/linux-64/latest \
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
      "numpy<2" \
      "openmm>=8.2" \
      "pdbfixer>=1.10" \
      pip \
      git \
    && micromamba clean -a -y

# Overwrite conda's hhblits with the patched build (see the builder stage above).
# The conda hhsuite package stays installed, so its shared libraries and the other
# hh-suite tools remain available; only the hhblits binary is replaced.
COPY --from=hhblits-builder /opt/hhsuite/bin/hhblits /opt/conda/bin/hhblits
RUN set -eux; \
    out="$(/opt/conda/bin/hhblits -h 2>&1 || true)"; \
    case "$out" in \
      *"HHblits 3.3.1"*) echo "patched hhblits in place" ;; \
      *) printf '%s\n' "$out" | head -3; echo "ERROR: patched hhblits missing"; exit 1 ;; \
    esac

#RUN micromamba run -n base python -m pip install --no-cache-dir "openmm==8.1.1"
RUN python -m pip install --no-cache-dir "setuptools<82" # setuptools>82 breaks pdbfixer at relaxation

# Install the exact checkout supplied as the Docker build context. In particular,
# pull-request image builds must test the submitted code rather than repository
# main.
WORKDIR /AlphaPulldown
COPY . /AlphaPulldown
RUN pip install --no-build-isolation .
# jax takes its CUDA runtime from the nvidia-* wheels, not from the base image, and
# `jax[cuda12]==0.5.3` puts no floor on them - which version you get depends on when
# the image was built. Blackwell cards (RTX PRO 4500/6000, compute capability 12.0 /
# sm_120) need ptxas >= 12.8 and cuDNN >= 9.8; against anything older the first kernel
# compilation dies with "ptxas does not support CC 12.0" / "ptxas too old" and JAX
# never reaches inference. Pin the floors so a rebuild cannot silently drop below
# Blackwell support. They are inert against today's index (pip already resolves
# nvcc 12.9.86 / cuDNN 9.24) and only bind if resolution would otherwise regress.
RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      "jax[cuda12]"==0.5.3 \
      "nvidia-cuda-nvcc-cu12>=12.8" \
      "nvidia-cudnn-cu12>=9.8" \
      "nvidia-cublas-cu12>=12.8" \
      "nvidia-cuda-runtime-cu12>=12.8"

# `pip install --upgrade "jax[cuda12]"==0.5.3` above drags in numpy 2.x, which makes
# AlphaFold multimer fail at runtime: alphafold/data/msa_pairing.py does
# `np.sum(x for x in feats)` and numpy>=2 raises
# `TypeError: Calling np.sum(generator) is deprecated`. The numpy<2 pins earlier in
# this file (conda + pyproject) run BEFORE the jax upgrade so they do not stick; re-pin
# numpy<2 as the LAST dependency step. jax 0.5.3 runs fine with numpy 1.26.x.
RUN pip install --no-cache-dir "numpy<2"

# Exercise the installed entry point without requiring a GPU during the build.
# This catches packaging/import regressions in the exact checkout copied above.
RUN run_structure_prediction_batch.py --helpshort >/dev/null

# AlphaFold's template code formats a hit's `sum_probs` with %.2f in the error
# path of `_process_single_hit`, but sum_probs is legitimately None for some hits
# - the same function's *warning* path already prints it with %s, and
# `_build_query_to_hit_index_mapping`'s caller guards for None explicitly. So any
# query whose template hits include one with sum_probs=None and a featurisation
# error dies with
#   TypeError: must be real number, not NoneType
# instead of skipping that template. Observed on 30 of 2337 UniProt queries in the
# feature-database backfill. Make the error path match the warning path.
RUN AF_TEMPLATES="$(python -c 'import alphafold.data.templates as t; print(t.__file__)')" \
 && grep -q "sum_probs: %.2f" "$AF_TEMPLATES" \
 && sed -i "s/(sum_probs: %\.2f, rank: %d)/(sum_probs: %s, rank: %s)/" "$AF_TEMPLATES" \
 && ! grep -q "sum_probs: %.2f" "$AF_TEMPLATES" \
 && python -c "import alphafold.data.templates" \
 && echo "patched sum_probs formatting in $AF_TEMPLATES"

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

#ENTRYPOINT ["bash"]
