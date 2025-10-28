# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=registry.git.embl.de/grp-cbbcs/abcdesktop-apps/base-image:ubuntu22-cuda-11-8
FROM ${BASE_IMAGE}
ARG CUDA=11.8.0

SHELL ["/bin/bash","-o","pipefail","-c"]

# ---- prevent docs/locales from being installed (much faster than deleting later)
RUN set -eux; \
  echo 'path-exclude=/usr/share/man/*'             >  /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-exclude=/usr/share/doc/*'             >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-exclude=/usr/share/locale/*'          >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  echo 'path-include=/usr/share/locale/en*'        >> /etc/dpkg/dpkg.cfg.d/01_nodoc; \
  printf 'Acquire::Languages "none";\n'            >  /etc/apt/apt.conf.d/99no-languages

ENV TZ=Etc/UTC \
    DEBIAN_FRONTEND=noninteractive

# ---- base runtime deps (cached apt metadata with BuildKit mounts; no leftover lists)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends tzdata wget ca-certificates bzip2 git openssh-client; \
    rm -rf /var/lib/apt/lists/*

# ---- micromamba bootstrap (pin to current 'latest' endpoint; change if you want a fixed version)
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    MAMBA_NO_BANNER=1 \
    CONDA_CHANNEL_PRIORITY=strict
RUN set -eux; \
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    micromamba --help >/dev/null

# ---- env creation (done in one layer; clean immediately)
RUN set -eux; \
    micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
        python=3.10 openmm=8 pdbfixer=1.9 kalign2 \
        importlib-metadata modelcif hmmer hhsuite pip; \
    micromamba clean -y --all

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ---- pip defaults
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ---- PyTorch (CUDA 11.8) from PyTorch index
ARG TORCH_VERSION=2.1.2
ARG TORCH_CUDA=cu118
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_CUDA}
RUN set -eux; \
    python -m pip install --upgrade pip; \
    python -m pip install \
        torch==${TORCH_VERSION}+${TORCH_CUDA} \
        --index-url ${TORCH_INDEX_URL}

# ---- project python deps (after torch)
# NOTE: if you want reproducibility, pin to tags/SHAs instead of @main
RUN set -eux; \
    python -m pip install --no-build-isolation \
        git+https://github.com/dptech-corp/Uni-Core.git@main; \
    python -m pip install --no-build-isolation \
        git+https://github.com/KosinskiLab/AlphaPulldown.git@main; \
    python -m pip install pytest; \
    # optional: strip tests, dist-info RECORDs, caches to shrink a bit
    find /opt/conda -type d -name "__pycache__" -prune -exec rm -rf {} +; \
    find /opt/conda -type f -name "*.pyc" -delete

# ---- trim build-time tools (kept until after git+pip installs)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    set -eux; \
    apt-get purge -y git openssh-client wget bzip2; \
    apt-get autoremove -y; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/* /root/.cache

# -------------- abcdesktop labels/env --------------
LABEL oc.icon="alphafold.svg"
LABEL oc.icondata="PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkSI[...]"
LABEL oc.keyword="AlphaLink"
LABEL oc.cat="development"
LABEL oc.desktopfile="org.kde.konsole.desktop"
LABEL oc.launch="af1.konsole"
LABEL oc.template="abcdesktopio/oc.template.ubuntu.nvidia.22.04"
LABEL oc.name="af1"
LABEL oc.displayname="AlphaLink"
LABEL oc.path="/usr/bin/konsole"
LABEL oc.type=app
LABEL oc.showinview="dock"
LABEL oc.rules="{\"homedir\":{\"default\":true}}"
LABEL oc.acl="{\"permit\":[\"all\"]}"

RUN if [ -d /usr/share/icons ]   && [ -x /composer/safelinks.sh ]; then cd /usr/share/icons;   /composer/safelinks.sh; fi
RUN if [ -d /usr/share/pixmaps ] && [ -x /composer/safelinks.sh ]; then cd /usr/share/pixmaps; /composer/safelinks.sh; fi

ENV APPNAME="af1" \
    APPBIN="/usr/bin/konsole" \
    APP="/usr/bin/konsole"

RUN mkdir -p /run/user/ && chmod 777 /run/user
RUN mkdir -p /etc/localaccount && for f in passwd shadow group gshadow; do \
      if [ -f /etc/$f ]; then cp /etc/$f /etc/localaccount; rm -f /etc/$f; ln -s /etc/localaccount/$f /etc/$f; fi; \
    done

# prefer abcdesktop entrypoint if present; otherwise bash
CMD ["/bin/bash","-lc","if [ -x /composer/appli-docker-entrypoint.sh ]; then exec /composer/appli-docker-entrypoint.sh; else exec bash; fi"]
