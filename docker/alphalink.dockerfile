# syntax=docker/dockerfile:1.4
ARG CUDA=11.8.0
FROM registry.git.embl.de/grp-cbbcs/abcdesktop-apps/base-image:ubuntu22-cuda-11-8
ARG CUDA

SHELL ["/bin/bash","-o","pipefail","-c"]

# Minimal runtime deps (+ bzip2 for micromamba tar)
# Note: Dropped cuda-command-line-tools-* to avoid duplicating CUDA with the torch cu118 wheel.
# If you actually need the CUDA toolchain at runtime, add it back.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      kalign tzdata wget ca-certificates bzip2 git openssh-client \
  && rm -rf /var/lib/apt/lists/*

# micromamba bootstrap
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN set -eux; \
    wget -qO- https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba; \
    micromamba --help >/dev/null

# faster dependency solve; strict priority
ENV CONDA_CHANNEL_PRIORITY=strict

# Create env at /opt/conda; no activation needed
RUN micromamba install -y -p /opt/conda -c conda-forge -c bioconda \
      python=3.10 openmm=8 pdbfixer=1.9 kalign2 \
      importlib-metadata modelcif hmmer hhsuite pip \
  && micromamba clean -y --all

ENV PATH=/opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# pip reliability defaults
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# PyTorch via pip (CUDA 11.8) BEFORE Uni-Core (Fix A)
# Torch from the PyTorch index; other deps from PyPI.
RUN python -m pip install --upgrade pip \
 && python -m pip install \
      torch==2.5.1+cu118 \
      --index-url https://download.pytorch.org/whl/cu118 \
      --extra-index-url https://pypi.org/simple

# Python packages (after torch)
# Optional: pin to SHAs for reproducibility instead of @main
RUN python -m pip install \
      git+https://github.com/dptech-corp/Uni-Core.git@main \
      git+https://github.com/KosinskiLab/AlphaPulldown.git@main \
      pytest

# Trim: drop build tools & caches
RUN apt-get purge -y git openssh-client wget bzip2 \
 && apt-get autoremove -y && apt-get clean \
 && micromamba clean -y --all \
 && python - <<'PY'
import os
for root, dirs, files in os.walk("/opt/conda", topdown=False):
    for f in files:
        if f.endswith((".pyc",".pyo")):
            try: os.remove(os.path.join(root,f))
            except: pass
    for d in list(dirs):
        if d == "__pycache__":
            try: os.rmdir(os.path.join(root,d))
            except: pass
PY
RUN rm -rf /root/.cache /var/cache/apt/* /usr/share/man /usr/share/doc

# Keep your abcdesktop labels/environment as needed

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
#LABEL oc.args="--hold -e bash --rcfile"
LABEL oc.type=app
LABEL oc.showinview="dock"
LABEL oc.rules="{\"homedir\":{\"default\":true}}"
LABEL oc.acl="{\"permit\":[\"all\"]}"

RUN if [ -d /usr/share/icons ]   && [ -x /composer/safelinks.sh ]; then cd /usr/share/icons;   /composer/safelinks.sh; fi
RUN if [ -d /usr/share/pixmaps ] && [ -x /composer/safelinks.sh ]; then cd /usr/share/pixmaps; /composer/safelinks.sh; fi

ENV APPNAME="af1"
ENV APPBIN="/usr/bin/konsole"
ENV APP="/usr/bin/konsole"

RUN mkdir -p /run/user/ && chmod 777 /run/user
RUN mkdir -p /etc/localaccount && for f in passwd shadow group gshadow; do \
      if [ -f /etc/$f ]; then cp /etc/$f /etc/localaccount; rm -f /etc/$f; ln -s /etc/localaccount/$f /etc/$f; fi; \
    done

CMD ["/composer/appli-docker-entrypoint.sh"]
# ENTRYPOINT ["bash"]
