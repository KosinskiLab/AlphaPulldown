# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Extended by Valentin Maurer <valentin.maurer@embl-hamburg.de>

ARG CUDA=11.8.0
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
ARG CUDA

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        git \
        hmmer \
        kalign \
        tzdata \
        wget \
        bc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"
ENV LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
RUN conda install -y -c conda-forge -c bioconda \
      openmm==8.0 \
      cudatoolkit==${CUDA_VERSION} \
      pdbfixer==1.9 \
      cctbx-base \
      kalign2 \
      importlib_metadata \
      pip \
      python=3.10 \
      && conda clean --all --force-pkgs-dirs --yes

RUN conda install -y -c nvidia cuda-nvcc

COPY . /app/alphafold
RUN wget -q -P /app/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
      alphapulldown==2.0.0b2 \
      pytest \
      jax==0.4.23 \
      jaxlib==0.4.23+cuda11.cudnn86 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN chmod u+s /sbin/ldconfig.real

ENTRYPOINT ["bash"]