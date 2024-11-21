# syntax = docker/dockerfile:1.4

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
# Using base image with BARD dependencies!
ARG CUDA=11.8.0
#FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu20.04
FROM registry.git.embl.de/grp-cbbcs/abcdesktop-apps/base-image:ubuntu22-cuda-11-8
ARG CUDA

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN apt update -y && apt upgrade -y
RUN apt install -y --reinstall libp11-kit0 libffi7
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        build-essential \
        cmake \
        cuda-command-line-tools-$(cut -f1,2 -d- <<< ${CUDA//./-}) \
        kalign \
        tzdata \
        wget \
        bc \
        openssh-client \
        konsole \
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

RUN conda install -y -c conda-forge -c bioconda --solver classic \
      openmm==8.0 \
      cudatoolkit==${CUDA_VERSION} \
      pdbfixer==1.9 \
      kalign2 \
      importlib_metadata \
      modelcif \
      pip \
      hmmer \
      hhsuite \
      python=3.10 \
      && conda clean --all --force-pkgs-dirs --yes

RUN conda install -y -c nvidia cuda-nvcc --solver classic

#COPY . /app/alphafold
#RUN wget -q -P /app/alphafold/alphafold/common/ \
#  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

#RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
##RUN ssh-keyscan github.com >> /root/.ssh/known_hosts
##RUN pip3 install git+https://github.com/KosinskiLab/AlphaPulldown.git@main
#RUN apt-get update && apt-get install -y git openssl
#RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
#RUN pip3 install git+https://github.com/KosinskiLab/AlphaPulldown.git@main

#RUN pip3 install --upgrade pip --no-cache-dir \
#    && pip3 install --upgrade --no-cache-dir \
#      -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
#      pytest \
#      jax==0.4.23 \
#      jaxlib==0.4.23+cuda11.cudnn86 \
#      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#RUN chmod u+s /sbin/ldconfig.real

RUN mkdir -p /root/.ssh && chmod 700 /root/.ssh
RUN apt-get update && apt-get install -y git openssl
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN pip3 install git+https://github.com/KosinskiLab/AlphaPulldown.git@main

RUN pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --upgrade --no-cache-dir \
      pytest
RUN pip3 install torch==2.5.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

RUN git clone https://github.com/dptech-corp/Uni-Core.git
WORKDIR /Uni-Core

RUN pip3 install .
RUN chmod u+s /sbin/ldconfig.real


LABEL oc.icon="alphafold.svg"
LABEL oc.icondata="PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPCFET0NUWVBFIHN2ZyBQVUJMSUMgIi0vL1czQy8vRFREIFNWRyAxLjEvL0VOIiAiaHR0cDovL3d3dy53My5vcmcvR3JhcGhpY3MvU1ZHLzEuMS9EVEQvc3ZnMTEuZHRkIj4KPHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZlcnNpb249IjEuMSIgd2lkdGg9IjI4MHB4IiBoZWlnaHQ9IjI4MHB4IiBzdHlsZT0ic2hhcGUtcmVuZGVyaW5nOmdlb21ldHJpY1ByZWNpc2lvbjsgdGV4dC1yZW5kZXJpbmc6Z2VvbWV0cmljUHJlY2lzaW9uOyBpbWFnZS1yZW5kZXJpbmc6b3B0aW1pemVRdWFsaXR5OyBmaWxsLXJ1bGU6ZXZlbm9kZDsgY2xpcC1ydWxlOmV2ZW5vZGQiIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIj4KPGc+PHBhdGggc3R5bGU9Im9wYWNpdHk6MSIgZmlsbD0iIzEyNTVkNiIgZD0iTSAxMjcuNSwtMC41IEMgMTM3LjUsLTAuNSAxNDcuNSwtMC41IDE1Ny41LC0wLjVDIDIxMi44MTksOS4zMTMwMiAyNTAuOTg1LDQwLjMxMyAyNzIsOTIuNUMgMjc0Ljk5NiwxMDIuMTU2IDI3Ny40OTYsMTExLjgyMyAyNzkuNSwxMjEuNUMgMjc5LjUsMTMzLjUgMjc5LjUsMTQ1LjUgMjc5LjUsMTU3LjVDIDI3MS41OTcsMjA2LjY2NCAyNDUuNTk3LDI0Mi42NjQgMjAxLjUsMjY1LjVDIDIwMC41MDgsMjY1LjY3MiAxOTkuODQyLDI2NS4zMzggMTk5LjUsMjY0LjVDIDE5OS41LDI2My44MzMgMTk5LjgzMywyNjMuNSAyMDAuNSwyNjMuNUMgMjAyLjQwNCwyNjMuMjYyIDIwMy43MzgsMjYyLjI2MiAyMDQuNSwyNjAuNUMgMjA4Ljg2MSwyNTUuODA3IDIxMy4xOTUsMjUxLjE0MSAyMTcuNSwyNDYuNUMgMjE4LjkwNiwyNDYuMDI3IDIxOS41NzMsMjQ1LjAyNyAyMTkuNSwyNDMuNUMgMjQwLjMxLDIxMS4wOCAyNDIuNjQzLDE3Ny40MTMgMjI2LjUsMTQyLjVDIDIyNi43MTUsMTQwLjgyMSAyMjYuMDQ4LDEzOS44MjEgMjI0LjUsMTM5LjVDIDIwMy42ODQsMTA1LjUxOSAxNzMuMzUsODkuMDE5NCAxMzMuNSw5MEMgMTA2LjczNSw5NS42NzE2IDkyLjA2ODIsMTEyLjE3MiA4OS41LDEzOS41QyA5MC4yMTIsMTUzLjkxNSA5NS41NDU0LDE2Ni4yNDkgMTA1LjUsMTc2LjVDIDEwNS41LDE3Ny4xNjcgMTA1LjE2NywxNzcuNSAxMDQuNSwxNzcuNUMgOTEuMjkzLDE3Mi4xNTQgNzkuNzkzLDE2NC4xNTQgNzAsMTUzLjVDIDQ4LjEyNjIsMTI3LjI5MSA0MS43OTI5LDk3LjYyMzkgNTEsNjQuNUMgNjMuNzg2NiwyNy44ODA3IDg5LjI4NjYsNi4yMTQwNiAxMjcuNSwtMC41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjEiIGZpbGw9IiM3MjkwZTUiIGQ9Ik0gNzYuNSwxNy41IEMgNzYuNSwxNi4xNjY3IDc1LjgzMzMsMTUuNSA3NC41LDE1LjVDIDc3LjczMzgsMTIuODgxMiA4MS40MDA1LDExLjM4MTIgODUuNSwxMUMgODIuNDcwNiwxMy4xNzYxIDc5LjQ3MDYsMTUuMzQyNyA3Ni41LDE3LjUgWiIvPjwvZz4KPGc+PHBhdGggc3R5bGU9Im9wYWNpdHk6MSIgZmlsbD0iIzEyNTVkNiIgZD0iTSA3NC41LDE1LjUgQyA3NS44MzMzLDE1LjUgNzYuNSwxNi4xNjY3IDc2LjUsMTcuNUMgNDcuMzU1Myw0NC4zNjk0IDM2Ljg1NTMsNzcuMzY5NCA0NSwxMTYuNUMgNTkuNTAzMywxNjAuNTkgODkuODM2NiwxODUuMjU3IDEzNiwxOTAuNUMgMTUyLjkxNiwxOTAuMDM5IDE2Ny4wODMsMTgzLjcwNSAxNzguNSwxNzEuNUMgMTc5LjkwNiwxNzEuMDI3IDE4MC41NzMsMTcwLjAyNyAxODAuNSwxNjguNUMgMTkzLjk5LDE0NS4zNzYgMTkxLjk5LDEyMy43MSAxNzQuNSwxMDMuNUMgMTc1LjIwOSwxMDIuNTk2IDE3Ni4yMDksMTAyLjI2MyAxNzcuNSwxMDIuNUMgMTkxLjEzNiwxMDguNjQ1IDIwMi40NywxMTcuNjQ1IDIxMS41LDEyOS41QyAyMTEuNzM4LDEzMS40MDQgMjEyLjczOCwxMzIuNzM4IDIxNC41LDEzMy41QyAyMjkuODExLDE1My45NiAyMzUuMzExLDE3Ni45NiAyMzEsMjAyLjVDIDIyMS43NjIsMjQ1LjU3MyAxOTUuNTk1LDI3MS4yNCAxNTIuNSwyNzkuNUMgMTQxLjgzMywyNzkuNSAxMzEuMTY3LDI3OS41IDEyMC41LDI3OS41QyA2MS4yMzA0LDI2OC40MDMgMjIuMzk3MSwyMzQuMDcgNCwxNzYuNUMgMi40NDUzNSwxNjkuNzEgMC45NDUzNDcsMTYzLjA0MyAtMC41LDE1Ni41QyAtMC41LDE0NS41IC0wLjUsMTM0LjUgLTAuNSwxMjMuNUMgNC4wMTY0NSw5MC4xMzMyIDE4LjM0OTgsNjEuNzk5OCA0Mi41LDM4LjVDIDQ1LjA4OTIsMzcuNTgwNiA0Ny4wODkyLDM1LjkxNCA0OC41LDMzLjVDIDU2LjExMjcsMjYuMDIzIDY0Ljc3OTMsMjAuMDIzIDc0LjUsMTUuNSBaIi8+PC9nPgo8Zz48cGF0aCBzdHlsZT0ib3BhY2l0eTowLjkxNCIgZmlsbD0iIzY5ODdlMyIgZD0iTSA0OC41LDMzLjUgQyA0Ny4wODkyLDM1LjkxNCA0NS4wODkyLDM3LjU4MDYgNDIuNSwzOC41QyA0My45MTA4LDM2LjA4NiA0NS45MTA4LDM0LjQxOTQgNDguNSwzMy41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjAuNjk0IiBmaWxsPSIjOTBhYmViIiBkPSJNIDE3Ny41LDEwMi41IEMgMTc2LjIwOSwxMDIuMjYzIDE3NS4yMDksMTAyLjU5NiAxNzQuNSwxMDMuNUMgMTcyLjQwMSwxMDIuNDgzIDE3MS40MDEsMTAxLjE1IDE3MS41LDk5LjVDIDE3My42MDQsMTAwLjUxOSAxNzUuNjA0LDEwMS41MTkgMTc3LjUsMTAyLjUgWiIvPjwvZz4KPGc+PHBhdGggc3R5bGU9Im9wYWNpdHk6MC45MDYiIGZpbGw9IiM4ODlmZTkiIGQ9Ik0gMjExLjUsMTI5LjUgQyAyMTMuMjYyLDEzMC4yNjIgMjE0LjI2MiwxMzEuNTk2IDIxNC41LDEzMy41QyAyMTIuNzM4LDEzMi43MzggMjExLjczOCwxMzEuNDA0IDIxMS41LDEyOS41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjAuODc4IiBmaWxsPSIjOTBhMWVhIiBkPSJNIDIyNC41LDEzOS41IEMgMjI2LjA0OCwxMzkuODIxIDIyNi43MTUsMTQwLjgyMSAyMjYuNSwxNDIuNUMgMjI1LjU5OSwxNDEuNzEgMjI0LjkzMiwxNDAuNzEgMjI0LjUsMTM5LjUgWiIvPjwvZz4KPGc+PHBhdGggc3R5bGU9Im9wYWNpdHk6MC44MTIiIGZpbGw9IiM4ZWE1ZWEiIGQ9Ik0gMTgwLjUsMTY4LjUgQyAxODAuNTczLDE3MC4wMjcgMTc5LjkwNiwxNzEuMDI3IDE3OC41LDE3MS41QyAxNzguNDI3LDE2OS45NzMgMTc5LjA5NCwxNjguOTczIDE4MC41LDE2OC41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjAuODU5IiBmaWxsPSIjN2Y5ZmU4IiBkPSJNIDEwNS41LDE3Ni41IEMgMTA2LjYyNywxNzcuMTIyIDEwNy42MjcsMTc3Ljk1NSAxMDguNSwxNzlDIDEwNi44NiwxNzkuMzQ4IDEwNS41MjcsMTc4Ljg0OCAxMDQuNSwxNzcuNUMgMTA1LjE2NywxNzcuNSAxMDUuNSwxNzcuMTY3IDEwNS41LDE3Ni41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjAuOCIgZmlsbD0iIzg5YTRlYSIgZD0iTSAyMTkuNSwyNDMuNSBDIDIxOS41NzMsMjQ1LjAyNyAyMTguOTA2LDI0Ni4wMjcgMjE3LjUsMjQ2LjVDIDIxNy40MjcsMjQ0Ljk3MyAyMTguMDk0LDI0My45NzMgMjE5LjUsMjQzLjUgWiIvPjwvZz4KPGc+PHBhdGggc3R5bGU9Im9wYWNpdHk6MC43NDkiIGZpbGw9IiM4ZGE1ZWEiIGQ9Ik0gMjA0LjUsMjYwLjUgQyAyMDMuNzM4LDI2Mi4yNjIgMjAyLjQwNCwyNjMuMjYyIDIwMC41LDI2My41QyAyMDEuMjYyLDI2MS43MzggMjAyLjU5NiwyNjAuNzM4IDIwNC41LDI2MC41IFoiLz48L2c+CjxnPjxwYXRoIHN0eWxlPSJvcGFjaXR5OjAuNzI1IiBmaWxsPSIjOGVhOGVhIiBkPSJNIDE5OS41LDI2NC41IEMgMTk5Ljg0MiwyNjUuMzM4IDIwMC41MDgsMjY1LjY3MiAyMDEuNSwyNjUuNUMgMTk5LjM2LDI2Ni45MDUgMTk3LjAyNiwyNjcuNzM4IDE5NC41LDI2OEMgMTk1Ljg0NiwyNjYuMzMxIDE5Ny41MTMsMjY1LjE2NCAxOTkuNSwyNjQuNSBaIi8+PC9nPgo8L3N2Zz4K"
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

RUN  if [ -d /usr/share/icons ]   && [ -x /composer/safelinks.sh ] && [ -d /usr/share/icons   ];  then cd /usr/share/icons;    /composer/safelinks.sh; fi
RUN  if [ -d /usr/share/pixmaps ] && [ -x /composer/safelinks.sh ] && [ -d /usr/share/pixmaps ];  then cd /usr/share/pixmaps;  /composer/safelinks.sh; fi
ENV APPNAME "af1"
ENV APPBIN "/usr/bin/konsole"
ENV APP "/usr/bin/konsole"
#ENV ARGS "--hold -e bash --rcfile /usr/local/bin/startide.sh"
ENV LD_LIBRARY_PATH "/opt/conda:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
ENV NVIDIA_DRIVER_CAPABILITIES "compute,utility"


RUN mkdir -p /run/user/
RUN chmod 777 /run/user
RUN mkdir -p /etc/localaccount
RUN for f in passwd shadow group gshadow ; do if [ -f /etc/$f ] ; then  cp /etc/$f /etc/localaccount; rm -f /etc/$f; ln -s /etc/localaccount/$f /etc/$f; fi; done


CMD [ "/composer/appli-docker-entrypoint.sh" ]




#ENTRYPOINT ["bash"]
