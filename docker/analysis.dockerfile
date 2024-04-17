ARG CUDA=11.1.1
FROM nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu18.04 AS cuda-base

FROM ghcr.io/dingquanyu/usearch_for_pi_score:latest as usearch
FROM rcayfordlbl/ccp4:latest as ccp4
FROM continuumio/anaconda3:latest

COPY --from=usearch /usr/bin/usearch /software/
COPY --from=ccp4 /ccp4/lib/* /software/lib64/
COPY --from=ccp4 /ccp4/etc/ /ccp4/etc/
COPY --from=ccp4 /ccp4/bin/pisa /software/
COPY --from=ccp4 /ccp4/bin/sc /software/
COPY --from=ccp4 /ccp4/include/* /ccp4/include/
COPY --from=ccp4 /ccp4/share/ /ccp4/share/
COPY --from=cuda-base /usr/local/cuda /usr/local/cuda
COPY software/ccp4-8.0/lib/data/ /software/lib64/data/
COPY software/ncbi-blast-2.13.0+-src/c++/ReleaseMT/bin/psiblast /software/
COPY software/ncbi-blast-2.13.0+-src/c++/ReleaseMT/bin/blastdbcmd /software/
COPY software/muscle-5.1/src/Linux/muscle /software/
COPY software/pi_score/ /software/pi_score/
COPY software/uniprot_sprot.fasta /database/
COPY alpha-analysis/pi_score_env.yml /app/pi_score_env.yml
COPY alpha-analysis/*py /app/alpha-analysis/
COPY alpha-analysis/*.sh /app/

RUN apt-get update && \
    apt-get install -y build-essential libcurl3-gnutls && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda env create -f /app/pi_score_env.yml && \
    pip install absl-py "jax[cuda]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
	conda run -n pi_score pip install numpy==1.16.6 scikit-learn==0.20.4 && \
    conda clean --all --force-pkgs-dirs --yes
RUN ["/bin/bash","-c","apt-get --reinstall install libcurl3-gnutls && apt-get update && (apt-get install -t buster-backports -y rate4site || apt-get install -y rate4site) && apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/*"]


ENV CCP4_MASTER="/" \
    CCP4="/ccp4" \
    CBIN="$CCP4/bin" \
    CLIB="/software/lib64" \
    CINCL="$CCP4/include" \
    CCP4_SCR="/tmp/root" \
    CETC="$CCP4/etc" \
    CLIBD="/software/lib64/data" \
    CCP4I_TOP="/ccp4/share/share/ccp4i" \
    MMCIFDIC="$CLIB/ccp4/cif_mmdic.lib" \
    CRANK="$CCP4I_TOP/crank" \
    PATH="/app:/software:/app/alpha-analysis:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    LD_LIBRARY_PATH="/software/lib64:$LD_LIBRARY_PATH"

RUN mkdir -p /tmp/root /software/lib64 && \
    test -d $CCP4_SCR || echo 'Unable to assign CCP4_SCR. This will cause probs.' && \
    cd /software/lib64 && \
    mv libz.so.1 libz.so.1.old && \
    ln -s /lib/x86_64-linux-gnu/libz.so.1 && \
    chmod +x /app/*.sh /app/alpha-analysis/*.py

ENTRYPOINT [ "bash" ]