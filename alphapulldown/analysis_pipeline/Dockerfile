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
COPY software/ccp4-8.0/lib/data/ /software/lib64/data/
COPY software/ncbi-blast-2.13.0+-src/c++/ReleaseMT/bin/psiblast /software/
COPY software/ncbi-blast-2.13.0+-src/c++/ReleaseMT/bin/blastdbcmd /software/
COPY software/muscle-5.1/src/Linux/muscle /software/
COPY software/pi_score/ /software/pi_score/
COPY software/uniprot_sprot.fasta /database/
COPY alpha-analysis/pi_score_env.yml /app/pi_score_env.yml
COPY alpha-analysis/*py /app/alpha-analysis/
COPY alpha-analysis/*.sh /app/
RUN apt-get update
RUN apt-get install -y build-essential
# RUN conda env create -f /app/programme_notebook.yml
RUN conda env create -f /app/pi_score_env.yml
RUN pip install absl-py
RUN pip install jax==0.4.16 jaxlib==0.4.16+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN ["/bin/bash","-c","source activate programme_notebook && conda install -c conda-forge cctbx-base"]
RUN ["/bin/bash","-c","source activate pi_score && pip install numpy==1.16.6 scikit-learn==0.20.4"]
RUN ["/bin/bash","-c","apt-get --reinstall install libcurl3-gnutls && apt-get update && (apt-get install -t buster-backports -y rate4site || apt-get install -y rate4site) && apt-get clean && apt-get purge && rm -rf /var/lib/apt/lists/* /tmp/*"]

# below need to set up env for sc
# RUN useradd -u 1000 -ms /bin/bash newuser && su newuser && mkdir /home/newuser
RUN ["mkdir","/tmp/root"]
ENV CCP4_MASTER="/"
ENV CCP4="/ccp4"
ENV CBIN="$CCP4/bin"
ENV CLIB="/software/lib64"
ENV CINCL="/ccp4/include"
ENV CCP4_SCR="/tmp/root"
ENV CETC="$CCP4/etc"
ENV CETC="$CCP4/etc"
ENV CLIBD="/software/lib64/data"
ENV CCP4I_TOP="/ccp4/share/share/ccp4i"
ENV MMCIFDIC="$CLIB/ccp4/cif_mmdic.lib"
ENV CRANK="$CCP4I_TOP/crank"
RUN ["/bin/bash","-c","test -d $CCP4_SCR || mkdir $CCP4_SCR"]
RUN ["/bin/bash","-c","test -d $CCP4_SCR || echo 'Unable to assign CCP4_SCR. This will cause probs.'"]
ENV CLIBD="/software/lib64/data"
ENV PATH="/app:$PATH"
ENV PATH="/software:$PATH"
ENV PATH="/app/alpha-analysis:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV LD_LIBRARY_PATH="/software/lib64:$LD_LIBRARY_PATH"
# Below have to relink the libz:
RUN ["/bin/bash","-c","cd /software/lib64 && mv libz.so.1 libz.so.1.old && ln -s /lib/x86_64-linux-gnu/libz.so.1"]
RUN chmod +x /app/run_get_good_pae.sh
RUN chmod +x /app/run_execute_notebook.sh
RUN chmod +x /app/run_pi_score.sh
RUN chmod +x /app/alpha-analysis/get_good_inter_pae.py
ENTRYPOINT [ "bash" ]
