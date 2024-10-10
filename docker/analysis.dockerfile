FROM kosinskilab/alpha_analysis_basis_jax0.4:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN mkdir -p /app/alpha-analysis/
RUN mkdir -p /tmp/root && chmod 777 /tmp/root
COPY ./alphapulldown/analysis_pipeline/*sh /app
COPY ./alphapulldown/analysis_pipeline/*py /app/alpha-analysis/

RUN chmod +x /app/run_get_good_pae.sh \
    && chmod +x /app/run_execute_notebook.sh \
    && chmod +x /app/run_pi_score.sh \
    && chmod +x /app/alpha-analysis/get_good_inter_pae.py

RUN rm -f /software/pisa /software/sc

ENV PATH="/app/alpha-analysis:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

ENTRYPOINT [ "bash" ]
