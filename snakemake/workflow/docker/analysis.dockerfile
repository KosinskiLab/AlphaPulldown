FROM geoffreyyu/alpha_analysis_basis_jax0.4:latest

RUN mkdir -p /app/alpha-analysis/
COPY alphapulldown/analysis_pipeline/*sh /app
COPY alphapulldown/analysis_pipeline/*py /app/alpha-analysis/

RUN chmod +x /app/run_get_good_pae.sh \
    && chmod +x /app/run_execute_notebook.sh \
    && chmod +x /app/run_pi_score.sh \
    && chmod +x /app/alpha-analysis/get_good_inter_pae.py

ENV PATH="/app/alpha-analysis:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

ENTRYPOINT [ "bash" ]