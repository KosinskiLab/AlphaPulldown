#!/bin/bash
# source activate programme_notebook
# python -c "from programme_notebook.utils import display_pae_plots"
# conda run -n programme_notebook 
mkdir -p /tmp/root
chmod go+rw /tmp/root
python /app/alpha-analysis/get_good_inter_pae.py $@
rm -rf /tmp/root