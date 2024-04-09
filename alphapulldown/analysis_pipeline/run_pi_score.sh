#!/bin/bash
while getopts "p:o:s:" arg; do
    case "${arg}" in 
    p)
        pdb_path=$OPTARG
        ;;
    o)
        output_dir=$OPTARG
        ;;
    s)
        surface_threshold=$OPTARG
        ;;

source activate pi_score
export PYTHONPATH=/software:$PYTHONPATH
python /software/pi_score/run_piscore_wc.py -p $pdb_path -o $output_dir -s $surface_threshold