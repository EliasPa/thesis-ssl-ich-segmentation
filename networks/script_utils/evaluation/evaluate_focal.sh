#!/bin/bash

#
# This script runs evaluation for all models in the given folder (indicated by experiment_name), grouped by their type.
# As a result, a LaTeX tabular table is created, which can then be included in a report.
# 
# Usage:
#   * navigate to /networks
#   * run command: source script_utils/evaluation/evaluate_mnist_mt_cut_mix.sh {experiment_name} {experiment_type} {gammas} {experiment_directory without /}

experiment_name=$1
experiment_type=$2
gammas=($3)
experiment_dir_arg=$4
experiment_dir=${experiment_dir_arg:-"../../remote_experiment_runs"}

source script_utils/setup_dirs.sh $experiment_name

if [ $experiment_type == "mnist" ]; then
    n_channels=3
else
    n_channels=1
fi

for gamma in ${gammas[@]};
do
    model_path="${experiment_dir}/${experiment_name}/models/*${gamma}*checkpoint_best*"

    if [ $gamma != "0.0" ]; then
        el_prefix="cross_entropy"
    else
        el_prefix="focal"
    fi

    log_prefix="${el_prefix}_bests"
    echo $log_prefix
    
    python evaluate.py \
        -m  "$model_path" \
        -c 2 \
        -ch $n_channels \
        --legacy-unet "False" \
        -el $log_prefix \
        --single-mode "False" \
        --visualize "False" \
        --relu-type normal \
        --model-type unet \
        -s 0 \
        -f 1 \
        --experiment-type $experiment_type \
        --experiment-name $experiment_name
done

python script_utils/post_process_results.py -n $experiment_name
