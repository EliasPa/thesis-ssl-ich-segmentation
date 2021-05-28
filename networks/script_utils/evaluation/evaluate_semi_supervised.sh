#!/bin/bash

#
# This script runs evaluation for all models in the given folder (indicated by experiment_name), grouped by their type.
# As a result, a LaTeX tabular table is created, which can then be included in a report.
# 
# Usage:
#   * navigate to /networks
#   * run command: source script_utils/evaluation/evaluate_mnist_mt_cut_mix.sh {experiment_name} {experiment_type} {lambdas} {experiment_directory without /}

experiment_name=$1
experiment_type=$2
lambdas=($3)
aug_arg=$4
aug=${aug_arg:-""}
loss_arg=$5
loss=${loss_arg:-""}
rand_aug_N_arg=$6
rand_aug_N=${rand_aug_N_arg:-""}
N_ramp_up_consistency_arg=$7
N_ramp_up_consistency=${N_ramp_up_consistency_arg:-""}
experiment_dir_arg=$8
experiment_dir=${experiment_dir_arg:-"../../remote_experiment_runs"}

if [ $experiment_type == "mnist" ]; then
    n_channels=3
else
    n_channels=1
fi

emas=("_" "_ema_")

for ema in ${emas[@]};
do
    for lambda in ${lambdas[@]};
    do
        model_path="${experiment_dir}/${experiment_name}/models/*${rand_aug_N}*${N_ramp_up_consistency}*${loss}*${aug}*${lambda}*checkpoint${ema}best.pth"

        if [ $ema != "_" ]; then
            el_postfix="_ema"
        else
            el_postfix=""
        fi

        if [[ $rand_aug_N != "" ]]; then
            el_prefix_4="${rand_aug_N}"
        else
            el_prefix_4=""
        fi

        if [ $lambda != "0.0" ]; then
            el_prefix_3="semi_${lambda}_"
        else
            el_prefix_3="base"
        fi

        if [[ $aug != "" ]]; then
            el_prefix_2="${aug}_"
        else
            el_prefix_2=""
        fi

        if [[ $loss != "" ]]; then
            el_prefix_1="${loss}_"
        else
            el_prefix_1=""
        fi
        
        if [[ $N_ramp_up_consistency != "" ]]; then
            el_prefix_0="${N_ramp_up_consistency}_"
        else
            el_prefix_0=""
        fi

        log_prefix="e_${el_prefix_0}${el_prefix_1}${el_prefix_2}${el_prefix_3}${el_prefix_4}${el_postfix}_bests"
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
done

python post_process_results.py -n $experiment_name -m ../../remote_experiment_runs/
