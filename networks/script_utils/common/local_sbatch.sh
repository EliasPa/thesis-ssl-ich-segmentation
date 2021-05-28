#!/bin/bash

# Usage:
# * To use this script, you must first modify script_utils/mt_cut_mix/seeds_cut_mix.sh so that it will run locally
# * Remove UNSUPERVISED_DATA_PATH in the script (let the new value specified here override the old value)

SLURM_ARRAY_TASK_ID=$1 # Used in script_to_run
script_to_run=$2
RUN_ENV="local" # Used in script_to_run
UNSUPERVISED_DATA_PATH="E:\\kaggle\\rsna-intracranial-hemorrhage-detection\\nifti_down_sampled\\"

echo "Running ${script_to_run} with array ID ${SLURM_ARRAY_TASK_ID}. Environment is ${RUN_ENV}."
source $script_to_run
