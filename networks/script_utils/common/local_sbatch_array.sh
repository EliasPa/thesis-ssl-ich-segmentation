#!/bin/bash

# Usage:
# $1    array in quotes, e.g. "0 1 8" would execute array ids 0, 1, and 8
# $2    path to script to run

ids=($1)
script_to_run=$2

for id in ${ids[@]};
do
    SLURM_ARRAY_TASK_ID=$id # Used in script_to_run
    echo "Running ${script_to_run} with array ID ${SLURM_ARRAY_TASK_ID}."
    source script_utils/common/local_sbatch.sh $SLURM_ARRAY_TASK_ID $script_to_run
done
