#!/bin/bash

# Creates directories required by models under network/
# The experiment name should be passed to this script as an argument.

base_path='../experiment_runs/'
experiment_name=$1

source script_utils/create_dir.sh "$base_path$experiment_name/out/"
source script_utils/create_dir.sh "$base_path$experiment_name/models/"
source script_utils/create_dir.sh "$base_path$experiment_name/evaluation/statistics"
source script_utils/create_dir.sh "../output/tasks/$experiment_name/evaluation"

echo "Done creating directories."
