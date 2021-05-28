#!/bin/bash

# Used to differentiate between local and remote environments, as scripts are executed differently.
# Note: Don't add echo commands in this script, as the $run_command variable's echo is used as a return value.

run_env=${RUN_ENV:-"remote"}
if [ $run_env == "local" ]; then
    # Running in local
    run_command="python"
elif [ $run_env == "remote" ]; then
    # Running in remote
    module load anaconda
    source activate image-segmentation

    run_command="srun --gres=gpu:1 python3"
fi

echo $run_command
