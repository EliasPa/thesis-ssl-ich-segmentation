#!/bin/bash

# This helper script ssh's into {server} with {username} and fetches all models inside the
# {experiment_name} directory, matching with {model_regexp} (glob pattern). Fetches models are placed
# into a corresponding directory in the local machine. This directory is created if it doesn't exist already.

username=$1
server=$2
experiment_name=$3
model_regexp=$4
target_path="../../remote_experiment_runs/$experiment_name/models/"
remote_path="/scratch/work/$username/diplomi/image-segmentation-playground/experiment_runs/$experiment_name/models/$model_regexp"

mkdir -p $target_path

scp $username@$server:$remote_path $target_path
