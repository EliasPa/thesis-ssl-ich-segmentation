#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/stable_baseline/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=4096
#SBATCH --array=0-14
#SBATCH --cpus-per-task=1
#SBATCH --constraint='pascal|volta'

# 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
# 123 123 123 234 234 234 345 345 345 456 456 456 567 567 567
# n   m   f   n   m   f   n   m   f   n   m   f   n   m   f 
#                             |           |           |

EXPERIMENT_NAME="stable_baseline"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

EPOCHS=200
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH=${UNSUPERVISED_DATA_PATH:-"../data/nifti_down_sampled_train/"}
N_CLASSES=2
START_FOLD=0
END_FOLD=1

LOSS_TYPE="dice"
LOSS_REDUCTION="no-bg"
learning_rate="0.00006"
zero_gradients_every=10
batch_size_s=8
use_legacy_unet="False"
optimizer="adam"

seeds=(123 234 345 456 567)
augmentations=("none" "mild" "full")

seed_combinations=()
augmentation_combinations=()

for i in ${!seeds[@]};
do
    for j in ${!augmentations[@]};
    do
        seed_combinations+=(${seeds[$i]})
        augmentation_combinations+=(${augmentations[$j]})
    done
done

seed=${seed_combinations[$SLURM_ARRAY_TASK_ID]}
augmentation=${augmentation_combinations[$SLURM_ARRAY_TASK_ID]}

LOG_PREFIX="Dinb_${seed}_${augmentation}_${zero_gradients_every}"
echo $LOG_PREFIX
echo $use_legacy_unet

run_command="`source script_utils/common/build_run_command.sh`"
echo "Using run command: ${run_command} in $run_env"

$run_command baseline_segmentation.py \
    --n-classes=$N_CLASSES \
    --experiment-log-prefix $LOG_PREFIX \
    -r $RELU_TYPE \
    -n $EXPERIMENT_NAME \
    -lt $LOSS_TYPE \
    --loss-reduction $LOSS_REDUCTION \
    -u $UNSUPERVISED_DATA_PATH \
    --epochs $EPOCHS \
    --start-fold $START_FOLD \
    --end-fold $END_FOLD \
    --seed $seed \
    --lr $learning_rate \
    --zero-gradients-every $zero_gradients_every \
    --batch-size-s $batch_size_s \
    --legacy-unet $use_legacy_unet \
    --augmentation-type $augmentation \
    --optimizer $optimizer \
