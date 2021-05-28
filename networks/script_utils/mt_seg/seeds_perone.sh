#!/bin/bash
#SBATCH --time=10:15:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/stable_mt_seg/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=9000
#SBATCH --array=0-14
#SBATCH --cpus-per-task=1

# 1 1 1 2 2 2 3 3 3 4 4  4  5  5  5 seeds
# 1 2 3 1 2 3 1 2 3 1 2  3  1  2  3 lambdas
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
#   |     |     |     |        |     <--- only 1.0 --array=1,4,7,10,13

EXPERIMENT_NAME="stable_mt_seg"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

N_UNSUPERVISED=4000
EPOCHS=200
N_ITER=10
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH=${UNSUPERVISED_DATA_PATH:-"../data/nifti_down_sampled_train/"}
N_CLASSES=2
skip_checkpoint="True"

use_dropout="False"
legacy_unet="False"
batch_size_s=8
batch_size_u=8
LOSS_TYPE="dice"
LOSS_REDUCTION="no-bg"
CONSISTENCY_LOSS_TYPE="cross_entropy"
CONSISTENCY_REDUCTION="null"
N_ramp_up_ema=50
N_ramp_up_consistency=100
base_augment_train="True"
consistency_augmentation_type="extreme"

n_channels=1
seeds=(123 234 345 456 567)
lambdas=("0.0" "5.0" "2.9")
learning_rate="0.00006"
experiment_type="ct"
zero_gradients_every=10
std="0.01"
optimizer="adam"

seed_combinations=()
lambda_combinations=()
ppc_combinations=()

for i in ${!seeds[@]};
do
    for j in ${!lambdas[@]};
    do
        seed_combinations+=(${seeds[$i]})
        lambda_combinations+=(${lambdas[$j]})
    done
done

seed=${seed_combinations[$SLURM_ARRAY_TASK_ID]}
lambd=${lambda_combinations[$SLURM_ARRAY_TASK_ID]}

LOG_PREFIX="DiXe_${consistency_augmentation_type}_${seed}_${lambd}"
echo $LOG_PREFIX

run_command="`source script_utils/common/build_run_command.sh`"

$run_command perone_seg_alt.py \
    --n-classes $N_CLASSES \
    --experiment-log-prefix $LOG_PREFIX \
    -r $RELU_TYPE \
    -n $EXPERIMENT_NAME \
    -lt $LOSS_TYPE \
    --loss-reduction $LOSS_REDUCTION \
    --consistency-loss-type $CONSISTENCY_LOSS_TYPE \
    --consistency-dice-reduction $CONSISTENCY_REDUCTION \
    -nu $N_UNSUPERVISED \
    -u $UNSUPERVISED_DATA_PATH \
    --epochs $EPOCHS \
    --use-dropout $use_dropout \
    --batch-size-s $batch_size_s \
    --batch-size-u $batch_size_u \
    --lr $learning_rate \
    --seed $seed \
    --skip-checkpoint $skip_checkpoint \
    --experiment-type $experiment_type \
    --lambd $lambd \
    -ch $n_channels \
    --std $std \
    --zero-gradients-every $zero_gradients_every \
    --legacy-unet $legacy_unet \
    --optimizer $optimizer \
    --N-ramp-up $N_ramp_up_ema \
    --N-ramp-up-consistency $N_ramp_up_consistency \
    --base-augment-train $base_augment_train \
    --consistency-augmentation-type $consistency_augmentation_type \
