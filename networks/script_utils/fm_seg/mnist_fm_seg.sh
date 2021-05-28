#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/mnist_fm_seg/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=9000
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1

# 123 123 234 234 345 345 456 456 567 567
# 0   1   2   3   4   5   6   7   8   9
# >--------------------< 3 seeds

EXPERIMENT_NAME="mnist_fm_seg"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

N_UNSUPERVISED=10000
EPOCHS=200
N_ITER=10
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH="../data/nifti_down_sampled_train/"
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
N_ramp_up_ema=80
N_ramp_up_consistency=80

N_SUPERVISED=10
n_channels=3
seeds=(123 234 345 456 567)
lambdas=("0.0" "2.9")
learning_rate="0.0012"
experiment_type="mnist"
fix_match_type="regular"
zero_gradients_every=5
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

LOG_PREFIX="DiXe_${N_SUPERVISED}_${seed}_${lambd}"
echo $LOG_PREFIX

run_command="`source script_utils/common/build_run_command.sh`"
echo "Using run command: ${run_command} in $run_env"

$run_command fixmatch_seg.py \
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
    -ns $N_SUPERVISED \
    -ch $n_channels \
    --std $std \
    --zero-gradients-every $zero_gradients_every \
    --legacy-unet $legacy_unet \
    --optimizer $optimizer \
    --N-ramp-up $N_ramp_up_ema \
    --N-ramp-up-consistency $N_ramp_up_consistency \
    --fix-match-type $fix_match_type \
