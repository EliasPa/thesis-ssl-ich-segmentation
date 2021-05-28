#!/bin/bash
#SBATCH --time=9:30:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/stable_mt_cut_mix/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=9000
#SBATCH --array=0-14
#SBATCH --cpus-per-task=1
#SBATCH --constraint="pascal|volta"

# 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
# 123 123 123 234 234 234 345 345 345 456 456 456 567 567 567
# 0   1   5   0   1   5   0   1   5   0   1   5   0   1   5 
#         |           |           |           |           |  <--- only 2.9's: --array=2,5,8,11,14


EXPERIMENT_NAME="stable_mt_cut_mix"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

N_UNSUPERVISED=4000
EPOCHS=150
N_ITER="-1"
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH=${UNSUPERVISED_DATA_PATH:-"../data/nifti_down_sampled_train/"}
N_CLASSES=2

LOSS_TYPE="dice"
use_dropout="False"
skip_checkpoint="True"
batch_size_s=8
batch_size_u=8
learning_rate="0.00006"
LOSS_REDUCTION="no-bg"
# CONSISTENCY_LOSS_TYPE="mse-manual-confidence"
CONSISTENCY_LOSS_TYPE="cross_entropy"
CONSISTENCY_REDUCTION="null"
n_channels=1
N_ramp_up_consistency=50
augmentation_type="full"
use_legacy_unet="False"
optimizer="adam"

N_SUPERVISED=10
experiment_type="ct"
shuffle="False"
std="0.01"
#zero_gradients_every=10

seeds=(123 234 345 456 567)
lambdas=("0.0" "1.0" "2.9")
per_pixel_confidences=("False")
zges=(5)

seed_combinations=()
lambda_combinations=()
#zge_combinations=()
#ppc_combinations=()

for i in ${!seeds[@]};
do
    for j in ${!lambdas[@]};
    do
        #for k in ${!per_pixel_confidences[@]};
        #do
        seed_combinations+=(${seeds[$i]})
        lambda_combinations+=(${lambdas[$j]})
        #ppc_combinations+=(${per_pixel_confidences[$k]})
        #zge_combinations+=(${zges[$j]})
        #done
    done
done

seed=${seed_combinations[$SLURM_ARRAY_TASK_ID]}
lambd=${lambda_combinations[$SLURM_ARRAY_TASK_ID]}
#zero_gradients_every=${zge_combinations[$SLURM_ARRAY_TASK_ID]}
zero_gradients_every=10
per_pixel_confidence="False"

#LOG_PREFIX="DiXe_${N_SUPERVISED}_${seed}_${lambd}_${per_pixel_confidence}"
LOG_PREFIX="DiXe_${CONSISTENCY_LOSS_TYPE}_${augmentation_type}_${N_SUPERVISED}_${seed}_${lambd}_${zero_gradients_every}"
echo $LOG_PREFIX

run_command="`source script_utils/common/build_run_command.sh`"
echo "Using run command: ${run_command} in $run_env"

$run_command cutmix_seg_alt.py \
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
    -ns $N_SUPERVISED \
    -ch $n_channels \
    --lambd $lambd \
    --shuffle $shuffle \
    --experiment-type $experiment_type \
    --std $std \
    --N-iter $N_ITER \
    --skip-checkpoint $skip_checkpoint \
    --per-pixel-confidence $per_pixel_confidence \
    --zero-gradients-every $zero_gradients_every \
    --N-ramp-up-consistency $N_ramp_up_consistency \
    --augmentation-type $augmentation_type \
    --legacy-unet $use_legacy_unet \
    --optimizer $optimizer \
