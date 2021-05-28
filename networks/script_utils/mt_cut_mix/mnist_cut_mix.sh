#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/mnist_mt_cut_mix/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=9000
#SBATCH --array=0-9
#SBATCH --cpus-per-task=1

EXPERIMENT_NAME="mnist_mt_cut_mix"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

N_UNSUPERVISED=10000
EPOCHS=150
N_ITER=10
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH="../data/nifti_down_sampled_train/"
N_CLASSES=2

LOSS_TYPE="dice"
use_dropout="False"
skip_checkpoint="True"
batch_size_s=8
batch_size_u=8
learning_rate="0.0012"
LOSS_REDUCTION="no-bg"
CONSISTENCY_LOSS_TYPE="mse-manual-confidence"
CONSISTENCY_REDUCTION="null"
n_channels=3

N_SUPERVISED=10
experiment_type="mnist"
shuffle="False"
std="0.01"
per_pixel_confidence="False"
use_legacy_unet="False"

seeds=(123 234 345 456 567)
N_s=(10)
lambdas=("0.0" "1.0")

seed_combinations=()
lambda_combinations=()

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
zero_gradients_every=1

LOG_PREFIX="DiXe_${N_SUPERVISED}_${seed}_${lambd}_${per_pixel_confidence}_${zero_gradients_every}"
echo $LOG_PREFIX

module load anaconda
source activate image-segmentation
conda list | grep torchvision

srun --gres=gpu:1 python3 cutmix_seg_alt.py \
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
    --legacy-unet $use_legacy_unet \
