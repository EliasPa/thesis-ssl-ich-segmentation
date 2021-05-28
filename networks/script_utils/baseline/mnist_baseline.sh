#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/mnist_baseline/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=4096
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --constraint='pascal|volta'

EXPERIMENT_NAME="mnist_baseline"
source script_utils/setup_dirs.sh $EXPERIMENT_NAME

EPOCHS=200
RELU_TYPE="normal"
UNSUPERVISED_DATA_PATH="../data/nifti_down_sampled_train/"
N_CLASSES=2
n_channels=3
START_FOLD=0
END_FOLD=1

LOSS_TYPE="dice"
LOSS_REDUCTION="no-bg"
learning_rate="0.0012"
zero_gradients_every=1
batch_size_s=8
use_legacy_unet="False"
experiment_type="mnist"

seeds=(123 234 345 456 567)

seed=${seeds[$SLURM_ARRAY_TASK_ID]}

LOG_PREFIX="Dinb_${seed}"
echo $LOG_PREFIX

module load anaconda
source activate image-segmentation
conda list | grep torchvision

srun --gres=gpu:1 python3 baseline_segmentation.py --n-classes=$N_CLASSES \
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
    --experiment-type $experiment_type \
    --n-channels $n_channels \
