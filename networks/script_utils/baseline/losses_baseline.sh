#!/bin/bash
#SBATCH --time=6:45:00
#SBATCH --output=/scratch/work/%u/diplomi/image-segmentation-playground/experiment_runs/losses_baseline/out/%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --mem=4096
#SBATCH --array=0-14
#SBATCH --cpus-per-task=1

# 123 123 123 234 234 234 345 345 345 456 456 456 567 567 567
# 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
# >--------------------------------< Pick these array elements if you want only three seeds

experiment_name="losses_baseline"
source script_utils/setup_dirs.sh $experiment_name

epochs=200
relu_type="normal"
unsupervised_data_path="../data/nifti_down_sampled_train/"
n_classes=2
start_fold=0
end_fold=1

zero_gradients_every=10
loss_reduction="no-bg" # Used for Dice losses only
learning_rate="0.00006"
batch_size_s=8
use_legacy_unet="False"
optimizer="adam"
augmentation="full"

seeds=(123 234 345 456 567)
loss_types=("dice" "focal" "focal")
focal_gammas=("-1" "0.0" "1.5")

seed_combinations=()
loss_combinations=()
gamma_combinations=()

for i in ${!seeds[@]};
do
    for j in ${!loss_types[@]};
    do
        seed_combinations+=(${seeds[$i]})
        gamma_combinations+=(${focal_gammas[$j]}) # select based on loss type
        loss_combinations+=(${loss_types[$j]})
    done
done

seed=${seed_combinations[$SLURM_ARRAY_TASK_ID]}
loss_type=${loss_combinations[$SLURM_ARRAY_TASK_ID]}
focal_gamma=${gamma_combinations[$SLURM_ARRAY_TASK_ID]}

log_prefix="${seed}_${loss_type}_${focal_gamma}"
echo $log_prefix

module load anaconda
source activate image-segmentation
conda list | grep torchvision

srun --gres=gpu:1 python3 baseline_segmentation.py \
    --n-classes=$n_classes \
    --experiment-log-prefix $log_prefix \
    -r $relu_type \
    -n $experiment_name \
    --loss-type $loss_type \
    --loss-reduction $loss_reduction \
    -u $unsupervised_data_path \
    --epochs $epochs \
    --start-fold $start_fold \
    --end-fold $end_fold \
    --lr $learning_rate \
    --zero-gradients-every $zero_gradients_every \
    --batch-size-s $batch_size_s \
    --legacy-unet $use_legacy_unet \
    --augmentation-type $augmentation \
    --optimizer $optimizer \
    --focal-gamma $focal_gamma \
    --seed $seed \


