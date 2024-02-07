#!/bin/bash
#SBATCH --array=0
#SBATCH --mem=40GB
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --qos=nic
#SBATCH --gres=gpu:1
#SBATCH -A nic
#SBATCH -c 4
#SBATCH --partition=nic

EROOT=""
RROOT=""

poetry run python $EROOT/main.py \
    --experiment_name "test" \
    --prev_results_dir $EROOT/previous_results/ \
    --save_path $EROOT/results/two_datasets/ \
    --path $RROOT/two_datasets/ \
    --data_dir /mfsnic/datasets/ \
    --num_rounds 20 \
    --init_priv 7 \
    --init_fair 0.05 \
    --dataset 'utkface' 'fairface' \
    --priority 'regulators' \
    --lambda_priv 0.01 \
    --lambda_fair 0.3 \
    --C_priv 3 \
    --C_fair 3 \
    --goal_priv 7 \
    --goal_fair 0.05 \
    --calibration 1 \
    --save 1