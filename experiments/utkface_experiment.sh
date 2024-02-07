#!/bin/bash
#SBATCH --array=0
#SBATCH --mem=20GB
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
    --save_path $EROOT/results/utkface/ \
    --path $RROOT/utkface/ \
    --data_dir /mfsnic/datasets/ \
    --num_rounds 20 \
    --init_priv 4 \
    --init_fair 0.1 \
    --dataset 'utkface' \
    --priority 'regulators' \
    --step_size_decay 1.5 \
    --builder_lambda 0.7 \
    --lambda_priv 0.01 \
    --lambda_fair 0.3 \
    --C_priv 3 \
    --C_fair 3 \
    --goal_priv 4 \
    --goal_fair 0.1 \
    --calibration 1 \
    --init_as_goal 1 \
    --save 1
