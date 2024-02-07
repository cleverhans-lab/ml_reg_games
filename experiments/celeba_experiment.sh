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
    --save_path $EROOT/results/celeba/ \
    --path $RROOT/celeba/ \
    --data_dir /mfsnic/datasets/ \
    --num_rounds 20 \
    --init_priv 3 \
    --init_fair 0.01 \
    --step_size_decay 1.5 \
    --dataset 'celebasensitive' \
    --priority 'regulators' \
    --builder_lambda 0.7 \
    --C_priv 0 \
    --C_fair 0 \
    --goal_priv 3 \
    --goal_fair 0.01 \
    --calibration 1 \
    --init_as_goal 1 \
    --save 1
