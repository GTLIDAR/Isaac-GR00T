#!/bin/bash

#SBATCH --job-name=ForceVLA_digit_v3_%j
#SBATCH --output=/coc/flash12/fwu91/Research/forcevla/Isaac-GR00T/logs/%j/output.log
#SBATCH --error=/coc/flash12/fwu91/Research/forcevla/Isaac-GR00T/logs/%j/error.log
#SBATCH --partition="wu-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu="48G"
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=feiyangwu@gatech.edu

conda activate gr00t
export CUDA_VISIBLE_DEVICES=0
cd /coc/flash12/fwu91/Research/forcevla/Isaac-GR00T

srun -u python gr00t/experiment/launch_finetune.py   
    --base-model-path ./nvidia/GR00T-N1.6-3B
    --dataset-path examples/Digit_v3/digit_loco_dataset/   
    --embodiment-tag NEW_EMBODIMENT   
    --modality-config-path examples/Digit_v3/modality_config.py   
    --num-gpus 1   
    --output-dir examples/Digit_v3/logs/0106-2   
    --save-total-limit 5   
    --save-steps 8000   
    --max-steps 8000   
    --use-wandb   
    --global-batch-size 32
    # --dataloader_num_workers 6 \
    # --color_jitter_params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08