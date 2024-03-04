#!/bin/bash
#SBATCH --job-name=train_vae # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --partition=gpu_p13
#SBATCH --ntasks=1          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=32:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=train_vae-rfrh%j.out # output file name
#SBATCH --error=train_vae-rfrh%j.err  # error file name
#SBATCH --qos=qos_gpu-t4

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u \
/gpfswork/rech/gft/umh25bv/latent-style_transfer/main_vae.py --mode train \
--data_dir data --dataset dataset_rhrh_4classes-jeanzay \
--model_save_dir ./vae_models-rfrh --batch_size 8 --epochs 200 --lr 1e-4 --beta 1e-3

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u \
# /gpfswork/rech/gft/umh25bv/latent-style_transfer/main_vae.py --mode test \
# --data_dir data --dataset dataset_rh_4classes-jeanzay \
# --model_save_dir ./vae_models --sample_save_dir ./vae_samples \
# --test_iter 83

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/latent-style_transfer/main_ddpm.py \
# --mode train --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir ddpm_models-eval \
# --batch_size 4 --lrate 1e-4 --n_epoch 50 --n_classes 4 \
# --sample_dir ddpm_samples --ae_param /gpfswork/rech/gft/umh25bv/latent-style_transfer/vae_models-no_sampling/model_83.pth \
# --model_param /gpfswork/rech/gft/umh25bv/latent-style_transfer/feature_extractor/models/model_b-64_lr-1e-04_epochs_150.pth 