#!/bin/bash
#SBATCH --job-name=train-vae # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --partition=gpu_p13
#SBATCH --ntasks=1          # number of MPI tasks per node
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=19:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=vae-train%j.out # output file name
#SBATCH --error=vae-train%j.err  # error file name

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u \
/gpfswork/rech/gft/umh25bv/latent-style_transfer/main_vae.py \
--data_dir data --dataset dataset_rh_4classes-jeanzay \
--model_save_dir ./vae_models-beta_1e-5 --batch_size 8 --epochs 200 --lr 1e-3 --beta 1e-5

# /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u /gpfswork/rech/gft/umh25bv/latent-style_transfer/main_ddpm.py \
# --mode train --dataset dataset_rh_4classes-jeanzay \
# --labels pipelines --model_save_dir ddpm_models \
# --batch_size 4 --lrate 1e-4 --n_epoch 200 --n_classes 4 \
# --sample_dir ddpm_samples --ae_param /gpfswork/rech/gft/umh25bv/latent-style_transfer/vae_models/model_17.pth \
# --model_param /gpfswork/rech/gft/umh25bv/latent-style_transfer/feature_extractor/models/model_b-64_lr-1e-04_epochs_150.pth 