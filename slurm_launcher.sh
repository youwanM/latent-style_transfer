#!/bin/bash
#SBATCH --job-name=train-vae # nom du job
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=32          # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --distribution=block:block   # we pin the tasks on contiguous cores
#SBATCH --time=19:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=vae-train%j.out # output file name
#SBATCH --error=vae-train%j.err  # error file name

source /gpfswork/rech/gft/umh25bv/miniconda3/bin/activate /gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv

/gpfswork/rech/gft/umh25bv/miniconda3/envs/workEnv/bin/python3 -u \
/gpfswork/rech/gft/umh25bv/latent-style_transfer/main_vae.py \
--data_dir data --dataset dataset_rh_4classes-jeanzay \
--model_save_dir vae_models --batch_size 4 --epochs 200 --lr 1e-4