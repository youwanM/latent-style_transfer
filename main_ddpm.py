from utils.datasets import ImageDataset, ClassifDataset
from torch.backends import cudnn 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import DataParallel

import os
import argparse
from ddpm import DDPM
from autoencoder.autoencoder import Autoencoder, Encoder, Decoder
import numpy as np 
import pandas as pd
import random
import nibabel as nib
import wandb
from tqdm import tqdm

def train(config):
    wandb.init(
        project="LDM",
        name='DDPM Training',
        config={  # Optional: Hyperparameter configuration
            "learning_rate": 0000,
            "batch_size": 0000,
        }
    )

    torch.manual_seed(42)

    if not os.path.isdir(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    ddpm = DDPM(config)

    # Data loader.
    dataset_file = f'{config.data_dir}/train-{config.dataset}.csv'
    data_flist = pd.read_csv(dataset_file)['filepaths']

    dataset = ImageDataset(
        data_flist
    )

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        )

    optim = torch.optim.Adam(
        ddpm.parameters(), 
        lr=config.lrate,
        weight_decay=1e-2
        )

    for ep in range(config.n_epoch):

        print(f'Epoch {ep}')
        if ddpm.device != 'cpu':
            if ddpm.device.type == 'cuda':
                torch.cuda.empty_cache()

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = config.lrate * (1 - ep / config.n_epoch)

        loss_ema = True

        for i, (x) in enumerate(tqdm(loader, desc=f"Epoch {ep + 1}", total=len(loader), dynamic_ncols=True)):

            optim.zero_grad()

            x = x.to(ddpm.device)

            loss = ddpm(x.float())
            wandb.log({"train_loss (iteration)": loss})
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            nn.utils.clip_grad_norm_(ddpm.nn_model.parameters(), max_norm=1.0)

            optim.step()

        print('Loss:', loss_ema)

        if ep %2 == 0 or ep == config.n_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(ddpm.nn_model.module.state_dict(), config.model_save_dir + f"/model_{ep}.pth")
            else:
                torch.save(ddpm.nn_model.state_dict(), config.model_save_dir + f"/model_{ep}.pth")
            
            ddpm.eval()

        with torch.no_grad():
            x_gen = ddpm.transfer(x)

            affine = np.array([[   4.,    0.,    0.,  -98.],
                       [   0.,    4.,    0., -134.],
                       [   0.,    0.,    4.,  -72.],
                       [   0.,    0.,    0.,    1.]])

            img_xgen = nib.Nifti1Image(
            np.array(
                x_gen.detach().cpu()
                )[0,0,:,:,:], 
            affine
            )

            img_xsrc = nib.Nifti1Image(
            np.array(
                x.detach().cpu()
                )[0,0,:,:,:], 
            affine
            )

            nib.save(img_xgen, f'{config.sample_dir}/gen-image_{i}-{config.dataset}.nii.gz')
            nib.save(img_xsrc, f'{config.sample_dir}/src-image_{i}-{config.dataset}.nii.gz')
            ddpm.train()

def transfer(config):
    # Load models
    ddpm = DDPM(config)

    ddpm.nn_model.load_state_dict(
        torch.load(
            config.model_save_dir + f"/model_{config.test_iter}.pth", 
            map_location=torch.device('cpu')
            )
        )

    # Data loader. 
    dataset_file = f'{config.data_dir}/test-{config.dataset}.csv'
    dataset = ClassifDataset(
        dataset_file, 
        config.labels)
    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')


    source_loader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        )

    for n, (x, c) in enumerate(source_loader):

        ddpm.eval()

        with torch.no_grad():
            x_gen = ddpm.transfer(x)

            affine = np.array([[   4.,    0.,    0.,  -98.],
                       [   0.,    4.,    0., -134.],
                       [   0.,    0.,    4.,  -72.],
                       [   0.,    0.,    0.,    1.]])

            img_xgen = nib.Nifti1Image(
            np.array(
                x_gen.detach().cpu()
                )[0,0,:,:,:], 
            affine
            )

            img_xsrc = nib.Nifti1Image(
            np.array(
                x.detach().cpu()
                )[0,0,:,:,:], 
            affine
            )

            nib.save(img_xgen, f'{config.sample_dir}/gen-image_{n}-{config.dataset}_ep{config.test_iter}.nii.gz')
            nib.save(img_xsrc, f'{config.sample_dir}/src-image_{n}-{config.dataset}_ep{config.test_iter}.nii.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='IXI-T1')
    parser.add_argument('--labels', type=str, help='conditions for generation',
                        default='pipelines')
    parser.add_argument('--sample_dir', type=str, default='ddpm_sampling')
    parser.add_argument('--model_save_dir', type=str, default='ddpm_temp')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'transfer'])
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=1000, help='number of total iterations')
    parser.add_argument('--lrate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--beta', type=tuple, default=(1e-4, 0.02), help='Beta Schedule for DDPM')
    parser.add_argument('--n_T', type=int, default=1000, help='number T')
    parser.add_argument('--ae_param', type=str, default='./vae_checkpoints/Jan_19_2025_95.pth',
        help='epoch of autoencoder')
    parser.add_argument('--test_iter', type=int, default=30, help='epochs to test')

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)
        wandb.finish()

    elif config.mode == 'transfer':
        transfer(config)
