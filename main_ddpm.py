from utils.datasets import ClassifDataset
from torch.backends import cudnn 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
from ddpm import DDPM
from autoencoder.autoencoder import Autoencoder, Encoder, Decoder
import numpy as np 
import pandas as pd
import random
import nibabel as nib 

def train(config):

    if not os.path.isdir(config.sample_dir):
        os.mkdir(config.sample_dir)
    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    ddpm = DDPM(config)

    # Data loader. 
    dataset_file = f'{config.data_dir}/{config.mode}-{config.dataset}.csv'

    dataset = ClassifDataset(
        dataset_file, 
        config.labels)

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    loader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        )

    optim = torch.optim.Adam(
        ddpm.parameters(), 
        lr=config.lrate
        )

    for ep in range(config.n_epoch):

        print(f'Epoch {ep}')
        if ddpm.device != 'cpu':
            if ddpm.device.type == 'cuda':
                torch.cuda.empty_cache()

        ddpm.train()

        # linear lrate decay
        optim.param_groups[0]['lr'] = config.lrate * (1 - ep / config.n_epoch)

        loss_ema = None

        for i, (x, c) in enumerate(loader):

            optim.zero_grad()

            x = x.to(ddpm.device)

            loss = ddpm(x.float())
            loss.backward()

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.95 * loss_ema + 0.05 * loss.item()

            optim.step()

        print('Loss:', loss_ema)

        if ep %10 == 0 or ep == config.n_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(ddpm.nn_model.module.state_dict(), config.model_save_dir + f"/model_{ep}.pth")
            else:
                torch.save(ddpm.nn_model.state_dict(), config.model_save_dir + f"/model_{ep}.pth")

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
            for i in range(config.n_classes):

                class_idx = [cl for cl in range(len(dataset.get_original_labels())) if dataset.get_original_labels()[cl]==dataset.label_list[i]]

                i_t_list = random.sample(class_idx, config.n_C)

                x_t_list = []

                for i_t in i_t_list:

                    x_t, c_t = dataset[i_t]

                    x_t_list.append(x_t)

                x_gen = ddpm.transfer(
                x, x_t_list
                )

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

                x_r, c_r = dataset[n//config.n_classes*config.n_classes+i]

                img_xreal = nib.Nifti1Image(
                    np.array(
                        x_r.detach().cpu()
                        )[0,:,:,:], 
                    affine
                    )

                img_xsrc = nib.Nifti1Image(
                    np.array(
                        x.detach().cpu()
                        )[0,0,:,:,:], 
                    affine
                    )

                c_idx = torch.argmax(c, dim=1)[0]
                c_t_idx = torch.argmax(c_t, dim=0)

                nib.save(img_xgen, f'{config.sample_dir}/gen-image_{n}-{config.dataset}_ep{config.test_iter}_n{config.n_C}-orig_{c_idx}-target_{c_t_idx}.nii.gz')
                nib.save(img_xreal, f'{config.sample_dir}/trg-image_{n}-{config.dataset}_ep{config.test_iter}_n{config.n_C}-orig_{c_idx}-target_{c_t_idx}.nii.gz')
                nib.save(img_xsrc, f'{config.sample_dir}/src-image_{n}-{config.dataset}_ep{config.test_iter}_n{config.n_C}-orig_{c_idx}-target_{c_t_idx}.nii.gz')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='dataset_rh_4classes')
    parser.add_argument('--labels', type=str, help='conditions for generation',
                        default='pipelines')
    parser.add_argument('--sample_dir', type=str, default='sampling directory')
    parser.add_argument('--model_save_dir', type=str, default='save directory')

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'transfer'])
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--n_epoch', type=int, default=500, help='number of total iterations')
    parser.add_argument('--lrate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--n_feat', type=int, default=64, help='number of features')
    parser.add_argument('--n_classes', type=int, default=24, help='number of classes')
    parser.add_argument('--beta', type=tuple, default=(1e-4, 0.02), help='number of classes')
    parser.add_argument('--n_T', type=int, default=500, help='number T')
    parser.add_argument('--n_C', type=int, default=10, help='number C')
    parser.add_argument('--model_param', type=str, default='./feature_extractor/models/model_b-64_lr-1e-04_epochs_150.pth', 
        help='epoch of classifier embedding')
    parser.add_argument('--ae_param', type=str, default='./vae_models-no_sampling/model_9.pth', 
        help='epoch of autoencoder')
    parser.add_argument('--test_iter', type=int, default=30, help='epochs to test')

    config = parser.parse_args()

    if config.mode == 'train':
        train(config)

    elif config.mode == 'transfer':
        transfer(config)