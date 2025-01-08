from utils.datasets import ImageDataset
from torch.backends import cudnn 
import torch
from torch.utils.data import DataLoader
import os
import argparse
from ddpm import DDPM
from autoencoder.autoencoder import Autoencoder, Encoder, Decoder, VAETrainer, VAETester
import numpy as np 
import pandas as pd
import random
import nibabel as nib
import wandb

def train(config):

    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    encoder = Encoder(z_channels=32,
                      in_channels=1,
                      channels=8,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=1)

    decoder = Decoder(out_channels=1,
                      z_channels=32,
                      channels=8,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=1)

    ae = Autoencoder(emb_channels=1,
                      encoder=encoder,
                      decoder=decoder,
                      z_channels=32)

    # Data loader. 
    dataset_file = f'{config.data_dir}/train-{config.dataset}.csv'
    data_flist = pd.read_csv(dataset_file)['filepaths']

    dataset = ImageDataset(
        data_flist
    )

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    trainer = VAETrainer(ae, config)
    trainer.train(dataset)


def test(config):
    if not os.path.isdir(config.sample_save_dir):
        os.mkdir(config.sample_save_dir)

    encoder = Encoder(z_channels=32,
                      in_channels=1,
                      channels=8,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=1)

    decoder = Decoder(out_channels=1,
                      z_channels=32,
                      channels=8,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=1)

    ae = Autoencoder(emb_channels=1,
                      encoder=encoder,
                      decoder=decoder,
                      z_channels=32)
    # Data loader. 
    dataset_file = f'{config.data_dir}/test-{config.dataset}.csv'
    data_flist = pd.read_csv(dataset_file)['filepaths']

    dataset = ImageDataset(
        data_flist
    )

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    tester = VAETester(ae, config)
    tester.test(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='IXI-T1full')
    parser.add_argument('--model_save_dir', type=str, default='./vae/vae-model-334epochs z32/')
    parser.add_argument('--sample_save_dir', type=str, default='./vae-samples')
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=1000, help='number of total iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=1e-3, help='weight factor for KLD')
    parser.add_argument('--test_iter', type=int, default=309, help='iteration to test')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--save_latent_vectors', type=bool, default=True)

    config = parser.parse_args()


    if config.mode == 'train':
        train(config)
    else:
        test(config)