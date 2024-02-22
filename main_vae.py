from utils.datasets import ImageDataset
from torch.backends import cudnn 
import torch
from torch.utils.data import DataLoader
import os
import argparse
from ddpm import DDPM
from autoencoder.autoencoder import Autoencoder, Encoder, Decoder, VAETrainer
import numpy as np 
import pandas as pd
import random
import nibabel as nib 

def train(config):

    if not os.path.isdir(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    encoder = Encoder(z_channels=4,
                      in_channels=1,
                      channels=128,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=2)

    decoder = Decoder(out_channels=1,
                      z_channels=4,
                      channels=128,
                      channel_multipliers=[1, 2, 4, 4],
                      n_resnet_blocks=2)

    ae = Autoencoder(emb_channels=4,
                      encoder=encoder,
                      decoder=decoder,
                      z_channels=4)

    # Data loader. 
    dataset_file = f'{config.data_dir}/train-{config.dataset}.csv'
    data_flist = pd.read_csv(dataset_file)['filepaths']

    dataset = ImageDataset(
        data_flist
    )

    print(f'Dataset {config.dataset}: \n {len(dataset)} images.')

    trainer = VAETrainer(ae, config)
    trainer.train(dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='dataset_rh_4classes')
    parser.add_argument('--model_save_dir', type=str, default='./vae-models')
    parser.add_argument('--batch_size', type=int, default=32, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of total iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--beta', type=float, default=1e-6, help='weight factor for KLD')

    config = parser.parse_args()

    train(config)