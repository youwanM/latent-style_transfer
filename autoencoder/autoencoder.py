"""
---
title: Autoencoder for Stable Diffusion
summary: >
 Annotated PyTorch implementation/tutorial of the autoencoder
 for stable diffusion.
---

# Autoencoder for [Stable Diffusion](../index.html)

This implements the auto-encoder model used to map between image space and latent space.

We have kept to the model definition and naming unchanged from
[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
so that we can load the checkpoints directly.
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import nibabel as nib
import numpy as np
from tqdm import tqdm
from PIL import Image
from monai.losses import PerceptualLoss
import wandb

class VAETester(nn.Module):
    """
    Tester for VAE.
    """
    def __init__(
        self, 
        ae, 
        config:dict):

        super().__init__()

        self.test_iter = config.test_iter
        self.model_save_dir = config.model_save_dir
        self.sample_save_dir = config.sample_save_dir
        self.save_latent_vectors = config.save_latent_vectors

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ae
        self.model.load_state_dict(
        torch.load(
            f'./{self.model_save_dir}/model_{self.test_iter}.pth', 
            map_location = self.device,
            weights_only=True,
            )
        )

        self.model.to(self.device)

        print('Loaded model from epoch:', self.test_iter)

    def sample_test(self, x, id):
        x_gen, mean, log_var = self.model(x.float().to(self.device))

        affine = np.array([[4., 0., 0., -98.],
                           [0., 4., 0., -134.],
                           [0., 0., 4., -72.],
                           [0., 0., 0., 1.]])

        img_xgen = nib.Nifti1Image(
            np.array(
                x_gen.detach().cpu()
            )[0, 0, :, :, :],
            affine
        )

        img_x = nib.Nifti1Image(
            np.array(
                x.detach().cpu()
            )[0, 0, :, :, :],
            affine
        )

        if self.save_latent_vectors == True:
            latent_vectors = {"mu": mean, "variance": log_var}
            torch.save(latent_vectors, f'{self.sample_save_dir}/latent-{id}.pth')
        nib.save(img_x,f'{self.sample_save_dir}/original-{id}.nii.gz')
        nib.save(img_xgen, f'{self.sample_save_dir}/reconstructed-{id}.nii.gz')

    def test(self, dataset):
        dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=False
            )

        self.model.eval()

        print('--- Start test ---')

        for idx, img in enumerate(tqdm(dataloader)):
            self.sample_test(img, idx)

        print('--- End of test ---')


class VAETrainer(nn.Module):
    """
    Trainer for VAE.
    """
    def __init__(
        self, 
        ae, 
        config:dict):

        super().__init__()

        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.lr = config.lr
        self.beta = config.beta

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.model = ae
        self.checkpoint = config.load_checkpoint_training

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model_save_dir = config.model_save_dir

    def loss_function(self, x, x_hat, mean, log_var):
        '''
        Get total loss. 

        Parameters
        ----------
        x : tensor, source image
        x_hat : tensor, generated image
        mean : float, mean of Gaussian distribution
        log_var : float, log of variance of Gaussian distribution 

        Returns 
        -------
        reproduction_loss + KLD : tensor loss
        '''
        mse = nn.MSELoss()
        #LPIPS = PerceptualLoss(spatial_dims = 3, network_type="medicalnet_resnet50_23datasets", is_fake_3d=False, cache_dir=None, pretrained=True, pretrained_path=None, pretrained_state_dict_key=None, channel_wise=False).to('cpu')
        reproduction_loss = mse(x, x_hat)
        #perceptual_loss = LPIPS(x.cpu(), x_hat.cpu())
        KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss  + self.beta * KLD

    def train_step(self, x):
        '''
        Perform a step of training. 

        Parameters
        ----------
        x : tensor, batch of images
        '''

        x = x.float().to(self.device)

        self.optimizer.zero_grad()

        x_hat, mean, log_var = self.model(x, sample_posterior=True)
        loss = self.loss_function(x, x_hat, mean, log_var)
                
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, dataset):
        '''
        Parameters
        ----------
        dataset : ImageDataset 
        '''
        # Initialize wandb
        wandb.init(
            project="VAE",
            name='VAE Training',
            config={  # Optional: Hyperparameter configuration
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
            }
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
            )
        if self.checkpoint != "None":
            self.model.load_state_dict(
                torch.load(
                    self.checkpoint,
                    map_location = self.device,
                    weights_only=True,
                    )
                )
            print('Loaded model from checkpoint:', self.checkpoint)

        self.model.train()

        print('---- Start training ----')
        print('\n')

        for epoch in range(self.epochs):
            overall_loss = 0

            for idx, x in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}", total=len(dataloader), dynamic_ncols=True)):
                loss = self.train_step(x).item()
                wandb.log({"train_loss (iteration)": loss})

            overall_loss += loss

            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(idx*self.batch_size))
            wandb.log({"train_loss (epoch)": overall_loss/(idx*self.batch_size) })

            if self.device != 'cpu':
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if epoch % 2 or epoch==self.epochs-1:
                if torch.cuda.device_count() > 1:
                    torch.save(self.model.module.state_dict(), 
                        self.model_save_dir + 
                        f"/model_{epoch}.pth")
                else:
                    torch.save(self.model.state_dict(), 
                        self.model_save_dir + 
                        f"/model_{epoch}.pth")
                self.sample(x, epoch)

        wandb.finish()

        return overall_loss

    def sample(self, x, epoch):
        x_gen, mean, log_var = self.model(x.float().to(self.device))

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
        img_x = nib.Nifti1Image(
            np.array(
                x.detach().cpu()
                )[0,0,:,:,:],
            affine
            )

        x_mid = img_x.get_fdata()[:, :, :]
        x_mid = x_mid[:, x_mid.shape[1] // 2, :]
        x_mid = (x_mid - np.min(x_mid)) / (np.max(x_mid) - np.min(x_mid))
        x_mid = (x_mid * 255).astype(np.uint8)
        x_mid = Image.fromarray(x_mid)

        xgen_mid = img_xgen.get_fdata()[:, :, :]
        xgen_mid = xgen_mid[:, xgen_mid.shape[1] // 2, :]
        xgen_mid = (xgen_mid - np.min(xgen_mid)) / (np.max(xgen_mid) - np.min(xgen_mid))
        xgen_mid = (xgen_mid * 255).astype(np.uint8)
        xgen_mid = Image.fromarray(xgen_mid)

        nib.save(img_xgen, f'{self.model_save_dir}/sample_epoch-{epoch}.nii.gz')
        wandb.log({
            "Original": wandb.Image(x_mid, caption="Original"),
            "Reconstructed": wandb.Image(xgen_mid, caption="Reconstructed"),
        })




class Autoencoder(nn.Module):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(self, encoder: 'Encoder', decoder: 'Decoder', emb_channels: int, z_channels: int):
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        :param emb_channels: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * emb_channels, 1, bias=True)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv3d(emb_channels, z_channels, 1, bias=True)

    def encode(self, img: torch.Tensor) -> 'GaussianDistribution':
        """
        ### Encode images to latent representation

        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return GaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        ### Decode images from latent representation

        :param z: is the latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z)

        return dec, posterior.mean, posterior.log_var


class Encoder(nn.Module):
    """
    ## Encoder module
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 in_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the first convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            subsequent blocks
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param in_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        n_resolutions = len(channel_multipliers)

        # Initial $3 \times 3$ convolution layer that maps the image to `channels`
        self.conv_in = nn.Conv3d(in_channels, channels, 3, stride=1, padding=1, bias=True)

        # Number of channels in each top level block
        channels_list = [m * channels for m in [1] + channel_multipliers]

        # List of top-level blocks
        self.down = nn.ModuleList()
        # Create top-level blocks
        for i in range(n_resolutions):
            # Each top level block consists of multiple ResNet Blocks and down-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i + 1]))
                channels = channels_list[i + 1]
            # Top-level block
            down = nn.Module()
            down.block = resnet_blocks
            # Down-sampling at the end of each top level block except the last
            if i != n_resolutions - 1:
                down.downsample = DownSample(channels)
            else:
                down.downsample = nn.Identity()
            #
            self.down.append(down)

        # Final ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # Map to embedding space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, 2 * z_channels, (2, 3, 2), stride=1, padding=(2, 2, 2), bias=True) #kernel size 2,3,2

    def forward(self, img: torch.Tensor):
        """
        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """

        # Map to `channels` with the initial convolution
        x = self.conv_in(img)

        # Top-level blocks
        for down in self.down:
            # ResNet Blocks
            for block in down.block:
                x = block(x)
            # Down-sampling
            x = down.downsample(x)

        # Final ResNet blocks with attention
        x = self.mid.block_1(x)
        x = self.mid.attn_1(x)
        x = self.mid.block_2(x)

        # Normalize and map to embedding space
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    """
    ## Decoder module
    """

    def __init__(self, *, channels: int, channel_multipliers: List[int], n_resnet_blocks: int,
                 out_channels: int, z_channels: int):
        """
        :param channels: is the number of channels in the final convolution layer
        :param channel_multipliers: are the multiplicative factors for the number of channels in the
            previous blocks, in reverse order
        :param n_resnet_blocks: is the number of resnet layers at each resolution
        :param out_channels: is the number of channels in the image
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()

        # Number of blocks of different resolutions.
        # The resolution is halved at the end each top level block
        num_resolutions = len(channel_multipliers)

        # Number of channels in each top level block, in the reverse order
        channels_list = [m * channels for m in channel_multipliers]

        # Number of channels in the  top-level block
        channels = channels_list[-1]

        # Initial $3 \times 3$ convolution layer that maps the embedding space to `channels`
        self.conv_in = nn.Conv3d(z_channels, channels, (5,4,5), stride=1, padding=(1,1,1), bias=True) #kernel size 5,4,5

        # ResNet blocks with attention
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(channels, channels)
        self.mid.attn_1 = AttnBlock(channels)
        self.mid.block_2 = ResnetBlock(channels, channels)

        # List of top-level blocks
        self.up = nn.ModuleList()
        # Create top-level blocks
        for i in reversed(range(num_resolutions)):
            # Each top level block consists of multiple ResNet Blocks and up-sampling
            resnet_blocks = nn.ModuleList()
            # Add ResNet Blocks
            for _ in range(n_resnet_blocks + 1):
                resnet_blocks.append(ResnetBlock(channels, channels_list[i]))
                channels = channels_list[i]
            # Top-level block
            up = nn.Module()
            up.block = resnet_blocks
            # Up-sampling at the end of each top level block except the first
            if i != 0:
                up.upsample = UpSample(channels)
            else:
                up.upsample = nn.Identity()
            # Prepend to be consistent with the checkpoint
            self.up.insert(0, up)

        # Map to image space with a $3 \times 3$ convolution
        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, out_channels, (3,3,3), stride=1, padding=(1,1,1), bias=True)

    def forward(self, z: torch.Tensor):
        """
        :param z: is the embedding tensor with shape `[batch_size, z_channels, z_height, z_height]`
        """

        # Map to `channels` with the initial convolution
        h = self.conv_in(z)

        # ResNet blocks with attention
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # Top-level blocks
        for up in reversed(self.up):
            # ResNet Blocks
            for block in up.block:
                h = block(h)
            # Up-sampling
            h = up.upsample(h)

        # Normalize and map to image space
        h = self.norm_out(h)
        h = swish(h)
        img = self.conv_out(h)

        #
        return img


class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters: torch.Tensor):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        # Split mean and log of variance
        self.mean, self.log_var = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        #self.log_var = torch.clamp(log_var, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.log_var)

    def sample(self):
        # Sample from the distribution
        return self.mean + self.std * torch.randn_like(self.std)

    def mode(self):
        return self.mean


class AttnBlock(nn.Module):
    """
    ## Attention block
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # Group normalization
        self.norm = normalization(channels)
        # Query, key and value mappings
        self.q = nn.Conv3d(channels, channels, 1, bias=True)
        self.k = nn.Conv3d(channels, channels, 1, bias=True)
        self.v = nn.Conv3d(channels, channels, 1, bias=True)
        # Final $1 \times 1$ convolution layer
        self.proj_out = nn.Conv3d(channels, channels, 1, bias=True)
        # Attention scaling factor
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        """
        :param x: is the tensor of shape `[batch_size, channels, height, width]`
        """
        # Normalize `x`
        x_norm = self.norm(x)
        # Get query, key and vector embeddings
        q = self.q(x_norm)
        k = self.k(x_norm)
        v = self.v(x_norm)

        # Reshape to query, key and vector embeedings from
        # `[batch_size, channels, height, width]` to
        # `[batch_size, channels, height * width]`
        b, c, h, w, d = q.shape
        q = q.view(b, c, h * w * d)
        k = k.view(b, c, h * w * d)
        v = v.view(b, c, h * w * d)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)$
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Compute $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_{key}}}\Bigg)V$
        out = torch.matmul(attn, v)

        # Reshape back to `[batch_size, channels, height, width, depth]`
        out = out.view(b, c, h, w, d)
        # Final $1 \times 1$ convolution layer
        out = self.proj_out(out)

        # Add residual connection
        return x + out

class UpSample(nn.Module):
    """
    ## Up-sampling layer
    """
    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv3d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """
    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.conv = nn.Conv3d(channels, channels, 3, stride=2, padding=0, bias=True) #padding initial a 0

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, depth, height, width]`
        """
        # Apply convolution
        out = self.conv(x)
        return out


class ResnetBlock(nn.Module):
    """
    ## ResNet Block
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of channels in the input
        :param out_channels: is the number of channels in the output
        """
        super().__init__()
        # First normalization and convolution layer
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1, bias=True)
        # Second normalization and convolution layer
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=True)
        # `in_channels` to `out_channels` mapping layer for residual connection
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv3d(in_channels, out_channels, 1, stride=1, padding=0, bias=True)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """

        h = x

        # First normalization and convolution layer
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        # Second normalization and convolution layer
        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        # Map and add residual
        return self.nin_shortcut(x) + h


def swish(x: torch.Tensor):
    """
    ### swish activation

    $$x \cdot \sigma(x)$$
    """
    return x * torch.sigmoid(x)


def normalization(channels: int):
    """
    ### Batch normalization

    This is a helper function, with fixed number of groups and `eps`.
    """
    return nn.BatchNorm3d(channels)