from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import numpy as np
from models.unet import UNetModel
import torch.nn.init as init
import importlib
from feature_extractor import model as md
from autoencoder.autoencoder import Autoencoder, Encoder, Decoder


def ddpm_schedules(beta1, beta2, T):
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    """
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

def init_weights_normal(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            init.zeros_(m.bias)

class DDPM(nn.Module):
    def __init__(self, config):
        super(DDPM, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.nn_model = UNetModel(in_channels=4,
                                  out_channels=4,
                                  channels=96,
                                  attention_levels=[0, 1],
                                  n_res_blocks=1,
                                  channel_multipliers=[1, 2],
                                  n_heads=1,
                                  tf_layers=1)

        self.betas = config.beta
        self.n_T = config.n_T

        if torch.cuda.device_count() > 1:
            self.nn_model = nn.DataParallel(self.nn_model)

        print("Let's use", torch.cuda.device_count(), "GPU(s)!")

        self.nn_model.to(self.device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(self.betas[0], self.betas[1], self.n_T).items():
            self.register_buffer(k, v)

        self.loss_mse = nn.MSELoss()

        encoder = Encoder(z_channels=4,
                          in_channels=1,
                          channels=16,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

        decoder = Decoder(out_channels=1,
                          z_channels=4,
                          channels=16,
                          channel_multipliers=[1, 2, 4, 4],
                          n_resnet_blocks=2)

        ae = Autoencoder(emb_channels=4,
                         encoder=encoder,
                         decoder=decoder,
                         z_channels=4)

        ae.load_state_dict(
            torch.load(
                config.ae_param,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.vae = ae.eval().to(self.device)
        # Weight initialization
        self.nn_model.apply(init_weights_normal)

    def forward(self, x_img):
        """
        this method is used in training, so samples t and noise randomly
        """
        # for sampling noise and real
        x = self.vae.encode(x_img.float().to(self.device)).mode()

        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
                self.sqrtab.to(self.device)[_ts, None, None, None, None] * x
                + self.sqrtmab.to(self.device)[_ts, None, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        t = _ts / self.n_T

        # return MSE between added noise, and our predicted noise
        with torch.cuda.amp.autocast():
            loss = self.loss_mse(noise, self.nn_model(x_t, t))
        return loss

    def transfer(self, source):

        x_i = self.vae.encode(source.to(self.device).float()).sample()

        noise = torch.randn_like(x_i)  # eps ~ N(0, 1)
        x_t = (
                self.sqrtab.to(self.device)[self.n_T] * x_i
                + self.sqrtmab.to(self.device)[self.n_T] * noise
        )

        for i in tqdm(range(self.n_T, 0, -1), desc="Sampling Timesteps"):
            t_is = torch.tensor([i / self.n_T]).to(self.device)
            # t_is = t_is.repeat(1,1,1,1,1)

            z = torch.randn(*x_t.shape).to(self.device) if i > 1 else 0

            # split predictions and compute weighting
            eps = self.nn_model(x_t.float(), t_is.float())

            x_t = (
                    self.oneover_sqrta[i] * (x_t - eps * self.mab_over_sqrtmab[i])
                    + self.sqrt_beta_t[i] * z
            )

        x_g = self.vae.decode(x_t)

        return x_g