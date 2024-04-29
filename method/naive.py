from .enc.img import get_img_enc
from .enc.txt import get_txt_enc
from .dec import get_decoder
from torch import nn

import torch.nn.functional as F
import torch


class Naive(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.img_enc = get_img_enc(args)
        self.txt_enc = get_txt_enc(args)
        self.decoder = get_decoder(args)
    
    def forward(self, img, txt, target):
        img_vector = self.img_enc(img)
        txt_vector = self.txt_enc(txt)

        latent = torch.cat([img_vector, txt_vector], dim=1)

        if self.args.fa:
            alpha = torch.rand_like(latent).to(latent.device)
            sigma1 = self.args.sig1 * torch.ones(alpha.size()).to(latent.device)
            torch.normal(mean=1, std=sigma1, out=alpha)

            beta = torch.rand_like(latent).to(latent.device)
            sigma2 = self.args.sig2 * torch.ones(alpha.size()).to(latent.device)
            torch.normal(mean=0, std=sigma2, out=beta)

            latent = alpha * latent + beta

        output = self.decoder(latent)

        return F.mse_loss(output, target), output