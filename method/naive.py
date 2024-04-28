from .enc.img import get_img_enc
from .enc.txt import get_txt_enc
from .dec import get_decoder
from torch import nn

import torch.nn.functional as F
import torch


class Naive(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.img_enc = get_img_enc(args)
        self.txt_enc = get_txt_enc(args)
        self.decoder = get_decoder(args)
    
    def forward(self, img, txt, target):
        img_vector = self.img_enc(img)
        txt_vector = self.txt_enc(txt)

        latent = torch.cat([img_vector, txt_vector], dim=1)

        output = self.decoder(latent)

        return F.mse_loss(output, target)