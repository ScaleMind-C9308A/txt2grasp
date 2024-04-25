from torch.utils.data import Dataset
from .utils import imgaug

import os, sys
import albumentations as A


ROOT = "/".join(__file__.split("/")[:-1]) + "/src"

class GAT(Dataset):
    def __init__(self, root = ROOT, train = True, img_size = ()) -> None:
        super().__init__()

        self.imgaug = A.Compose(imgaug(), p = 0.9)
        self.imgres = A.Compose([A.Resize(img_size[0], img_size[1]), A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) # H, W

        if train:
            self.part = 'seen'
        else:
            self.part = 'unseen'
        
        self.dir = root + f"/{self.part}"

        