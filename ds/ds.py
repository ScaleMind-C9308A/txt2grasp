from torch.utils.data import Dataset, DataLoader
from .utils import imgaug, read_pickle
from glob import glob
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer

import os
import albumentations as A
import cv2
import torch
import numpy as np


CURR = "/".join(__file__.split("/")[:-1])
ROOT = CURR + "/src"

class GAT(Dataset):
    def __init__(self, root = ROOT, train = True, img_size = 224, vocab=None, aug=False) -> None:
        super().__init__()

        if train:
            self.part = 'seen'
        else:
            self.part = 'unseen'

        self.aug = A.Compose(imgaug(), p = 0.9, bbox_params=A.BboxParams(format='yolo'))
        self.res = A.Compose([A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        
        self.dir = root + f"/{self.part}"

        self.inss = glob(self.dir + "/grasp_instructions/*")
        self.lbls = [self.dir + f"/grasp_label/{os.path.basename(x).replace('.pkl', '')}.pt" for x in self.inss]
        self.imgs = [self.dir + f"/image/{os.path.basename(x).split('_')[0]}.jpg" for x in self.inss]

        self.vocab_path = CURR + "/vocab.pkl"
        if not os.path.exists(self.vocab_path):
            raise ValueError(f"No Vocab Found at {self.vocab_path}")

        self.vocab = vocab
        self.aug = aug
        self.tr = train
        self.sz = img_size

        self.pad_idx = self.vocab['<pad>']
        self.bos_idx = self.vocab['<bos>']
        self.eos_idx = self.vocab['<eos>']

        self.tokenizer = get_tokenizer("basic_english")
    
    def __len__(self):
        return len(self.inss)

    def __getitem__(self, index):
        ins_path = self.inss[index]
        lbl_path = self.lbls[index]
        img_path = self.imgs[index]

        img = cv2.imread(img_path)
        W, H, _ = img.shape
        x, y, w, h, a = self.lblproc(lbl_path)
        ins = read_pickle(ins_path)

        if self.tr and self.aug:
            aug_transformed = self.aug(image=img)
            transformed = self.res(image = aug_transformed['image'])
            transformed_image = transformed['image']
        else:
            transformed = self.res(image=img)
            transformed_image = transformed['image']

        tokens = torch.tensor([self.vocab[token] for token in self.tokenizer(ins)], dtype=torch.long)
        
        img = torch.from_numpy(transformed_image).permute(-1, 0, 1).float()
        lbl = torch.from_numpy(np.array([x/W, y/W, w/W, h/W, a/180])).float()

        return img, lbl, tokens

    @staticmethod
    def lblproc(path):
        data = torch.load(path)
        quality = data[:, 0]
        ext_data = data[quality.argmax()].tolist()
        return ext_data[1:]

def get_data(args):

    vocab_path = CURR + "/vocab.pkl"
    if not os.path.exists(vocab_path):
        raise ValueError(f'Vocan is not found at {vocab_path}')

    vocab = read_pickle(vocab_path)

    args.vcs = len(vocab)

    train_ds = GAT(train=True, img_size=args.sz, vocab=vocab, aug=args.aug)
    valid_ds = GAT(train=False, img_size=args.sz, vocab=vocab, aug=args.aug)

    PAD_IDX = vocab['<pad>']
    BOS_IDX = vocab['<bos>']
    EOS_IDX = vocab['<eos>']

    def generate_batch(data_batch):
        tok_batch = []
        lbl_batch = []
        img_batch = []
        for (img, lbl, tok) in data_batch:
            tok_batch.append(torch.cat([torch.tensor([BOS_IDX]), tok, torch.tensor([EOS_IDX])], dim=0))
            img_batch.append(img)
            lbl_batch.append(lbl)
        tok_batch = pad_sequence(tok_batch, padding_value=PAD_IDX).T
        img_batch = torch.stack(img_batch)
        lbl_batch = torch.stack(lbl_batch)
        return img_batch, tok_batch, lbl_batch 

    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)
    valid_ld = DataLoader(valid_ds, batch_size=args.bs, shuffle=True, collate_fn=generate_batch, pin_memory=args.pm)

    return args, train_ld, valid_ld