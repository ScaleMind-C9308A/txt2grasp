from trainer import trainer

import argparse
import numpy as np
import torch
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='MTLDOG - Domain Generalization for Multi-task Learning')

    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--wk', type=str, default=1, help='number of workers')
    parser.add_argument('--pm', action='store_true', help='pin memory')
    parser.add_argument('--sz', type=int, default=224, help='size of processed image (h, w)')
    parser.add_argument('--aug', action='store_true', help='augmentation')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--idx', type=int, default=0, help='device index')
    parser.add_argument('--epoch', type=int, default=1, help='#epoch')

    parser.add_argument('--method', type=str, default='naive', help='method')
    parser.add_argument('--txtenc', type=str, default='gpt2', help='text encoder')
    parser.add_argument('--imgenc', type=str, default='resnet18', help='img encoder')
    parser.add_argument('--dec', type=str, default='basev0', help='decoder')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--log', action='store_true', help='wandb logging')

    # gpt2
    parser.add_argument('--bls', type=int, default=16, help='blocksize')
    parser.add_argument('--nl', type=int, default=4, help='#layers')
    parser.add_argument('--nh', type=int, default=8, help='#heads')
    parser.add_argument('--ne', type=int, default=512, help='embedding size')

    # imgencoder
    parser.add_argument('--w', action='store_true', help='using ImageNet pretrained weight')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    trainer(args=args)