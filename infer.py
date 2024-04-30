from ds import get_data, draw_point, draw_rec
from method import get_method
from alive_progress import alive_it
from torchvision import transforms

import wandb
import argparse
import torch
import os
import numpy as np
import cv2

invnorm = transforms.Compose(
    [
        transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.]),
    ]
)

def output_decode(x):
    x[:-1] *= 419
    x[-1] *= 180

    return x.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--run', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--set', type=str, required=True)

    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(f"truelove/grasp/{args.run}")
    artifact = run.use_artifact(api.artifact(f"truelove/grasp/{args.name}"))

    mode = args.name.split('-')[1]
    set = args.set
    
    downloaded_model_path = artifact.download()
    model_path = downloaded_model_path + f"/{mode}.pt"

    data = torch.load(model_path)
    print(f'load model :{model_path}')

    args = data['args']
    args.bs = 1 # inference

    args, train_ld, valid_ld = get_data(args)

    print(f"#TRAIN Batch: {len(train_ld)}")
    print(f"#VALID Batch: {len(valid_ld)}")

    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    loader = train_ld if set == 'train' else valid_ld

    sv_dir = downloaded_model_path + f"/{mode}/{set}"
    os.makedirs(sv_dir, exist_ok=True)

    model = get_method(args).to(device)
    model.eval()
    with torch.no_grad():
        for idx, (img, txt, lbl) in alive_it(enumerate(loader)):
            img = img.to(device)
            txt = txt.to(device)
            lbl = lbl.to(device)
            
            loss, output = model.predict(img, txt, lbl)

            output = output_decode(output[0])
            label = output_decode(lbl[0])

            output = [round(x) for x in output]
            label = [round(x) for x in label]

            img_np = invnorm(img[0]).permute(1, -1, 0).cpu().numpy()
            image = np.ascontiguousarray(img_np, dtype=np.uint8)

            print(output, label)

            img_draw_label = draw_rec(image, label, (0, 0, 255))
            img_draw_output = draw_rec(img_draw_label, output, (0, 255, 0))

            cv2.imwrite(sv_dir + f"/{idx}.jpg", img_draw_output)