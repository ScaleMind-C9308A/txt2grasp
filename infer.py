from ds import get_data
from method import get_method
from alive_progress import alive_it

import wandb
import argparse
import torch
import os

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

    loader = train_ld if args.set == 'train' else valid_ld

    sv_dir = downloaded_model_path + f"/{mode}/{args.set}"
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

