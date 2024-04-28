from ds import get_data
from method import get_method
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from alive_progress import alive_it
from metrics import probiou

import torch
import wandb
import json
import hashlib
import os

def get_hash(args):
    args_str = json.dumps(vars(args), sort_keys=True)
    args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
    return args_hash

def trainer(args):

    if torch.cuda.is_available():
        device = torch.device("cuda", index=args.idx)
    else:
        device = torch.device("cpu")

    args, train_ld, valid_ld = get_data(args)

    print(f"#TRAIN Batch: {len(train_ld)}")
    print(f"#VALID Batch: {len(valid_ld)}")

    run_name = get_hash(args)

    if args.log:
        run = wandb.init(
            project='grasp',
            entity='truelove',
            config=args,
            name=run_name,
            force=True
        )
    
    run_dir = os.getcwd() + '/runs'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)
    
    sv_dir = run_dir + f"/{run_name}"
    if not os.path.exists(sv_dir):
        os.mkdir(sv_dir)
    
    best_model_path = sv_dir + f'/best.pt'
    last_model_path = sv_dir + f'/last.pt'

    model = get_method(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params: {total_params}")
    total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Params: {total_train_params}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, len(train_ld) * args.epoch)


    old_valid_loss = 1e26
    for epoch in range(args.epoch):
        log_dict = {}
        
        model.train()
        total_loss = 0
        for img, txt, lbl in alive_it(train_ld):
            img = img.to(device)
            txt = txt.to(device)
            lbl = lbl.to(device)

            loss, _ = model(img, txt, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            scheduler.step()
            
            total_loss += loss.item()
        
        train_mean_loss = total_loss / len(train_ld)
        
        log_dict['train/loss'] = train_mean_loss

        print(f"Epoch: {epoch} - Train Loss: {train_mean_loss}")

        model.eval()
        with torch.no_grad():
            total_loss = 0
            for img, txt, lbl in alive_it(valid_ld):
                img = img.to(device)
                txt = txt.to(device)
                lbl = lbl.to(device)

                loss, _ = model(img, txt, lbl)
                total_loss += loss.item()
        
        valid_mean_loss = total_loss / len(valid_ld)
        log_dict['valid/loss'] = valid_mean_loss
        print(f"Epoch: {epoch} - Valid Loss: {valid_mean_loss}")

        save_dict = {
            'args' : args,
            'model_state_dict': model.state_dict()
        }

        if valid_mean_loss < old_valid_loss:
            old_valid_loss = valid_mean_loss
            
            torch.save(save_dict, best_model_path)
        torch.save(save_dict, last_model_path)

        if args.log:
            run.log(log_dict)