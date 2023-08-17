import argparse
import os
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.distributed as dist
import torch.optim as optim
from losses.loss import Loss
from optimizers.lr_scheduler import WarmupCosineSchedule
from torch.cuda.amp import GradScaler, autocast

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from utils.ops import *
from networks.vision_transformer import SwinUnet, HybridSwinUnet
from utils.data_utils import InpaintingDataset
from utils.my_config import load_yaml
from utils.config import get_config


def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)
        
    def train(args, global_step, train_loader, val_best, scaler):
        model.train()
        loss_train_recon = []
        loss_valid_recon = []
        for step, (masked_img, img) in enumerate(train_loader):
            t1 = time()
            x, img = masked_img.cuda(), img.cuda()
            x_aug = aug_rand(img)
            with autocast(enabled=args.amp):
                imgs_recon = model(x_aug)
                loss = loss_function(imgs_recon, img)
                
            loss_train_recon.append(loss.item())
            
            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            
            if global_step % args.print_step == 0:
                print("Step:{}/{}, Loss:{:.4f}, Time:{:.4f}".format(global_step, args.num_steps, loss, time() - t1))

            global_step += 1
            
            val_cond = global_step % 100 == 0

            if val_cond:
                val_loss, img_list = validation(args, global_step, valid_loader)
                
                val_loss_recon = np.mean(val_loss)
                writer.add_scalar("Validation/loss_recon", scalar_value=val_loss_recon, global_step=global_step)
                writer.add_scalar("train/loss_recon", scalar_value=np.mean(loss_train_recon), global_step=global_step)

                writer.add_image("Validation/x1_gt", img_list[0], global_step, dataformats="CHW")
                writer.add_image("Validation/x1_aug", img_list[1], global_step, dataformats="CHW")
                
                if val_loss_recon < val_best:
                    val_best = val_loss_recon
                    checkpoint = {
                        "global_step": global_step,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_ckp(checkpoint, logdir + "/model_bestValRMSE.pt")
                    print("Model was saved ! Best Recon. Val Loss: {:.4f}".format(val_best))
                else:
                    print("Model was not saved ! Best Recon. Val Loss: {:.4f}".format(val_best))
                loss_valid_recon.append(val_loss_recon)
                
        return global_step, loss_train_recon, val_best, loss_valid_recon
    
    def validation(args, global_step, test_loader):
        model.eval()
        loss_val_recon = []
        with torch.no_grad():
            for step, (masked_img, img) in enumerate(test_loader):
                val_inputs, img = masked_img.cuda(), img.cuda()
                val_inputs = aug_rand(img)
                with autocast(enabled=args.amp):
                    imgs_recon = model(val_inputs)
                    loss = loss_function(imgs_recon, img)
                
                loss_val_recon.append(loss.item())
                x_gt = img.detach().cpu().numpy()
                xgt = x_gt[0] * 255.0
                xgt = xgt.astype(np.uint8)
                
                rec_x1 = imgs_recon.detach().cpu().numpy()
                rec_x1 = (rec_x1 - np.min(rec_x1)) / (np.max(rec_x1) - np.min(rec_x1))
                recon = rec_x1[0] * 255.0
                recon = recon.astype(np.uint8)
                img_list = [xgt, recon]
                
                if step % 100 == 0:
                    print("Validation step:{}, Loss Reconstruction:{:.4f}".format(step, loss))
                
                if step == 0 or step == len(test_loader)-1:
                    np.save(f'/home/hyukiggle/Documents/workspace/pretrain/reconstruction/results/{global_step}_{step}gt.npy', xgt)
                    np.save(f'/home/hyukiggle/Documents/workspace/pretrain/reconstruction/results/{global_step}_{step}pred.npy', recon)
        
        return loss_val_recon, img_list
    
    parser = argparse.ArgumentParser(description="ImageNet pretraining")
    
    # -----------------------------------------------------------------------------
    # Model settings
    # -----------------------------------------------------------------------------
    parser.add_argument("--cfg", type = str, metavar="FILE", help="Path to config file")
    # parser.add_argument("--img_size", default=256, type=int)
    # parser.add_argument("--patch_size", default=4, type =int)
    # parser.add_argument("--in_channels", default=3, type=int)
    # parser.add_argument("--embed_dim", default=96, type=int)
    # parser.add_argument("--window_size", default=7, type=int)
    # parser.add_argument("--mlp_ratio", default=4.0, type=float)
    # parser.add_argument("--qkv_bias", default=True, type=bool)
    # parser.add_argument("--qk_scale", default=None, type=float)
    # parser.add_argument("--drop_rate", default=0, type=float)
    # parser.add_argument("--drop_path_rate", default=0.1, type=float)
    # parser.add_argument("--ape", default=False, help="absolute positional embedding")
    # parser.add_argument("--patch_norm", default=True, type=bool)
    # parser.add_argument("--use_checkpoint", default=False, type=bool)    
    
    # # -----------------------------------------------------------------------------
    # # Training settings
    # # -----------------------------------------------------------------------------
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--out_channel", default=3, type=int)
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--epochs", default=150, type = int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_steps", default=500000, type=int, help='number of training iterations')
    parser.add_argument("--warmup_steps", default = 500, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--print_step", default=100, type=int)
    parser.add_argument("--opts", default='adamw', type=str)
    parser.add_argument("--lrdecay", action="store_true", help="enable learing rate scheduling")
    parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
    parser.add_argument("--noamp", action="store_true")
    parser.add_argument("--n_gpu", default=1, type=int)
    parser.add_argument("--root_dir", default = "/home/hyukiggle/Documents/data/ImageNet", type=str)
    
    args = parser.parse_args()
        
    args.amp = not args.noamp
    logdir = os.path.join('./runs', args.logdir)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
    config = get_config(args)
    print(config)
    
    model = SwinUnet(config=config,img_size=args.img_size, num_classes=args.out_channel).cuda()
    loss_function = Loss(batch_size=args.batch_size*args.batch_size, args=args)
    
    if args.opts == 'adamw':
        optimizer = optim.AdamW(params=model.parameters(), lr = args.learning_rate, 
                                weight_decay=0.1)
    elif args.opts == 'adam':
        optimizer = optim.Adam(params=model.parameters(), lr = args.learning_rate, 
                               weight_decay=0.1)
    elif args.opts == 'sgd':
        optimizer = optim.SGD(params=model.parameters(), lr = args.learning_rate,
                              momentum=0.9, weight_decay=0.1)
    
    if args.lrdecay:
        if args.lr_schedule == "warmup_cosine":
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

        elif args.lr_schedule == "poly":

            def lambdas(epoch):
                return (1 - float(epoch) / float(args.epochs)) ** 0.9

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambdas)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = InpaintingDataset(os.path.join(args.root_dir, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    valid_dataset = InpaintingDataset(os.path.join(args.root_dir, 'val'),transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"Length of train loader {len(train_loader)} and validation loader {len(valid_loader)}")
    global_step = 0
    best_val = 1e8
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    while global_step < args.num_steps:
        global_step, loss, best_val, val_loss_list = train(args, global_step, train_loader, best_val, scaler)
    np.save('/home/hyukiggle/Documents/workspace/pretrain/reconstruction/results/train_loss.npy', loss)  
    np.save('/home/hyukiggle/Documents/workspace/pretrain/reconstruction/results/val_loss.npy', val_loss_list)  
    
    checkpoint = {"step": args.num_steps, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_ckp(checkpoint, logdir+"/model_final_epoch.pt")
    
if __name__ == "__main__":
    main()