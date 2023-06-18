import random

from config import Params
from utils import train_iter, val_iter, UNet, Augs, SSLDataset
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional
import cv2
import albumentations as A
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from datetime import datetime

np.random.seed(Params.seed)
torch.manual_seed(Params.seed)
random.seed(Params.seed)
torch.set_deterministic(True)

if __name__ == '__main__':
    augs = Augs()
    train_dataset = SSLDataset(auto_mask_dir=Params.auto_mask_dir, img_root_dir=Params.img_root_dir, transforms=augs)
    train_loader = DataLoader(train_dataset, batch_size=Params.batch_size, shuffle=True)
    model = UNet(out_classes=1, unsup_dim=Params.unsup_emb).double().to(Params.device)
    optimizer = optim.AdamW(model.parameters(), lr=Params.lr, weight_decay=Params.weight_decay)
    lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=Params.scheduler_factor,
                                     patience=Params.scheduler_patience)
    ce_loss = nn.CosineEmbeddingLoss(margin=Params.ce_margin)
    segm_loss = nn.BCEWithLogitsLoss()
    best_metric = float(np.inf)
    train_params = {k: v for k, v in Params.__dict__.items() if not k.startswith('__')}
    Params.pred_save_dir.mkdir(exist_ok=True, parents=True)
    Params.ckpt_save_path.mkdir(exist_ok=True, parents=True)
    Params.log_dir.mkdir(exist_ok=True, parents=True)
    if Params.ckpt_load_path is not None:
        model.load_state_dict(torch.load(Params.ckpt_load_path)['state_dict'], strict=True)
        optimizer.load_state_dict(torch.load(Params.ckpt_load_path)['opt_dict'])
    for epoch in tqdm(range(1, Params.epochs + 1), desc='Training'):
        loss_dict = train_iter(model=model, train_loader=train_loader, optimizer=optimizer,
                               loss_sup=segm_loss, loss_unsup=ce_loss, device=Params.device,
                               print_interval=Params.loss_print_interval, img_save_dir=Params.pred_save_dir,
                               weights_save_dir=Params.ckpt_save_path, train_params=train_params,
                               log_dir=Params.log_dir, save_interval=Params.ckpt_save_interval)
        lr_scheduler.step(loss_dict['total_loss'], epoch=epoch)
        print(f'Segm loss: {loss_dict["segm_loss"]}, SSL loss: {loss_dict["ssl_loss"]}, '
              f'total: {loss_dict["total_loss"]}')
        if loss_dict["total_loss"] < best_metric:
            best_metric = loss_dict["total_loss"]
            save_dict = {
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
                'params': {k: v for k, v in Params.__dict__.items() if not k.startswith('__')},
                'curr_epoch': epoch,
                'LR': optimizer.param_groups[0]['lr'],
            }
            torch.save(save_dict, Params.ckpt_save_path / 'weights.pth')
