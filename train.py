import os
import numpy as np
from argparse import ArgumentParser
from torch.optim.optimizer import Optimizer
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset.idr_dataset import IDRDataset
from network.avatar_fb import AvatarNet
from network.lpips import LPIPS

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *


class NeuSScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer,
                 decay_iter, 
                 alpha=0.05, 
                 last_epoch=-1):
        self.alpha = alpha
        self.decay_iter = decay_iter
        super(NeuSScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        progress = min(self.last_epoch / self.decay_iter, 1.0)
        learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - self.alpha) + self.alpha
        lrs = [base_lr * learning_factor for base_lr in self.base_lrs]
        return lrs


class AvatarModel(pl.LightningModule):
    def __init__(self,
                 # model params
                 smpl_pos_map_dir: str,
                 cano_smplx_path: str,
                 use_random_bg_color: bool = True,
                 use_flame_mask: bool = False,
                 use_mano_mask: bool = False,
                #  pretrained_path: str = None,
                 # loss params
                 l1_loss_weight: float = 1.0,
                 lpips_loss_weight: float = 0.1,
                 offset_loss_weight: float = 0.005,
                 # optimizer params
                 lr: float = 5e-4,
                 alpha: float = 0.05,
                 decay_iter: int = None,
                 ):
        super().__init__()
        self.use_random_bg_color = use_random_bg_color
        self.use_flame_mask = use_flame_mask
        self.use_mano_mask = use_mano_mask

        self.l1_loss_weight = l1_loss_weight
        self.lpips_loss_weight = lpips_loss_weight
        self.offset_loss_weight = offset_loss_weight

        self.lr = lr
        self.alpha = alpha
        self.decay_iter = decay_iter

        self.net = AvatarNet(
            smpl_pos_map_dir=smpl_pos_map_dir,
            cano_smplx_path=cano_smplx_path,
            use_flame_mask=use_flame_mask,
            use_mano_mask=use_mano_mask,
        )
        # pretrain = torch.load(pretrained_path)['state_dict']
        # for k, v in pretrain.items():
        #     if 'net.' in k:
        #         pretrain[k.replace('net.', '')] = v.copy()
        #         del pretrain[k]
        # self.net.load_state_dict(pretrain)

        self.lpips = LPIPS(net='vgg')
        for p in self.lpips.parameters():
            p.requires_grad = False
    
    def forward(self, batch):
        output = self.net.render(batch)
        return output

    def training_step(self, batch, batch_idx):
        batch = dotdict(batch)
        if self.use_random_bg_color:
            batch.bg_color = torch.rand(3, dtype=torch.float32, device=self.device)
        else:
            batch.bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
        output = self.forward(batch)

        gt_rgb = batch['rgb'] * batch['msk'] + (1 - batch['msk']) * batch.bg_color.view(1, 3, 1, 1)
        gt_rgb = gt_rgb * (1. - batch['bd_msk']) + batch['bd_msk'] * batch.bg_color.view(1, 3, 1, 1)
        pred_rgb = output.rgb * (1. - batch['bd_msk']) + batch['bd_msk'] * batch.bg_color.view(1, 3, 1, 1)

        l1_loss = torch.abs(pred_rgb - gt_rgb).mean()
        # random_path = False if self.global_step < 300000 else True
        # patch_pred, patch_gt = self.crop_image(batch['msk'], 512, random_patch, pred_rgb, gt_rgb)
        lpips_loss = self.lpips(pred_rgb, gt_rgb, normalize=True).mean()
        offset_loss = torch.norm(output.offset, dim=-1).mean()
        loss = self.l1_loss_weight * l1_loss + self.lpips_loss_weight * lpips_loss + self.offset_loss_weight * offset_loss

        self.log('l1_loss', l1_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('lpips_loss', lpips_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('offset_loss', offset_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = dotdict(batch)
        output = self.net.render(batch)
        gt_rgb = batch.rgb * batch.msk
        gt_rgb = gt_rgb * (1. - batch.bd_msk)
        if self.local_rank == 0:
            frame_index = batch.frame_index[0].item()
            camera_index = batch.camera_index[0]
            name = f'{frame_index:06d}_{camera_index}'
            rgb = torch.cat([output.rgb, gt_rgb], dim=-1)
            grid = torchvision.utils.make_grid(rgb)
            self.logger.experiment.add_image(f'val/rgb_{name}', grid, self.global_step)
            acc = output.acc.expand(-1, 3, -1, -1)
            msk = torch.cat([acc, batch.msk], dim=-1)
            grid = torchvision.utils.make_grid(msk)
            self.logger.experiment.add_image(f'val/msk_{name}', grid, self.global_step)
            grid = torchvision.utils.make_grid(output.dpt, normalize=True)
            self.logger.experiment.add_image(f'val/dpt_{name}', grid, self.global_step)

    def configure_optimizers(self):
        optimizer = Adam(self.net.parameters(), lr=self.lr)
        scheduler = NeuSScheduler(optimizer, decay_iter=self.decay_iter, alpha=self.alpha)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


def main(opt):
    train_dataset = IDRDataset(
        **opt['train']['dataset']
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=opt['train']['dataloader']['batch_size'],
        shuffle=True,
        num_workers=opt['train']['dataloader']['num_workers'],
    )
    val_dataset = IDRDataset(
        **opt['val']['dataset']
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt['train']['dataloader']['batch_size'],
        shuffle=False,
        num_workers=opt['train']['dataloader']['num_workers'],
    )
    
    model = AvatarModel(
        smpl_pos_map_dir=opt['model']['smpl_pos_map_dir'],
        cano_smplx_path=opt['model']['cano_smplx_path'],
        use_flame_mask=opt['model'].get('use_flame_mask', False),
        use_mano_mask=opt['model'].get('use_mano_mask', False),
        l1_loss_weight=opt['model']['l1_loss_weight'],
        lpips_loss_weight=opt['model']['lpips_loss_weight'],
        offset_loss_weight=opt['model']['offset_loss_weight'],
        lr=opt['model']['lr'],
        decay_iter=opt['model']['decay_iter'],
    )
    model.load_state_dict(torch.load(opt['model']['pretrained_path'])['state_dict'], strict=False)
    
    logger = TensorBoardLogger(opt['trainer']['log_dir'], name='train')
    
    modelckpt = ModelCheckpoint(
        dirpath=opt['trainer']['save_dir'],
        save_last=True,
        verbose=True,
        every_n_train_steps=opt['trainer']['every_n_train_steps'],
        save_on_train_epoch_end=True,
    )
    lrmonitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        accelerator='gpu',
        strategy=opt['trainer']['strategy'],
        devices=opt['trainer']['devices'],
        logger=logger,
        max_steps=opt['trainer']['max_steps'],
        num_sanity_val_steps=opt['trainer']['num_sanity_val_steps'],
        log_every_n_steps=opt['trainer']['log_every_n_steps'],
        val_check_interval=opt['trainer']['val_check_interval'],
        callbacks=[modelckpt, lrmonitor],
    )
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    torch.manual_seed(31359)
    np.random.seed(31359)
    
    parser = ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str)
    args = parser.parse_args()

    opt = yaml.load(open(args.config_path, encoding='UTF-8'))
    main(opt)
