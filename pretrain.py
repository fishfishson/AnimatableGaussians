import os
import numpy as np
from argparse import ArgumentParser
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset.idr_dataset import IDRDataset
from network.avatar_fb import AvatarNet

from easyvolcap.utils.base_utils import *
from easyvolcap.utils.console_utils import *


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AvatarModel(pl.LightningModule):
    def __init__(self,
                 # model params
                 smpl_pos_map_dir: str,
                 cano_smplx_path: str,
                 use_random_bg_color: bool = True,
                 use_flame_mask: bool = False,
                 use_mano_mask: bool = False,
                 # optimizer params
                 lr: float = 5e-4,
                 ):
        super().__init__()
        self.use_random_bg_color = use_random_bg_color
        self.use_flame_mask = use_flame_mask
        self.use_mano_mask = use_mano_mask
        self.lr = lr

        self.net = AvatarNet(
            smpl_pos_map_dir=smpl_pos_map_dir,
            cano_smplx_path=cano_smplx_path,
            use_flame_mask=use_flame_mask,
            use_mano_mask=use_mano_mask,
        )
        self.loss_func = nn.L1Loss()
    
    def forward(self, batch):
        smpl_pos_map = batch['smpl_pos_map'][:, :3]
        delta_position = self.net.get_positions(smpl_pos_map, batch=batch, return_delta=True)
        delta_opacity, delta_scaling, delta_rotation = self.net.get_others(smpl_pos_map, batch=batch, return_delta=True)
        return delta_position, delta_opacity, delta_scaling, delta_rotation

    def training_step(self, batch, batch_idx):
        delta_position, delta_opacity, delta_scaling, delta_rotation = self.forward(batch)
        position_loss = self.loss_func(delta_position, torch.zeros_like(delta_position))
        opacity_loss = self.loss_func(delta_opacity, torch.zeros_like(delta_opacity))
        scaling_loss = self.loss_func(delta_scaling, torch.zeros_like(delta_scaling))
        rotation_loss = self.loss_func(delta_rotation, torch.zeros_like(delta_rotation))
        loss = position_loss + opacity_loss + scaling_loss + rotation_loss

        self.log('p_loss', position_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('o_loss', opacity_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('s_loss', scaling_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('r_loss', rotation_loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)
        self.log('loss', loss, on_step=True, prog_bar=True, logger=True, rank_zero_only=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        batch = dotdict(batch)
        if self.use_random_bg_color:
            batch.bg_color = torch.rand(3, dtype=torch.float32, device=self.device)
        else:
            batch.bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)
        output = self.net.render(batch)
        if self.local_rank == 0:
            frame_index = batch.frame_index[0].item()
            camera_index = batch.camera_index[0]
            name = f'{frame_index:06d}_{camera_index}'
            rgb = torch.cat([output.rgb, batch.rgb], dim=-1)
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
        return {
            "optimizer": optimizer,
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
        lr=opt['model']['lr'],
    )
    
    logger = TensorBoardLogger(opt['trainer']['log_dir'], name='pretrain')
    
    modelckpt = ModelCheckpoint(
        dirpath=opt['trainer']['save_dir'],
        save_last=True,
        verbose=True,
        every_n_train_steps=opt['trainer']['every_n_train_steps'],
        save_on_train_epoch_end=True,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=opt['trainer']['devices'],
        logger=logger,
        max_steps=opt['trainer']['max_steps'],
        num_sanity_val_steps=opt['trainer']['num_sanity_val_steps'],
        log_every_n_steps=opt['trainer']['log_every_n_steps'],
        val_check_interval=opt['trainer']['val_check_interval'],
        callbacks=[modelckpt],
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
