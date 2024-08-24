
# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader
from timm.utils import NativeScaler
from torch.utils.data import DataLoader

# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig


# *basic
import os
import time
import shutil
import json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger

# *metric
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image
import argparse
from hpman.m import _
import hpargparse
from create_dataloaders import create_dataloaders

def train_model(config, args): 
    device = torch.device(args.device)
    train_dataloader, dev_dataloader, test_dataloader = create_dataloaders(config, args)
    print(f"Creating model: ")
    device = args.device

    ## SLRCLIP needs to be coded
    model = SLRCLIP(config = config).to(device)

	## Optionally load a pre-trained model for fine-tuning
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    model_without_ddp = model
    
    # There is no distributed training so we will comment this out first   
    # if args.distributed:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #     model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')
    
	 # Create the optimizer and learning rate scheduler
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    text_decoder = Text_Decoder(config).to(device)
    
    optimizer_td = AdamW(text_decoder.module.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98))

    lr_scheduler_td = scheduler.CosineAnnealingLR(
        optimizer=optimizer_td,
        eta_min=1e-8,
        T_max=args.epochs,
    )
    TD_train_dict = dict(
        optimizer=optimizer_td,
        lr_scheduler=lr_scheduler_td,
        text_decoder=text_decoder
    )

    criterion = utils.KLLoss()
    loss_scaler = NativeScaler()

    # Resume training from a checkpoint if provided
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    # Evaluate the model if evaluation mode is enabled
    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                             UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, criterion, config, args.start_epoch,
                              UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return  # break the function here so that it will not go into training
    
    # Training loop for multiple epochs
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        # Train the model for one epoch
        train_stats = train_one_epoch(
            args, model, criterion, train_dataloader, optimizer, device, epoch, config,
            PAD_IDX, loss_scaler, TD_train_dict
        )

        # Step the learning rate scheduler
        lr_scheduler.step(epoch)
        TD_train_dict['lr_scheduler'].step(epoch)

        # Save checkpoints during training
        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        # Evaluate the model on the validation set
        test_stats = evaluate(
            args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
            SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict
        )

        # Save the best model checkpoint based on validation loss
        if min_loss > test_stats["loss"]:
            min_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'text_decoder': TD_train_dict['text_decoder'].state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

        print(f"* DEV loss {test_stats['loss']:.3f} Min DEV loss {min_loss}")

