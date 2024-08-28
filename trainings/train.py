
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
from definition import * 
import sentencepiece
import transformers
# *transformers and models 
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig
from models import * 

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
import utils 
from prep_args import get_args_parser
print (torch.__version__)
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

# Contrastive loss 
from SignCL import SignCL
if torch.cuda.is_available():
    device = "cuda"
else: 
    device = "mps"
def train_model(args, config ):
    torch.cuda.empty_cache() 
    cl_criterion =  SignCL(max_distance=32.0, pos_samples=2, neg_samples=4)
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
    print(f'Model created! number of params: {n_parameters}M')
    
	 # Create the optimizer and learning rate scheduler
    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    text_decoder = Text_Decoder(config).to(device)
    
    optimizer_td = AdamW(text_decoder.parameters(), lr=1e-3, weight_decay=0, betas=(0.9, 0.98))

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
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(args.start_epoch, args.epochs):

        # Train the model for one epoch
        print(f"training epoch {epoch}")
        train_stats = train_one_epoch(
            args, model, criterion, cl_criterion, train_dataloader, optimizer, device, epoch, config,
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

        # Optionally log metrics to an external service like Weights & Biases
        if args.run:
            args.run.log({
                'epoch': epoch + 1,
                'training/train_loss': train_stats['loss'],
                'training/masked_lm_loss': train_stats['masked_lm_loss'],
                'dev/dev_loss': test_stats['loss'],
                'dev/min_loss': min_loss
            })

        # Log training statistics to a file
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    # Final evaluation after training completes
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        torch.distributed.barrier()  # Synchronize all processes before final evaluation
        checkpoint = torch.load(args.output_dir + '/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        dev_stats = evaluate(
            args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
            SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict
        )
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(
            args, test_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX,
            SPECIAL_SYMBOLS, PAD_IDX, device, TD_train_dict
        )
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Function to train the model for one epoch
def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss, cl_criterion, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, PAD_IDX, loss_scaler, TD_train_dict, max_norm: float = 0,
                    set_training_mode=True):
    
    # Set the model to training mode
    model = model.to(device)
    model.train(set_training_mode)

    # Initialize metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10  # Frequency of logging training status

    # Define the loss functions
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)
    #enumerate(tqdm(train_loader, desc=f'Training Epoch {epoch+1}/{num_epochs}', leave=False))
    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)):
        print(f"current batch: {step}")
        print (src_input['input_ids'].shape)

        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            # Calculate contrastive loss and margin adjustment
            margin = max(10, int((frames_feature.shape[1] // tgt_input['input_ids'].shape[1] + 1) * 2.3)) * 2
            num_negative = 30
            margin = min(margin, int((frames_feature.shape[1] - num_negative) / 2))  # Ensure margin for negative sampling
            cl_loss = cl_criterion(frames_feature, margin=margin)

            # Combine image-text losses and add contrastive loss
            ml_loss = (loss_imgs + loss_texts) / 2.
            total_loss = ml_loss + 0.01 * cl_loss
            print(f"total loss calculated: {total_loss}")

        # Backward pass and optimization step
        loss_scaler(ml_loss, optimizer)

        # Update text decoder parameters periodically
        if step % 5 == 0:
            TD_train_dict['optimizer'].zero_grad()
            with torch.cuda.amp.autocast():
                lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model.model_txt)
                masked_lm_loss = loss_fct(
                    lm_logits.view(-1, lm_logits.shape[-1]),
                    tgt_input['input_ids'].to(device).view(-1)
                ) * args.loss_lambda
            loss_scaler(masked_lm_loss, TD_train_dict['optimizer'])

        # Log and handle potential training issues
        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(cl_loss=cl_loss.item())
        metric_logger.update(masked_lm_loss=masked_lm_loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(td_lr=TD_train_dict['optimizer'].param_groups[0]["lr"])

        # Optionally visualize the training process
        if (step + 1) % 10 == 0 and utils.is_main_process():
            visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
            utils.visualization([visual_map, ])

    if args.run:
        args.run.log({
            'epoch': epoch + 1,
            'epoch/train_loss': loss_value,
            'epoch/masked_lm_loss': masked_lm_loss.item()
        })

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# Function to evaluate the model on the validation or test dataset
def evaluate(args, dev_dataloader, model, model_without_ddp, criterion, config, epoch, UNK_IDX, SPECIAL_SYMBOLS,
             PAD_IDX, device, TD_train_dict):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10  # Frequency of logging evaluation status

    # Define the loss functions
    loss_img = criterion
    loss_txt = criterion
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.2)

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for step, (src_input, tgt_input, masked_tgt_input) in enumerate(
                metric_logger.log_every(dev_dataloader, print_freq, header)):

            logits_per_image, logits_per_text, ground_truth, frames_feature = model(src_input, tgt_input)
            loss_imgs = loss_img(logits_per_image, ground_truth)
            loss_texts = loss_txt(logits_per_text, ground_truth)

            # Compute the text decoder's loss
            lm_logits = TD_train_dict['text_decoder'](tgt_input, masked_tgt_input, model_without_ddp.model_txt)
            masked_lm_loss = loss_fct(
                lm_logits.view(-1, lm_logits.shape[-1]), tgt_input['input_ids'].to(device).view(-1)
            )
            total_loss = (loss_imgs + loss_texts) / 2.

            metric_logger.update(loss=total_loss.item())
            metric_logger.update(masked_lm_loss=masked_lm_loss.item())

            # Optionally visualize the evaluation process
            if (step + 1) % 10 == 0 and utils.is_main_process():
                visual_map = torch.cat((logits_per_image.unsqueeze(0), logits_per_text.unsqueeze(0)))
                utils.visualization([visual_map, ])

    if args.run:
        args.run.log({'epoch': epoch + 1, 'epoch/dev_loss': total_loss.item()})

    metric_logger.synchronize_between_processes()
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            #run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run


if __name__ == '__main__':
    # before running, remember to set the cd correctly to the folder "trainings"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_model(args, config)
    # main_extract_features(args, config)
        

