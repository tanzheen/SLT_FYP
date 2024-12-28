# *basic
import os
from scripts.train_SpaEMo_utils import * 
from pathlib import Path
import sys 
import torch.distributed

from scripts.logger import setup_logger
from timm.optim import create_optimizer_v2
from timm.scheduler import  create_scheduler, create_scheduler_v2
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm

# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# *transformers
from transformers import MBartForConditionalGeneration, MBartTokenizer,MBartConfig
from transformers.models.mbart.modeling_mbart import shift_tokens_right

import utils as utils

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import test as test
import wandb
import copy
from pathlib import Path
from typing import Iterable, Optional
import math, sys
from loguru import logger

from hpman.m import _
import hpargparse

# *metric
from sacrebleu.metrics import BLEU, CHRF, TER
# try:
#     from nlgeval import compute_metrics
# except:
#     print('Please install nlgeval package.')


# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler
import torch.distributed as dist
from definition import * 

def get_args_parser():
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    # * distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "sgd"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, metavar='WD',
                        help = 'weight decay (default: 1e-4)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # *Drop out params
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    
    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * visualization
    parser.add_argument('--visualize', action='store_true')

    return parser

def main(args, config):
    #utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    print(config)
    print(f"Creating dataset:")
    # Create dataloaders
    tokenizer = MBartTokenizer.from_pretrained(config['model']['tokenizer'], src_lang = 'de_DE', tgt_lang = 'de_DE')
    train_dataloader, dev_dataloader, test_dataloader = create_dataloader(config, logger, tokenizer)

    # Initialize model
    model = create_SpaEMo(config, logger).to(device)
    model_without_ddp = model
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f"Model has {n_parameters} parameters")
    # Create optimizer and scheduler
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler = scheduler.CosineAnnealingLR(
                optimizer=optimizer,  
                eta_min=1e-8,
                T_max=args.epochs,
            )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=0.2, num_classes=2172)

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
    output_dir = Path(args.output_dir)

    if args.resume:
        print('Resuming Model Parameters... ')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f} ")
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BLEU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler, mixup_fn)
        lr_scheduler.step(epoch)

        if args.output_dir and utils.is_main_process():
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)
        
        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")
        
        if max_accuracy < test_stats["belu4"]:
            max_accuracy = test_stats["belu4"]
            if args.output_dir and utils.is_main_process():
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            
        print(f'Max BELU-4: {max_accuracy:.2f}%')
        # if utils.is_main_process():
        #     wandb.log({'epoch':epoch+1,'training/train_loss':train_stats['loss'], 'dev/dev_loss':test_stats['loss'], 'dev/Bleu_4':test_stats['belu4'], 'dev/Best_Bleu_4': max_accuracy})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir+'/best_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        test_stats = evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['belu4']:.2f}")
        
        test_stats = evaluate(args, test_dataloader, model, model_without_ddp, tokenizer, criterion, config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"BELU-4 of the network on the {len(test_dataloader)} test videos: {test_stats['belu4']:.2f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    sys.path.append("..")
    print(torch.__version__)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    config = get_config()
    parser = argparse.ArgumentParser('Gloss-free Sign Language Translation script', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()
    
    # if utils.is_main_process():
    #     wandb.init(project='GF-SLT',config=config)
    #     wandb.run.name = args.output_dir.split('/')[-1]
    #     wandb.define_metric("epoch")
    #     wandb.define_metric("training/*", step_metric="epoch")
    #     wandb.define_metric("dev/*", step_metric="epoch")
    # Set default environment variables if not already set
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    # Initialize the distributed process group
 
    # Set the local device
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)

'''
python train_SpaEMo_script.py --config=configs/SpaEMo_P14_config.yaml --experiment.project="SpaEMo_P14" --experiment.name="SpaEMo_P14_run1" --experiment.output_dir="SpaEMo_P14_run1"
  
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network train_SpaEMo_script.py config=configs/SpaEMo_P14_config.yaml --experiment.project="SpaEMo_P14" --experiment.name="SpaEMo_P14_run1" --experiment.output_dir="SpaEMo_P14_run1"
'''










