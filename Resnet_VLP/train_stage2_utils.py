from omegaconf import OmegaConf
from torchinfo import summary
import torch
import os 
from torch.optim import AdamW
from signdata import SignTransDataset
from transformers import MBart50Tokenizer
from torch.utils.data import DataLoader
import json 
from pathlib import Path
import time 
from collections import defaultdict

from tqdm import tqdm 

import glob
import sacrebleu
from logger import setup_logger
from sacrebleu.metrics import BLEU 
import numpy as np 
from lr_schedulers import get_scheduler
from torch.distributed import broadcast
from torch.nn.utils.rnn import pad_sequence
from definition import *
import torch.nn as nn
import torch.nn.functional as F
from SLT_CLIP import * 
import sys 
from definition import * 
import math 


def get_config():
    """Reads configs from a yaml file and terminal."""
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


class AverageMeter(object):
    """Computes and stores the average and current value.
    
    This class is borrowed from
    https://github.com/pytorch/examples/blob/main/imagenet/main.py#L423
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def create_optimizer(config, logger, model):
    """Creates optimizer for Sign2Text model."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer = AdamW(model.parameters(),
        lr=config.optimizer.params.learning_rate,  # Default learning rate, not used as we set per group
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2)
    )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")



    return optimizer


def create_signloader(config, logger,accelerator, tokenizer, device =  None): 
    if device is None: 
        device = accelerator.device
    batch_size = config.training.per_gpu_batch_size 
    logger.info(f"Creating Signloaders. Batch_size = {batch_size}")

    train_dataset = SignTransDataset(tokenizer, config,  'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=config.dataset.params.num_workers, collate_fn=train_dataset.collate_fn,
                                   generator=torch.Generator(device=device))
    train_dataloader = accelerator.prepare(train_dataloader)
    print("train dataloader done!")

    dev_dataset = SignTransDataset(tokenizer, config,  'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=config.dataset.params.num_workers, collate_fn=dev_dataset.collate_fn, 
                                 generator=torch.Generator(device=device) )
    dev_dataloader = accelerator.prepare(dev_dataloader)
    print("dev dataloader done!")

    test_dataset = SignTransDataset(tokenizer, config, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=config.dataset.params.num_workers, collate_fn=test_dataset.collate_fn, 
                                  generator=torch.Generator(device=device))
    test_dataloader = accelerator.prepare(test_dataloader)
    print("train dataloader done!")

    return train_dataloader, dev_dataloader, test_dataloader



def auto_resume(config, logger, accelerator, 
             strict=True):
    """Auto resuming the training."""
    global_step = 0
    first_epoch = 0
    # If resuming training.
    if config.experiment.resume:            
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator,
                logger=logger,
                strict=strict
            )
        
    return global_step, first_epoch


def save_checkpoint(model,  output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
   
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        logger.info(f"SLT saved state to {save_path}")

        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        
    accelerator.save_state(save_path)
    return save_path

def load_checkpoint(checkpoint_path: Path, accelerator, logger, strict=True):
    '''need to check where will the metadata.json being printed'''

    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path, strict=strict)
    
    with open(checkpoint_path / "metadata.json", "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step

def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


def translate_images(model, src, tgt, accelerator, config, global_step, output_dir, logger, tokenizer):
    logger.info("Translating images...")
    model = accelerator.unwrap_model(model)

    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    ## Generate some examples
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            src,
            decoder_start_token_id=tokenizer.lang_code_to_id[config.dataset.lang],
            num_beams=4,
            max_new_tokens=150
        )

        # Decode the generated token IDs
        generated_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
        tgt_batch  = tokenizer.batch_decode(tgt['input_ids'], skip_special_tokens=True)

        # Log translations locally to text files
        root = Path(output_dir) / "train_translations"
        os.makedirs(root, exist_ok=True)
        for i, (name,gen,  pred,gt) in enumerate(zip(src['name_batch'],generated, generated_batch, tgt_batch)):
            filename = f"{global_step:08}_t-{i:03}.txt"
            path = os.path.join(root, filename)
            
            # Save each translation as a separate text file
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Sample {i + 1}:\n")
                f.write(f"Name: {name}\n")
                f.write(f"tokens : {gen}\n")
                f.write(f"Ground Truth: {gt}\n")
                f.write(f"Prediction  : {pred}\n")
        
        print(f"Translations saved locally in {root}")






def create_scheduler(config, logger, accelerator, optimizer, len_data):
    """Creates learning rate scheduler for the optimizer."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.num_epochs*len_data * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
 
    return lr_scheduler



def train_one_epoch(config, accelerator, model, tokenizer,
                    train_dataloader, dev_dataloader, optimizer,
                    logger ,  scheduler , global_step, early_stop
                    ):
    
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    total_loss = 0
    transformer_logs = defaultdict(float)


    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)

    for step, (src_input, tgt_input, masked_tgt_input) in enumerate(tqdm(train_dataloader, desc=f"Training!")):
        model.train()
        # print(f"batch len: {len(batch)}")
        # print("batch shape: " ,batch.shape)
        
     
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model, loss_fct]):
            out_logits = model(src_input, tgt_input)
            label = tgt_input['input_ids'].reshape(-1)
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            loss = loss_fct(logits, label.to(accelerator.device, non_blocking=True))

            optimizer.zero_grad()
            accelerator.backward(loss)
            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            optimizer.step()
            scheduler.step()

            if (
                accelerator.sync_gradients
                and (global_step + 1) % (config.experiment.log_grad_norm_every * len(train_dataloader)/config.training.gradient_accumulation_steps) == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)
        

            loss_value = loss.item()
            if not math.isfinite(loss_value ):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            total_loss += loss.item()
            transformer_logs = {}
            transformer_logs ['train current loss'] = loss.item()
            transformer_logs ['train total_loss'] = total_loss


        
        
        if accelerator.sync_gradients:
            

            if (global_step + 1) % int(config.experiment.log_every * len(train_dataloader)/config.training.gradient_accumulation_steps)  == 0:
                
                lr = scheduler.get_last_lr()[0]

                logger.info(
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {transformer_logs['train total_loss']:0.4f} "
                    f"Recon Loss: {transformer_logs['train current loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "time/data_time": data_time_meter.val,
                    "time/batch_time": batch_time_meter.val,
                }
                logs.update(transformer_logs)

                accelerator.log(logs, step=global_step + 1)

                # Reset batch / data time meters per log window.
                batch_time_meter.reset()
                data_time_meter.reset()


            ## Generate some examples
            if (global_step + 1) % int(config.experiment.translate_every * len(train_dataloader)/config.training.gradient_accumulation_steps)== 0 and accelerator.is_main_process:

                # Save a batch of translated images to check by reading
                translate_images(model=model, 
                                 src=src_input, 
                                 tgt=tgt_input, 
                                 accelerator=accelerator,
                                 global_step= global_step, 
                                 config=config, output_dir=config.experiment.output_dir, 
                                 logger = logger, 
                                 tokenizer = tokenizer)


            # Evaluate reconstruction.
            if dev_dataloader is not None and (global_step + 1) % int(config.experiment.eval_every* len(train_dataloader)/config.training.gradient_accumulation_steps) == 0:
                logger.info(f"Computing metrics on the validation set.")
                
                     
                # Eval for non-EMA.
                total_val_loss = eval_SLT(
                    model=model,
                    dev_dataloader=dev_dataloader,
                    accelerator=accelerator,
                    criterion=loss_fct
                )
                        
                accelerator.wait_for_everyone()
                

                # gather all val losses to synchronise across all processes
                total_val_loss = accelerator.gather(total_val_loss)
                total_val_loss = total_val_loss.mean().item()
                if accelerator.is_main_process:
                    should_save = early_stop(total_val_loss)
                else: should_save = False

                should_save = torch.tensor([int(should_save)], dtype=torch.int, device=accelerator.device)
                if torch.cuda.device_count() > 1:
                    broadcast(should_save, src=0)
                if bool(should_save.item()): # save only if lower validation loss and this function will return True 
                    save_path = save_checkpoint(model=model,
                                                output_dir= config.experiment.output_dir,
                                                accelerator= accelerator,
                                                global_step= global_step + 1,
                                                logger=logger)
            
                    # Wait for everyone to save their checkpoint.
                    accelerator.wait_for_everyone()

            global_step += 1

    return global_step, early_stop


class EarlyStopping:
    '''
    This class helps to keep track of the validation losses so far
    This class will also stop the training if the validation loss did not decrease at all for the past 10 epochs
    This will prevent overfitting to training data and allow easy initialization of the best weights for the model
    '''
    def __init__(self, patience=10, verbose=True ):
        """
        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
                            Here it means how many previous epochs to consider.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.val_loss_min = np.Inf
        self.val_loss_history = []
        self.early_stop = False
        self.counter = 0 

    def __call__(self, val_loss):
        """
        Call method to update the early stopping status based on the validation loss.

        Args:
            val_loss (float): Current epoch's validation loss.
        """
        self.val_loss_history.append(val_loss)
        print (f"Validation loss: {val_loss}")
        # Ensure we only keep the last 'patience' number of validation losses
        if len(self.val_loss_history) > self.patience:
            self.val_loss_history.pop(0)
        
        # Check if the current validation loss is higher than the last 'patience' number of validation losses
        if len(self.val_loss_history) == self.patience and all(val_loss > loss for loss in self.val_loss_history[:-1]):
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. Validation loss {val_loss} is higher than the last {self.patience} validation losses: {self.val_loss_history[:-1]}")
        
        save_boo = val_loss < self.val_loss_min # true if new validation loss is smaller (to be returned)

        if self.verbose and save_boo:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
            self.val_loss_min = val_loss # updating the minimum validation loss 
        
        return save_boo
    
'''FOR NOW DONT NEED TO SAVE'''
# def save_predictions_and_references(pred, ref,names,  output_dir, filename="predictions.txt"):
#     """
#     Save predictions and references to a text file for later inspection.
    
#     Args:
#         pred (list): List of predicted sentences.
#         ref (list): List of reference sentences.
#         output_dir (str): Directory where the file will be saved.
#         filename (str): Name of the file to save the predictions and references.
#     """
#     # Ensure the output directory exists
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
    
#     output_file = Path(output_dir) / filename
    
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("Names\tPrediction\tReference\n")  # Header row (optional)
#         for n ,p, r in zip(names, pred, ref):
#             f.write(f"{n}\t{p}\t{r}\n")  # Tab-separated values
    
#     print(f"Predictions and references saved to {output_file}")



def create_SLT(config, logger, accelerator, model_type="SLT"):
    """Creates Sign2Text model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "SLT":
        model = gloss_free_model(config)

    if config.experiment.init_weight:
        # If loading a pretrained weight
        # Create special way to load it

        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")
        


    return model


def eval_SLT(model,dev_dataloader,accelerator, criterion): 
    '''
    translate images in the dev_loader 
    Calculate metrics using BLEU
    Output BLEU and Rouge values
    '''
    
    model.eval()
    local_model = accelerator.unwrap_model(model)
    total_val_loss = 0 
    with torch.no_grad(): 
        for i, (src,tgt, masked_tgt) in enumerate(tqdm(dev_dataloader, desc = f"Validation!")): 
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
            out_logits = local_model(src, tgt)
            label = tgt['input_ids'].reshape(-1)
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            loss = criterion(logits, label.to(accelerator.device, non_blocking=True))
            total_val_loss += loss.item()
    model.train()
    total_val_loss = torch.tensor(total_val_loss, device=accelerator.device)
    return total_val_loss
