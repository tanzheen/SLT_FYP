from omegaconf import OmegaConf
import torch
import os 
from torch.utils.data import DataLoader
import json 
from pathlib import Path
import time 
from collections import defaultdict
import csv 
from tqdm import tqdm 

import glob
import sacrebleu
from sacrebleu.metrics import BLEU 
import numpy as np 

from torch.distributed import broadcast
import torch.nn as nn
import torch.nn.functional as F
import sys 
import math 
from torch.optim import SGD
from data.SignVideoData import SignVideoDataset
from Combined_model.SpaEMo_model import * 
import time
import math
import sys
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
import torch.distributed as dist
import utils as utils 
'''
Might not need to have create scheduler and optimizer in this file
'''
def preprocess_cli_args():
    """
    Preprocess CLI arguments to strip `--` prefixes and make them OmegaConf compatible.
    Returns:
        dict: Processed CLI arguments.
    """
    args = sys.argv[1:]
    processed_args = {}
    for arg in args:
        if arg.startswith("--"):
            key, value = arg[2:].split("=", 1)
            processed_args[key] = value
        else:
            raise ValueError(f"Invalid argument format: {arg}")
    return processed_args

def get_config():
    """
    Load and preprocess the configuration.
    Returns:
        OmegaConf: Configuration object.
    """
    cli_args = preprocess_cli_args()
    cli_conf = OmegaConf.create(cli_args)

    # Ensure 'config' key exists
    if "config" not in cli_conf:
        raise ValueError("The '--config' argument is missing. Please provide a valid YAML configuration file.")

    yaml_conf = OmegaConf.load(cli_conf.config)
    config = OmegaConf.merge(yaml_conf, cli_conf)
    return config

# def get_config():
#     """Reads configs from a yaml file and terminal."""
#     cli_conf = OmegaConf.from_cli()
#     print("CLI Arguments:", cli_conf)
#     yaml_conf = OmegaConf.load(cli_conf.config)
#     conf = OmegaConf.merge(yaml_conf, cli_conf)

#     return conf


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


def create_dataloader(config, logger, tokenizer, device =  None): 

    batch_size = config.training.per_gpu_batch_size 
    logger.info(f"Creating Signloaders. Batch_size = {batch_size}")




    train_dataset = SignVideoDataset(tokenizer, config,  'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=config.dataset.params.num_workers,
                                  collate_fn=train_dataset.collate_tokens if config.training.token_usage else train_dataset.collate_fn,
                                  generator=torch.Generator(device=device))
    #train_dataloader = accelerator.prepare(train_dataloader)
    print("train dataloader done!")

    dev_dataset = SignVideoDataset(tokenizer, config,  'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=config.dataset.params.num_workers,
                                collate_fn=dev_dataset.collate_tokens if config.training.token_usage else dev_dataset.collate_fn, 
                                generator=torch.Generator(device=device) )
    #dev_dataloader = accelerator.prepare(dev_dataloader)
    print("dev dataloader done!")

    test_dataset = SignVideoDataset(tokenizer, config, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=config.dataset.params.num_workers, 
                                  collate_fn=test_dataset.collate_tokens if config.training.token_usage else test_dataset.collate_fn, 
                                  generator=torch.Generator(device=device))
    #test_dataloader = accelerator.prepare(test_dataloader)
    print("train dataloader done!")

    return train_dataloader, dev_dataloader, test_dataloader



# def auto_resume(config, logger,  
#              strict=True):
#     """Auto resuming the training."""
#     global_step = 0
#     first_epoch = 0
#     # If resuming training.
#     if config.experiment.resume:            
#         accelerator.wait_for_everyone()
#         local_ckpt_list = list(glob.glob(os.path.join(
#             config.experiment.output_dir, "checkpoint*")))
#         logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
#         if len(local_ckpt_list) >= 1:
#             if len(local_ckpt_list) > 1:
#                 fn = lambda x: int(x.split('/')[-1].split('-')[-1])
#                 checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
#             else:
#                 checkpoint_paths = local_ckpt_list
#             global_step = load_checkpoint(
#                 Path(checkpoint_paths[0]),
#                 accelerator,
#                 logger=logger,
#                 strict=strict
#             )
        
#     return global_step, first_epoch

def auto_resume(config, logger, model=None, optimizer=None, scheduler=None, strict=True):
    """
    Auto resuming the training.
    Need to revise this to ensure that without accelerator, the model still works
    """
    global_step = 0
    first_epoch = 0

    # If resuming training.
    if config.experiment.resume:
        # Ensure synchronization if using distributed training.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.output_dir, "checkpoint*")))
        logger.info(f"All globbed checkpoints are: {local_ckpt_list}")
        
        if len(local_ckpt_list) >= 1:
            # Sort checkpoint paths based on global step in filename.
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:
                checkpoint_paths = local_ckpt_list
            
            # Load the latest checkpoint.
            checkpoint_path = Path(checkpoint_paths[0])
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Example of how you might restore state from a checkpoint.
            # Ensure you adapt this part to match your training setup.
            global_step = checkpoint.get('global_step', 0)
            first_epoch = checkpoint.get('epoch', 0)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return global_step, first_epoch


# def save_checkpoint(model,  output_dir,  global_step, logger) -> Path:
#     save_path = Path(output_dir) / f"checkpoint-{global_step}"

#     state_dict = accelerator.get_state_dict(model)
   
#     if accelerator.is_main_process:
#         unwrapped_model = accelerator.unwrap_model(model)
#         unwrapped_model.save_pretrained_weight(
#             save_path / "unwrapped_model",
#             save_function=accelerator.save,
#             state_dict=state_dict,
#         )
#         logger.info(f"SLT saved state to {save_path}")

#         json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        
#     accelerator.save_state(save_path)
#     return save_path

def save_checkpoint(model, output_dir, global_step, logger, optimizer=None, scheduler=None):
    """Save checkpoint for model, optimizer, and scheduler states."""
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    save_path.mkdir(parents=True, exist_ok=True)

    # Save model state dict
    model_save_path = save_path / "model_state.pth"
    torch.save(model.state_dict(), model_save_path)
    logger.info(f"Model state saved to {model_save_path}")

    # Save optimizer state dict (if provided)
    if optimizer is not None:
        optimizer_save_path = save_path / "optimizer_state.pth"
        torch.save(optimizer.state_dict(), optimizer_save_path)
        logger.info(f"Optimizer state saved to {optimizer_save_path}")

    # Save scheduler state dict (if provided)
    if scheduler is not None:
        scheduler_save_path = save_path / "scheduler_state.pth"
        torch.save(scheduler.state_dict(), scheduler_save_path)
        logger.info(f"Scheduler state saved to {scheduler_save_path}")

    # Save metadata (e.g., global step)
    metadata_path = save_path / "metadata.json"
    with metadata_path.open("w") as metadata_file:
        json.dump({"global_step": global_step}, metadata_file)
    logger.info(f"Metadata saved to {metadata_path}")

    return save_path

# def load_checkpoint(checkpoint_path: Path,  logger, strict=True):
#     '''need to check where will the metadata.json being printed'''

#     logger.info(f"Load checkpoint from {checkpoint_path}")

#     accelerator.load_state(checkpoint_path, strict=strict)
    
#     with open(checkpoint_path / "metadata.json", "r") as f:
#         global_step = int(json.load(f)["global_step"])

#     logger.info(f"Resuming at global_step {global_step}")
#     return global_step

def load_checkpoint(checkpoint_path: Path, model, logger, optimizer=None, scheduler=None, strict=True):
    """
    Load the model, optimizer, and scheduler states from a checkpoint.

    Args:
        checkpoint_path (Path): Path to the checkpoint directory.
        model (torch.nn.Module): Model to load the state into.
        logger: Logger for logging messages.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        scheduler (torch.optim.lr_scheduler, optional): Scheduler to load state into.
        strict (bool): Whether to strictly enforce that the keys in the state dict match the model.
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    # Load model state
    model_state_path = checkpoint_path / "model_state.pth"
    model.load_state_dict(torch.load(model_state_path, map_location="cpu"), strict=strict)
    logger.info(f"Model state loaded from {model_state_path}")

    # Load optimizer state (if provided)
    if optimizer is not None:
        optimizer_state_path = checkpoint_path / "optimizer_state.pth"
        optimizer.load_state_dict(torch.load(optimizer_state_path, map_location="cpu"))
        logger.info(f"Optimizer state loaded from {optimizer_state_path}")

    # Load scheduler state (if provided)
    if scheduler is not None:
        scheduler_state_path = checkpoint_path / "scheduler_state.pth"
        scheduler.load_state_dict(torch.load(scheduler_state_path, map_location="cpu"))
        logger.info(f"Scheduler state loaded from {scheduler_state_path}")

    # Load metadata (e.g., global step)
    metadata_path = checkpoint_path / "metadata.json"
    with open(metadata_path, "r") as f:
        global_step = int(json.load(f)["global_step"])

    logger.info(f"Resuming at global_step {global_step}")
    return global_step

# def log_grad_norm(model, accelerator, global_step):
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             grads = param.grad.detach().data
#             grad_norm = (grads.norm(p=2) / grads.numel()).item()
#             accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)

def log_grad_norm(model, logger, global_step):
    """
    Log gradient norms for each parameter in the model.

    Args:
        model (torch.nn.Module): The model whose gradients to log.
        logger: Logger for logging gradient information.
        global_step (int): The current global training step.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach()
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            logger.info(f"Step {global_step} | Grad norm for {name}: {grad_norm}")


# def translate_images(model, src, tgt,  config, global_step, output_dir, logger, tokenizer):
#     logger.info("Translating images...")
#     model = accelerator.unwrap_model(model)

#     dtype = torch.float32
#     if accelerator.mixed_precision == "fp16":
#         dtype = torch.float16
#     elif accelerator.mixed_precision == "bf16":
#         dtype = torch.bfloat16does 

#     ## Generate some examples
#     model.eval()
#     with torch.no_grad():
#         generated = model.generate(
#             src,
#             num_beams=4,
#             max_length = 150 
#         )

#         # Decode the generated token IDs
#         generated_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
#         tgt_batch  = tokenizer.batch_decode(tgt['input_ids'], skip_special_tokens=True)

#         # Log translations locally to text files
#         root = Path(output_dir) / "train_translations"
#         os.makedirs(root, exist_ok=True)
#         for i, (name, pred,gt) in enumerate(zip(src['name_batch'], generated_batch, tgt_batch)):
#             filename = f"{global_step:08}_t-{i:03}.txt"
#             path = os.path.join(root, filename)
            
#             # Save each translation as a separate text file
#             with open(path, "w", encoding="utf-8") as f:
#                 f.write(f"Sample {i + 1}:\n")
#                 f.write(f"Name        : {name}\n")
#                 f.write(f"Ground Truth: {gt}\n")
#                 f.write(f"Prediction  : {pred}\n")
        
#         print(f"Translations saved locally in {root}")


def translate_images(model, src, tgt, config, global_step, output_dir, logger, tokenizer, mixed_precision="fp32"):
    """
    Translate images using the model and log the results to files.

    Args:
        model (torch.nn.Module): The model to perform translation.
        src (dict): Source batch containing input data and metadata.
        tgt (dict): Target batch containing ground truth data.
        config: Configuration object with translation parameters.
        global_step (int): Current global step for naming files.
        output_dir (str): Directory to save translations.
        logger: Logger for logging information.
        tokenizer: Tokenizer to decode token IDs.
        mixed_precision (str): Precision to use (e.g., "fp32", "fp16", "bf16").
    """
    logger.info("Translating images...")

    # Set the model precision
    dtype = torch.float32
    if mixed_precision == "fp16":
        dtype = torch.float16
    elif mixed_precision == "bf16":
        dtype = torch.bfloat16

    model.eval()
    with torch.no_grad():
        # Generate translations
        generated = model.generate(
            src,
            num_beams=4,
            max_length=150
        )

        # Decode the generated token IDs
        generated_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
        tgt_batch = tokenizer.batch_decode(tgt['input_ids'], skip_special_tokens=True)

        # Log translations locally to text files
        root = Path(output_dir) / "train_translations"
        os.makedirs(root, exist_ok=True)
        for i, (name, pred, gt) in enumerate(zip(src['name_batch'], generated_batch, tgt_batch)):
            filename = f"{global_step:08}_t-{i:03}.txt"
            path = os.path.join(root, filename)

            # Save each translation as a separate text file
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"Sample {i + 1}:\n")
                f.write(f"Name        : {name}\n")
                f.write(f"Ground Truth: {gt}\n")
                f.write(f"Prediction  : {pred}\n")

        logger.info(f"Translations saved locally in {root}")

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
    

# def eval_SpaEMo(model,dev_dataloader): 
#     '''
#     translate sample images in the dev_loader 
#     '''
    
#     model.eval()
#     local_model = accelerator.unwrap_model(model)
#     total_val_loss = 0 
#     with torch.no_grad(): 
#         for i, (src,tgt) in enumerate(tqdm(dev_dataloader, desc = f"Validation!")): 

#             output = local_model(src, tgt)
#             loss = output.loss
#             total_val_loss += loss.item()
#     model.train()
#     total_val_loss = torch.tensor(total_val_loss, device=accelerator.device)
#     return total_val_loss

def eval_SpaEMo(model, dev_dataloader, device):
    """
    Translate sample images in the dev_dataloader and calculate validation loss.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dev_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to perform computation on.

    Returns:
        torch.Tensor: Total validation loss as a tensor.
    """
    model.eval()
    total_val_loss = 0

    # Ensure model is on the correct device
    model.to(device)
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(tqdm(dev_dataloader, desc="Validation")):
            # Move inputs and targets to the specified device

            # Forward pass
            output = model(src, tgt)
            loss = output.loss
            total_val_loss += loss.item()

    model.train()

    # Wrap total_val_loss in a tensor for consistency
    total_val_loss = torch.tensor(total_val_loss, device=device)
    return total_val_loss

def generate_bleu(model, dev_dataloader,  tokenizer, output_dir, phase): 
    model.eval() 
    name = [] 
    refs = [] 
    gen = [] 
    for step, (src, tgt) in enumerate(tqdm(dev_dataloader, desc=f"Generation!")):
        with torch.no_grad():
            generated = model.generate(
                src,
                num_beams=5,
                max_length = 60 
            )

            # Decode the generated token IDs
            generated_batch = tokenizer.batch_decode(generated, skip_special_tokens=True)
            tgt_batch  = tokenizer.batch_decode(tgt['input_ids'], skip_special_tokens=True)
            
            name.extend(src['name_batch'])
            refs.extend(tgt_batch)
            gen.extend(generated_batch)

    # Calculate BLEU score
    print (gen)
    print([refs])
    blue = sacrebleu.corpus_bleu(gen, [refs])
    print(f"BLEU score: {blue}")




    # Save lists as a CSV file
    if phase == "dev": 
        filepath = os.path.join(output_dir, "dev_bleu.csv")
    else: 
        filepath = os.path.join(output_dir, "test_bleu.csv")

    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file)
        # Write header
        writer.writerow(["Name", "Reference", "Generated"])
        # Write rows
        writer.writerows(zip(name, refs, gen))
    

    print(f"Translations saved locally in {filepath}")


def create_SpaEMo(config, logger):
    """Creates Sign2Text model and loss module."""
    logger.info("Creating model and loss module.")

    model = SpaEMo(config)

    if config.experiment.init_weight:
        # If loading a pretrained weight
        # Create special way to load it

        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")
        

    return model


# def train_one_epoch(config,  model, tokenizer,
#                     train_dataloader, dev_dataloader, optimizer,
#                     logger ,  scheduler , global_step, early_stop, current_epoch
#                     ):
    
#     batch_time_meter = AverageMeter()
#     data_time_meter = AverageMeter()
#     end = time.time()

#     total_loss = 0
#     transformer_logs = defaultdict(float)


#     for step, (src_input, tgt_input) in enumerate(tqdm(train_dataloader, desc=f"Training!")):
#         model.train()
#         # print(f"batch len: {len(batch)}")
#         # print("batch shape: " ,batch.shape)
        
     
#         data_time_meter.update(time.time() - end)

#         with accelerator.accumulate([model]):
            
#             # alignment starts first for a certain number of steps 
#             if config.training.vt_align and global_step < config.training.vt_steps:
#                 logits_per_text = model.vis_text_align(src_input, tgt_input)
#                 loss = clip_loss(logits_per_text)

#             # then the transformer model starts training
#             else: 
#                 outputs = model(src_input, tgt_input)
#                 loss = outputs.loss


#             optimizer.zero_grad()
#             accelerator.backward(loss)
#             if config.training.max_grad_norm is not None and accelerator.sync_gradients:
#                 accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
#             optimizer.step()
#             scheduler.step(current_epoch)

#             if (
#                 accelerator.sync_gradients
#                 and (global_step + 1) % (config.experiment.log_grad_norm_every * len(train_dataloader)/config.training.gradient_accumulation_steps) == 0
#                 and accelerator.is_main_process
#             ):
#                 log_grad_norm(model, accelerator, global_step + 1)
        

#             loss_value = loss.item()
#             if not math.isfinite(loss_value ):
#                 print("Loss is {}, stopping training".format(loss_value))
#                 sys.exit(1)

#             total_loss += loss.item()
#             transformer_logs = {}
#             transformer_logs ['train current loss'] = loss.item()
#             transformer_logs ['train total_loss'] = total_loss
        
        
#         if accelerator.sync_gradients:
            

#             if (global_step + 1) % int(config.experiment.log_every * len(train_dataloader)/config.training.gradient_accumulation_steps)  == 0:
                
              

#                 logger.info(
#                     f"Batch (t): {batch_time_meter.val:0.4f} "
#                     f"Step: {global_step + 1} "
#                     f"Total Loss: {transformer_logs['train total_loss']:0.4f} "
#                     f"Recon Loss: {transformer_logs['train current loss']:0.4f} "
#                 )
#                 logs = {
#                     "time/data_time": data_time_meter.val,
#                     "time/batch_time": batch_time_meter.val,
#                 }
#                 logs.update(transformer_logs)

#                 accelerator.log(logs, step=global_step + 1)

#                 # Reset batch / data time meters per log window.
#                 batch_time_meter.reset()
#                 data_time_meter.reset()


#             ## Generate some examples
#             if (global_step + 1) % int(config.experiment.translate_every * len(train_dataloader)/config.training.gradient_accumulation_steps)== 0 and accelerator.is_main_process:

#                 # Save a batch of translated images to check by reading
#                 translate_images(model=model, 
#                                  src=src_input, 
#                                  tgt=tgt_input, 
#                                  accelerator=accelerator,
#                                  global_step= global_step, 
#                                  config=config, output_dir=config.experiment.output_dir, 
#                                  logger = logger, 
#                                  tokenizer = tokenizer)


#             # Evaluate reconstruction.
#             if dev_dataloader is not None and \
#                 (global_step + 1) % int(config.experiment.eval_every* len(train_dataloader)/config.training.gradient_accumulation_steps) == 0:
                           
#                 if global_step < config.training.vt_steps: 
#                     global_step += 1
#                     continue # Skip evaluation if aligning vis-text is not over
                
#                 logger.info(f"Computing metrics on the validation set.")
                
                     
#                 # Eval for non-EMA.
#                 total_val_loss = eval_SpaEMo(
#                     model=model,
#                     dev_dataloader=dev_dataloader,
#                     accelerator=accelerator,
#                 )
                        
#                 accelerator.wait_for_everyone()
                

#                 # gather all val losses to synchronise across all processes
#                 total_val_loss = accelerator.gather(total_val_loss)
#                 total_val_loss = total_val_loss.mean().item()
#                 if accelerator.is_main_process:
#                     should_save = early_stop(total_val_loss)
#                 else: should_save = False

#                 should_save = torch.tensor([int(should_save)], dtype=torch.int, device=accelerator.device)
#                 if torch.cuda.device_count() > 1:
#                     broadcast(should_save, src=0)
#                 if bool(should_save.item()): # save only if lower validation loss and this function will return True 
#                     save_path = save_checkpoint(model=model,
#                                                 output_dir= config.experiment.output_dir,
#                                                 accelerator= accelerator,
#                                                 global_step= global_step + 1,
#                                                 logger=logger)
            
#                     # Wait for everyone to save their checkpoint.
#                     accelerator.wait_for_everyone()

#             global_step += 1

#     return global_step, early_stop


def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, mixup_fn=None, max_norm: float = 1.0,
                    set_training_mode=True, gradient_accumulation_steps=4):
    """
    Train the model for one epoch.

    Args:
        config: Configuration object.
        model (torch.nn.Module): The model to train.
        tokenizer: Tokenizer for input/output processing.
        train_dataloader: DataLoader for the training dataset.
        dev_dataloader: DataLoader for the validation dataset.
        optimizer: Optimizer for training.
        logger: Logger for logging progress.
        scheduler: Learning rate scheduler.
        global_step (int): The global training step.
        early_stop: Early stopping object/function.
        current_epoch (int): The current epoch.
        device: The device to use for training (e.g., "cuda" or "cpu").

    Returns:
        global_step (int): Updated global step.
        early_stop: Updated early stopping object.
    """
    
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10

    model.to(device)

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Calculate loss
        logits_per_text = model.vis_text_align(src_input, tgt_input)
        cliploss = clip_loss(logits_per_text)
        out_logits = model(src_input, tgt_input).logits
        label = tgt_input['input_ids'].reshape(-1)
        logits = out_logits.reshape(-1, out_logits.shape[-1])
        textloss = criterion(logits, label.to(device, non_blocking=True))   # Scale loss
        
        loss = textloss + 0.3 * cliploss
        loss.backward()

        # Perform optimizer step and zero gradients every `gradient_accumulation_steps` steps
        if (step + 1) % gradient_accumulation_steps == 0:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # Gradient clipping
            optimizer.step()
            optimizer.zero_grad()

        # Log metrics
        loss_value = loss.item()  # Scale back for logging
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if len(optimizer.param_groups) > 1:  # If there are multiple parameter groups
            metric_logger.update(lr_mbart=round(float(optimizer.param_groups[1]["lr"]), 8))

        if (step + 1) % 10 == 0 and args.visualize and utils.is_main_process():
            utils.visualization(model.module.visualize())
        
    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return global_step, early_stop

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

def evaluate(args, dev_dataloader, model, model_without_ddp, tokenizer, criterion,  config, UNK_IDX, SPECIAL_SYMBOLS, PAD_IDX, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []
 
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, 10, header)):

            out_logits = model(src_input, tgt_input).logits
            
            label = tgt_input['input_ids'].reshape(-1)
            total_loss = 0.0
            logits = out_logits.reshape(-1,out_logits.shape[-1])
            tgt_loss = criterion(logits, label.to(device))
            
            total_loss += tgt_loss

            metric_logger.update(loss=total_loss.item())
            
            output = model_without_ddp.generate(src_input, max_length=150, num_beams = 4,
                        )

            tgt_input['input_ids'] = tgt_input['input_ids'].to(device)
            for i in range(len(output)):
                tgt_pres.append(output[i,:])
                tgt_refs.append(tgt_input['input_ids'][i,:])
            
            if (step+1) % 10 == 0 and args.visualize and utils.is_main_process():
                utils.visualization(model_without_ddp.visualize())

    pad_tensor = torch.ones(200-len(tgt_pres[0])).to(device)
    tgt_pres[0] = torch.cat((tgt_pres[0],pad_tensor.long()),dim = 0)
    tgt_pres = pad_sequence(tgt_pres,batch_first=True,padding_value=PAD_IDX)

    pad_tensor = torch.ones(200-len(tgt_refs[0])).to(device)
    tgt_refs[0] = torch.cat((tgt_refs[0],pad_tensor.long()),dim = 0)
    tgt_refs = pad_sequence(tgt_refs,batch_first=True,padding_value=PAD_IDX)

    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)
    tgt_refs = tokenizer.batch_decode(tgt_refs, skip_special_tokens=True)

    bleu = BLEU()
    bleu_s = bleu.corpus_score(tgt_pres, [tgt_refs]).score
    # metrics_dict['belu4']=bleu_s

    metric_logger.meters['belu4'].update(bleu_s)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* BELU-4 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.belu4, losses=metric_logger.loss))
    
    if utils.is_main_process() and utils.get_world_size() == 1 and args.eval:
        with open(args.output_dir+'/tmp_pres.txt','w') as f:
            for i in range(len(tgt_pres)):
                f.write(tgt_pres[i]+'\n')
        with open(args.output_dir+'/tmp_refs.txt','w') as f:
            for i in range(len(tgt_refs)):
                f.write(tgt_refs[i]+'\n')
        print('\n'+'*'*80)
        # metrics_dict = compute_metrics(hypothesis=args.output_dir+'/tmp_pres.txt',
        #                    references=[args.output_dir+'/tmp_refs.txt'],no_skipthoughts=True,no_glove=True)
        print('*'*80)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}