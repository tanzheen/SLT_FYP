from omegaconf import OmegaConf
from torchinfo import summary
from seq_model import SignModel
import torch
from ema_model import EMAModel
import os 
from torch.optim import AdamW
from signdata import SignTransDataset
from transformers import MBart50Tokenizer
from torch.utils.data import DataLoader
import json 
from pathlib import Path
import time 
from collections import defaultdict
import pprint
from tqdm import tqdm 
from accelerate import Accelerator
import glob
import sacrebleu
from logger import setup_logger
from sacrebleu.metrics import BLEU 
import numpy as np 
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from lr_schedulers import get_scheduler
from torch.distributed import broadcast
from torch.nn.utils.rnn import pad_sequence
from definition import *
import torch.nn as nn

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


def create_model(config, logger, accelerator, model_type="Sign2Text"):
    """Creates Sign2Text model and loss module."""
    logger.info("Creating model and loss module.")
    if model_type == "Sign2Text":
        model_cls = SignModel
    else: 
        raise ValueError(f"Unsupported model_type {model_type}")
    model = model_cls(config)

    if config.experiment.init_weight:
        # If loading a pretrained weight
        model_weight = torch.load(config.experiment.init_weight, map_location="cpu")
        msg = model.load_state_dict(model_weight, strict=False)
        logger.info(f"loading weight from {config.experiment.init_weight}, msg: {msg}")

    # Create the EMA model.
    ema_model = None
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999,
                            model_cls=model_cls, config=config)
        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"),
                                                  model_cls=model_cls, config=config)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    return model, ema_model



def create_optimizer(config, logger, model):
    """Creates optimizer for Sign2Text model."""
    logger.info("Creating optimizers.")
    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate

    # Condition to set a different learning rate when to train the MBart and LLM adaptor with different learning rates
    if optimizer_config.lr_diff:
        # set adapter lr to specified value from config file 
        lr_adapt = optimizer_config.adapt_lr 
    else: 
        # else just follow the same learning rate as LLM
        lr_adapt = learning_rate 

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")


    # Separate the parameters for MBart, LLMAdapter and the Titok visial encoder
    mbart_params = [p for n, p in model.Mbart.named_parameters() if p.requires_grad] #self.Mbart
    titok_params = [p for n, p in model.titok.named_parameters() if p.requires_grad]
    llm_adaptor_params = [p for n, p in model.adapter.named_parameters() if p.requires_grad] #self.adapter

    # Create optimizer with different learning rates and weight decay settings
    combined_params = mbart_params + titok_params
    optimizer = optimizer_cls(
        [
    # Combined MBart and TiTok parameters with their respective learning rate
            {"params": combined_params, "lr": learning_rate, "weight_decay": optimizer_config.weight_decay},
            # LLM Adapter parameters with its own learning rate
            {"params": llm_adaptor_params, "lr": lr_adapt, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=config.optimizer.params.learning_rate,  # Default learning rate, not used as we set per group
        betas=(config.optimizer.params.beta1, config.optimizer.params.beta2)
    )

    return optimizer


def create_signloader(config, logger,accelerator, tokenizer): 
    batch_size = config.training.per_gpu_batch_size 
    logger.info(f"Creating Signloaders. Batch_size = {batch_size}")

    train_dataset = SignTransDataset(tokenizer, config,  'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=config.dataset.params.num_workers, collate_fn=train_dataset.collate_fn,
                                   generator=torch.Generator(device=accelerator.device))
    train_dataloader = accelerator.prepare(train_dataloader)
    print("train dataloader done!")

    dev_dataset = SignTransDataset(tokenizer, config,  'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=config.dataset.params.num_workers, collate_fn=dev_dataset.collate_fn, 
                                 generator=torch.Generator(device=accelerator.device) )
    dev_dataloader = accelerator.prepare(dev_dataloader)
    print("dev dataloader done!")

    test_dataset = SignTransDataset(tokenizer, config, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=config.dataset.params.num_workers, collate_fn=test_dataset.collate_fn, 
                                  generator=torch.Generator(device=accelerator.device))
    test_dataloader = accelerator.prepare(test_dataloader)
    print("train dataloader done!")

    return train_dataloader, dev_dataloader, test_dataloader

def continue_from_frozen(config, logger, accelerator, ema_model, strict = True): 
    '''Continue training from a frozen model'''
    global_step = 0
    first_epoch = 0
    if config.experiment.previous_frozen: 
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(
            config.experiment.previous_frozen, "checkpoint*")))
        logger.info(f"Taking the weights from the frozen runs. All globbed checkpoints are: {local_ckpt_list}")
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
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = 1
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch



def auto_resume(config, logger, accelerator, ema_model, 
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
            if config.training.use_ema:
                ema_model.set_step(global_step)
            first_epoch = 1
        else:
            logger.info("Training from scratch.")
    return global_step, first_epoch


def save_checkpoint(model, output_dir, accelerator, global_step, logger) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"

    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained_weight(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

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

def translate_images(model, images,tgt_labels,input_attn,tgt_attn,  src_length ,config, accelerator, global_step, output_dir, logger, tokenizer): 
    logger.info("Translating images...")
    model = accelerator.unwrap_model(model)
    images = torch.clone(images)
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        output = model(src_input = images,tgt_input = tgt_labels, src_attn = input_attn, tgt_attn = tgt_attn, src_length = src_length)
        # Output logits (predictions before softmax) from the decoder
        # Get the predicted token IDs by taking the argmax of the logits along the vocabulary dimension
        logits = output['logits']
        probs = logits.softmax(dim=-1)
        values, pred = torch.topk(probs,k=1, dim = -1)
        predictions = pred.reshape(config.training.per_gpu_batch_size,-1).squeeze()
        

        # pad_tensor = torch.ones(200-len(predictions[0])).to(accelerator.device)
        # predictions[0] = torch.cat((predictions[0],pad_tensor.long()),dim = 0)
        # predictions = pad_sequence(predictions,batch_first=True,padding_value=PAD_IDX)

        
        pred_translations  = tokenizer.batch_decode(predictions, skip_special_tokens = True)

        gt_translations = tokenizer.batch_decode(tgt_labels, skip_special_tokens = True)
        #print(gt_translations)

    
    # Log translations using wandb or tensorboard, if enabled
    if config.training.enable_wandb:
        # Log translations to wandb (as text)
        accelerator.get_tracker("wandb").log(
            {f"Train Translations": pred_translations, 
             f"Train truth": gt_translations},
            step=global_step
        )
    else:
        # Log translations to tensorboard (you may need a different logging method for text)
        accelerator.get_tracker("tensorboard").log(
            {"Train Translations": pred_translations, 
             f"Train truth": gt_translations}, step=global_step
        )

    # Log translations locally to text files
    root = Path(output_dir) / "train_translations"
    os.makedirs(root, exist_ok=True)
    for i, (pred,gt) in enumerate(zip(pred_translations, gt_translations)):
        filename = f"{global_step:08}_t-{i:03}.txt"
        path = os.path.join(root, filename)
        
        # Save each translation as a separate text file
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Sample {i + 1}:\n")
            f.write(f"Ground Truth: {gt}\n")
            f.write(f"Prediction  : {pred}\n")
    
    print(f"Translations saved locally in {root}")


    return pred_translations



def eval_translation(model,dev_dataloader,accelerator, tokenizer , config ): 
    '''
    translate images in the dev_loader 
    Calculate metrics using BLEU
    Output BLEU and Rouge values
    '''
    loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    model.eval()
    local_model = accelerator.unwrap_model(model)
    predictions = [] # this is just a list 
    references = [] # this will be a list of a list
    name_lst = [] 
    total_val_loss = 0 
    with torch.no_grad(): 
        for i, (src,tgt) in enumerate(tqdm(dev_dataloader, desc = f"Validation!")): 
            batch = src['input_ids']
            names = src['name_batch']
            src_length = src['src_length_batch']
            tgt_attn = tgt.attention_mask
            
            images = batch.to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            tgt_input = tgt['input_ids'].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            input_attn = src['attention_mask'].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            tgt_attn = tgt_attn.to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            
            #tgt_input[tgt_attn== 0] = -100
            original_images = torch.clone(images)
            output = local_model(original_images, tgt_input, input_attn , tgt_attn, src_length) ## is the tgt_attn for the loss calculation?

        
            label = tgt_input.reshape(-1)
            logits = output.logits.reshape(-1,output.logits.shape[-1])
            #print(f"Logits shape: {logits.shape}, Label shape: {label.shape}")

    
            val_loss = loss_fct(logits, label)
            total_val_loss += val_loss

            # ## Should use generated output to calculate the bleu score
            generated = local_model.generate(src_input = original_images, src_attn = input_attn, 
                                       src_length= src_length, 
                                       max_new_tokens=150, num_beams=4, 
                                       decoder_start_token_id=tokenizer.lang_code_to_id[config.dataset.lang])
            tgt_input = tgt_input.to(accelerator.device)
            for i in range(len(generated)): 
                predictions.append(generated[i, :])
                references.append(tgt_input[i, :])

    pad_tensor = torch.ones(200-len(predictions[0])).to(accelerator.device)
    predictions[0] = torch.cat((predictions[0],pad_tensor.long()),dim = 0)
    predictions = pad_sequence(predictions,batch_first=True,padding_value=PAD_IDX)

    pad_tensor = torch.ones(200-len(references[0])).to(accelerator.device)
    references[0] = torch.cat((references[0],pad_tensor.long()),dim = 0)
    references = pad_sequence(references,batch_first=True,padding_value=PAD_IDX)

    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    references = tokenizer.batch_decode(references, skip_special_tokens=True)

    ## Compare the score 
    bleu = BLEU()
    bleu_score = bleu.corpus_score(predictions, [references])
    print(f" BLEU scores: {bleu_score}")
    model.train()

    return bleu_score, total_val_loss, predictions, references, name_lst



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


def train_one_epoch(config, logger, accelerator, model, ema_model, optimizer,scheduler,
                     train_dataloader, dev_dataloader, test_dataloader, tokenizer, global_step, early_stop): 
    
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()
    total_loss = 0
    transformer_logs = defaultdict(float)
    loss_fct = nn.CrossEntropyLoss(ignore_index=PAD_IDX ,label_smoothing=0.1)

    for i, (src, tgt) in enumerate(tqdm(train_dataloader, desc=f"Training!")):
        model.train()
        # print(f"batch len: {len(batch)}")
        # print("batch shape: " ,batch.shape)

        batch = src['input_ids']
        src_length = src['src_length_batch']
        tgt_attn = tgt.attention_mask
        tgt_input = tgt['input_ids']
        input_attn = src['attention_mask']
        
        images = batch.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        tgt_input = tgt['input_ids'].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        input_attn = input_attn.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        tgt_attn = tgt_attn.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
        
        data_time_meter.update(time.time() - end)

        with accelerator.accumulate([model]):
     
            output = model(images, tgt_input, input_attn , tgt_attn, src_length)
     
            label = tgt_input.reshape(-1)
            logits = output.logits.reshape(-1,output.logits.shape[-1])
            #print(f"Logits shape: {logits.shape}, Label shape: {label.shape}")
            loss = loss_fct(logits, label)
            
            
            optimizer.zero_grad(set_to_none=True)
    
            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            
            optimizer.step()
            scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % (config.experiment.log_grad_norm_every * len(train_dataloader)/config.training.gradient_accumulation_steps) == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)
            
            # Gather the losses across all processes for logging.
            # loss = accelerator.gather(loss)
            # loss = loss.mean().item()
            total_loss += loss 
            transformer_logs = {}
            transformer_logs ['train current loss'] = loss 
            transformer_logs ['train total_loss'] = total_loss # Total loss does not really matter until towards the end

            

        # not train_generator specifies we finish both a generator step and a discriminator step
        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % int(config.experiment.log_every * len(train_dataloader)/config.training.gradient_accumulation_steps)  == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )
                lr = scheduler.get_last_lr()[0]

                logger.info(
                    f"Data (t): {data_time_meter.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                    f"Batch (t): {batch_time_meter.val:0.4f} "
                    f"LR: {lr:0.6f} "
                    f"Step: {global_step + 1} "
                    f"Total Loss: {transformer_logs['train total_loss']:0.4f} "
                    f"Recon Loss: {transformer_logs['train current loss']:0.4f} "
                )
                logs = {
                    "lr": lr,
                    "lr/generator": lr,
                    "samples/sec/gpu": samples_per_second_per_gpu,
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
                # Store the model parameters temporarily and load the EMA parameters to perform inference.
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())

                # Save a batch of translated images to check by reading
                translate_images(
                    model=model,
                    images=images,
                    tgt_labels=tgt_input,
                    input_attn=input_attn, 
                    tgt_attn=tgt_attn,
                    src_length=src_length,
                    config=config,
                    accelerator=accelerator,
                    global_step=global_step + 1,
                    output_dir=config.experiment.output_dir,
                    logger=logger, 
                    tokenizer=tokenizer
                )


                if config.training.get("use_ema", False):
                    # Switch back to the original model parameters for training.
                    ema_model.restore(model.parameters())
            
            # Evaluate reconstruction.
            if dev_dataloader is not None and (global_step + 1) % int(config.experiment.eval_every* len(train_dataloader)/config.training.gradient_accumulation_steps) == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores, total_val_loss, dev_pred, dev_ref , name_lst  = eval_translation(
                        model=model,
                        dev_dataloader=dev_dataloader,
                        accelerator=accelerator,
                        tokenizer=tokenizer, 
                        config = config 
                    )

                    logger.info(
                        f"EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/BLEU': eval_scores}
                        accelerator.log(eval_log, step=global_step + 1)
                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())

                else:
                     
                    # Eval for non-EMA.
                    eval_scores, total_val_loss, dev_pred, dev_ref, name_lst = eval_translation(
                        model=model,
                        dev_dataloader=dev_dataloader,
                        accelerator=accelerator,
                        tokenizer=tokenizer , 
                        config = config 
                    )

                    logger.info(
                        f"Non-EMA EVALUATION "
                        f"Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {f'ema_eval/BLEU': eval_scores}
                        accelerator.log(eval_log, step=global_step + 1)
                        
                accelerator.wait_for_everyone()
                # save dev prediction and dev references in a txt file regardless if validation loss decreased
                save_predictions_and_references(pred = dev_pred,ref =  dev_ref,names=name_lst,
                                                output_dir= config.experiment.output_dir,
                                                filename= "dev_pred.txt")

                # gather all val losses to synchronise across all processes
                total_val_loss = accelerator.gather(total_val_loss)
                total_val_loss = total_val_loss.mean().item()
                if accelerator.is_main_process:
                    should_save = early_stop(total_val_loss)
                else: should_save = False
                should_save = torch.tensor([int(should_save)], dtype=torch.int, device=accelerator.device)
                broadcast(should_save, src=0)
                if bool(should_save.item()): # save only if lower validation loss and this function will return True 
                    save_path = save_checkpoint(model=model,output_dir= config.experiment.output_dir,accelerator= accelerator,global_step= global_step + 1,logger=logger)
            
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
    

def save_predictions_and_references(pred, ref,names,  output_dir, filename="predictions.txt"):
    """
    Save predictions and references to a text file for later inspection.
    
    Args:
        pred (list): List of predicted sentences.
        ref (list): List of reference sentences.
        output_dir (str): Directory where the file will be saved.
        filename (str): Name of the file to save the predictions and references.
    """
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_file = Path(output_dir) / filename
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("Names\tPrediction\tReference\n")  # Header row (optional)
        for n ,p, r in zip(names, pred, ref):
            f.write(f"{n}\t{p}\t{r}\n")  # Tab-separated values
    
    print(f"Predictions and references saved to {output_file}")