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
from lr_schedulers import get_scheduler


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
    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    exclude = (lambda n, p: p.ndim < 2 or "ln" in n or "bias" in n or 'latent_tokens' in n 
               or 'mask_token' in n or 'embedding' in n or 'norm' in n or 'gamma' in n)
    include = lambda n, p: not exclude(n, p)
    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    optimizer = optimizer_cls(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": optimizer_config.weight_decay},
        ],
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2)
    )

    return optimizer


def create_signloader(config, logger,accelerator, tokenizer): 
    batch_size = config.training.per_gpu_batch_size 
    logger.info(f"Creating Signloaders. Batch_size = {batch_size}")

    train_dataset = SignTransDataset(tokenizer, config,  'train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                  num_workers=config.dataset.params.num_workers, collate_fn=train_dataset.collate_fn,
                                   generator=torch.Generator(device=accelerator.device) )
    train_dataloader = accelerator.prepare(train_dataloader)
    print("train dataloader done!")

    dev_dataset = SignTransDataset(tokenizer, config,  'dev')
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=config.dataset.params.num_workers, collate_fn=dev_dataset.collate_fn, 
                                 generator=torch.Generator(device=accelerator.device) )
    dev_dataloader = accelerator.prepare(dev_dataloader)
    print("dev dataloader done!")

    test_dataset = SignTransDataset(tokenizer, config, 'test')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=config.dataset.params.num_workers, collate_fn=test_dataset.collate_fn, 
                                  generator=torch.Generator(device=accelerator.device))
    test_dataloader = accelerator.prepare(test_dataloader)
    print("train dataloader done!")

    return train_dataloader, dev_dataloader, test_dataloader


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

def translate_images(model, images,tgt_labels,input_attn, tgt_attn, src_length ,config, accelerator, global_step, output_dir, logger, tokenizer): 
    logger.info("Translating images...")
    images = torch.clone(images)
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        output = model(images, tgt_labels,input_attn, tgt_attn, src_length)
        # Output logits (predictions before softmax) from the decoder
        # Get the predicted token IDs by taking the argmax of the logits along the vocabulary dimension
        logits = output['logits']
        probs = logits.softmax(dim=-1)
        values, predictions = torch.topk(probs,k=1, dim = -1)
        #print(f"output shape: {output.shape}")
        predictions = predictions.reshape(config.training.per_gpu_batch_size,-1).squeeze()
        with tokenizer.as_target_tokenizer():
            pred_translations  = tokenizer.batch_decode(predictions, skip_special_tokens = True)

        # Decode the predicted token IDs into sentences
        #print(f"target labels: {tgt_labels}")
        tgt_labels[tgt_labels== -100] = 0 
        #print(tgt_labels)
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




def eval_translation(model,dev_dataloader,accelerator, tokenizer, batch_size =4 ): 
    '''
    translate images in the dev_loader 
    Calculate metrics using BLEU
    Output BLEU and Rouge values
    '''

    model.eval()
    local_model = accelerator.unwrap_model(model)
    predictions = [] # this is just a list 
    references = [] # this will be a list of a list
    for i, (src,tgt) in enumerate(tqdm(dev_dataloader, desc = f"Validation!")): 
        batch = src['input_ids']
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
        tgt_input[tgt_attn== 0] = -100
        original_images = torch.clone(images)
        output = local_model(original_images, tgt_input, input_attn , tgt_attn, src_length)
        #must decode the output and tgt_input 

        logits = output['logits']
        probs = logits.softmax(dim=-1)
        values, prediction = torch.topk(probs,k=1, dim = -1)
        prediction = prediction.reshape(batch_size,-1).squeeze()
        with tokenizer.as_target_tokenizer():
            sentence = tokenizer.batch_decode(prediction, skip_special_tokens = True)

        tgt_input[tgt_input==-100] = 0 
        gt_translation = tokenizer.batch_decode(tgt_input, skip_special_tokens=True)
        
        predictions.extend(sentence)
        references.extend(gt_translation)


    bleu_score = sacrebleu.corpus_bleu(predictions, [references])
    print(f" BLEU scores: {bleu_score}")
    model.train()

    return bleu_score

def create_lr_scheduler(config, logger, accelerator, optimizer, len_data):
    """Creates learning rate scheduler for Sign2Text."""
    logger.info("Creating lr_schedulers.")
    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.num_epochs * len_data/config.training.gradient_accumulation_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
        base_lr=config.lr_scheduler.params.learning_rate,
        end_lr=config.lr_scheduler.params.end_lr,
    )
    return lr_scheduler

def train_one_epoch(config, logger, accelerator, model, ema_model, optimizer,lr_scheduler,
                     train_dataloader, dev_dataloader, test_dataloader, tokenizer, global_step): 
    
    batch_time_meter = AverageMeter()
    data_time_meter = AverageMeter()
    end = time.time()

    model.train()
    total_loss = 0
    transformer_logs = defaultdict(float)

    for i, (src, tgt) in enumerate(tqdm(train_dataloader, desc=f"Training!")):
        model.train()
        # print(f"batch len: {len(batch)}")
        # print("batch shape: " ,batch.shape)

        batch = src['input_ids']
        src_length = src['src_length_batch']
        tgt_attn = tgt.attention_mask
        tgt_input = tgt['input_ids']
        tgt_input[tgt_attn== 0] = -100
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
            loss = output.loss 
            total_loss += loss 
            # Gather the losses across all processes for logging.
            transformer_logs = {}
            transformer_logs ['train current loss'] = loss 
            transformer_logs ['train total_loss'] = total_loss # Total loss does not really matter until towards the end

            accelerator.backward(loss)

            if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

            # Log gradient norm before zeroing it.
            if (
                accelerator.sync_gradients
                and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                and accelerator.is_main_process
            ):
                log_grad_norm(model, accelerator, global_step + 1)

            optimizer.zero_grad(set_to_none=True)

        # not train_generator specifies we finish both a generator step and a discriminator step
        if accelerator.sync_gradients:
            if config.training.use_ema:
                ema_model.step(model.parameters())
            batch_time_meter.update(time.time() - end)
            end = time.time()

            if (global_step + 1) % int(config.experiment.log_every * len(train_dataloader))  == 0:
                samples_per_second_per_gpu = (
                    config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_meter.val
                )

                lr = lr_scheduler.get_last_lr()[0]
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
            
            # Save model checkpoint.
            if (global_step + 1) % int(config.experiment.save_every * len(train_dataloader)) == 0:
                save_path = save_checkpoint(
                    model, config.experiment.output_dir, accelerator, global_step + 1, logger=logger)
                # Wait for everyone to save their checkpoint.
                accelerator.wait_for_everyone()

            ## Generate some examples
            if (global_step + 1) % int(config.experiment.translate_every * len(train_dataloader))== 0 and accelerator.is_main_process:
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
            if dev_dataloader is not None and (global_step + 1) % int(config.experiment.eval_every* len(train_dataloader)) == 0:
                logger.info(f"Computing metrics on the validation set.")
                if config.training.get("use_ema", False):
                    ema_model.store(model.parameters())
                    ema_model.copy_to(model.parameters())
                    # Eval for EMA.
                    eval_scores = eval_translation(
                        model=model,
                        dev_dataloader=dev_dataloader,
                        accelerator=accelerator,
                        tokenizer=tokenizer
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
                    eval_scores = eval_translation(
                        model=model,
                        dev_dataloader=dev_dataloader,
                        accelerator=accelerator,
                        tokenizer=tokenizer 
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

            global_step += 1

    return global_step


