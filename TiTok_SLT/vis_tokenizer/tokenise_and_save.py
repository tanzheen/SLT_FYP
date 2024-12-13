import torch
from pathlib import Path
from transformers import MBartTokenizer  # Assuming you're using a tokenizer like MBart
from torchvision import transforms  # Assuming frames are images
from accelerate.utils import set_seed
from accelerate import Accelerator
# Assuming the TiTok model is imported from your project
from modeling.titok import TiTok # Replace with the actual path to your model class
from utils.logger import setup_logger
from utils.train_utils import (
    get_config, create_pretrained_tokenizer, 
    create_model_and_loss_module,
    create_optimizer, create_lr_scheduler,  create_dataloader,
    create_evaluator, auto_resume, save_checkpoint, 
    train_one_epoch)
import os 
from omegaconf import OmegaConf
from tqdm import tqdm 
import sys 


def main(): 
    workspace = os.environ.get('WORKSPACE', '')
    if workspace:
        torch.hub.set_dir(workspace + "/models/hub")
    
    config = get_config()
    # Enable TF32 on Ampere GPUs.
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
     
    #torch.distributed.init_process_group(backend="nccl")  # Or "gloo" if you are using CPU
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    config.experiment.logging_dir = os.path.join(output_dir, "logs")

    # Whether logging to Wandb or Tensorboard.
    tracker = "tensorboard"
    if config.training.enable_wandb:
        tracker = "wandb"

    # Download the weights of the model


    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    logger = setup_logger(name="TiTok", log_level="INFO",
    output_file=f"{output_dir}/log{accelerator.process_index}.txt")

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    pretrained_tokenizer = create_pretrained_tokenizer(config,
                                                    logger,
                                                    accelerator)

    model, ema_model, loss_module = create_model_and_loss_module(
        config, logger, accelerator, model_type="titok")
    train_dataloader, eval_dataloader, test_dataloader = create_dataloader(config, logger, accelerator)
    # Check on dataloader 
    print(train_dataloader.dataset[0])
    
    # Set up evaluator.
    evaluator = create_evaluator(config, logger, accelerator)
    model= accelerator.prepare(
            model)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model,
        strict=True)
    model.eval()
    # Using the same weights, tokenize train set, dev set and test set
    with torch.no_grad():
        for i, (batch, name) in enumerate(tqdm(train_dataloader, desc=f"Tokenisation!")): 
            images = batch.to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            with accelerator.accumulate([model]): 
                encoded_tokens = model.encode(images)[1]['min_encoding_indices'].squeeze()
                # for loop to also save the encoded tokens with a modified name
                for token, fname in zip(encoded_tokens, name): 
                    emb_name = os.path.splitext(fname)[0]
                    emb_name = f"{emb_name}.pt" # change the filename from jpeg to .pt for saving later
                    # save token with the name  
                    # Save the token tensor as a .pt file
                    torch.save(token.cpu(), emb_name)
                    print(f"Saved tokens for {fname} to {emb_name}")

        # Using the same weights, tokenize train set, dev set and test set
        for i, (batch, name) in enumerate(tqdm(eval_dataloader, desc=f"Tokenisation!")): 
            images = batch.to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            with accelerator.accumulate([model]): 
                encoded_tokens = model.encode(images)[1]['min_encoding_indices'].squeeze()
                # for loop to also save the encoded tokens with a modified name
                for token, fname in zip(encoded_tokens, name): 
                    emb_name = os.path.splitext(fname)[0]
                    emb_name = f"{emb_name}.pt" # change the filename from jpeg to .pt for saving later
                    # save token with the name  
                    # Save the token tensor as a .pt file
                    torch.save(token.cpu(), emb_name)
                    print(f"Saved tokens for {fname} to {emb_name}")

        # Using the same weights, tokenize train set, dev set and test set
        for i, (batch, name) in enumerate(tqdm(test_dataloader, desc=f"Tokenisation!")): 
            images = batch.to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            with accelerator.accumulate([model]): 
                encoded_tokens = model.encode(images)[1]['min_encoding_indices'].squeeze()
                # for loop to also save the encoded tokens with a modified name
                for token, fname in zip(encoded_tokens, name): 
                    emb_name = os.path.splitext(fname)[0]
                    emb_name = f"{emb_name}.pt" # change the filename from jpeg to .pt for saving later
                    # save token with the name  
                    # Save the token tensor as a .pt file
                    torch.save(token.cpu(), emb_name)
                    print(f"Saved tokens for {fname} to {emb_name}")


if __name__ == "__main__":
 
    sys.path.append("..")
    print(torch.__version__)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        torch.set_default_device('cuda')
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
    else:
        torch.set_default_device('mps')
        print("CUDA is not available. Using MPS.")
    torch.cuda.empty_cache()
    main()

'''
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network tokenise_and_save.py config=configs/training/stage1/titok_l32_CSL.yaml
    --experiment.project="tokenise_titok" 
    --experiment.name="tokenise_titok" 
    --experiment.output_dir="titok_l32_CSL_tokenise"
'''
