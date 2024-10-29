'''
Set up LLM adaptor and pass the embeddings thru the adapter 
'''

from train_sign_utils import * 
import os 
from accelerate import Accelerator
from logger import setup_logger
from accelerate.utils import set_seed
import sys 
from transformers import MBart50Tokenizer
import torch.multiprocessing as mp


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



    # Create model 
    model, ema_model = create_model(config, logger, accelerator)
    # Create signloaders 
    tokenizer = MBart50Tokenizer.from_pretrained(config.training.tokenizer,
                                               src_lang=config.dataset.lang,
                                                 tgt_lang= config.dataset.lang)
    train_dataloader, dev_dataloader, test_dataloader = create_signloader(config, logger, accelerator, tokenizer)

    model= accelerator.prepare(
            model)
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model,
        strict=True)
    model.eval()

    with torch.no_grad(): 
        titok = model.titok
        llm_adapter = model.adapter
        for i , (src, tgt) in enumerate(tqdm(train_dataloader, desc=f"After adapter tokenising!")): 
            batch = src['input_ids']
            src_length = src['src_length_batch']
            names = src['name_batch']
            images = batch.to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            encoded_tokens = titok.encode(x=images)[1]['min_encoding_indices'].squeeze()
            hidden_values = llm_adapter(encoded_tokens.float(), src_length).squeeze()
            input_attn = src['attention_mask'].to(
                    accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
                )
            print(hidden_values.shape)
            for j, nam in enumerate(names): 
                encoded_grp = hidden_values[j]
                ## save each pth file 
                for i, enc in enumerate(encoded_grp): 
                    save_path = os.path.join(output_dir, f"{nam}_{i}.pth") ## need to double check on output_dir
                    torch.save(enc, save_path)
                    print(f"Saved {nam}_{i}.pth")
                




    

