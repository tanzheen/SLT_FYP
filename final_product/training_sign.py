import torch.distributed
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
        split_batches=False
    )

    logger = setup_logger(name="Sign2Text", log_level="INFO",
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
    # Create optimizer
    optimizer = create_optimizer(config, logger, model)
    # Create lr_scheduler 
    scheduler= create_scheduler(
        config, logger, accelerator, optimizer,len(train_dataloader))


    #Prepare everything with accelerator.
    logger.info("Preparing model, optimizer and dataloaders")
    
    
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    # Start training.
    logger.info("***** Running training *****")
    #logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"mixed precision = {config.training.mixed_precision}")
    logger.info(f"""  Total train batch size (w. parallel, distributed & accumulation) = {(
        config.training.per_gpu_batch_size *
        accelerator.num_processes *
        config.training.gradient_accumulation_steps)}""")
    logger.info(f"accelerator device: {accelerator.device}")
    global_step = 0
    first_epoch = 0
   
    
    # Freeze both model's weights again just in case
    # Assuming 'model' is your original model wrapped in DDP
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model.module.freeze_Titok_weights()
    else:
        model.freeze_Titok_weights()

    if config.model.MBart_model.freeze_MBart: 
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.freeze_MBart_weights()
        else:
            model.freeze_MBart_weights()
    
    # Prepare modules with accelerator
    model, optimizer,  scheduler= accelerator.prepare(
            model, optimizer,scheduler 
        )
    
    
    # Auto resume from training 
    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model,
        strict=True)

    num_train_epochs = config.training.num_epochs
    early_stopping = EarlyStopping(verbose=True)
    for current_epoch in range(first_epoch, num_train_epochs):

        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step, early_stopping = train_one_epoch(config=config,
                                      logger= logger,
                                      accelerator= accelerator,model= model,
                                       ema_model= ema_model,optimizer= optimizer,
                                        scheduler = scheduler,
                                       train_dataloader=train_dataloader,
                                        dev_dataloader= dev_dataloader,
                                         test_dataloader= test_dataloader, 
                                         tokenizer=tokenizer,
                                         global_step= global_step, 
                                         early_stop = early_stopping) # the early stopping will be passed back in again 
        
    #accelerator.wait_for_everyone()
    # Save checkpoint at the end of training.
    save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if config.training.use_ema:
            ema_model.copy_to(model.parameters())
        model.save_pretrained_weight(output_dir)
    accelerator.end_training()


if __name__ == "__main__":
 
    sys.path.append("..")
    print(torch.__version__)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        torch.set_default_device('cuda')
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")

        mp.set_start_method('spawn', force=True)
    else:
        torch.set_default_device('mps')
        print("CUDA is not available. Using CPU.")
    torch.cuda.empty_cache()
    main()

'''
# Training for Sign2Text
# Stage 1
$env:WANDB_MODE="offline"
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network training_sign.py config=configs/Sign2Text_CSL_config.yaml --experiment.project="Sign2Text_CSL" --experiment.name="Sign2Text_CSL_run1" --experiment.output_dir="Sign2Text_CSL_run1" 
'''
    

