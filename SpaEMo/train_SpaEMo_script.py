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


def main(rank, world_size, config):
    # Initialize DDP if world_size > 1
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Set up logging
    output_dir = config.experiment.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(
        name=config.experiment.project,
        log_level="INFO",
        output_file=f"{output_dir}/log{rank}.txt" if world_size > 1 else f"{output_dir}/log.txt", 
        use_accelerate=False
    )

    # Config logging
    if rank == 0:
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Set random seed for reproducibility
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
    tokenizer.padding_side = 'right'
    if config.model.multilingual:
        tokenizer.src_lang = config.dataset.lang
        tokenizer.tgt_lang = config.dataset.lang

    # Create dataloaders
    train_dataloader, dev_dataloader, test_dataloader = create_dataloader(config, logger, tokenizer)

    # Initialize model
    model = create_SpaEMo(config, logger).to(device)

    # Wrap model in DDP if world_size > 1
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Create optimizer and scheduler
    optimizer = create_optimizer_v2(
        model.parameters(),
        opt=config.optimizer.name,
        lr=config.optimizer.params.learning_rate,
        weight_decay=config.optimizer.params.weight_decay
    )

    scheduler, _ = create_scheduler_v2(
        optimizer=optimizer,
        sched="cosine",
        min_lr=1e-5,
        warmup_lr=1e-6,
        warmup_epochs=5,
        decay_epochs=10,
        decay_rate=0.1
    )

    # Resume training if applicable
    global_step, first_epoch = auto_resume(config, logger, model, optimizer, scheduler)

    # Start training
    num_train_epochs = config.training.num_epochs
    early_stop = EarlyStopping(verbose=True)

    for current_epoch in range(first_epoch, num_train_epochs):
        torch.cuda.empty_cache()
        logger.info(f"Epoch {current_epoch}/{num_train_epochs - 1} started.")
        global_step, early_stop = train_one_epoch(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            optimizer=optimizer,
            logger=logger,
            scheduler=scheduler,
            global_step=global_step,
            early_stop=early_stop,
            current_epoch=current_epoch,
            device=device
        )

    # Save final checkpoint
    if rank == 0:
        save_checkpoint(model, output_dir, global_step, logger=logger)

    # Cleanup DDP
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.path.append("..")
    print(torch.__version__)
    world_size = torch.cuda.device_count()
    mp.set_start_method('spawn', force=True)
    config = get_config()
    if world_size > 1:
        mp.spawn(
            main,
            args=(world_size, get_config()),
            nprocs=world_size,
            join=True
        )
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        main(rank=0, world_size=1, config = config)

'''
python train_SpaEMo_script.py --config=configs/SpaEMo_P14_config.yaml --experiment.project="SpaEMo_P14" --experiment.name="SpaEMo_P14_run1" --experiment.output_dir="SpaEMo_P14_run1"
  
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network train_SpaEMo_script.py config=configs/SpaEMo_P14_config.yaml --experiment.project="SpaEMo_P14" --experiment.name="SpaEMo_P14_run1" --experiment.output_dir="SpaEMo_P14_run1"
'''


    

# def main ():

#     workspace = os.environ.get('WORKSPACE', '')
#     if workspace:
#         torch.hub.set_dir(workspace + "/models/hub")
    
#     config = get_config()

#     # Enable TF32 on Ampere GPUs.
#     if config.training.enable_tf32:
#         torch.backends.cuda.matmul.allow_tf32 = True
#         torch.backends.cudnn.allow_tf32 = True
#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.deterministic = False
    
#     output_dir = config.experiment.output_dir
#     os.makedirs(output_dir, exist_ok=True)
#     config.experiment.logging_dir = os.path.join(output_dir, "logs")

#     # Whether logging to Wandb or Tensorboard.
#     tracker = "tensorboard"
#     if config.training.enable_wandb:
#         tracker = "wandb"
#     kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

#     ## need convert this part to pure pytorch code
#     accelerator = Accelerator(
#         gradient_accumulation_steps=config.training.gradient_accumulation_steps,
#         mixed_precision=config.training.mixed_precision,
#         log_with=tracker,
#         project_dir=config.experiment.logging_dir,
#         split_batches=False,
#         kwargs_handlers= [kwargs]
#     )

#     logger = setup_logger(name=config.experiment.project, log_level="INFO",
#         output_file=f"{output_dir}/log{accelerator.process_index}.txt")
    
#     ## here also 
#     if accelerator.is_main_process:
#         accelerator.init_trackers(config.experiment.name)
#         config_path = Path(output_dir) / "config.yaml"
#         logger.info(f"Saving config to {config_path}")
#         OmegaConf.save(config, config_path)
#         logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

#     # If passed along, set the training seed now.
#     if config.training.seed is not None:
#         set_seed(config.training.seed, device_specific=True)

#     tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
#     tokenizer.padding_side= 'right'

#     if config.model.multilingual:
#         tokenizer.src_lang = config.dataset.lang
#         tokenizer.tgt_lang = config.dataset.lang

#     train_dataloader, dev_dataloader, test_dataloader = create_dataloader(config, 
#                                                                           logger, 
#                                                                           accelerator,
#                                                                             tokenizer)
    
#     model = create_SpaEMo(config, logger)
#     # Create an optimizer
#     optimizer = create_optimizer_v2(
#         model.parameters(),
#         opt=config.optimizer.name,
#         lr=config.optimizer.params.learning_rate,
#         weight_decay=config.optimizer.params.weight_decay
#     )

#     scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine", 
#                                        min_lr=1e-5, 
#                                         warmup_lr=1e-6,
#                                         warmup_epochs=5,
#                                         decay_epochs=10,
#                                         decay_rate=0.1,
#                                        )

#     # model,optimizer, scheduler= accelerator.prepare(model,
#     #                                                    optimizer,
#     #                                                    scheduler
#     #                                                    )
    
#     # Auto resume from training 
#     global_step, first_epoch = auto_resume(
#         config, logger, accelerator, 
#         strict=True)
    
#     ## Start training
#     num_train_epochs = config.training.num_epochs
#     early_stop = EarlyStopping(verbose=True)

#     for current_epoch in range(first_epoch, num_train_epochs):
#         torch.cuda.empty_cache()

#         # remove accelerator print 
#         accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
#         global_step, early_stop= train_one_epoch(config, accelerator, model, tokenizer,
#                     train_dataloader, dev_dataloader, optimizer,
#                     logger ,  scheduler , global_step, early_stop, current_epoch ) # the early stopping will be passed back in again 
    

#     # Save the final trained checkpoint
#     # don't need this, just save 
#     if accelerator.is_main_process:
#         model = accelerator.unwrap_model(model)
#         save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
#     accelerator.end_training()

# if __name__ == "__main__":
 
#     sys.path.append("..")
#     print(torch.__version__)
#     if torch.cuda.is_available():
#         current_device = torch.cuda.current_device()
#         torch.set_default_device('cuda')
#         print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")

#         mp.set_start_method('spawn', force=True)
#     else:
#         torch.set_default_device('mps')
#         print("CUDA is not available. Using CPU.")
#     torch.cuda.empty_cache()
#     main()







