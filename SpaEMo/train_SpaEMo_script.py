# *basic
import os
from scripts.train_SpaEMo_utils import * 
from pathlib import Path
import sys 
import torch.distributed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from logger import setup_logger
from timm.optim import create_optimizer_v2
from timm.scheduler import  create_scheduler, create_scheduler_v2
import torch.multiprocessing as mp

def main ():
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
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=tracker,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
        kwargs_handlers= [kwargs]
    )

    logger = setup_logger(name=config.experiment.project, log_level="INFO",
        output_file=f"{output_dir}/log{accelerator.process_index}.txt")
    
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed, device_specific=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)

    train_dataloader, dev_dataloader, test_dataloader = create_dataloader(config, 
                                                                          logger, 
                                                                          accelerator,
                                                                            tokenizer)
    model = create_SpaEMo(config, logger)
    # Create an optimizer
    optimizer = create_optimizer_v2(
        model.parameters(),
        opt=config.optimizer.name,
        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler, _ = create_scheduler_v2(optimizer=optimizer, sched="cosine", 
                                       min_lr=1e-5, 
                                        warmup_lr=1e-6,
                                        warmup_epochs=5,
                                        decay_epochs=10,
                                        decay_rate=0.1,
                                       )

    model,optimizer, scheduler= accelerator.prepare(model,
                                                       optimizer,
                                                       scheduler
                                                       )
    
    # Auto resume from training 
    global_step, first_epoch = auto_resume(
        config, logger, accelerator, 
        strict=True)
    
    ## Start training
    num_train_epochs = config.training.num_epochs
    early_stop = EarlyStopping(verbose=True)
    for current_epoch in range(first_epoch, num_train_epochs):
        torch.cuda.empty_cache()

        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        global_step, early_stop= train_one_epoch(config, accelerator, model, tokenizer,
                    train_dataloader, dev_dataloader, optimizer,
                    logger ,  scheduler , global_step, early_stop) # the early stopping will be passed back in again 
    

    # Save the final trained checkpoint
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        save_checkpoint(model, output_dir, accelerator, global_step, logger=logger)
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
'accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network train_SpaEMo_script.py config=configs/SpaEMo_P14_config.yaml --experiment.project="SpaEMo_P14" --experiment.name="SpaEMo_P14_run1" --experiment.output_dir="SpaEMo_P14_run1"'
'''


    









