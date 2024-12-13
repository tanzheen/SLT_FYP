# *basic
import os
from scripts.train_SpaEMo_utils import * 
from pathlib import Path
import sys 
import torch.distributed
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed
from scripts.logger import setup_logger
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
    tokenizer.padding_side= 'right'

    train_dataloader, dev_dataloader, test_dataloader = create_dataloader(config, 
                                                                          logger, 
                                                                            tokenizer, 
                                                                            "cuda")
    model = create_SpaEMo(config, logger)

    # tokenise training set 
    for i , (src_input,tgt_input) in enumerate(train_dataloader): 
        model.save_embeddings(src_input, phase = 'train')


    # tokenise dev set 
    for i , (src_input,tgt_input) in enumerate(dev_dataloader): 
        model.save_embeddings(src_input, phase = 'dev')


    # tokenise test set 
    for i , (src_input,tgt_input) in enumerate(dev_dataloader): 
        model.save_embeddings(src_input, phase = 'test')


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
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network tokenise_all.py config=configs/SpaEMo_P14_config.yaml 
'''




