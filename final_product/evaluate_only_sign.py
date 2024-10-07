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
    
    # Initialize trackers on the main process.
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
                                                 tgt_lang=config.dataset.lang)
    _, dev_dataloader, test_dataloader = create_signloader(config, logger, accelerator, tokenizer)

    # Prepare everything with the accelerator.
    logger.info("Preparing model and dataloaders for evaluation")
    
    model = accelerator.prepare(model)
    global_step, first_epoch = auto_resume(
        config, logger, accelerator, ema_model,
        strict=True)

    # Call evaluation function
    logger.info("***** Running evaluation *****")
    print("Checking Dev set")
    _, _ , dev_pred, dev_ref = eval_translation(model=model, dev_dataloader=dev_dataloader, accelerator=accelerator, tokenizer=tokenizer, config=config)
    save_predictions_and_references(dev_pred, dev_ref, output_dir, filename="dev_predictions.txt")
    print("Checking Test set")
    _, _, test_pred, test_ref = eval_translation(model=model, dev_dataloader=test_dataloader, accelerator=accelerator, tokenizer=tokenizer, config=config)
    
    save_predictions_and_references(test_pred, test_ref, output_dir, filename="test_predictions.txt")
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