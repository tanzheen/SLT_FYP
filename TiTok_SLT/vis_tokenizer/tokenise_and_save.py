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
from modeling.titok import TiTok
from PIL import Image
import gzip 
import pickle 
import numpy as np

def main(): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset_file(filename):
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    all_paths = []
    train_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.train'
    train_labels = load_dataset_file(train_labels)
    val_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.dev'
    val_labels = load_dataset_file(val_labels)
    test_labels = 'E:\GFSLT-VLP\data\Phonexi-2014T/labels.test'
    test_labels = load_dataset_file(test_labels)

    for value in train_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    for value in val_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    for value in test_labels.values(): 
        path_lst = value['imgs_path']
        all_paths.extend(path_lst)

    #print(all_paths)

    # Open up the label files and collate all the labels
    

    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet").to(device)
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)


    print(f"number of images: {len(all_paths)}")

    def tokenize_and_save(img_path):
        img_path = os.path.join('E:\PHOENIX-2014-T-release-v3\PHOENIX-2014-T/features/fullFrame-210x260px/', img_path)
        print(img_path)
        original_image = Image.open(img_path)
        original_image = original_image.resize((256, 256))  
        image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"].squeeze()

        # save the encoded tokens 
        new_file_path = os.path.splitext(img_path)[0] + ".pt"
        torch.save(encoded_tokens, new_file_path)
    
    def tokenize_and_quantize(img_path): 
        img_path = os.path.join('E:\PHOENIX-2014-T-release-v3\PHOENIX-2014-T/features/fullFrame-210x260px/', img_path)
        print(img_path)
        original_image = Image.open(img_path)
        original_image = original_image.resize((256, 256)) 
        image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        encoded_tokens = titok_tokenizer.encode(image.to(device))[1]["min_encoding_indices"]
        quantized_image = titok_tokenizer.codebook_quantize_tokens(encoded_tokens).squeeze()
        # save the encoded tokens 
        new_file_path = os.path.splitext(img_path)[0] + "_quant.pt"
        torch.save(quantized_image, new_file_path)
    print("Tokenizing! ")
    for img_path in all_paths: 
        tokenize_and_quantize(img_path)


    


if __name__ == "__main__":
 
    sys.path.append("..")
    print(torch.__version__)
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device: {torch.cuda.get_device_name(current_device)}")
        print(torch.backends.cuda.flash_sdp_enabled())
    
    torch.cuda.empty_cache()
    main()

'''
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network tokenise_and_save.py config=configs/training/stage1/titok_l32_P14.yaml
    --experiment.project="tokenise_titok" 
    --experiment.name="tokenise_titok" 
    --experiment.output_dir="titok_l32_P14_tokenise"
'''
