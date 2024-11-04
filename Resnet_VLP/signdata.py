

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from definition import *
import pickle
from vidaug import augmentors as va


def load_annot_file(file_path):
    """
    Load the annotation file from the provided path.
    
    Args:
        file_path (str): The path to the annotation file.
        
    Returns:
        data: The deserialized annotation data from the file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

class SignTransDataset(Dataset):
    """
    Custom dataset class for sign language translation. 
    
    This class handles loading and processing the sign language video frames and their corresponding text labels, 
    as well as preparing the data for training, validation, and testing stages.
    
    Attributes:
        config (dict): Configuration dictionary containing paths, settings, and model details.
        training_refurbish (bool): Flag indicating whether to add noise for data augmentation.
        phase (str): Current phase ('train', 'dev', or 'test').
        raw_data (dict): Contains information about the dataset, including max length and file paths.
        tokenizer: The tokenizer used for encoding text labels.
        img_path (str): The path to the image directory for the current phase.
        data_list (list): List of information about each sample, including file names and labels.
        crop_width (int): Desired width for cropping the image.
        crop_height (int): Desired height for cropping the image.
    """

    def __init__(self, tokenizer, config,  phase, training_refurbish=True ):
        """
        Initialize the SignTrans_Dataset with the necessary parameters.
        
        Args:
            tokenizer: Tokenizer for text data (used for embedding the text labels).
            config (dict): Configuration for the dataset and models.
            phase (str): The current phase ('train', 'dev', 'test') to determine which split to load.
        """
        self.config = config  # Config contains paths, settings, and model details.

        self.phase = phase  # Can be 'train', 'dev', or 'test'.

        # Load the annotations (e.g., translations) from the annotation file corresponding to the current phase.
        self.raw_data = load_annot_file(config['dataset'][phase])
        
        # Setup tokenizer and set the target language from config.
        self.tokenizer = tokenizer
        
        self.max_length = self.raw_data['max_length']  # Maximum sequence length.
        self.img_path = os.path.join(config['dataset']['img_path'], phase)  # Path to the images directory for the phase.
        self.data_list = self.raw_data['info']  # List of all data samples (videos).
        self.dataset_name = config['dataset']['name']  # The dataset type (used to apply specific transformations).
        self.crop_width = config.dataset.preprocessing.person_size  # Image crop width.
        self.crop_height = config.dataset.preprocessing.person_size  # Image crop height.
        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10)),

            # sometimes(Brightness(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),

        ])
        self.training_refurbish = training_refurbish  # Flag to determine whether to add noise for data augmentation.  
    def __len__(self):
        """
        Get the total number of videos in the dataset.
        
        Returns:
            int: Number of videos in the dataset.
        """
        return len(self.data_list)
    
    def __getitem__(self, index): 
        """
        Get an individual item (video and its corresponding label) from the dataset.
        
        Args:
            index (int): The index of the video to retrieve.
        
        Returns:
            tuple: A tuple containing the video name, the loaded images as a tensor, and the text label.
        """
        file = self.data_list[index]  # Get the data entry for the specified index.
        name = file['name']  # Name of the video file.
        text_label = file['translation']  # Corresponding translation (text label).
        length = file['length']  # Number of frames in the video.
        
        # Load the images for the video and pad as necessary.
        img_sample, path_lst= self.load_imgs(os.path.join(self.img_path, name))
        if img_sample.shape[0] != length: 
            print(f"Name: {name}, Length mismatch: Retrieved {img_sample.shape[0]} vs Recorded {length}") 
        # else: 
        #     print(f"Name list: {path_lst}, Name: {name}, Length: {length}") 
        
        return name, img_sample, text_label
    
    def load_imgs(self, dir_path):
        """
        Load all image frames from a specified video directory, apply necessary transformations,
        and pad the sequence if it exceeds the maximum length.
        
        Args:
            dir_path (str): The directory path where the video frames are stored.
        
        Returns:
            torch.Tensor: A tensor containing the video frames, padded to the required size.
        """
        # Set up dataset-specific transformations.
        # No normalization applied (all values remain between 0 and 1).

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]
        data_transform = transforms.Compose([
            transforms.Resize((self.config.dataset.preprocessing.resize_shorter_edge, 
                                self.config.dataset.preprocessing.resize_shorter_edge)),
            transforms.RandomCrop((self.config.dataset.preprocessing.crop_size, self.config.dataset.preprocessing.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std)
        ])
    
        # Filter the frames to ensure only photos are loaded
        filtered_paths  = []
        
        #Sort the frame paths to maintain the order.
        paths = sorted(os.listdir(dir_path)) 
        for path in paths:
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')): 
                    filtered_paths.append(path)
        paths = sorted(filtered_paths) 
        #print (paths)
        # If the video contains more frames than the max length, sample randomly up to max_length.
        if len(paths) > self.max_length:
            sampled_indices = sorted(random.sample(range(len(paths)), k=self.max_length))
            paths = [paths[i] for i in sampled_indices]

        # Create an empty tensor to store the images.
        imgs = torch.zeros(len(paths), 3, 
                        self.config.dataset.preprocessing.crop_size, 
                        self.config.dataset.preprocessing.crop_size)
        batch_image = []

        # Load each image, apply transformations, and crop if necessary.
        for i, img_path in enumerate(paths):
            img_path = os.path.join(dir_path, img_path)
            img = Image.open(img_path).convert("RGB")
            
            # Crop the image based on the provided config (center horizontal, lower vertical crop).
            x_start = int((img.size[0] - self.crop_width) // 2)
            x_end = x_start + self.crop_width
            y_start = img.size[1] - self.crop_height
            y_end = img.size[1]
            img = img.crop((x_start, y_start, x_end, y_end))

            batch_image.append(img)

        # Apply transformations and stack the images into a tensor.
        if self.phase == 'train':
            batch_image = self.seq(batch_image)



        for i, img in enumerate(batch_image):
            img = data_transform(img).unsqueeze(0)  # Apply the dataset-specific transformation.
            imgs[i, :, :, :] = img
        
        return imgs, paths  # Return the tensor (frames, RGB channels, height, width).

        


    
    def __str__(self):
        """
        Get a string representation of the dataset (number of samples).
        
        Returns:
            str: A string indicating the number of videos in the current phase.
        """
        return f'#total {self.phase} set: {len(self.data_list)}.'
  
    def collate_fn(self, batch):
        """
        Collate function for batching multiple samples.
        
        Pads the images to the same length, and tokenizes the text labels.
        
        Args:
            batch (list): A list of samples to be batched together.
        
        Returns:
            dict: A dictionary containing the batched input data and other details.
        """
        tgt_batch, img_tmp, src_length_batch, name_batch = [], [], [], []

        # Separate the video frames and labels.
        for name_sample, img_sample, tgt_sample in batch:
            name_batch.append(name_sample)
            img_tmp.append(img_sample)
            tgt_batch.append(tgt_sample)

        # Determine the maximum video length in the batch.
        max_len = max([len(vid) for vid in img_tmp])
        video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 16 for vid in img_tmp])
        left_pad = 8
        right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        
        
        # Pad the videos to match the length of the longest video in the batch.
        # Although the transfomer can indeed take in varying lengths of video, we need to still make their length uniform in the batch 
        padded_video = [torch.cat(
            (
                vid[0][None].expand(left_pad, -1, -1, -1),  # Padding at the start with the first frame.
                vid,
                vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),  # Padding at the end.
            ), dim=0) for vid in img_tmp]
        
        # Ensure each video has the same length.
        img_tmp = [padded_video[i][0:video_length[i], :, :, :] for i in range(len(padded_video))]
            

        # Track the length of each padded video.
        for i in range(len(img_tmp)):
            src_length_batch.append(len(img_tmp[i]))
        src_length_batch = torch.tensor(src_length_batch)
        
        #print("img tmp", img_tmp[0].shape)
        # Stack all images into a single tensor.
        img_batch = torch.cat(img_tmp, 0)
        #print("imgbatch", img_batch.shape)
        # Compute the new source lengths after some processing.
        new_src_lengths = (((src_length_batch - 5 + 1) / 2) - 5 + 1) / 2
        new_src_lengths = new_src_lengths.long()

        # Create a mask for padding. These will be for the sequences into the MBart model
        mask_gen = []
        for i in new_src_lengths:
            tmp = torch.ones([i]) + 7
            mask_gen.append(tmp)
        mask_gen = pad_sequence(mask_gen, padding_value=PAD_IDX, batch_first=True)
        img_padding_mask = (mask_gen != PAD_IDX).long()

        # Tokenize the text labels.
        with self.tokenizer.as_target_tokenizer():
            tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)
        
        # Pack everything into a dictionary to return.
        src_input = {
            'input_ids': img_batch,
            'attention_mask': img_padding_mask,
            'name_batch': name_batch,
            'src_length_batch': src_length_batch,
            'new_src_length_batch': new_src_lengths
        }

        if self.training_refurbish:
        
            masked_tgt = NoiseInjecting(tgt_batch, noise_rate=0.15, noise_type='omit_last', random_shuffle=False , is_train=(self.phase=='train'))
            with self.tokenizer.as_target_tokenizer():
                masked_tgt_input = self.tokenizer(masked_tgt, return_tensors="pt", padding = True,  truncation=True)
            return src_input, tgt_input, masked_tgt_input

        return src_input, tgt_input
    

def NoiseInjecting(raw_gloss, noise_rate=0.15, noise_type='omit_last', random_shuffle=False, is_train=True):
    new_gloss = []

    for ii, gloss in enumerate(raw_gloss):
        text = gloss.split()

        if noise_type == 'omit':
            # del noise
            if random.uniform(0, 1) <= 1. and is_train:
                index = sampler_func(len(text), int(len(text)*(1. - noise_rate)), random_choice=is_train)
                noise_gloss = []
                noise_idx = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
                        noise_idx.append(i)
            else:
                noise_gloss = [d for d in text]

        elif noise_type == 'omit_last' :
            if random.uniform(0, 1) <= 1.0 and is_train:
                index = np.arange(0, len(text) - int(np.ceil(len(text)*(np.random.uniform(0,noise_rate,(1,))))), 1, dtype=int)
                noise_gloss = []
                for i, d in enumerate(text):
                    if i in index:
                        noise_gloss.append(d)
                    else:
                        noise_gloss.append(WORD_MASK)
            else:
                noise_gloss = [d for d in text]
        
        if is_train and random_shuffle and random.uniform(0, 1) > 0.5:
            random.shuffle(noise_gloss) # random shuffle sequence

        new_gloss.append(' '.join(noise_gloss))
    return new_gloss


def sampler_func(clip, sn, random_choice=True):
    if random_choice:
        f = lambda n: [(lambda n, arr: n if arr == [] else np.random.choice(arr))(n * i / sn,
                                                                                range(int(n * i / sn),
                                                                                        max(int(n * i / sn) + 1,
                                                                                            int(n * (
                                                                                                    i + 1) / sn))))
                        for i in range(sn)]
    else:
        f = lambda n: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                                max(int(
                                                                                                    n * i / sn) + 1,
                                                                                                    int(n * (
                                                                                                            i + 1) / sn))))
                        for i in range(sn)]
    return f(clip)