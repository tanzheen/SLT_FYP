from ctypes import util
from cv2 import IMREAD_GRAYSCALE
import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
import math
from torchvision import transforms
from PIL import Image
import cv2
import os
import random
import numpy as np
import lmdb
import io
import time
from vidaug import augmentors as va
from loguru import logger
from augmentation import *
from definition import *
import pickle

def load_annot_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



class S2T_Dataset(Dataset.Dataset):
    def __init__ (self, tokenizer, config, args, phase, training_refurbish = False):
        self.config = config # this is the config file that will contain paths 
        self.args = args 
        self.training_refurbish = training_refurbish
        self.phase = phase 
        
        self.raw_data = load_annot_file(config['data'][phase])  # path will be the split set to retrieve (train, dev, test)
        self.tokenizer = tokenizer
        self.max_length = self.raw_data['max_length']
        self.img_path = os.path.join(config['data']['img_path'], phase)
        self.data_list = self.raw_data['info']

        # the following parameters are for data augmentation which is proven to improve the training process
        sometimes = lambda aug: va.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
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

        self.seq_color = va.Sequential([
            sometimes(Brightness(min=0.1, max=1.5)),
            sometimes(Color(min=0.1, max=1.5)),
            # sometimes(Contrast(min=0.1, max=2.0)),
            # sometimes(Sharpness(min=0.1, max=2.))
        ])
        # self.seq = SomeOf(self.seq_geo, self.seq_color)

    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, index): 
        file = self.data_list[index]
        name = file['name']
        text_label = file['translation']
        length = file['length']
        img_folder_path = os.path.join(self.img_path , name)
        
        img_sample = self.load_imgs(img_folder_path) ## load_imgs will retrieve all the images from the folder to consolidate into one folder
        
        # need to pad according to max length
        max_len = self.max_length
        left_pad = 8 
        right_pad = right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 8
        max_len = max_len + left_pad + right_pad
        padded_video = torch.cat(
            (
                img_sample[0][None].expand(left_pad, -1, -1, -1), # repeated first frame 
                img_sample,
                img_sample[-1][None].expand(max_len - len(img_sample) - left_pad, -1, -1, -1), # repeated last frame
            )
            , dim=0)
            



        
        return name, img_sample, text_label
    
    def load_imgs(self, dir_path):
        print('img directory: ', dir_path )
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ## these values are the mean and s.d of RGB channels from the ImageNet dataset
        ])

        paths = sorted(os.listdir(dir_path)) 
        
        if len(paths) > self.max_length:
            tmp = sorted(random.sample(range(len(paths)), k=self.max_length))
            new_paths = []
            for i in tmp:
                new_paths.append(paths[i])
            paths = new_paths

        # number of frames, RGB, Height, Width 
        imgs = torch.zeros(len(paths ), 3, self.args.input_size, self.args.input_size)
        crop_rect, resize = utils.data_augmentation(resize=(self.args.resize, self.args.resize),
                                                    crop_size=self.args.input_size, is_train=(self.phase == 'train'))
        
        batch_image = []
        for i, img_path in enumerate(paths):
            img_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            batch_image.append(img)

        if self.phase == 'train':
            batch_image = self.seq(batch_image)

        for i, img in enumerate(batch_image):
            img = img.resize(resize)
            img = data_transform(img).unsqueeze(0)
            imgs[i, :, :, :] = img[:, :, crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]
        
        return imgs # frames, RGB, height, width 
    

    def __str__(self):
        return f'#total {self.phase} set: {len(self.data_list)}.'
    




        


        

        




        
