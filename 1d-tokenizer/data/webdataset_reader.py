"""This file contains the definition of data loader using webdataset.

This file may have been modified by Bytedance Ltd. and/or its affiliates (“Bytedance's Modifications”).
All Bytedance's Modifications are Copyright (year) Bytedance Ltd. and/or its affiliates. 

Reference:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text

from torch.utils.data import default_collate
from torchvision import transforms
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms




class SimpleImageDataset(Dataset):
    def __init__(self, root_dir,phase,   transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = os.path.join(root_dir,phase) 
        self.transform = transform
        self.image_paths = self._gather_image_paths(root_dir)

    def _gather_image_paths(self, root_dir):
        """
        Recursively collects all image file paths from the root directory.
        """
        image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            image: The image corresponding to the given index.
            path: The path of the image.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path