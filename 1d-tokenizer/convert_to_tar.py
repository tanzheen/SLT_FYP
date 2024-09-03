import os
import tarfile

def create_tar_from_folders(root_dir, output_dir, chunk_size=1000):
    """
    Create tar files from subdirectories of images in the root directory.

    Args:
        root_dir (str): Root directory containing subdirectories of images.
        output_dir (str): Directory where the output tar files will be stored.
        chunk_size (int): Number of samples per tar file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over each subdirectory in the root directory
    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if os.path.isdir(subdir_path):
            files = sorted([f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
            for i in range(0, len(files), chunk_size):
                chunk_files = files[i:i + chunk_size]
                tar_name = os.path.join(output_dir, f"{subdir}_{i // chunk_size}.tar")
                with tarfile.open(tar_name, 'w') as tar:
                    for file_name in chunk_files:
                        file_path = os.path.join(subdir_path, file_name)
                        tar.add(file_path, arcname=file_name)
                print(f"Created {tar_name}")

# # Replace with your directories
# root_dir = '../../CSL-Daily/sentence/frames_512x512/train'
# output_dir = '../../CSL-Daily/sentence/frames_512x512/train'
# create_tar_from_folders(root_dir, output_dir)

# # Replace with your directories
# root_dir = '../../CSL-Daily/sentence/frames_512x512/dev'
# output_dir = '../../CSL-Daily/sentence/frames_512x512/dev'
# create_tar_from_folders(root_dir, output_dir) 

# # Replace with your directories
# root_dir = '../../CSL-Daily/sentence/frames_512x512/train'
# output_dir = '../../CSL-Daily/sentence/frames_512x512/train'
# create_tar_from_folders(root_dir, output_dir)

import webdataset as wds
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def create_webdataset_loader(output_dir, transform=None, batch_size=64, num_workers=4):
    dataset = (
        wds.DataPipeline.ResampledShards(f"{output_dir}/*.tar")
        .decode("pil")
        .to_tuple("jpg", "png")
        .map_tuple(transform)
    dataloader = wds.WebLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return dataloader

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

tar_paths = '../../CSL-Daily/sentence/frames_512x512/train'
# Alternatively, you can use a wildcard pattern: tar_paths = "path_to_tar_files/*.tar"
dataloader = create_webdataset_loader(tar_paths, transform=transform)
for images in dataloader: 
    print (images.shape)
