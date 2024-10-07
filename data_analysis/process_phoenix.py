import cv2
import torch
import json
import torchvision.transforms as transforms
import pandas as pd
from transformers import MBartTokenizer
from PIL import Image
import torch.utils.data.dataset as Dataset
import random 
import vidaug.augmentors as va
from torch.nn.utils.rnn import pad_sequence
import os 
import pickle 
'''
Given: 
- paths to different image files 
- their gloss and translation in a CSV file

Create: 
yaml config 
function to combine the images into folders 
dataset class
'''


def read_csv_annotations(csv_path):
    dataset = pd.read_csv(csv_path,delimiter = '|')
    return dataset

def create_phoenix_annot(csv_input_path, output_path, img_path):
    dataset= read_csv_annotations(csv_input_path)
    max_length = 0 
    # Initialize an empty list to store the results
    video_info_list = []
    # Iterate over each row in the dataset
    for index, row in dataset.iterrows():
        video_name = row['name']
        translation = row['translation']
 
        video_folder = os.path.join(img_path, row['name'] ) # folder to get to the video
        
        # Count the number of frames (png files) in the video folder
        try:
            frame_count = len([f for f in os.listdir(video_folder) if f.endswith('.png')])
            if frame_count>max_length: 
                max_length = frame_count
            video_info_list.append({
            'name': video_name,
            'translation': translation,
            'length': frame_count
        })
            # Convert the list of dictionaries into a DataFrame for easier handling
            # info contains the names of the videos and the respective translation 
            # max_length is the number of frames in the longest video, this number will be used for padding later on
            video_info_dict = {'info':video_info_list, 
                       'max_length': max_length}

        except FileNotFoundError:
            print(f"file {video_name} not found") # In case the folder doesn't exist
        

    # Save the dictionary as a pkl file
    save_results(video_info_dict, output_path)
    return video_info_dict


def save_results(video_info_dict, output_path):
    # Save the dictionary as a pkl file
    with open(output_path, 'wb') as pkl_file:
        pickle.dump(video_info_dict, pkl_file)


def main():
    output_folder = '../../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/processed/'

    for ty in ['train', 'dev', 'test']: 
        ## repeat itself for train, dev and test set
        dataset_path = f'../../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.{ty}.corpus.csv'  
        os.makedirs(output_folder, exist_ok=True)
        output_path = f'../../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/processed/labels_{ty}.pkl'
        img_path = f'../../PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/{ty}/'
        
        # Process the dataset and save the results
        create_phoenix_annot(dataset_path, output_path, img_path)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    '''
    Running this file will process the labels from the csv files 
    The data will process the following: 
    - the folder name containing the video images 
    - the translation of the video 
    - the number of frames in case there is the need to add padding frames 
    '''
    main()