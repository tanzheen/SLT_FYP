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
import shutil

'''
1. How to process the compressed data in the folder[sentence]?
a. download the splited files of it.
    1- download csl-daily-frames-512x512.tar.gz_00 ---- csl-daily-frames-512x512.tar.gz_09 
    2- cat these files on Ubuntu.
        cat csl-daily-frames-512x512.tar.gz_* >  csl-daily-frames-512x512.tar.gz
b. extract the file
    tar -xvzf csl-daily-frames-512x512.tar.gz

2. How to read the annotation file?
import pickle
with open('csl2020ct_v2.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())
print(data['info'][0])

3. What's the meaning of keys in the above file?
info: all the annotations
    name: video name
    length: the number of frames in the video
    label_gloss: gloss sequence stored in [List] for sign language recognition
    label_char: char sequence stored in [List] for sign language translation. In our experiments, we use this as SLT target.
    label_word: word sequence stored in [List] for sign language translation. Just for reference.
    signer: the id of some signer. It starts from 0.
    time: how many times the same signer performs the same sentence. It starts from 0. Actually, no signer performs the same sentence twice.
gloss_map: vocabulary for glosses
char_map: vocabulary for chars
word_map: vocabulary for words


1. Open split file then split the folders accordingly by moving them into the train, dev, test
2. While moving also check the csl annotation: combine the sentence to save them and then also save the video name and the number of frames 
3. save the new annotations
'''

def read_CSL_annotations(CSL_annot_path):
    
    with open(CSL_annot_path, 'rb') as f:
        data = pickle.load(f)

    return data

def save_results(video_info_dict, output_path):
    # Save the dictionary as a pkl file
    with open(output_path, 'wb') as pkl_file:
        pickle.dump(video_info_dict, pkl_file)

def find_entry_by_name(data, name):
    # Iterate through the list of dictionaries in 'info'
    for entry in data['info']:
        if entry['name'] == name:
            return entry
    return None

def split_and_move_data(CSL_annot_path, split_txt_path, img_path, output_label_path):
    # initialise the dictionary for label saving later on

    info = read_CSL_annotations(CSL_annot_path)

    # read text file as a dataframe
    split_df = pd.read_csv(split_txt_path, delimiter='|')  # Adjust 'delimiter' as needed

    # create the train, dev, test folders 
    all_info_dict = {}
    for ty in ['train', 'dev', 'test']: 
        all_info_dict[ty] = {'info': [], 
                             'max_length':0}
    
    # Define the target directories for train, dev, and test
    train_dir = os.path.join(img_path, 'train')
    dev_dir = os.path.join(img_path, 'dev')
    test_dir = os.path.join(img_path, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    print(train_dir)
    # Iterate over each row in the DataFrame
    for index, row in split_df.iterrows():
        folder_name = row['name']
        set_type = row['split']
        row_info = find_entry_by_name(info, folder_name) ## might have been wrong here to assume index which led me to the wrong translation
        text =''.join(row_info['label_char']) 
        # Determine the target directory based on the set type
        if set_type == 'train':
            target_dir = train_dir
        elif set_type == 'dev':
            target_dir = dev_dir
        elif set_type == 'test':
            target_dir = test_dir
        else:
            #print(f"Unknown set type: {set_type} for folder {folder_name}")
            continue
        
        # move the folder accordingly 
        # Construct full paths
        src_path = os.path.join(img_path, folder_name)
        dest_path = os.path.join(target_dir, folder_name)
        

        # Move the folder
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
            print(f"Moved {folder_name} to {dest_path}")
        else:

            print(f"Folder {folder_name} does not exist in {img_path}")

        #Count the number of frames (png files) in the video folder
        try:
            frame_count = len([f for f in os.listdir(dest_path) if f.endswith('.jpg')]) # note that the CSL daily dataset is in jpg
            
            if frame_count>all_info_dict[set_type]['max_length']: 
                all_info_dict[set_type]['max_length']= frame_count

            all_info_dict[set_type]['info'].append({'name': folder_name, 
                                                'translation': text, 
                                                'length':frame_count}) # frame count is 
        except FileNotFoundError:
            print(f"file {folder_name} not found")   # In case the folder doesn't exist
        
    
    # save the dictionaries
    for ty in ['train', 'dev', 'test']: 
        to_save_dict = all_info_dict[ty]
        # save file
        save_results(to_save_dict, os.path.join(output_label_path, f'labels_{ty}.pkl'))


def main():
    output_label_path= '../../CSL-Daily/sentence_label/processed'
    # os.makedirs(output_label_path, exist_ok=True)
    img_path = '../../CSL-Daily/sentence/frames_512x512'
    CSL_annot_path = '../../CSL-Daily/sentence_label/csl2020ct_v2.pkl'
    split_txt_path= '../../CSL-Daily/sentence_label/split_1.txt'
    split_and_move_data(CSL_annot_path, split_txt_path, img_path, output_label_path)
    


if __name__ == "__main__":
    '''
    Running this file will process the labels from the csv files 
    The data will process the following: 
    - the folder name containing the video images 
    - the translation of the video 
    - the number of frames in case there is the need to add padding frames 
    '''
    main()