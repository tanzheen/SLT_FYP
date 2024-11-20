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
import math 
import cv2
from transformers import AutoImageProcessor, AutoProcessor

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

class FaceDetectorYunet():
    def __init__(self,
                  model_path='./face_detection_yunet_2023mar.onnx',
                  img_size=(300, 300),
                  threshold=0.5):
        self.model_path = model_path
        self.img_size = img_size
        self.fd = cv2.FaceDetectorYN_create(str(model_path),
                                            "",
                                            img_size,
                                            score_threshold=threshold)

    def draw_faces(self,
                   image,
                   faces,
                   draw_landmarks=False,
                   show_confidence=False):
        for face in faces:
            color = (0, 0, 255)
            thickness = 2
            cv2.rectangle(image, (face['x1'], face['y1']), (face['x2'], face['y2']), color, thickness, cv2.LINE_AA)

            if draw_landmarks:
                landmarks = face['landmarks']
                for landmark in landmarks:
                    radius = 5
                    thickness = -1
                    cv2.circle(image, landmark, radius, color, thickness, cv2.LINE_AA)

            if show_confidence:
                confidence = face['confidence']
                confidence = "{:.2f}".format(confidence)
                position = (face['x1'], face['y1'] - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.5
                thickness = 2
                cv2.putText(image, confidence, position, font, scale, color, thickness, cv2.LINE_AA)
        return image

    def scale_coords(self, image, prediction):
        ih, iw = image.shape[:2]
        rw, rh = self.img_size
        a = np.array([
                (prediction['x1'], prediction['y1']),
                (prediction['x1'] + prediction['x2'], prediction['y1'] + prediction['y2'])
                    ])
        b = np.array([iw/rw, ih/rh])
        c = a * b
        prediction['img_width'] = iw
        prediction['img_height'] = ih
        prediction['x1'] = int(c[0,0].round())
        prediction['x2'] = int(c[1,0].round())
        prediction['y1'] = int(c[0,1].round())
        prediction['y2'] = int(c[1,1].round())
        prediction['face_width'] = (c[1,0] - c[0,0])
        prediction['face_height'] = (c[1,1] - c[0,1])
        # prediction['face_width'] = prediction['x2'] - prediction['x1']
        # prediction['face_height'] = prediction['y2'] - prediction['y1']
        prediction['area'] = prediction['face_width'] * prediction['face_height']
        prediction['pct_of_frame'] = prediction['area']/(prediction['img_width'] * prediction['img_height'])
        return prediction

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(str(image))
        img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        img = cv2.resize(img, self.img_size)
        self.fd.setInputSize(self.img_size)
        _, faces = self.fd.detect(img)
        if faces is None:
            return None
        else:
            predictions = self.parse_predictions(image, faces)
            return predictions

    def parse_predictions(self,
                          image,
                          faces):
        data = []
        for num, face in enumerate(list(faces)):
            x1, y1, x2, y2 = list(map(int, face[:4]))
            landmarks = list(map(int, face[4:len(face)-1]))
            landmarks = np.array_split(landmarks, len(landmarks) / 2)
            positions = ['left_eye', 'right_eye', 'nose', 'right_mouth', 'left_mouth']
            landmarks = {positions[num]: x.tolist() for num, x in enumerate(landmarks)}
            confidence = face[-1]
            datum = {'x1': x1,
                     'y1': y1,
                     'x2': x2,
                     'y2': y2,
                     'face_num': num,
                     'landmarks': landmarks,
                     'confidence': confidence,
                     'model': 'yunet'}
            d = self.scale_coords(image, datum)
            data.append(d)
        return data


class SignVideoDataset(Dataset): 
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
    """

    def __init__(self, tokenizer, config, phase, training_refurbish = True): 
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
        sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
        self.seq = va.Sequential([
            # va.RandomCrop(size=(240, 180)), # randomly crop video with a size of (240 x 180)
            # va.RandomRotate(degrees=10), # randomly rotates the video with a degree randomly choosen from [-10, 10]  
            sometimes(va.RandomRotate(30)),
            sometimes(va.RandomResize(0.2)),
            # va.RandomCrop(size=(256, 256)),
            sometimes(va.RandomTranslate(x=10, y=10))]
        )
        self.facial_processor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        self.image_processor =  AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-large-finetuned-kinetics")
        self.transformer_type = config.model.transformer_type 

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
        if self.transformer_type == 'causal': 
            text_label = " " + text_label
        length = file['length']  # Number of frames in the video.
        
        # Load the images for the video and pad as necessary.
        img_sample, path_lst= self.load_imgs(os.path.join(self.img_path, name))
        if len(img_sample) != length: # Number of PIL Images not matching indicated length
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
            batch_image: A list of PIL images to be processed by the AutoImageProcessor 
            paths: A list of the paths of the images in the batch_image
        """
        # Set up dataset-specific transformations.
        # No normalization applied (all values remain between 0 and 1).
    
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
       
        batch_image = []
        batch_face = []
        # Load each image, apply transformations, and crop if necessary.
        for i, img_path in enumerate(paths):
            img_path = os.path.join(dir_path, img_path)
            img = Image.open(img_path).convert("RGB")
            
            batch_image.append(img)

        
        return batch_image , paths 

    def extract_faces(self, img_sample, face_detector):
        """
        Extract faces from the images using the provided face detector.
        
        Args:
            img_sample (list): A list of PIL images to extract faces from.
            face_detector: The face detector to use for
        Returns:
            list: A list of cropped face images. (same length as img_sample)
        """ 
        transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224))
                ])
        plots= [] 
        faces_PIL = []
        for img in img_sample:
            img_width , img_height = img.size
            img = np.array(img)
            faces = face_detector.detect(img)
            if len(faces)>1: # Sometimes hands might be detected as faces, the face will have the lowest y2
                faces = [min(faces, key=lambda x: x['y2'])]
            face = faces[0]
            x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
        # Calculate width and height of the bounding box
        width = x2 - x1
        height = y2 - y1

        # Determine the size of the square side (maximum of width and height)
        side_length = max(width, height)

        # Calculate the center of the original bounding box
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        # Adjust the bounding box to make it a square, centered around the original box
        new_x1 = center_x - side_length // 2
        new_y1 = center_y - side_length // 2
        new_x2 = new_x1 + side_length
        new_y2 = new_y1 + side_length

        # Ensure the new bounding box stays within the image boundaries
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(img_width, new_x2)
        new_y2 = min(img_height, new_y2)
        
        # Crop the face from the image
        cropped_face = img[new_y1:new_y2, new_x1:new_x2]
        plots.append(cropped_face)
        # Convert the cropped face to a tensor
        face_tensor = transform(cropped_face) ## need to convert back to PIL image after cropping
        faces_PIL.append(face_tensor)

        return faces_PIL, plots
    

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

        Pad the images to the same length, and tokenizes the text labels.

        Args: 
            batch (list): A list of individual samples to batch together. 
                - a list of tuples; each entry is a tuple containing the name, img_sample and tgt_sample 

        Returns: 
            src_input: A dictionary containing the batched data with the following keys:
                - 'name_batch': The names of the videos.
                - 'emo_batch': The batched cropped faces.
                - 'image_batch': The batched images.
                - 'clip_batch': list of clips; each clip is a list containing 16 images  
                - 'num_frames_batch': list of number of frames in each video.
                - 'num_clips_batch': list of number of clips in each video.
                - 'tgt_batch': The batched text labels.
        """
        name_batch, emo_batch, image_batch, clip_batch, num_frames_batch, num_clips_batch, tgt_batch = [], [], [], [], [], [], []
        for name, img_sample, text_label in batch: 
            name_batch.append(name)
            tgt_batch.append(text_label)
            

            # Cut up video properly to a list of list of 16 images
            
            num_clips = math.ceil((len(img_sample)-8) / 8)
            num_frames = num_clips * 8 +8

            # if not enough frames, repeat the last frame
            img_sample.extend([img_sample[-1]]*(num_frames-len(img_sample)))
            
            # chop up the video into clips of 16 frames with 8 frame overlap
            clip_sample = [img_sample[i:i+16] for i in range(0, len(img_sample), 8) 
                           if len(img_sample[i:i+16])==16 if len(img_sample)-i>8]


            # Add the clips and images to the batch
            clip_batch.extend(clip_sample)
            num_clips_batch.append(num_clips)
            num_frames_batch.append(num_frames)
            image_batch.extend(img_sample)

            # Extract facial features 
            face_detector = FaceDetectorYunet()
            faces_PIL, plots = self.extract_faces(img_sample, face_detector)

            emo_batch.extend(faces_PIL)

        # Transform images using AutoProcessors 
        clip_batch = self.video_processor(clip_batch, return_tensors="pt")
        image_batch = self.image_processor(image_batch, return_tensors="pt")
        emo_batch = self.facial_processor(emo_batch, return_tensors="pt")
        
        # After the batches are aranged in a sequence, 
        # we can pad the clips and images for the adapter later to process them for LLM 
        ## Still must settle attention masks and padding for the clips, However, we do that directly in the model


        # Tokenize the text labels
        tgt_input = self.tokenizer(tgt_batch, return_tensors="pt", padding=True, truncation=True)
        src_input = {
            'name_batch': name_batch,
            'emo_batch': emo_batch,
            'image_batch': image_batch,
            'clip_batch': clip_batch,
            'num_frames_batch': num_frames_batch,
            'num_clips_batch': num_clips_batch,
            'tgt_batch': tgt_input
        }

        return src_input, tgt_input



        


        
    
