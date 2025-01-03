from transformers import AutoModelForCausalLM,  AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer , AutoModelForZeroShotImageClassification, VideoMAEForVideoClassification, AutoModelForSeq2SeqLM
import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pad_sequence
import copy 
from peft import LoraConfig, get_peft_model, TaskType
import os 
import math
from typing import Union, Callable, Dict, Optional



def create_mask(seq_lengths: list, device="cpu"):
    """
    Creates a mask tensor based on sequence lengths.

    Args:
        seq_lengths (list): A list of sequence lengths.
        device (str): The device to create the mask on.

    Returns:
        torch.Tensor: A mask tensor.
    """
    max_len = max(seq_lengths)
    mask = torch.arange(max_len, device=device)[None, :] < torch.tensor(seq_lengths, device=device)[:, None]
    return mask.to(torch.bool)


class Emo_extractor(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.emotion_model = AutoModelForImageClassification.from_pretrained(config.model.emotion_model).to('cuda')
        # Freeze all parameters in the emotion model

        # for param in self.emotion_model.vit.parameters():
        #     param.requires_grad = False

    def forward(self, faces): 
        with torch.no_grad():
            #print("faces shape: ", faces.pixel_values.shape)
            embeddings = self.emotion_model.vit.embeddings(faces.pixel_values.cuda())  # Obtain embeddings from the ViT backbone
            encoder_outputs = self.emotion_model.vit.encoder(embeddings)  # Pass through the encoder layers
            features = self.emotion_model.vit.layernorm(encoder_outputs[0][:, 0, :])  # Extract [CLS] token feature
        
        #print(f"emo features shape: {features.shape}")
        return features # hidden size = 768
    
class Spatio_extractor(nn.Module): 
    def __init__(self, config): 
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.spatio_extractor = AutoModelForZeroShotImageClassification.from_pretrained(config.model.spatio_model).to(device).get_image_features
        ## Settle S2 wrapper too

        # Freeze all parameters in the spatio extractor model
        # for param in self.spatio_extractor.parameters():
        #     param.requires_grad = False
    
    def forward(self, images): 
        with torch.no_grad():
            spatial_features = self.spatio_extractor(images.pixel_values.cuda())  # Shape: [batch_size, num_patches, feature_dim]
        return spatial_features # hidden size = 768

class Motion_extractor(nn.Module):
    def __init__(self, config): 
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.motion_model = VideoMAEForVideoClassification.from_pretrained(config.model.motion_model).to(device)
        # Freeze all parameters in the motion model
        # for param in self.motion_model.parameters():
        #     param.requires_grad = False

    def forward(self, video_tensor): 
        with torch.no_grad():
            # Forward pass up to the encoder and normalization layer
            features = self.motion_model.videomae(video_tensor.pixel_values.cuda()).last_hidden_state  # Shape: [batch_size, num_patches, feature_dim]
            # Apply the final LayerNorm before the classifier
            features = self.motion_model.fc_norm(features.mean(dim=1))  # Shape: [batch_size, 1024]
        return features 
    

class SignAdaptor (nn.Module): 
    def __init__(self, config): 
        super().__init__()
        self.config = config
        
    def forward(self, src_input, pad_idx): 
        # batch_size, seq_len, hidden_size
        if self.config.training.token_usage: # use the other function is we are using the saved embeddings 
            return self.tokenised_forward(src_input, pad_idx = pad_idx)

        name_batch = src_input['name_batch']
        emo_batch = src_input['emo_batch']
        image_batch = src_input['image_batch']
        clip_batch = src_input['clip_batch']
        num_frames_batch = src_input['num_frames_batch']
        num_clips_batch = src_input['num_clips_batch']
        
        embeddings_batch = []
        src_length = [] 
        start_clips = 0 
        start_frames = 0 
        for i , (num_clips, num_frames) in enumerate(zip(num_clips_batch, num_frames_batch)): 
            name = name_batch[i]
            #print(f"num_clips: {num_clips}, num_frames: {num_frames}, repeat factor: {repeat_factor}")
            # Sectioning out the frames after the visual features extractor
            end_clips = start_clips + num_clips
            end_frames = start_frames  + num_frames
            current_images = image_batch[start_frames :end_frames]
            current_emos = emo_batch[start_frames :end_frames]
            current_clips = clip_batch [start_clips :end_clips]

            # repeat the number of clips accordingly (each frame will be concatenated to a similar clip a few times)
            repeat_factor = math.floor(num_frames/num_clips) # number of times to repeat clips 
            expanded_clips = current_clips.unsqueeze(1).repeat(1, repeat_factor, 1)
            expanded_clips = expanded_clips.view(-1, current_clips.size(1))


            #repeat the last clip again if required
            if repeat_factor*num_clips < num_frames:
                last_clip = current_clips[-1].unsqueeze(0).repeat(num_frames - repeat_factor*num_clips, 1)
                expanded_clips = torch.cat([expanded_clips, last_clip], dim=0)
            
            #print(f"shape of current_emos: {current_emos.shape}, shape of current_images: {current_images.shape}, shape of expanded_clips: {expanded_clips.shape}")
            # Concat the features along the dimensions
            combined_features = torch.cat([ current_images, expanded_clips], dim=1) 
            embeddings_batch.append(combined_features)
            src_length.append(combined_features.size(0))

            start_clips = end_clips
            start_frames = end_frames
        
        x = pad_sequence(embeddings_batch, batch_first=True, padding_value=pad_idx)
        return x , src_length
    
    def tokenised_forward(self, src_input, pad_idx):
        name_batch = src_input['name_batch']
        emo_batch = src_input['emo_batch']
        image_batch = src_input['image_batch']
        clip_batch = src_input['clip_batch']
        num_frames_batch = src_input['num_frames_batch']
        num_clips_batch = src_input['num_clips_batch']
        
        embeddings_batch = [] 
        src_length = [] 
        
        for i , (num_clips, num_frames) in enumerate(zip(num_clips_batch, num_frames_batch)):
            # if its tokenised already, then don't need to batch out the clips and frames with start and end indices

            current_clips = clip_batch[i]
            current_images = image_batch[i]
            current_emos = emo_batch[i]

            # repeat the number of clips accordingly (each frame will be concatenated to a similar clip a few times)
            repeat_factor = math.floor(num_frames/num_clips) # number of times to repeat clips 
            expanded_clips = current_clips.unsqueeze(1).repeat(1, repeat_factor, 1)
            expanded_clips = expanded_clips.view(-1, current_clips.size(1))

            #repeat the last clip again if required
            if repeat_factor*num_clips < num_frames:
                last_clip = current_clips[-1].unsqueeze(0).repeat(num_frames - repeat_factor*num_clips, 1)
                expanded_clips = torch.cat([expanded_clips, last_clip], dim=0)
            
            #print(f"shape of current_emos: {current_emos.shape}, shape of current_images: {current_images.shape}, shape of expanded_clips: {expanded_clips.shape}")
            # Concat the features along the dimensions
            combined_features = torch.cat([current_images, expanded_clips], dim=1) 
            embeddings_batch.append(combined_features)
            src_length.append(combined_features.size(0))

        x = pad_sequence(embeddings_batch, batch_first=True, padding_value=pad_idx)
        return x , src_length
    
class SignAdaptorV2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatio_proj = nn.Linear(self.config.model.spatio_hiddim,
                                     self.config.model.llm_hiddim)
        self.motion_proj = nn.Linear(self.config.model.motion_hiddim,
                                     self.config.model.llm_hiddim)

    def forward(self, src_input, pad_idx):
        # batch_size, seq_len, hidden_size
        if self.config.training.token_usage:
            return self.tokenised_forward(src_input, pad_idx=pad_idx)
        
        name_batch = src_input['name_batch']
        emo_batch = src_input['emo_batch']
        image_batch = src_input['image_batch']
        clip_batch = src_input['clip_batch']
        num_frames_batch = src_input['num_frames_batch']
        num_clips_batch = src_input['num_clips_batch']

        embeddings_batch = []
        src_length = []
        start_clips = 0 
        start_frames = 0 

        ## If not tokenised, then batch out the clips and frames with start and end indices
        for i , (num_clips, num_frames) in enumerate(zip(num_clips_batch, num_frames_batch)): 
            name = name_batch[i]
            #print(f"num_clips: {num_clips}, num_frames: {num_frames}, repeat factor: {repeat_factor}")
            # Sectioning out the frames after the visual features extractor
            end_clips = start_clips + num_clips
            end_frames = start_frames  + num_frames
            current_images = image_batch[start_frames :end_frames]
            current_emos = emo_batch[start_frames :end_frames]
            current_clips = clip_batch [start_clips :end_clips]

            # However, this time we do not concat the features on dim 1, but rather on dim 0
            # Therefore, don't need to repeat the clips

            # Go through the different projection layers
            spatio_emo = torch.cat([current_images], dim=1)
            spatio_proj = self.spatio_proj(spatio_emo)
            motion_proj = self.motion_proj(current_clips)

            # Concat the features along the temporal dimensions
            combined_features = torch.cat([  spatio_proj, motion_proj], dim=0)

            embeddings_batch.append(combined_features)
            src_length.append(combined_features.size(0))

            start_clips = end_clips
            start_frames = end_frames
    
        x = pad_sequence(embeddings_batch, batch_first=True, padding_value=pad_idx)
        return x , src_length
    
    def tokenised_forward(self, src_input, pad_idx = 0):
        name_batch = src_input['name_batch']
        emo_batch = src_input['emo_batch']
        image_batch = src_input['image_batch']
        clip_batch = src_input['clip_batch']
        num_frames_batch = src_input['num_frames_batch']
        num_clips_batch = src_input['num_clips_batch']
        
        embeddings_batch = [] 
        src_len =  []
        
        for i , (num_clips, num_frames) in enumerate(zip(num_clips_batch, num_frames_batch)):
            # if its tokenised already, then don't need to batch out the clips and frames with start and end indices

            current_clips = clip_batch[i]
            current_images = image_batch[i]
            current_emos = emo_batch[i]

            # Go through the different projection layers
            spatio_emo = torch.cat([current_images], dim=1)
            spatio_proj = self.spatio_proj(spatio_emo)
            motion_proj = self.motion_proj(current_clips)
            
            # Concat the features along the dimensions
            combined_features = torch.cat([spatio_proj,  motion_proj], dim=0)
            embeddings_batch.append(combined_features)
            src_len.append(combined_features.size(0))


        x = pad_sequence(embeddings_batch, batch_first=True, padding_value=pad_idx)
        return x , src_len

    
    def save_embeddings(self, src_input, save_path): 
        # batch_size, seq_len, hidden_size
        name_batch = src_input['name_batch']
        emo_batch = src_input['emo_batch']
        image_batch = src_input['image_batch']
        clip_batch = src_input['clip_batch']
        num_frames_batch = src_input['num_frames_batch']
        num_clips_batch = src_input['num_clips_batch']
        
        embeddings_batch = []
        start_clips = 0 
        start_frames = 0 
        for i , (num_clips, num_frames) in enumerate(zip(num_clips_batch, num_frames_batch)): 
            name = name_batch[i]
            #print(f"num_clips: {num_clips}, num_frames: {num_frames}, repeat factor: {repeat_factor}")
            end_clips = start_clips + num_clips
            end_frames = start_frames  + num_frames
            current_images = image_batch[start_frames :end_frames]
            current_emos = emo_batch[start_frames :end_frames]
            current_clips = clip_batch [start_clips :end_clips]

            final_save_path = os.path.join(save_path, name)

            emo_save_path = os.path.join(final_save_path, "emo_embeddings")
            image_save_path = os.path.join(final_save_path, "image_embeddings")
            clip_save_path = os.path.join(final_save_path, "clip_embeddings")
            # make directories if does not exist 
            os.makedirs(emo_save_path, exist_ok=True)
            os.makedirs(image_save_path, exist_ok=True)
            os.makedirs(clip_save_path, exist_ok=True)

            torch.save(current_emos, os.path.join(emo_save_path, "emo.pt"))
            torch.save(current_images, os.path.join(image_save_path, "image.pt"))
            torch.save(current_clips, os.path.join(clip_save_path, "clip.pt"))
            print(f"Saved embeddings for {name}")



class TemporalConv(nn.Module): 
    def __init__(self, input_size, hidden_size, conv_type=2): 
        super(TemporalConv, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', "P2"]
        elif self.conv_type == 4:
            self.kernel_size = ['K5', 'K5']
        elif self.conv_type == 5:
            self.kernel_size = ['K5', "P2", 'K5']
        elif self.conv_type == 6:
            self.kernel_size = ["P2", 'K5', 'K5']
        elif self.conv_type == 7:
            self.kernel_size = ["P2", 'K5', "P2", 'K5']
        elif self.conv_type == 8:
            self.kernel_size = ["P2", "P2", 'K5', 'K5']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 or self.conv_type == 6 and layer_idx == 1 or self.conv_type == 7 and layer_idx == 1 or self.conv_type == 8 and layer_idx == 2 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
            
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)

    def update_lgt(self, lgt):
        lgt = torch.tensor(lgt)
        feat_len = copy.deepcopy(lgt)  # Deep copy the input
        for ks in self.kernel_size:
            if ks[0] == 'P':
                feat_len = torch.div(feat_len, 2)
            else:
                feat_len -= int(ks[1]) - 1
        feat_len = feat_len.tolist()
        feat_len = [int(x) for x in feat_len]
        return feat_len

    def forward(self, frame_feat, lgt):
        # need to make sure it is permuted correctly
        # frame_feat: batch_size, hidden_size, seq_len
        visual_feat = self.temporal_conv(frame_feat.permute(0,2,1))
        lgt = self.update_lgt(lgt)
        attn_masks = create_mask(seq_lengths = lgt)
        return {
            "visual_feat": visual_feat.permute(0, 2, 1),
            "feat_len": lgt,
            "src_attn": attn_masks
        }
    
class MLP(nn.Module):
    def __init__(
        self,
        config,
        in_features,
        out_features,
        hidden_features=None,
        act_layer=nn.GELU,
        drop=0.1
    ):
        ''''''
        super().__init__()
        self.config = config
        self.in_features= in_features
        self.hidden_features = hidden_features or self.in_features
        self.out_features = out_features
        print(f"MLP: self.in_features: {self.in_features}, hidden_features: {self.hidden_features}")
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class BaseModel(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def save_pretrained_weight(
        self,
        save_directory: Union[str, os.PathLike],
        save_function: Callable = None,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Saves a model and its configuration file to a directory.

        Args:
            save_directory: A string or os.PathLike, directory to which to save. 
                Will be created if it doesn't exist.
            save_function: A Callable function, the function to use to save the state dictionary.
                Useful on distributed training like TPUs when one need to replace `torch.save` by
                another method. Can be configured with the environment variable `DIFFUSERS_SAVE_MODE`.
            state_dict: A dictionary from str to torch.Tensor, the state dictionary to save.
                If `None`, the model's state dictionary will be saved.
        """
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        model_to_save = self

        if state_dict is None:
            state_dict = model_to_save.state_dict()
        weights_name = "pytorch_model.bin"

        save_function(state_dict, os.path.join(save_directory, weights_name))

        print(f"Model weights saved in {os.path.join(save_directory, weights_name)}")

    def load_pretrained_weight(
        self,
        pretrained_model_path: Union[str, os.PathLike],
        strict_loading: bool = True,
        torch_dtype: Optional[torch.dtype] = None
    ):
        r"""Instantiates a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you should first set it back in training mode with `model.train()`.

        Args:
            pretrained_model_path: A string or os.PathLike, a path to a *directory* or *file* containing model weights.

        Raises:
            ValueError: If pretrained_model_path does not exist.
        """
        # If pretrained_model_path is a file, set model_file to this file.
        if os.path.isfile(pretrained_model_path):
            model_file = pretrained_model_path
        # If pretrained_model_path is a directory, set model_file to the path of the 
        # file "pytorch_model.bin" in this directory.
        elif os.path.isdir(pretrained_model_path):
            pretrained_model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.isfile(pretrained_model_path):
                model_file = pretrained_model_path
            else:
                raise ValueError(f"{pretrained_model_path} does not exist")
        else:
            raise ValueError(f"{pretrained_model_path} does not exist")

        # Load model state from checkpoint.
        checkpoint = torch.load(model_file, map_location="cpu")
        # Load state dictionary into self.
        msg = self.load_state_dict(checkpoint, strict=strict_loading)
        # Print information about loading weights.
        print(f"loading weight from {model_file}, msg: {msg}")
        # If torch_dtype is specified and is a valid torch.dtype, convert self to this dtype.
        if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
            raise ValueError(
                f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
            )
        elif torch_dtype is not None:
            self.to(torch_dtype)

        # Set model in evaluation mode to deactivate DropOut modules by default.
        self.eval()



class SpaEMo(BaseModel):

    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.combined_dim =  config.model.spatio_hiddim + config.model.motion_hiddim
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))
        self.emo_extractor = Emo_extractor(config)
        self.spatio_extractor = Spatio_extractor(config)
        self.motion_extractor = Motion_extractor(config)
        
        if self.config.model.adaptor_type == 1:
            self.adaptor =  SignAdaptor(config)
            self.tconv = TemporalConv(input_size = self.combined_dim, hidden_size=self.combined_dim)
            self.mlp = MLP(config, in_features=self.combined_dim, out_features=self.config.model.llm_hiddim)

        elif self.config.model.adaptor_type == 2:
            self.adaptor = SignAdaptorV2(config)  
            self.tconv = TemporalConv(input_size = self.config.model.llm_hiddim, hidden_size=self.config.model.llm_hiddim)
            self.mlp = MLP(config, in_features=self.config.model.llm_hiddim, out_features=self.config.model.llm_hiddim)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer)
        if self.config.model.transformer_type == 'causal':
            self.lora_config =  LoraConfig(
                                        task_type=TaskType.CAUSAL_LM,   # Type of task
                                        inference_mode=False,    # Enable training
                                        r=8,                     # Low-rank dimension
                                        lora_alpha=16,           # Scaling factor
                                        lora_dropout=0.1,        # Dropout rate for LoRA
                                        target_modules=["q_proj", "v_proj"]  # Target modules (Gemma specific)
                                        )
            model = AutoModelForCausalLM.from_pretrained(config.model.llm)

        elif self.config.model.transformer_type == 'seq2seq':
            self.lora_config = LoraConfig(
                                task_type=TaskType.SEQ_2_SEQ_LM,  # Task type for sequence-to-sequence learning
                                inference_mode=False,
                                r=16,  # Low-rank dimension
                                lora_alpha=32,  # Scaling factor
                                lora_dropout=0.1,  # Dropout probability
                                target_modules=["q_proj", "v_proj"]
                                )
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model.llm)
        
        elif self.config.model.transformer_type == 'seq2seq_v2': 
            self.lora_config =LoraConfig(
                                        r=16,
                                        lora_alpha=32,
                                        target_modules=["q", "v"],
                                        lora_dropout=0.1,
                                        bias="none",
                                        task_type=TaskType.SEQ_2_SEQ_LM
                                        )
            model = AutoModelForSeq2SeqLM.from_pretrained(config.model.llm)

        self.lora_model = model 
        # self.lora_model = get_peft_model(model, self.lora_config)
        # self.lora_model.print_trainable_parameters()

    def create_causal_inputs(self, visual_feats ,visual_attn,  tgt_ids,tokenizer , prompt): 
        '''This function is only for causal language modeling.'''
        # The space will be added by the model itself
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_len = len(prompt_ids["input_ids"][0])
        #print(f"prompt_len: {prompt_len}")
        #print(f"visual feats shape: {visual_feats.shape}")

        prompt_embeds = self.lora_model.get_input_embeddings()(prompt_ids["input_ids"]).squeeze()
        tgt_embeds = self.lora_model.get_input_embeddings()(tgt_ids["input_ids"])

        #print(f"tgt embeds: {tgt_embeds.shape}")
        tgt_attn = tgt_ids["attention_mask"]
        #print(f"tgt attn: {tgt_attn.shape}")
        new_input_embeds = [] 
        new_labels = []

        for i in range(tgt_embeds.shape[0]): # batch size
            curr_vis_feats = visual_feats[i] # take the current visual features
            curr_vis_attn = visual_attn[i] # take the current visual attention mask
            curr_vis_feats = curr_vis_feats[curr_vis_attn==1] # only take the visual features that are attended to

            curr_vis_len = curr_vis_feats.shape[0] # get the length of the visual features

            curr_tgt_embeds = tgt_embeds[i] # take the current target embeddings

            curr_tgt_feats = curr_tgt_embeds[tgt_attn[i]==1][1:] # only take the target embeddings that are attended to, remove the bos token 
    
            combined_embeds = torch.cat((prompt_embeds, curr_vis_feats, curr_tgt_feats), dim = 0) #Concat all the embeddings
            #print(f"combined_embeds: {combined_embeds}")
            new_input_embeds.append(combined_embeds) 
            negate_tgt = torch.full((1, prompt_len + curr_vis_len), -100) # create the -100 labels for the model (only the target text is not -100)
            #print(f"negate_tgt: {negate_tgt.shape}")
            labels =torch.cat([negate_tgt,  tgt_ids["input_ids"][i][tgt_attn[i]==1][1:].clone().unsqueeze(0)], dim =1).permute(1,0) # Concat both the -100s and the target text

            new_labels.append(labels) # append the labels
            assert labels.shape[0] == len(combined_embeds), f"len labels: {labels.shape} vs len combined_embeds: {combined_embeds.shape}" 
            # assert the length of the labels is the same as the combined embeddings
        
        # perform padding for the batch before returning
        new_input_embeds = pad_sequence(new_input_embeds, batch_first=True, padding_value=0)
        #print("HERE", [labels.shape for labels in new_labels])
        new_labels = pad_sequence(new_labels, batch_first=True, padding_value=-100).squeeze()
        new_labels[new_labels==0]=-100
        return new_input_embeds, new_labels

    def forward(self, src_input,tgt_input):
        '''
        2 types of forward pass: 
        a. CausalLM: need to prep the visual embeddings and label tokens such that they are in consecutive order
        b. Seq2Seq: need to prep the visual embeddings and label tokens separately, might also need to prepare for the attention mask
        '''

        # This part if tokens are not saved
        if self.config.training.token_usage == False:
            src_input['emo_batch'] = self.emo_extractor(src_input['emo_batch'])
            src_input['image_batch']= self.spatio_extractor(src_input['image_batch'])
            src_input['clip_batch']= self.motion_extractor(src_input['clip_batch'])

        # Create a new dictionary for transferring of results
        emb_input = {'name_batch': src_input['name_batch'],
                     'emo_batch': src_input['emo_batch'], 
                     'image_batch': src_input['image_batch'], 
                     'clip_batch': src_input['clip_batch'], 
                     'num_clips_batch': src_input['num_clips_batch'],
                     'num_frames_batch': src_input['num_frames_batch'],}
        prompt = self.config.dataset.prompt
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        #print(f"prompt_len: {prompt_len}")
        #print(f"visual feats shape: {visual_feats.shape}")

        #prompt_embeds = self.lora_model.get_input_embeddings()(prompt_ids["input_ids"]).squeeze()
        combined_features, src_length = self.adaptor(emb_input, self.tokenizer.pad_token_id)

        # if tokenised, just need arrangement and then the visual embeddings go straight right into the temporal conv


        temporal_features = self.tconv(combined_features.cuda(), src_length)
        vis_features = self.mlp(temporal_features['visual_feat'].cuda())
        if self.config.model.transformer_type == 'causal': 
            new_input_embeds , new_labels = self.create_causal_inputs(vis_features, temporal_features['src_attn'], tgt_input, 
                                                                      self.tokenizer, self.config.dataset.prompt )
            outputs = self.lora_model(inputs_embeds = new_input_embeds, labels = new_labels)
            
        elif self.config.model.transformer_type == 'seq2seq' or self.config.model.transformer_type == 'seq2seq_v2':
            new_input_embeds = vis_features
            new_labels = tgt_input["input_ids"]
            label_attn = tgt_input["attention_mask"]
            new_labels = new_labels.masked_fill(new_labels == self.tokenizer.pad_token_id, -100)
            outputs = self.lora_model(inputs_embeds = new_input_embeds.cuda(),
                                      attention_mask= temporal_features['src_attn'].cuda(), 
                                      decoder_attention_mask = label_attn.cuda(),
                                       labels = new_labels.cuda())

        
        return outputs
    
    def save_embeddings(self, src_input, phase = None ):
        assert phase is not None, "Please specify the phase of the model"
        emo_features = self.emo_extractor(src_input['emo_batch'])
        spatio_features = self.spatio_extractor(src_input['image_batch'])
        motion_features = self.motion_extractor(src_input['clip_batch'])

        # Create a new dictionary for transferring of results
        emb_input = {'name_batch': src_input['name_batch'],
                     'emo_batch': emo_features, 
                     'image_batch': spatio_features, 
                     'clip_batch': motion_features, 
                     'num_clips_batch': src_input['num_clips_batch'],
                     'num_frames_batch': src_input['num_frames_batch'],}
        save_path = os.path.join(self.config.dataset.img_path, phase)
        self.adaptor.save_embeddings(emb_input, save_path)
    
    def create_generation_inputs(self, visual_feats ,visual_attn,  tokenizer , prompt): 
        # The space will be added by the model itself
        prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        prompt_len = len(prompt_ids["input_ids"][0])
        #print(f"prompt_len: {prompt_len}")
        #print(f"visual feats shape: {visual_feats.shape}")
        prompt_embeds = self.lora_model.get_input_embeddings()(prompt_ids["input_ids"]).squeeze()

        #print(f"tgt embeds: {tgt_embeds.shape}")
        #print(f"tgt attn: {tgt_attn.shape}")
        new_input_embeds = [] 


        for i in range(visual_feats.shape[0]): # batch size
            curr_vis_feats = visual_feats[i] # take the current visual features
            curr_vis_attn = visual_attn[i] # take the current visual attention mask
            curr_vis_feats = curr_vis_feats[curr_vis_attn==1] # only take the visual features that are attended to

            curr_vis_len = curr_vis_feats.shape[0] # get the length of the visual features
    
            combined_embeds = torch.cat((prompt_embeds, curr_vis_feats), dim = 0) #Concat all the embeddings
            #print(f"combined_embeds: {combined_embeds}")
            new_input_embeds.append(combined_embeds) 
       
            # assert the length of the labels is the same as the combined embeddings
        
        # perform padding for the batch before returning
        new_input_embeds = torch.nn.utils.rnn.pad_sequence(new_input_embeds, batch_first=True, padding_value=0)
        #print("HERE", [labels.shape for labels in new_labels])

        return new_input_embeds

    def generate(self, src_input, num_beams =5, max_length = 50): 
        with torch.no_grad(): 
            if self.config.training.token_usage == False:
                src_input['emo_batch'] = self.emo_extractor(src_input['emo_batch'])
                src_input['image_batch'] = self.spatio_extractor(src_input['image_batch'])
                src_input['clip_batch']= self.motion_extractor(src_input['clip_batch'])

            # Create a new dictionary for transferring of results
            emb_input = {'name_batch': src_input['name_batch'],
                        'emo_batch': src_input['emo_batch'], 
                        'image_batch': src_input['image_batch'], 
                        'clip_batch': src_input['clip_batch'], 
                        'num_clips_batch': src_input['num_clips_batch'],
                        'num_frames_batch': src_input['num_frames_batch'],}
            
            combined_features, src_len = self.adaptor(emb_input, self.tokenizer.pad_token_id)
            temporal_features = self.tconv(combined_features,src_len)
            vis_features = self.mlp(temporal_features['visual_feat'])
            if self.config.model.transformer_type == "causal": 
                new_input_embeds  = self.create_generation_inputs(vis_features, temporal_features['src_attn'],  self.tokenizer, self.config.dataset.prompt )
            else: 
                new_input_embeds = vis_features.cuda()
            outputs = self.lora_model.generate(inputs_embeds = new_input_embeds,attention_mask = temporal_features['src_attn'].cuda(), num_beams = num_beams, max_length = max_length)

        return outputs
    

    def vis_text_align(self, src_input, tgt_input):

        ##Get visual features 
        if self.config.training.token_usage == False:
            src_input['emo_batch'] = self.emo_extractor(src_input['emo_batch'])
            src_input['image_batch']= self.spatio_extractor(src_input['image_batch'])
            src_input['clip_batch']= self.motion_extractor(src_input['clip_batch'])

        # Create a new dictionary for transferring of results
        emb_input = {'name_batch': src_input['name_batch'],
                     'emo_batch': src_input['emo_batch'], 
                     'image_batch': src_input['image_batch'], 
                     'clip_batch': src_input['clip_batch'], 
                     'num_clips_batch': src_input['num_clips_batch'],
                     'num_frames_batch': src_input['num_frames_batch'],}
        
        combined_features , src_len = self.adaptor(emb_input, self.tokenizer.pad_token_id)
        temporal_features = self.tconv(combined_features, src_len)
        image_embeds = self.mlp(temporal_features['visual_feat'])

        # Get textual features 
        text_embeds = self.lora_model.get_input_embeddings()(tgt_input["input_ids"].cuda())
        
        image_embeds = image_embeds.mean(1)  # pooler_output
        text_embeds = text_embeds.mean(1)  # pooler_output

        # align the visual features with the text features, don't need to perform causal modelling
        # normalized features
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.T
        
        return logits_per_text

