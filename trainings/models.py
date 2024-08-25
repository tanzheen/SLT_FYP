from torch import Tensor
import torch
import torch.nn as nn
from torch import nn, einsum
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import math
# from utils import create_mask

import torchvision
from torch.nn.utils.rnn import pad_sequence
#import pytorchvideo.models.x3d as x3d
import utils as utils

""" PyTorch MBART model."""
from transformers import MBartForConditionalGeneration, MBartPreTrainedModel, MBartModel, MBartConfig
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.models.mbart.modeling_mbart import shift_tokens_right

from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding, MBartEncoderLayer, _expand_mask

from collections import OrderedDict


import copy
import math
import random
from typing import List, Optional, Tuple, Union
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import numpy as np

# global definition
from definition import *

from hpman.m import _
from pathlib import Path

# Position encodings are used to add position information to the token embeddings
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Generate positional encodings
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))

        # Cosine and sine are used to generate the position embedding 
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        # positional encoding to the token embeddings then apply dropout
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


def make_resnet(name='resnet50'):
    if name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = torchvision.models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif name == 'resnet101':
        model = torchvision.models.resnet101(pretrained=True)
    else:
        raise Exception('There are no supported resnet model {}.'.format(_('resnet')))

    inchannel = model.fc.in_features
    model.fc = nn.Identity()
    return model

class resnet(nn.Module):
    def __init__(self):
        super(resnet, self).__init__()
        self.resnet = make_resnet(name='resnet18')

    def forward(self, x, lengths):
        x = self.resnet(x)
        x_batch = []
        start = 0
        for length in lengths:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        x = pad_sequence(x_batch,padding_value=PAD_IDX,batch_first=True)
        return x
  
# Temportal convolutional network for sequence modelling (analysing the sequence)
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
        
        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            # Notice that they are all 1D 
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    # This temp CNN is not dilated as it does not skip over any steps (stride = 1)
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def forward(self, x):
        x = self.temporal_conv(x.permute(0,2,1))
        return x.permute(0,2,1)

# Utility function to create the final layer of the model
def make_head(inplanes, planes, head_type):
    '''
    Args: 
        inplanes: the number of input features to the head. size of the feature vector that the network will process at this stage.
        planes: number of output features. In the context of NLP, the size of the embedding to output. 
    '''
    if head_type == 'linear':
        return nn.Linear(inplanes, planes, bias=False)
    else:
        return nn.Identity()

class TextCLIP(nn.Module):
    '''
    Part of the larger NN that will process and aligns textual and visual data
    TextCLIP handles the textual part of the multimodal learning 
    '''
    def __init__(self, config=None, inplanes=1024, planes=1024, head_type='identity'):
        super(TextCLIP, self).__init__()

        # Load the text encoder from a pretrained MBart model
        self.model_txt = MBartForConditionalGeneration.from_pretrained(config['model']['transformer']).get_encoder() 

        self.lm_head = make_head(inplanes, planes, head_type)

    def forward(self, tgt_input):
        # Encode the input text using MBart 
        txt_logits = self.model_txt(input_ids=tgt_input['input_ids'].cuda(), attention_mask=tgt_input['attention_mask'].cuda())[0]
        
        # Extract features from encoded text at the position of the last token
        output = txt_logits[torch.arange(txt_logits.shape[0]), tgt_input['input_ids'].argmax(dim=-1)]
        return self.lm_head(output), txt_logits


class ImageCLIP(nn.Module):
    """
    ImageCLIP class processes visual inputs and extracts feature representations 
    that can be aligned with textual features from the TextCLIP model.

    Attributes:
        config (dict): Configuration dictionary containing model parameters.
        model (nn.Module): A feature extractor model, typically a CNN (e.g., ResNet) 
                           followed by a temporal convolutional network.
        trans_encoder (MBartEncoder): An MBart encoder that further processes the visual features.
        cls_token (torch.nn.Parameter): A learnable class token appended to the sequence of visual features.
        lm_head (nn.Module): The head of the model, which could be a linear layer or an identity layer, 
                             depending on the configuration.
    """
    
    def __init__(self, config, inplanes=1024, planes=1024, head_type='linear') :
        """
        Initializes the ImageCLIP model with the provided configuration.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            inplanes (int): Number of input features to the head layer.
            planes (int): Number of output features from the head layer.
            head_type (str): Type of the head layer ('linear' for nn.Linear or 'identity' for nn.Identity).
        """

        super(ImageCLIP, self).__init__()
        self.config = config
        self.model =  FeatureExtracter() 
        # Initialize the MBart encoder for further processing of visual features, though I am a little confused by this, 
        # should check the shape inputted and outputted 
        self.trans_encoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_encoder()
        self.cls_token = nn.Parameter(torch.randn(1, 1, inplanes))
        self.lm_head = make_head(inplanes, planes, head_type)
        
    def forward(self, src_input):
        """
        Forward pass of the ImageCLIP model.

        Args:
            src_input (Tensor): Input tensor containing images.

        Returns:
            Tensor: Processed visual features after passing through the ViT and head.
        """
        
        # Extract features from the visual input using the feature extractor model 
        x = self.model(src_input['input_ids'].cuda(), src_input['src_length_batch']) # [b, n, c]
        # Attention mask for the visual input
        attention_mask = src_input['attention_mask']

        # Add the class token to the sequence of features
        B, N, C = x.shape  # B: batch size, N: sequence length, C: number of channels (features)

        ## !! why is this like that? 
        cls_token = repeat(self.cls_token, '() n d -> b n d', b=B)  # Repeat the class token for each element in the batch
        x = torch.cat((cls_token, x), dim=1)  # Concatenate class token to the beginning of the sequence
        
        # Adjust the attention mask to account for the added class token
        attention_mask = F.pad(attention_mask.flatten(1), (1, 0), value=1.0)  # [batch_size, N] -> [batch_size, N+1]
        
        # Pass the concatenated features through the MBart encoder
        outs = self.trans_encoder(inputs_embeds=x, attention_mask=attention_mask.cuda(), return_dict=True)
        last_hidden_state = outs['last_hidden_state']  # Extract the hidden states from the encoder
        
        # Pass the first token's hidden state (corresponding to the class token) through the head
        output = self.lm_head(last_hidden_state[:, 0, :])  # [batch_size, feature_dim]
        return output

class Text_Decoder(nn.Module):
    def __init__(self, config):
        super(Text_Decoder, self).__init__()
        self.text_decoder = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_decoder()
        self.lm_head = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).get_output_embeddings()
        self.register_buffer("final_logits_bias", torch.zeros((1, MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder']).model.shared.num_embeddings)))

    
    def forward(self, tgt_input, masked_tgt_input, model_txt):
        with torch.no_grad():
            _, encoder_hidden_states = model_txt(masked_tgt_input)

        decoder_input_ids = shift_tokens_right(tgt_input['input_ids'].cuda(), self.text_decoder.config.pad_token_id)
        decoder_out = self.text_decoder(
                    input_ids = decoder_input_ids,
                    attention_mask = tgt_input['attention_mask'].cuda(),
                    encoder_hidden_states = encoder_hidden_states,
                    encoder_attention_mask = masked_tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        lm_logits = self.lm_head(decoder_out[0]) + self.final_logits_bias

        return lm_logits
    
        
class SLRCLIP(nn.Module):
    def __init__(self, config, embed_dim=1024) :
        super(SLRCLIP, self).__init__()
        self.model_txt = TextCLIP(config, inplanes=embed_dim, planes=embed_dim)
        self.model_images = ImageCLIP(config, inplanes=embed_dim, planes=embed_dim)

        # logit scale is learnable parameter used to scale and control the sharpeness of the cosine similarity distribution
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def get_model_txt(self):
        return self.model_txt
    
    def get_model_image(self): 
        return self.model_images
    
    @property
    def get_encoder_hidden_states(self):
        return self.encoder_hidden_states
    
    def forward(self, src_input, tgt_input):
        # generate embeddings representing the visual input
        image_features = self.model_images(src_input)

        # generate embeddings representing textual input
        text_features, self.encoder_hidden_states = self.model_txt(tgt_input)

        # normalized both image and text features to unit length 
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits to map them together 
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # Naturally if correctly mapped then the the logits would be all 1
        ground_truth = torch.eye(logits_per_image.shape[0], device=logits_per_text.device, dtype=logits_per_image.dtype, requires_grad=False)

        return logits_per_image, logits_per_text, ground_truth

class FeatureExtracter(nn.Module):
    '''
    uses the resnet and the temporal Conv to extract 
    Is it possible to use 3D networks so that can learn the video better?
    '''
    def __init__(self, frozen=False):
        super(FeatureExtracter, self).__init__()
        self.conv_2d = resnet() # InceptionI3d()
        self.conv_1d = TemporalConv(input_size=512, hidden_size=1024, conv_type=2)

        if frozen:
            for param in self.conv_2d.parameters():
                param.requires_grad = False

    def forward(self,
                src: Tensor,
                src_length_batch
                ):
        src = self.conv_2d(src,src_length_batch)
        src = self.conv_1d(src)

        return src

class V_encoder(nn.Module):
    def __init__(self,
                 emb_size,
                 feature_size,
                 config,
                 ):
        super(V_encoder, self).__init__()
        
        self.config = config

        self.src_emb = nn.Linear(feature_size, emb_size)
        modules = []
        modules.append(nn.BatchNorm1d(emb_size))
        modules.append(nn.ReLU(inplace=True))
        self.bn_ac = nn.Sequential(*modules)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d,nn.Linear)):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self,
                src: Tensor,
                ):
      
        src = self.src_emb(src)
        src = self.bn_ac(src.permute(0,2,1)).permute(0,2,1)

        return src

def config_decoder(config):
    from transformers import AutoConfig
    
    decoder_type = _('decoder_type', 'LD', choices=['LD', 'LLMD'])
    if decoder_type == 'LD':
        
        return MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['visual_encoder'])/'config.json'))
    elif decoder_type == 'LLMD':
        return MBartForConditionalGeneration.from_pretrained(config['model']['transformer'], ignore_mismatched_sizes = True, config = AutoConfig.from_pretrained(Path(config['model']['transformer'])/'LLMD_config.json'))
    
class gloss_free_model(nn.Module):
    def __init__(self, config, args, embed_dim=1024, pretrain=None):
        super(gloss_free_model, self).__init__()
        self.config = config
        self.args = args

        self.backbone = FeatureExtracter(frozen=_('freeze_backbone', False))
        # self.mbart = MBartForConditionalGeneration.from_pretrained(config['model']['visual_encoder'])
        self.mbart = config_decoder(config)
 
        if config['model']['sign_proj']:
            self.sign_emb = V_encoder(emb_size=embed_dim,feature_size=embed_dim, config = config)
            self.embed_scale = math.sqrt(embed_dim) if config['training']['scale_embedding'] else 1.0
        else:
            self.sign_emb = nn.Identity()
            self.embed_scale = 1.0
        
    def share_forward(self, src_input):
        
        frames_feature = self.backbone(src_input['input_ids'].cuda(), src_input['src_length_batch'])
        attention_mask = src_input['attention_mask']

        inputs_embeds = self.sign_emb(frames_feature)
        inputs_embeds = self.embed_scale * inputs_embeds

        return inputs_embeds, attention_mask

    def forward(self,src_input, tgt_input ):
        
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),
                    # decoder_input_ids = tgt_input['input_ids'].cuda(),
                    labels = tgt_input['input_ids'].cuda(),
                    decoder_attention_mask = tgt_input['attention_mask'].cuda(),
                    return_dict = True,
                    )
        return out['logits']
    

    def generate(self,src_input,max_new_tokens,num_beams,decoder_start_token_id ):
        inputs_embeds, attention_mask = self.share_forward(src_input)

        out = self.mbart.generate(inputs_embeds = inputs_embeds,
                    attention_mask = attention_mask.cuda(),max_new_tokens=max_new_tokens,num_beams = num_beams,
                                decoder_start_token_id=decoder_start_token_id
                            )
        return out