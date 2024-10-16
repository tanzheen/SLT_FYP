'''
1. VQ-VAE take in video frames and output the multiple 32 tokens 
2. Temporal module for the relationship among frames for the 32 tokens
3. Adaptor for the 32 tokens into MBart LLM
4. LLM produces final output and cross entropy loss
'''
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration
from vis_tokenizer.modeling.titok import TiTok
from torch.nn.utils.rnn import pad_sequence
from definition import * 
from omegaconf import OmegaConf
from pathlib import Path
import json 
from base_model import BaseModel
from huggingface_hub import PyTorchModelHubMixin


class SignModel(BaseModel, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config 
        self.titok = TiTok(config)
        self.load_Titok_weights(config.model.vq_model.init_weight)  # Load pretrained weights for the Titok model
        self.freeze_Titok_weights()  # Freeze Titok weights here
        self.Mbart = MBartForConditionalGeneration.from_pretrained(config.model.MBart_model.init_weight)
        if config.model.MBart_model.freeze_MBart: 
            self.freeze_MBart_weights()
        
        # Can add adapter type next time
        if config.model.MBart_model.adapt_type == 1:
            self.adapter = LLMAdapter(config.model.vq_model.num_latent_tokens, hidden_dim= 1024)
        elif config.model.MBart_model.adapt_type == 2:
            self.adapter = LLMAdapter2(config.model.vq_model.num_latent_tokens, hidden_dim= 1024)
        elif config.model.MBart_model.adapt_type == 3:
            self.adapter = LLMAdapter3(config.model.vq_model.num_latent_tokens, hidden_dim= 1024)
        # Replace the embedding layer in the encoder    

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights and config to a local directory."""
        # Assume 'self.config' is your DictConfig object
        # Convert to a regular dictionary
        dict_config = OmegaConf.to_container(self.config)
        # Save as JSON
        file_path = Path(save_directory) / "config.json"
        with open(file_path, 'w') as json_file:
            json.dump(dict_config, json_file, indent=4)
        super()._save_pretrained(save_directory)

    def _init_weights(self, module):
        """ Initialize the weights.
            :param:
                module -> torch.nn.Module: module to initialize
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data = nn.init.trunc_normal_(module.weight.data, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def load_Titok_weights(self, titok_weight_path): 
        """
        Load pretrained weights for the Titok model from the given path.
        """
        state_dict = torch.load(titok_weight_path, map_location=torch.device('cpu'))  # adjust to 'cuda' if using GPU
        self.titok.load_state_dict(state_dict)
        print("Titok weights loaded successfully from:", titok_weight_path)

    
    def freeze_MBart_weights(self):
        """ Freeze the weights of the MBart model. """
        for param in self.Mbart.parameters():
            param.requires_grad = False
        print("MBart weights are frozen!")


    def freeze_Titok_weights(self):
        # Freeze the weights of the Titok model
        for param in self.titok.parameters():
            param.requires_grad = False
        print("TiTok weights are frozen chowwww!")

    def forward(self, src_input,tgt_input, src_attn, tgt_attn, src_length): 
        '''
        Shape of: 
            src_input: (batch_size, sequence_len, embedding_dim)
            tgt_input: (batch_size, sequence_length)
        '''
        encoded_tokens = self.titok.encode(x=src_input)[1]['min_encoding_indices'].squeeze()
        hidden_values = self.adapter(encoded_tokens.float(), src_length).squeeze()
        sign_translation = self.Mbart(inputs_embeds = hidden_values, attention_mask = src_attn, 
                                      decoder_attention_mask = tgt_attn,  labels= tgt_input)

        return sign_translation
    
    def generate(self, src_input, src_attn, src_length, max_new_tokens=150, num_beams=4, decoder_start_token_id=None): 
        encoded_tokens = self.titok.encode(x=src_input)[1]['min_encoding_indices'].squeeze()
        hidden_values = self.adapter(encoded_tokens.float(), src_length).squeeze()
        generated_tokens = self.Mbart.generate(inputs_embeds = hidden_values, attention_mask = src_attn,
                                               max_new_tokens = max_new_tokens, num_beams = num_beams, decoder_start_token_id = decoder_start_token_id)
        
        return generated_tokens
        



class LLMAdapter(nn.Module):
    '''
    LLM adapter here aims to capture temporal relations and at the same time transform the 32 tokens into 1024 for the Mbart
    32 tokens per frame will be transformed into 1024 tokens for the LLM to take in 
    At the same time, information about temporal relations will be understood
    '''
    def __init__(self, num_tokens= 32, hidden_dim= 1024, kernel_size=5):
        super(LLMAdapter, self).__init__()
        # Temporal convolution over the time dimension
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.proj = nn.Linear(self.num_tokens, self.hidden_dim//2)
        self.temporal_conv= nn.Sequential(
                                    nn.Conv1d(self.hidden_dim//2, self.hidden_dim, kernel_size= 5, stride = 1 , padding = 0), 
                                     nn.BatchNorm1d(self.hidden_dim), 
                                     nn.ReLU(inplace = True), 
                                     nn.AvgPool1d(kernel_size=2, ceil_mode=False), 
        
                                     nn.Conv1d(self.hidden_dim,self.hidden_dim, kernel_size=5, stride = 1, padding =0), 
                                     nn.BatchNorm1d(self.hidden_dim), 
                                     nn.ReLU(inplace=True), 
                                     nn.AvgPool1d(kernel_size=2, ceil_mode=False))
 

    def forward(self, x, src_length):
        # Input shape: (num_frames, num_tokens)
        start=0 
        x_batch = []
        for length in src_length: 
            end = start + length 
            x_batch.append(x[start:end])
            start = end 
        x = pad_sequence(x_batch, padding_value = PAD_IDX, batch_first = True)
        # We need to apply Conv1d over the temporal dimension, so we transpose to (batch_size, num_tokens, num_frames)
        # sequence: {x.shape}")
        x = self.proj(x)
        x = x.permute(0,2,1)
        x = self.temporal_conv(x)  # Shape: (batch_size, hidden_dim, num_frames)
        x = x.permute(0,2,1 )  # Convert back to (batch_size, num_frames, hidden_dim)

        return x
    

class LLMAdapter2(nn.Module):
    '''
    LLM adapter aims to capture temporal relations and transform 32 tokens into 1024 tokens.
    This version introduces an additional projection layer between the two convolution layers.
    '''
    def __init__(self, num_tokens=32, hidden_dim=1024, kernel_size=5):
        super(LLMAdapter2, self).__init__()
        
        # Store parameters
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # First projection from input tokens to hidden_dim/2
        self.proj = nn.Linear(self.num_tokens, self.hidden_dim // 2)

        # First convolutional block
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(self.hidden_dim // 2, self.hidden_dim // 2, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, ceil_mode=False)
        )

        # New projection layer between convolution layers
        self.intermediate_proj = nn.Linear(self.hidden_dim // 2, self.hidden_dim)

        # Second convolutional block
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=1, padding=0),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=2, ceil_mode=False)
        )


    def forward(self, x, src_length):
        # Input shape: (batch_size, num_frames, num_tokens)
        
        # Split the input into individual batches according to src_length
        start = 0
        x_batch = []
        for length in src_length:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        
        # Pad sequences to ensure uniform batch sizes
        x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        
        # Apply the initial projection layer
        x = self.proj(x)  # Shape: (batch_size, num_frames, hidden_dim / 2)
        
        # Permute to (batch_size, hidden_dim / 2, num_frames) for Conv1d
        x = x.permute(0, 2, 1)
        
        # First convolutional block
        x = self.conv_block_1(x)  # Shape: (batch_size, hidden_dim / 2, reduced_num_frames)
        
        # Apply the intermediate projection layer
        x = x.permute(0, 2, 1)  # Back to (batch_size, reduced_num_frames, hidden_dim / 2)
        x = self.intermediate_proj(x)  # Shape: (batch_size, reduced_num_frames, hidden_dim)
        x = x.permute(0, 2, 1)  # Back to (batch_size, hidden_dim, reduced_num_frames)
        
        # Second convolutional block
        x = self.conv_block_2(x)  # Shape: (batch_size, hidden_dim, further_reduced_num_frames)
        
        # Convert back to (batch_size, further_reduced_num_frames, hidden_dim)
        x = x.permute(0, 2, 1)


        return x
    




class LLMAdapter3(nn.Module):
    def __init__(self, num_tokens=32, hidden_dim=1024, kernel_size=5):
        super(LLMAdapter3, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        
        # Temporal convolution over the time dimension
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.num_tokens, self.num_tokens * 2, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(self.num_tokens * 2),  # Channels must match Conv1d output channels
            nn.ReLU(inplace=True),
            # Reduce kernel size for pooling to avoid sequence collapse
            nn.AvgPool1d(kernel_size=1, ceil_mode=False),  

            nn.Conv1d(self.num_tokens * 2, self.num_tokens * 4, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm1d(self.num_tokens * 4),  # Channels must match Conv1d output channels
            nn.ReLU(inplace=True),
            nn.AvgPool1d(kernel_size=1, ceil_mode=False)  # Adjusted pooling to avoid reducing size to zero
        )
        
        # Final projection layer
        self.final_proj = nn.Sequential(
            nn.Linear(self.num_tokens * 4, self.hidden_dim)
        )
        self.out = nn.Sequential(nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(inplace=True))

    def forward(self, x, src_length):
        start = 0
        x_batch = []
        for length in src_length:
            end = start + length
            x_batch.append(x[start:end])
            start = end
        #print(f"Before padding: {x.shape}") 
        x = pad_sequence(x_batch, padding_value=PAD_IDX, batch_first=True)
        #print(f"After padding: {x.shape}")  # Print shape after padding
        
        # Permute to match Conv1d expected shape: (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)
        #print(f"After permute: {x.shape}")  # Shape should now be (batch_size, num_tokens, num_frames)
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        #print(f"After temporal_conv: {x.shape}")  # Check shape after convolution
        
        # Permute back to (batch_size, sequence_length, hidden_dim)
        x = x.permute(0, 2, 1)
        #print(f"After second permute: {x.shape}")  # Shape should be (batch_size, num_frames, num_tokens*4)
        
        # Apply final projection (we need to flatten or reshape input to match Linear input requirements)
     
        x = self.final_proj(x)
        #x = self.final_proj(x.reshape(batch_size * seq_len, hidden_dim))
        #print(f"After final_proj: {x.shape}")  # Check final shape

        #print(f"before out shape : {x.shape}")
        x = self.out(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x



        
        