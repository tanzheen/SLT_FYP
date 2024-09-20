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

class SignModel(nn.Module): 
    def __init__(self, Config):
        super().__init__()
        self.titok = TiTok(Config)
        self.load_Titok_weights(Config.model.vq_model.init_weight)  # Load pretrained weights for the Titok model
        self.freeze_Titok_weights()  # Freeze Titok weights here
        self.Mbart = MBartForConditionalGeneration.from_pretrained(Config.model.MBart_model.init_weight)
        self.adapter = LLMAdapter(Config.model.vq_model.num_latent_tokens, hidden_dim= 1024)
        # Replace the embedding layer in the encoder
    
    def load_Titok_weights(self, titok_weight_path): 
        """
        Load pretrained weights for the Titok model from the given path.
        """
        state_dict = torch.load(titok_weight_path, map_location=torch.device('cpu'))  # adjust to 'cuda' if using GPU
        self.titok.load_state_dict(state_dict)
        print("Titok weights loaded successfully from:", titok_weight_path)


    def freeze_Titok_weights(self):
        # Freeze the weights of the Titok model
        for param in self.titok.parameters():
            param.requires_grad = False

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
        self.temporal_conv= nn.Sequential(nn.Conv1d(self.num_tokens, self.hidden_dim//4, kernel_size= 5, stride = 1 , padding = 0), 
                                     nn.BatchNorm1d(self.hidden_dim//4), 
                                     nn.ReLU(inplace = True), 
                                     nn.MaxPool1d(kernel_size=2, ceil_mode=False), 
                                     nn.Conv1d(self.hidden_dim//4,self.hidden_dim, kernel_size=5, stride = 1, padding =0), 
                                     nn.BatchNorm1d(self.hidden_dim), 
                                     nn.ReLU(inplace=True), 
                                     nn.MaxPool1d(kernel_size=2, ceil_mode=False))

    def forward(self, x, src_length):
        # Input shape: (batch_size, num_frames, num_tokens)
        start=0 
        x_batch = []
        for length in src_length: 
            end = start + length 
            x_batch.append(x[start:end])
            start = end 
        x = pad_sequence(x_batch, padding_value = PAD_IDX, batch_first = True)
        # We need to apply Conv1d over the temporal dimension, so we transpose to (batch_size, num_tokens, num_frames)
        print(f"after padded sequence: {x.shape}")
        x = x.permute(0,2,1)
        x = self.temporal_conv(x)  # Shape: (batch_size, hidden_dim, num_frames)
        x = x.permute(0,2,1 )  # Convert back to (batch_size, num_frames, hidden_dim)

        return x



