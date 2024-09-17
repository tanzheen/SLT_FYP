'''
1. VQ-VAE take in video frames and output the multiple 32 tokens 
2. Temporal module for the relationship among frames for the 32 tokens
3. Adaptor for the 32 tokens into MBart LLM
4. LLM produces final output and cross entropy loss
'''
import torch
import torch.nn as nn
from transformers import MBartForConditionalGeneration, MBartTokenizer

class SignModel(nn.Module): 
    def __init__(self, Titok_config): 
        super(SignModel, self).__init__()
        self.Titok = Titok(Titok_config)
        self.Adapted_Mbart = Adapted_Mbart(Mbart_config)
    
    def forward(self, input_frames): 
        encoded_tokens = Titok.encode(input_frames)[1]["min_encoding_indices"]




class LLMAdapter(nn.Module):
    '''
    LLM adapter here aims to capture temporal relations and at the same time transform the 32 tokens into 1024 for the Mbart
    32 tokens per frame will be transformed into 1024 tokens for the LLM to take in 
    At the same time, information about temporal relations will be understood
    '''
    def __init__(self, num_tokens= 32, hidden_dim= 1024, kernel_size=3):
        super(LLMAdapter, self).__init__()
        # Temporal convolution over the time dimension
        self.temporal_conv = nn.Conv1d(
            in_channels=num_tokens,    # 32 tokens per frame
            out_channels=hidden_dim,   # Output dimension
            kernel_size=kernel_size,   # Temporal window
            stride=1,
            padding=kernel_size // 2   # Padding to maintain sequence length
        )
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input shape: (batch_size, num_frames, num_tokens)
        # We need to apply Conv1d over the temporal dimension, so we transpose to (batch_size, num_tokens, num_frames)
        x = x.transpose(1, 2)
        x = self.temporal_conv(x)  # Shape: (batch_size, hidden_dim, num_frames)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = x.transpose(1, 2)  # Convert back to (batch_size, num_frames, hidden_dim)
        return x


class Adapted_Mbart(MBartForConditionalGeneration):
    def __init__(self, num_vq_tokens = 32, hidden_size= 1024 , *args, **kwargs):
        super().__init__(*args, **kwargs)
        # initiate the LLM Adaptor 
        self.adapter = LLMAdapter(num_vq_tokens, hidden_size)
        # Replace the embedding layer in the encoder
        self.model.encoder.embed_tokens = self.adapter
    
    def forward(self, input_tokens, **kwargs):
        # Forward pass using the modified MBart with custom adapter
        return super().forward(input_ids=None, inputs_embeds=self.adapter(input_tokens), **kwargs)


# Load the MBart model and replace the word embedding with the LLM adapter
model_name = "facebook/mbart-large-50"
mbart_model = Adapted_Mbart().from_pretrained(model_name)

