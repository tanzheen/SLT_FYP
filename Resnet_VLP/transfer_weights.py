import torch

# Load the model weights
text_decoder_weights = torch.load("text_decoder/pytorch_model.bin", map_location="cpu")



# Load the model weights
CLIP_weights = torch.load("unwrapped_model/pytorch_model.bin", map_location="cpu")

from transformers import MBartForConditionalGeneration

# Load a pre-trained MBart model
mbart_model = MBartForConditionalGeneration.from_pretrained("pretrain_models/MBart_proun")

# Create a new state dict for MBart decoder
mbart_decoder_weights = {}

# Assuming the text decoder keys align with MBart's decoder keys (check naming carefully)
for key, value in text_decoder_weights.items():
    # Replace text_decoder's layer names to match MBart's decoder layers
    new_key = key.replace("text_decoder", "model.decoder")
    mbart_decoder_weights[new_key] = value

# Create a new state dict for MBart encoder
mbart_encoder_weights = {}

# Assuming the visual encoder keys align with MBart's encoder keys (check naming carefully)
for key, value in CLIP_weights.items():
    # Replace visual_encoder's layer names to match MBart's encoder layers
    new_key = key.replace("model_images.trans_encoder", "model.encoder")
    mbart_encoder_weights[new_key] = value

# Initialize an empty dictionary to store only the relevant encoder weights
filtered_mbart_encoder_weights = {}

# Loop through the mapped keys and filter those that start with "model.encoder"
for key, value in mbart_encoder_weights.items():
    if key.startswith("model.encoder"):
        filtered_mbart_encoder_weights[key] = value

# Get the full MBart state dict
mbart_state_dict = mbart_model.state_dict()

# Update MBart's encoder and decoder with the respective weights
mbart_state_dict.update(filtered_mbart_encoder_weights )
mbart_state_dict.update(mbart_decoder_weights)

# Load the merged state dict into the MBart model
mbart_model.load_state_dict(mbart_state_dict, strict=False)

# Save the merged model
mbart_model.save_pretrained("transferred_mbart_model")

