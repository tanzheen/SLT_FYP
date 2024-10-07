from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# Define the model and tokenizer you want to download
model_name = "facebook/mbart-large-50-many-to-many-mmt"

# Download the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50Tokenizer.from_pretrained(model_name)

# Save the model and tokenizer locally for offline use
model.save_pretrained("./mbart_model")
tokenizer.save_pretrained("./mbart_model")