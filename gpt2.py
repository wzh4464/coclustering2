import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set the model to evaluation mode
model.eval()

# Generate some text
prompt = 'Hello, world!'
input_ids = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(input_ids, max_length=50, do_sample=True)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)