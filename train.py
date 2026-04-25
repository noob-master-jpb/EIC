import torch
from transformers import AutoTokenizer, AutoProcessor

MODEL_ID = "./models/gemma-4-E2B-it"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

chat_messages = [
    {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
    {"role": "model", "content": [{"type": "text", "text": "The capital of France is Paris."}]}
]

# tokenize=False returns the raw formatted string
formatted_prompt = processor.apply_chat_template(chat_messages, tokenize=False)
print(formatted_prompt)
inputs = tokenizer(formatted_prompt, return_tensors="pt")

print("\n--- Tokenized Output ---")
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"First 10 Token IDs: {inputs['input_ids'][0][:10].tolist()}")