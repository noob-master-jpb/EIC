import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoProcessor, AutoModelForCausalLM
import torch 



MODEL_ID = "./models/gemma-4-E2B-it"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load model
processor = AutoProcessor.from_pretrained(MODEL_ID)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map={"": 0},
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)
# Prompt
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Generate a code for fibonacci series in python for n=50."}]},
]

# Process input
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 
    enable_thinking=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse and print output
print("--- Raw Response ---")
print(response)
print("--- Parsed Response ---")
parsed_response = processor.parse_response(response)
print(parsed_response)
