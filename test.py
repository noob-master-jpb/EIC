import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from transformers import AutoProcessor, AutoModelForCausalLM
import torch 
import gemma



MODEL_ID = "./models/gemma-4-E2B-it"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Configuration
TOKENIZE = True
ENABLE_THINKING = True

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
    {"role": "user", "content": [{"type": "text", "text": """
Explain the logic of a binary search algorithm.
    """}]},
]

# Process input
inputs_data = processor.apply_chat_template(
    messages, 
    tokenize=TOKENIZE, 
    add_generation_prompt=True, 
    enable_thinking=ENABLE_THINKING,
    return_dict=True,
    return_tensors="pt" if TOKENIZE else None
)

if not TOKENIZE:
    print("--- Raw Prompt (Tokenizer Off) ---")
    print(inputs_data)
    # Manually tokenize for the model
    inputs = processor(text=inputs_data, return_tensors="pt", add_special_tokens=False).to(model.device)
else:
    inputs = inputs_data.to(model.device)

input_len = inputs["input_ids"].shape[-1]
print(f"Input Token Length: {input_len}")

# Generate output
outputs = model.generate(**inputs, max_new_tokens=1024)
response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

# Parse and print output
print("--- Raw Response ---")
print(response)
print("--- Parsed Response (Native Gemma 4 Parser) ---")
# Using the native parser implemented in gemma.py
parser = gemma.Gemma4ReasoningParser()
parsed_data = parser.parse(response)

if parsed_data["thinking"]:
    print(f"THINKING:\n{parsed_data['thinking']}\n")
print(f"ANSWER:\n{parsed_data['answer']}")

print(parsed_data)