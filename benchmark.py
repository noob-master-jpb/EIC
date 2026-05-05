from unsloth import FastLanguageModel
import torch
import re

# 1. Load the Model and the LoRA Adapter
max_seq_length = 2048
model_path = "/root/EIC/gemma-4-31B-it-finetuned"

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path, # This loads your LoRA and its base model automatically
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False, # Turn this OFF. You have 192GB, let's use it!
)

# 2. Define our "Smoke Prompt"
cuda_code = """
#include <cooperative_groups.h>
#include <cuda_runtime.h>
namespace cg = cooperative_groups;

__global__ void warp_reduce_kernel(float* d_ptr, float* d_out) {
    cg::thread_block cta = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);
    float val = d_ptr[blockIdx.x * blockDim.x + threadIdx.x];
    for (int offset = tile.size() / 2; offset > 0; offset /= 2) {
        val += tile.shfl_down(val, offset);
    }
    if (tile.thread_rank() == 0) {
        d_out[blockIdx.x] = val;
    }
}
"""

# Format it using the Gemma 4 Chat Template
messages = [
    {"role": "user", "content": f"Convert this CUDA Warp-Shuffle kernel to ROCm/HIP:\n\n```cpp\n{cuda_code}\n```"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text=prompt, return_tensors="pt").to("cuda")

# Helper function to "Parse" (extract) the C++ code from the model's chatty response
def extract_code(text):
    match = re.search(r'```(?:cpp|c\+\+)?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No code block found. Raw output:\n" + text.strip()

print("\n" + "="*50)
print(" 🛑 RUNNING BASE MODEL (Adapters Disabled)")
print("="*50)

# 3. Generate with the BASE Model (Turn off your fine-tuning)
FastLanguageModel.for_inference(model) # Optimize for fast generation
with model.disable_adapter():
    base_outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, use_cache=True)
    base_response = tokenizer.batch_decode(base_outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    
base_code = extract_code(base_response)
print(base_code)

print("\n" + "="*50)
print(" 🚀 RUNNING LORA MODEL (Adapters Enabled)")
print("="*50)

# 4. Generate with the LORA Model (Your fine-tuning is active)
lora_outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9, use_cache=True)
lora_response = tokenizer.batch_decode(lora_outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]

lora_code = extract_code(lora_response)
print(lora_code)
