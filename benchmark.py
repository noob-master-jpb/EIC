from unsloth import FastLanguageModel
import torch
import time
import gc

model_path = "/root/EIC/gemma-4-31B-it-finetuned"
max_seq_length = 4096  # Increased to handle longer NVIDIA code + reasoning

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False, # Full precision for the 192GB MI300X
)
FastLanguageModel.for_inference(model)

# --- 1. DATASETS (THE CODE) ---

# A. The Smoke Prompt (Warp Shuffle)
smoke_instruction = "Task: The kernel below has a correctness bug that only manifests at runtime under specific launch configurations. Port it to HIP targeting AMD CDNA architecture, and fix all bugs. Do not add comments explaining what you changed."

smoke_cuda = """
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

# B. Official NVIDIA Codebase (Optimized Shared Memory Reduction)
# A classic, highly optimized kernel from the NVIDIA CUDA Samples
nvidia_cuda = """
#include <cuda_runtime.h>
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *sdata, unsigned int tid) {
    if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16) sdata[tid] += sdata[tid +  8];
    if (blockSize >=  8) sdata[tid] += sdata[tid +  4];
    if (blockSize >=  4) sdata[tid] += sdata[tid +  2];
    if (blockSize >=  2) sdata[tid] += sdata[tid +  1];
}

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + tid;
    unsigned int gridSize = blockSize*2*gridDim.x;
    sdata[tid] = 0;
    while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+blockSize]; i += gridSize; }
    __syncthreads();
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }
    if (tid < 32) warpReduce<blockSize>(sdata, tid);
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
"""

# --- 2. PROMPT BUILDER ---
def prepare_batch(thinking_enabled=False):
    prompts = []
    
    if thinking_enabled:
        base_instruction = "Think step-by-step. Analyze the CUDA memory models, hardware constraints, and warp primitives. Document your reasoning, then output the final ROCm/HIP C++ code."
    else:
        base_instruction = "Directly convert this CUDA code to ROCm/HIP. Do not provide explanations, output only the C++ code."

    smoke_msg = f"{base_instruction}\n\n{smoke_instruction}\n\n```cpp\n{smoke_cuda}\n```"
    nvidia_msg = f"{base_instruction}\n\n```cpp\n{nvidia_cuda}\n```"

    for msg in [smoke_msg, nvidia_msg]:
        messages = [{"role": "user", "content": msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    # Left padding is required for batched generation.
    tokenizer.padding_side = "left"
    # Ensure pad token is set to avoid errors during batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer(text=prompts, return_tensors="pt", padding=True).to("cuda")

# --- 3. BENCHMARK ENGINE ---
def run_benchmark(model_name, inputs, description):
    print(f"\n{'='*80}")
    print(f" BENCHMARK: {model_name} | {description}")
    print(f"{'='*80}")
    
    # Reset GPU memory stats for accurate measurement
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    start_mem = torch.cuda.memory_allocated()
    start_time = time.time()
    
    # Generate batch (Batch Size = 2)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024, # Increased to handle "thinking" text
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    end_time = time.time()
    
    # Calculate Performance Metrics
    peak_mem = torch.cuda.max_memory_allocated()
    mem_used_gb = (peak_mem - start_mem) / (1024**3)
    
    total_time = end_time - start_time
    
    # Calculate tokens (subtracting the input length to only measure generated tokens)
    input_len = inputs['input_ids'].shape[1]
    generated_tokens_per_seq = outputs.shape[1] - input_len
    total_generated_tokens = generated_tokens_per_seq * outputs.shape[0] # Multiply by batch size
    
    tps = total_generated_tokens / total_time
    
    print(f"Total Time:      {total_time:.2f} seconds")
    print(f"Tokens Gen:      {total_generated_tokens} tokens")
    print(f"Speed:           {tps:.2f} tokens/second")
    print(f"VRAM Spike:      {mem_used_gb:.2f} GB (Inference overhead)")
    
    # Print Snippets to verify the quality and intent
    response1 = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    response2 = tokenizer.decode(outputs[1][input_len:], skip_special_tokens=True)
    
    print("\n--- Snippet: Smoke Prompt Result ---")
    print(response1[:300].strip() + "...\n")
    print("--- Snippet: NVIDIA Codebase Result ---")
    print(response2[:300].strip() + "...\n")

# --- 4. EXECUTION FLOW ---

print("Preparing Batch Tensors...")
inputs_no_think = prepare_batch(thinking_enabled=False)
inputs_think = prepare_batch(thinking_enabled=True)

# 1. Base Model Benchmarks
with model.disable_adapter():
    run_benchmark("BASE MODEL", inputs_no_think, "Mode: Thinking DISABLED (Batch=2)")
    run_benchmark("BASE MODEL", inputs_think, "Mode: Thinking ENABLED  (Batch=2)")

# 2. LoRA Model Benchmarks
run_benchmark("LORA MODEL", inputs_no_think, "Mode: Thinking DISABLED (Batch=2)")
run_benchmark("LORA MODEL", inputs_think, "Mode: Thinking ENABLED  (Batch=2)")

print("\n" + "="*80)
print(" BENCHMARK COMPLETE.")
print("="*80)
