from unsloth import FastLanguageModel
import torch
import time
import gc

model_path = "/root/EIC/gemma-4-31B-it-finetuned"
max_seq_length = 4096 

print("Loading model and tokenizer...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = False, # Full precision for the 192GB MI300X
)
FastLanguageModel.for_inference(model)

# --- 1. DATASETS (THE CODE) ---

# 1. Smoke Prompt (Warp Shuffle + CDNA Bug)
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

# 2. NVIDIA Reduction Template
nvidia_reduce = """
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

# 3. NVIDIA cuBLAS (Native Compatibility Test)
nvidia_cublas = """
#include <cublas_v2.h>
#include <cuda_runtime.h>
void perform_gemm(float* A, float* B, float* C, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, m, B, k, &beta, C, m);
    cublasDestroy(handle);
}
"""

# 4. NVIDIA cuFFT (Native Compatibility Test)
nvidia_cufft = """
#include <cufft.h>
#include <cuda_runtime.h>
void perform_fft(cufftComplex *d_data, int batch_size) {
    cufftHandle plan;
    cufftPlan1d(&plan, 1024, CUFFT_C2C, batch_size);
    cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    cufftDestroy(plan);
}
"""

# 5. NVIDIA cuRAND (Native Compatibility Test)
nvidia_curand = """
#include <curand.h>
#include <cuda_runtime.h>
void generate_random(float *d_data, int size) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_data, size);
    curandDestroyGenerator(gen);
}
"""

# --- 2. PROMPT BUILDER ---
def prepare_batch(thinking_enabled=False):
    prompts = []
    
    if thinking_enabled:
        base_instruction = "Think step-by-step. Analyze the CUDA memory models, library dependencies, and hardware constraints. Document your reasoning, then output the final ROCm/HIP C++ code with native compatibility."
    else:
        base_instruction = "Directly convert this CUDA code to natively compatible ROCm/HIP. Change all headers and library calls appropriately. Do not provide explanations, output only the C++ code."

    msgs = [
        f"{base_instruction}\n\n{smoke_instruction}\n\n```cpp\n{smoke_cuda}\n```",
        f"{base_instruction}\n\n```cpp\n{nvidia_reduce}\n```",
        f"{base_instruction}\n\n```cpp\n{nvidia_cublas}\n```",
        f"{base_instruction}\n\n```cpp\n{nvidia_cufft}\n```",
        f"{base_instruction}\n\n```cpp\n{nvidia_curand}\n```"
    ]

    for msg in msgs:
        messages = [{"role": "user", "content": msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        
    tokenizer.padding_side = "left"
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
    
    # Generate batch
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9, 
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    end_time = time.time()
    
    # Calculate Performance Metrics
    peak_mem = torch.cuda.max_memory_allocated()
    
    # Detailed VRAM Breakdown for the Lead
    base_model_gb = start_mem / (1024**3)
    inference_overhead_gb = (peak_mem - start_mem) / (1024**3)
    total_peak_gb = peak_mem / (1024**3)
    
    total_time = end_time - start_time
    
    input_len = inputs['input_ids'].shape[1]
    generated_tokens_per_seq = outputs.shape[1] - input_len
    batch_size = outputs.shape[0]
    total_generated_tokens = generated_tokens_per_seq * batch_size
    
    tps = total_generated_tokens / total_time
    
    print(f"Total Time:         {total_time:.2f} seconds")
    print(f"Tokens Gen:         {total_generated_tokens} tokens (Batch Size: {batch_size})")
    print(f"Speed:              {tps:.2f} tokens/second")
    print(f"Base Model VRAM:    {base_model_gb:.2f} GB (The static weights in 16-bit)")
    print(f"Inference Overhead: {inference_overhead_gb:.2f} GB (The dynamic 'Spike' for KV Cache)")
    print(f"Total Peak VRAM:    {total_peak_gb:.2f} GB (What you see in rocm-smi)\n")
    
    # Decode and print snippets from the native library conversions
    print("--- Snippet: cuBLAS Native Compatibility Test ---")
    print(tokenizer.decode(outputs[2][input_len:], skip_special_tokens=True)[:250].strip() + "...\n")
    print("--- Snippet: cuFFT Native Compatibility Test ---")
    print(tokenizer.decode(outputs[3][input_len:], skip_special_tokens=True)[:250].strip() + "...\n")

# --- 4. EXECUTION FLOW ---

print("Preparing Batch Tensors (Batch Size = 5)...")
inputs_no_think = prepare_batch(thinking_enabled=False)
inputs_think = prepare_batch(thinking_enabled=True)

# 1. Base Model Benchmarks
with model.disable_adapter():
    run_benchmark("BASE MODEL", inputs_no_think, "Mode: Thinking DISABLED (Batch=5)")
    run_benchmark("BASE MODEL", inputs_think, "Mode: Thinking ENABLED  (Batch=5)")

# 2. LoRA Model Benchmarks
run_benchmark("LORA MODEL", inputs_no_think, "Mode: Thinking DISABLED (Batch=5)")
run_benchmark("LORA MODEL", inputs_think, "Mode: Thinking ENABLED  (Batch=5)")

print("\n" + "="*80)
print(" BENCHMARK COMPLETE.")
print("="*80)
