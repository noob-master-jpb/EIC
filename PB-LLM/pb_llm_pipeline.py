import os
import subprocess
import logging
import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# CLOUD & GLOBAL CONFIGURATION
# ==========================================

# --- Task Toggles ---
RUN_QUANTIZATION_PIPELINE = True  # Run the Python quantization/training pipeline
RUN_OLLAMA = False                # Manage models via Ollama

# --- Quantization Algorithm ---
# Options:
# "pb-llm-1bit" : Pure PyTorch 1-bit quantization (Best for AMD ROCm servers & Hackathon logic)
# "bnb-4bit"    : BitsAndBytes 4-bit + QLoRA (Requires NVIDIA/CUDA, lower memory, slower training)
QUANTIZATION_METHOD = "pb-llm-1bit"

# --- Hardware Scaling ---
# "auto" : Automatically spans across multiple GPUs on high-end cloud servers (e.g. 8x MI300X or 8x H100)
# {"": 0}: Forces the model onto GPU 0 (Useful for strict single-GPU local testing)
DEVICE_MAP = "auto" 
ENABLE_GRADIENT_CHECKPOINTING = True # Saves massive VRAM on large models

# --- Model & Pipeline Settings ---
INPUT_MODEL_ID = "google/gemma-4-E2B-it"    

PERFORM_QAT = True                 # Fine-tune the model to fix quantization degradation
QAT_EPOCHS = 10                    # Number of training loops
SALIENT_RATIO = 0.50               # For PB-LLM: Protect 50% of weights in high precision

SAVE_QUANTIZED_MODEL = True             
SAVE_MODEL_DIR = "./quantized_cloud_model" 

GENERATE_TEST_OUTPUT = True             
PROMPT_TEXT = "Explain the concept of quantum computing in simple terms:" 

# ==========================================
# DUMMY DATASET
# ==========================================
DUMMY_CORPUS = [
    "Explain the concept of quantum computing in simple terms: Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers.",
    "Artificial intelligence leverages computers and machines to mimic the problem-solving and decision-making capabilities of the human mind.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a branch of artificial intelligence and computer science which focuses on the use of data and algorithms to imitate the way that humans learn.",
    "In mathematics, a matrix is a rectangular array or table of numbers, symbols, or expressions, arranged in rows and columns.",
    "Large language models use deep neural networks to generate human-like text based on the patterns they learned during training."
]

# ==========================================
# OLLAMA MANAGER
# ==========================================
OLLAMA_MODEL_NAME = "qwen3:0.6b"      
AUTO_DOWNLOAD = True       
MANUAL_PATH = None         

class OllamaManager:
    def __init__(self, auto_download: bool = True):
        self.auto_download = auto_download

    def check_model_exists(self, model_name: str) -> bool:
        try:
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
            return model_name in result.stdout
        except FileNotFoundError:
            logging.error("Ollama is not installed.")
            return False
        except subprocess.CalledProcessError:
            return False

    def pull_model(self, model_name: str) -> bool:
        logging.info(f"Downloading model '{model_name}'...")
        try:
            subprocess.run(['ollama', 'pull', model_name], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def create_model_from_path(self, custom_model_name: str, file_path: str) -> bool:
        modelfile_path = f"Modelfile_{custom_model_name.replace(':', '_')}"
        try:
            with open(modelfile_path, "w") as f:
                f.write(f"FROM {os.path.abspath(file_path)}\n")
            subprocess.run(['ollama', 'create', custom_model_name, '-f', modelfile_path], check=True)
            return True
        except Exception:
            return False
        finally:
            if os.path.exists(modelfile_path):
                os.remove(modelfile_path)

    def run_model(self, model_name: str, manual_path: Optional[str] = None):
        if manual_path:
            self.create_model_from_path(model_name, manual_path)
        elif not self.check_model_exists(model_name) and self.auto_download:
            self.pull_model(model_name)
        
        try:
            subprocess.run(['ollama', 'run', model_name])
        except KeyboardInterrupt:
            logging.info("\nExiting model execution.")

# ==========================================
# 1-BIT PB-LLM CORE (PURE PYTORCH / AMD READY)
# ==========================================

class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class PBLLMLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, salient_ratio: float = 0.50):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
            
        k = max(int(self.weight.numel() * salient_ratio), 1)
        threshold, _ = torch.topk(self.weight.abs().flatten(), k)
        threshold_val = threshold[-1].item()
        
        self.register_buffer('salient_mask', self.weight.abs() >= threshold_val)

    def forward(self, x):
        non_salient_weights = self.weight[~self.salient_mask]
        
        if non_salient_weights.numel() > 0:
            alpha = (non_salient_weights.to(torch.float32).abs().mean().to(self.weight.dtype) + 1e-9).detach()
        else:
            alpha = torch.tensor(1.0, device=self.weight.device, dtype=self.weight.dtype)
            
        binarized_non_salient = SignSTE.apply(self.weight) * alpha
        quantized_weight = torch.where(self.salient_mask, self.weight, binarized_non_salient)
        return nn.functional.linear(x, quantized_weight, self.bias)

def replace_with_pbllm_layers(model: nn.Module, salient_ratio: float = 0.50) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if any(x in name for x in ["lm_head", "embed_out", "output"]):
                continue
            if getattr(module.weight, "is_meta", False) or module.weight.device.type == 'meta':
                logging.info(f"Skipping layer '{name}' (offloaded to meta).")
                continue
            setattr(model, name, PBLLMLinear(module, salient_ratio))
        else:
            replace_with_pbllm_layers(module, salient_ratio)
    return model

def freeze_non_linear_parameters(model: nn.Module):
    for name, param in model.named_parameters():
        if not any(x in name for x in ["weight", "bias"]):
            param.requires_grad = False
        if any(x in name.lower() for x in ["norm", "embed", "vision"]):
            param.requires_grad = False

# ==========================================
# PIPELINE ORCHESTRATION
# ==========================================

def run_pipeline():
    try:
        logging.info(f"Loading tokenizer for {INPUT_MODEL_ID}...")
        tokenizer = AutoTokenizer.from_pretrained(INPUT_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if QUANTIZATION_METHOD == "bnb-4bit":
            # NVIDIA / CUDA Specific 4-Bit Loading
            import ctypes
            try:
                # Pre-load required library to avoid bitsandbytes CUDA 13 crash on some systems
                lib_path = os.path.abspath("./Arya-Files/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13")
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
                
            from transformers import BitsAndBytesConfig
            from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
            
            logging.info(f"Loading model '{INPUT_MODEL_ID}' in 4-bit (BitsAndBytes)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                INPUT_MODEL_ID, 
                quantization_config=bnb_config,
                device_map=DEVICE_MAP,
                low_cpu_mem_usage=True
            )
            if ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                
            if PERFORM_QAT:
                model = prepare_model_for_kbit_training(model)
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                    target_modules=["q_proj", "v_proj"]
                )
                model = get_peft_model(model, peft_config)
                
        elif QUANTIZATION_METHOD == "pb-llm-1bit":
            # Hardware Agnostic (AMD ROCm / NVIDIA CUDA) 1-Bit PB-LLM
            logging.info(f"Loading model '{INPUT_MODEL_ID}' in High Precision (PB-LLM Pre-Step)...")
            model = AutoModelForCausalLM.from_pretrained(
                INPUT_MODEL_ID, 
                device_map=DEVICE_MAP, 
                torch_dtype=compute_dtype,
                low_cpu_mem_usage=True
            )
            if ENABLE_GRADIENT_CHECKPOINTING and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
                
            logging.info("Injecting Custom PB-LLM 1-Bit Layers...")
            model = replace_with_pbllm_layers(model, salient_ratio=SALIENT_RATIO)
            freeze_non_linear_parameters(model)
            
        else:
            raise ValueError(f"Unknown Quantization Method: {QUANTIZATION_METHOD}")

        # Clear Cache Before Training
        torch.cuda.empty_cache()
        gc.collect()

        # Training Phase
        if PERFORM_QAT:
            logging.info(f"--- Starting Quantization-Aware Training ({QAT_EPOCHS} Epochs) ---")
            model.train()
            
            if QUANTIZATION_METHOD == "pb-llm-1bit":
                # SGD prevents floating point explosion on full 1-bit training
                optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, momentum=0.0)
            else:
                # AdamW is standard for QLoRA
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

            for epoch in range(QAT_EPOCHS):
                total_loss = 0
                for sentence in DUMMY_CORPUS:
                    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
                    outputs = model(**inputs, labels=inputs.input_ids)
                    loss = outputs.loss
                    
                    if torch.isnan(loss):
                        logging.warning(f"NaN loss detected at epoch {epoch+1}. Skipping.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                        
                    loss.backward()
                    
                    if QUANTIZATION_METHOD == "pb-llm-1bit":
                        for p in model.parameters():
                            if p.grad is not None:
                                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
                                
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    total_loss += loss.item()
                    del outputs
                    torch.cuda.empty_cache()
                
                logging.info(f"Epoch {epoch+1}/{QAT_EPOCHS} | Avg Loss: {(total_loss / len(DUMMY_CORPUS)):.4f}")
            logging.info("--- Training Complete! ---")

        # Save Phase
        model.eval()
        if SAVE_QUANTIZED_MODEL:
            logging.info(f"Saving Model to '{SAVE_MODEL_DIR}'...")
            model.save_pretrained(SAVE_MODEL_DIR)
            tokenizer.save_pretrained(SAVE_MODEL_DIR)
            logging.info("Save successful.")

        # Generation Phase
        if GENERATE_TEST_OUTPUT:
            logging.info(f"Testing with prompt: '{PROMPT_TEXT}'")
            inputs = tokenizer(PROMPT_TEXT, return_tensors="pt").to(model.device)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            if hasattr(model, "generation_config"):
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=100,
                    do_sample=True, 
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    repetition_penalty=2.0
                )
                
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("\n" + "="*50)
            print("--- GENERATED OUTPUT ---")
            print("="*50)
            print(generated_text)
            print("="*50 + "\n")
            
    except ImportError as e:
        logging.error(f"Missing library: {e}")
        logging.error("Ensure transformers, torch, accelerate, bitsandbytes, and peft are installed.")

def main():
    if RUN_QUANTIZATION_PIPELINE:
        print(f"--- Running Global Pipeline on {INPUT_MODEL_ID} ---")
        run_pipeline()

    if RUN_OLLAMA:
        manager = OllamaManager(auto_download=AUTO_DOWNLOAD)
        manager.run_model(model_name=OLLAMA_MODEL_NAME, manual_path=MANUAL_PATH)

if __name__ == "__main__":
    main()
