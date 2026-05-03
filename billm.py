import os
import gc

# Must be set before torch initializes the CUDA allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random

# =========================
# CONFIG
# =========================
KEEP_RATIO = {
    "q_proj": 0.06, "k_proj": 0.06, "v_proj": 0.06,
    "o_proj": 0.03, "up_proj": 0.02, "down_proj": 0.02, "gate_proj": 0.02,
    "default": 0.02
}
RESIDUAL_STEPS             = 1
EPSILON                    = 1e-8
SEQ_LEN                    = 512    # Reduced: 128 samples × 512 tokens is ample for Hessian diagonal
N_CALIBRATION_SAMPLES      = 128    # 128 samples converges well for Hessian stats
CALIBRATION_BATCH_SIZE     = 1      # One sample at a time — peak VRAM = activations for ONE sequence
MODEL_NAME                 = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR                 = "qwen3.5-0.8B-billm"

# toggles
USE_SENSITIVITY_SCHEDULING = False
USE_DYNAMIC_KEEP_RATIO     = False
USE_CLIP_SEARCH            = False
USE_BLOCK_NORMALIZATION    = False

FORCE_CPU                  = False
CPU_THREADS                = 40     # match your logical core count

torch.manual_seed(42)
random.seed(42)


# =========================
# DEVICE DETECTION  (CUDA/ROCm-safe)
# =========================

def get_device():
    """
    Detects the best available accelerator.
    - PyTorch ROCm builds expose MI300X via torch.cuda (HIP aliases).
    - Prints the ROCm/HIP version when available so you can confirm the build.
    """
    if FORCE_CPU:
        return torch.device("cpu")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_gb  = props.total_memory / 1024**3
        free_gb   = (props.total_memory - torch.cuda.memory_allocated(0)) / 1024**3
        alloc_gb  = torch.cuda.memory_allocated(0) / 1024**3
        hip_ver = getattr(torch.version, "hip", None)
        runtime  = f"ROCm {hip_ver}" if hip_ver else f"CUDA {torch.version.cuda}"
        print(f"[Device] {runtime} — {name}")
        print(f"[VRAM]   total={total_gb:.1f} GB | already allocated={alloc_gb:.1f} GB | free={free_gb:.1f} GB")
        if alloc_gb > 1.0:
            print("[WARN]   >1 GB already allocated at startup — another process may be holding GPU memory.")
        return dev

    return torch.device("cpu")


# =========================
# CORE
# =========================

def compute_saliency(w, h):
    return (w ** 2) / (h.unsqueeze(0) + EPSILON)


def apply_mask(saliency, keep_ratio):
    flat = saliency.view(-1)
    k = max(1, int(flat.numel() * keep_ratio))
    mask = torch.zeros_like(flat, dtype=torch.bool)
    idx = torch.topk(flat, k).indices
    mask[idx] = True
    return mask.view_as(saliency)


def compute_alpha(w):
    return w.abs().mean(dim=1, keepdim=True)


def clip_if_enabled(w):
    if not USE_CLIP_SEARCH:
        return w

    w_abs = w.abs()
    base = torch.quantile(w_abs, 0.99)
    cands = base * torch.tensor([0.8, 1.0, 1.2], device=w.device)

    errs = []
    for c in cands:
        w_c = torch.clamp(w, -c, c)
        alpha = w_c.abs().mean()
        b = alpha * torch.sign(w_c)
        errs.append(torch.norm(w - b))

    best = cands[torch.argmin(torch.stack(errs))]
    return torch.clamp(w, -best, best)


def billm_binarize(w, mask):
    w = clip_if_enabled(w)

    alpha = compute_alpha(w)
    sign = torch.sign(w)

    b = w.clone()
    rows, cols = torch.where(~mask)
    b[rows, cols] = alpha[rows, 0] * sign[rows, cols]

    return b


def apply_residual(w, b, mask):
    for _ in range(RESIDUAL_STEPS):
        r = (w - b) * (~mask)

        alpha = compute_alpha(r)
        sign = torch.sign(r)

        rows, cols = torch.where(~mask)
        update = torch.zeros_like(w)
        update[rows, cols] = alpha[rows, 0] * sign[rows, cols]

        b = b + update

    return b


# =========================
# CALIBRATION
# =========================

@torch.no_grad()
def get_data(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.select(range(min(10000, len(dataset))))
    text = " ".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")

    samples = []
    for _ in range(N_CALIBRATION_SAMPLES):
        i = random.randint(0, enc.input_ids.shape[1] - SEQ_LEN - 1)
        samples.append(enc.input_ids[:, i:i+SEQ_LEN])
    return samples


@torch.no_grad()
def collect_hessian(model, samples, device):
    """
    Single-sample Hessian collection.

    Memory strategy:
    - Model is already on CPU (loaded that way in download_and_load_model).
    - We move the model to GPU, run ONE sample at a time (CALIBRATION_BATCH_SIZE=1),
      then move it back to CPU.  Peak VRAM = model weights + activations for
      exactly one sequence — the absolute minimum possible.
    """
    hess = {}
    hooks = []

    def hook(name):
        def fn(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])  # (B*T, D) — B=1 here
            h = torch.mean(x.float() ** 2, dim=0).cpu() + EPSILON
            del x  # release GPU tensor immediately
            if name not in hess:
                hess[name] = h
            else:
                hess[name].add_(h)
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook(name)))

    if device.type == "cuda":
        print("[Calibration] Moving model to GPU...")
        model.to(device)
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"[Calibration] Model on GPU — {alloc:.2f} GB allocated")

    for i, sample in enumerate(tqdm(samples, desc="Calibrating")):
        # sample shape: (1, SEQ_LEN)  — send one sequence at a time
        inp = sample.to(device)
        with torch.inference_mode():
            model(inp, use_cache=False)
        del inp

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        if i % 16 == 0:
            gc.collect()

    for h in hooks:
        h.remove()

    # Normalise — use out-of-place division to escape the inference-mode flag
    # that was baked into these tensors when the hooks ran inside inference_mode().
    n = len(samples)
    for k in hess:
        hess[k] = hess[k] / n

    # Move model back to CPU; GPU is fully free for evaluation later
    if device.type == "cuda":
        print("[Calibration] Moving model back to CPU...")
        model.to("cpu")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    return hess


# =========================
# QUANTIZATION
# =========================

def get_keep_ratio(name):
    for k, v in KEEP_RATIO.items():
        if k in name:
            return v
    return KEEP_RATIO["default"]


@torch.no_grad()
def quantize(model, hessians):

    modules = dict(model.named_modules())

    if USE_SENSITIVITY_SCHEDULING:
        sensitivities = {}
        for name, m in modules.items():
            if isinstance(m, nn.Linear) and name in hessians:
                w = m.weight.data
                h = hessians[name].to(w.device)
                sensitivities[name] = ((w**2)/(h+EPSILON)).mean().item()

        layer_order = sorted(sensitivities, key=lambda x: sensitivities[x])
    else:
        layer_order = [n for n, m in modules.items() if isinstance(m, nn.Linear)]

    accumulated_error = 0.0

    for name in tqdm(layer_order, desc="Quantizing layers"):
        if name not in hessians:
            continue

        m = modules[name]
        w = m.weight.data.cpu()
        h = hessians[name]  # already on CPU

        keep_ratio = get_keep_ratio(name)

        if USE_DYNAMIC_KEEP_RATIO:
            keep_ratio = min(
                keep_ratio * (1 + min(0.3, accumulated_error * 0.5)),
                keep_ratio * 2
            )

        saliency = compute_saliency(w, h)
        mask = apply_mask(saliency, keep_ratio)

        b = billm_binarize(w, mask)
        b = apply_residual(w, b, mask)

        if USE_BLOCK_NORMALIZATION:
            ratio = w.norm() / (b.norm() + EPSILON)
            ratio = torch.clamp(ratio, 0.85, 1.2)
            b *= ratio

        m.weight.data.copy_(b)

        err = torch.norm(w - b) / (torch.norm(w) + EPSILON)
        accumulated_error += err.item()

        print(f"{name} | err={err:.4f} | keep={keep_ratio:.4f}")


# =========================
# PERPLEXITY
# =========================

@torch.no_grad()
def perplexity(model, tokenizer, device):
    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids

    stride = SEQ_LEN
    max_len = model.config.max_position_embeddings

    nlls = []

    for i in range(0, enc.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = i + stride

        input_ids = enc[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss * stride)

    return torch.exp(torch.stack(nlls).sum() / (len(nlls) * stride)).item()


# =========================
# MODEL LOADING
# =========================

from huggingface_hub import snapshot_download

def download_and_load_model(model_name, local_dir="models"):
    model_path = os.path.join(local_dir, model_name.split("/")[-1])
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"Downloading model {model_name} to '{model_path}'...")
        os.makedirs(model_path, exist_ok=True)
        snapshot_download(
            repo_id=model_name,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
    else:
        print(f"Model found in '{model_path}', loading from local directory...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Always load to CPU first.
    # We move to GPU only during calibration forward passes and final eval,
    # so VRAM is never occupied by idle model weights.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return tokenizer, model


# =========================
# MAIN
# =========================

def main():
    device = get_device()

    if device.type == "cpu":
        torch.set_num_threads(CPU_THREADS)
        print(f"[CPU] Using {CPU_THREADS} threads.")
    else:
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[GPU] Total VRAM: {total_vram:.1f} GB")

    tokenizer, model = download_and_load_model(MODEL_NAME, "models")
    model.config.use_cache = False

    samples = get_data(tokenizer)
    hess = collect_hessian(model, samples, device)

    # collect_hessian already moved the model to CPU.
    # Quantization is weight-only (no activations) — CPU is fine.
    print("Quantizing on CPU (weight-only, no activations needed)...")
    gc.collect()

    quantize(model, hess)

    print(f"Moving model to {device} for evaluation...")
    model = model.to(device)

    ppl = perplexity(model, tokenizer, device)
    print("Perplexity:", ppl)

    if OUTPUT_DIR:
        print(f"Saving quantized model and tokenizer to '{OUTPUT_DIR}'...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()