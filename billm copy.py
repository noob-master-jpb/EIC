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
SEQ_LEN                    = 128
N_CALIBRATION_SAMPLES      = 128
MODEL_NAME                 = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR                 = "qwen3.5-0.8B-billm"

# toggles
USE_SENSITIVITY_SCHEDULING = False
USE_DYNAMIC_KEEP_RATIO     = False
USE_CLIP_SEARCH            = False
USE_BLOCK_NORMALIZATION    = False

FORCE_CPU                  = False
CPU_THREADS                = 16  # Number of CPU cores to use if running on CPU
QUANTIZE_ON_GPU            = True  # Set to True to run quantization on GPU (uses more VRAM)

torch.manual_seed(42)
random.seed(42)


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
    hess = {}
    hooks = []

    def hook(name):
        def fn(module, inp, out):
            x = inp[0]
            if x.dim() == 3:
                x = x.view(-1, x.shape[-1])

            # .clone() escapes inference-mode so we get a normal tensor
            h = torch.mean(x**2, dim=0).cpu().clone() + EPSILON
            if name not in hess:
                hess[name] = h
            else:
                hess[name] = hess[name] + h   # out-of-place: avoids inference-tensor inplace error
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook(name)))

    for s in tqdm(samples):
        with torch.inference_mode():
            model(s.to(device))
        del s
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    for k in hess:
        hess[k] = hess[k] / len(samples)   # out-of-place: tensors may be inference-mode tagged

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

    # optional sensitivity scheduling
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

    for name in tqdm(layer_order):
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

        # optional normalization
        if USE_BLOCK_NORMALIZATION:
            ratio = w.norm() / (b.norm() + EPSILON)
            ratio = torch.clamp(ratio, 0.85, 1.2)
            b *= ratio

        m.weight.data.copy_(b)

        err = torch.norm(w - b) / (torch.norm(w) + EPSILON)
        accumulated_error += err.item()

        print(f"{name} | err={err:.4f} | keep={keep_ratio:.4f}")


# =========================
# PERPLEXITY (CORRECT)
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
# MAIN
# =========================

import os
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
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

def main():
    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cpu":
        torch.set_num_threads(CPU_THREADS)
        print(f"Running on CPU with {CPU_THREADS} threads.")
    else:
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    tokenizer, model = download_and_load_model(MODEL_NAME, "models")
    model.config.use_cache = False

    samples = get_data(tokenizer)
    hess = collect_hessian(model, samples, device)

    if QUANTIZE_ON_GPU and device.type == "cuda":
        print("Quantizing on GPU...")
        model = model.to(device)
    else:
        print("Moving model to CPU for quantization...")
        model = model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

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