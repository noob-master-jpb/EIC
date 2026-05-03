import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random

# =========================
# CONFIG
# =========================
CONFIG = {
    "keep_ratio": {
        "q_proj": 0.06, "k_proj": 0.06, "v_proj": 0.06,
        "o_proj": 0.03, "up_proj": 0.02, "down_proj": 0.02, "gate_proj": 0.02,
        "default": 0.02
    },
    "residual_steps": 1,
    "epsilon": 1e-8,
    "seq_len": 256,
    "n_calibration_samples": 256,
    "model_name": "Qwen/Qwen3.5-0.8B",

    # toggles
    "use_sensitivity_scheduling": False,
    "use_dynamic_keep_ratio": False,
    "use_clip_search": False,
    "use_block_normalization": False,
}

torch.manual_seed(42)
random.seed(42)


# =========================
# CORE
# =========================

def compute_saliency(w, h):
    return (w ** 2) / (h.unsqueeze(0) + CONFIG["epsilon"])


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
    if not CONFIG["use_clip_search"]:
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
    b[~mask] = (alpha * sign)[~mask]

    return b


def apply_residual(w, b, mask):
    for _ in range(CONFIG["residual_steps"]):
        r = (w - b) * (~mask)

        alpha = compute_alpha(r)
        sign = torch.sign(r)

        update = torch.zeros_like(w)
        update[~mask] = (alpha * sign)[~mask]

        b = b + update

    return b


# =========================
# CALIBRATION
# =========================

@torch.no_grad()
def get_data(tokenizer):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(dataset["text"])
    enc = tokenizer(text, return_tensors="pt")

    samples = []
    for _ in range(CONFIG["n_calibration_samples"]):
        i = random.randint(0, enc.input_ids.shape[1] - CONFIG["seq_len"] - 1)
        samples.append(enc.input_ids[:, i:i+CONFIG["seq_len"]])
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

            h = torch.mean(x**2, dim=0) + CONFIG["epsilon"]
            hess[name] = hess.get(name, 0) + h
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook(name)))

    for s in tqdm(samples):
        model(s.to(device))

    for h in hooks:
        h.remove()

    for k in hess:
        hess[k] /= len(samples)

    return hess


# =========================
# QUANTIZATION
# =========================

def get_keep_ratio(name):
    for k, v in CONFIG["keep_ratio"].items():
        if k in name:
            return v
    return CONFIG["keep_ratio"]["default"]


@torch.no_grad()
def quantize(model, hessians):

    modules = dict(model.named_modules())

    # optional sensitivity scheduling
    if CONFIG["use_sensitivity_scheduling"]:
        sensitivities = {}
        for name, m in modules.items():
            if isinstance(m, nn.Linear) and name in hessians:
                w = m.weight.data
                h = hessians[name].to(w.device)
                sensitivities[name] = ((w**2)/(h+CONFIG["epsilon"])).mean().item()

        layer_order = sorted(sensitivities, key=lambda x: sensitivities[x])
    else:
        layer_order = [n for n, m in modules.items() if isinstance(m, nn.Linear)]

    accumulated_error = 0.0

    for name in tqdm(layer_order):
        if name not in hessians:
            continue

        m = modules[name]
        w = m.weight.data
        h = hessians[name].to(w.device)

        keep_ratio = get_keep_ratio(name)

        if CONFIG["use_dynamic_keep_ratio"]:
            keep_ratio = min(
                keep_ratio * (1 + min(0.3, accumulated_error * 0.5)),
                keep_ratio * 2
            )

        saliency = compute_saliency(w, h)
        mask = apply_mask(saliency, keep_ratio)

        b = billm_binarize(w, mask)
        b = apply_residual(w, b, mask)

        # optional normalization
        if CONFIG["use_block_normalization"]:
            ratio = w.norm() / (b.norm() + CONFIG["epsilon"])
            ratio = torch.clamp(ratio, 0.85, 1.2)
            b *= ratio

        m.weight.data.copy_(b)

        err = torch.norm(w - b) / (torch.norm(w) + CONFIG["epsilon"])
        accumulated_error += err.item()

        print(f"{name} | err={err:.4f} | keep={keep_ratio:.4f}")


# =========================
# PERPLEXITY (CORRECT)
# =========================

@torch.no_grad()
def perplexity(model, tokenizer, device):
    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids.to(device)

    stride = CONFIG["seq_len"]
    max_len = model.config.max_position_embeddings

    nlls = []

    for i in range(0, enc.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = i + stride

        input_ids = enc[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss * stride)

    return torch.exp(torch.stack(nlls).sum() / (len(nlls) * stride)).item()


# =========================
# MAIN
# =========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

    samples = get_data(tokenizer)
    hess = collect_hessian(model, samples, device)

    quantize(model, hess)

    ppl = perplexity(model, tokenizer, device)
    print("Perplexity:", ppl)


if __name__ == "__main__":
    main()