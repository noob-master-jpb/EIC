import os
import gc
import sys
import torch
import torch.nn as nn
import unittest
import unittest.mock as mock
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
from huggingface_hub import snapshot_download

# Must be set before torch initializes the CUDA allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
SEQ_LEN                    = 512
N_CALIBRATION_SAMPLES      = 128
CALIBRATION_BATCH_SIZE     = 1
MODEL_NAME                 = "Qwen/Qwen3.5-0.8B"
OUTPUT_DIR                 = "qwen3.5-0.8B-billm"

# VRAM safety: reserve this many GB as overhead buffer before moving to GPU
VRAM_OVERHEAD_RESERVE_GB   = 0.5
# Cap perplexity eval tokens; None = no cap (full test set)
MAX_EVAL_TOKENS            = 1024

# toggles
USE_SENSITIVITY_SCHEDULING = False
USE_DYNAMIC_KEEP_RATIO     = False
USE_CLIP_SEARCH            = False
USE_BLOCK_NORMALIZATION    = False

FORCE_CPU                  = os.environ.get("FORCE_CPU", "False").lower() == "true"
CPU_THREADS                = int(os.environ.get("CPU_THREADS")) if os.environ.get("CPU_THREADS") else None

torch.manual_seed(42)
random.seed(42)


# =========================
# CPU SPEC DETECTION
# =========================

def detect_cpu_threads():
    """Auto-detect optimal PyTorch thread count from CPU specs."""
    logical = os.cpu_count() or 4
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical // 2
        ram_gb = psutil.virtual_memory().total / 1024**3
        # Reserve 1 physical core for OS; prefer physical over hyperthreads
        safe = max(1, physical - 1)
        print(f"[CPU] {physical} physical / {logical} logical cores | {ram_gb:.1f} GB RAM")
    except ImportError:
        safe = max(1, logical - 2)
        print(f"[CPU] {logical} logical cores detected (install psutil for physical core info)")
    print(f"[CPU] Using {safe} threads")
    return safe


# =========================
# DEVICE DETECTION (CUDA/ROCm-safe)
# =========================

def get_device():
    """
    Detects the best available accelerator.
    PyTorch ROCm builds expose MI300X via torch.cuda (HIP aliases).
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
        print(f"[VRAM]   total={total_gb:.1f} GB | allocated={alloc_gb:.1f} GB | free={free_gb:.1f} GB")
        if alloc_gb > 1.0:
            print("[WARN]   >1 GB already allocated at startup — another process may be holding GPU memory.")
        return dev

    return torch.device("cpu")


def estimate_model_vram_gb(model):
    """Estimate model VRAM footprint in GB (params + 15% overhead buffer)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return (total_bytes * 1.15) / 1024**3


def can_fit_on_gpu(model, reserve_gb=VRAM_OVERHEAD_RESERVE_GB):
    """Return True only if free VRAM >= model size + reserve."""
    if not torch.cuda.is_available():
        return False
    free_gb = (torch.cuda.get_device_properties(0).total_memory
               - torch.cuda.memory_allocated(0)) / 1024**3
    needed_gb = estimate_model_vram_gb(model) + reserve_gb
    if free_gb < needed_gb:
        print(f"[VRAM]   Need ~{needed_gb:.1f} GB (incl. {reserve_gb:.1f} GB reserve), "
              f"only {free_gb:.1f} GB free — will use CPU.")
        return False
    return True


def _is_oom(exc):
    """True for CUDA out-of-memory on any PyTorch version."""
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type and isinstance(exc, oom_type):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


# =========================
# CORE  (BiLLM principle)
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
    global RESIDUAL_STEPS
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
def _run_calibration_on_device(model, samples, device):
    """
    Inner calibration loop — registers hooks, collects Hessian diagonals,
    then removes hooks unconditionally (even on OOM).
    """
    hess = {}
    hooks = []

    def hook(name):
        def fn(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            h = torch.mean(x.float() ** 2, dim=0).cpu() + EPSILON
            del x
            if name not in hess:
                hess[name] = h
            else:
                hess[name].add_(h)
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook(name)))

    try:
        for i, sample in enumerate(tqdm(samples, desc=f"Calibrating [{device.type.upper()}]")):
            inp = sample.to(device)
            with torch.inference_mode():
                model(inp, use_cache=False)
            del inp

            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if i % 16 == 0:
                gc.collect()
    finally:
        # Always remove hooks — prevents double-counting on OOM retry
        for h in hooks:
            h.remove()

    n = len(samples)
    for k in hess:
        hess[k] = hess[k] / n

    return hess


@torch.no_grad()
def collect_hessian(model, samples, device):
    """
    Collect Hessian diagonal estimates via forward hooks.
    Moves model to GPU if VRAM allows; falls back to CPU on OOM or tight memory.
    """
    if device.type == "cuda" and can_fit_on_gpu(model):
        print("[Calibration] Moving model to GPU...")
        model.to(device)
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 1024**3
        print(f"[Calibration] Model on GPU — {alloc:.2f} GB allocated")
        try:
            hess = _run_calibration_on_device(model, samples, device)
        except Exception as e:
            if not _is_oom(e):
                raise
            print("[OOM] CUDA out of memory during calibration — falling back to CPU.")
            torch.cuda.empty_cache()
            gc.collect()
            model.to("cpu")
            torch.cuda.empty_cache()
            hess = _run_calibration_on_device(model, samples, torch.device("cpu"))
        else:
            print("[Calibration] Moving model back to CPU...")
            model.to("cpu")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    else:
        print("[Calibration] Running on CPU (insufficient VRAM or CPU mode)...")
        hess = _run_calibration_on_device(model, samples, torch.device("cpu"))

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
        h = hessians[name]

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

        # Free intermediate tensors immediately to reduce peak RAM
        del w, saliency, mask, b

        print(f"{name} | err={err:.4f} | keep={keep_ratio:.4f}")


# =========================
# PERPLEXITY
# =========================

@torch.no_grad()
def _eval_nll(model, enc, device):
    """Inner NLL loop — caller handles device placement."""
    stride = SEQ_LEN
    # Cap window to 2048 to prevent OOM on long-context models during eval
    max_len = min(model.config.max_position_embeddings, 2048)
    nlls = []

    for i in range(0, enc.size(1), stride):
        begin = max(i + stride - max_len, 0)
        end = i + stride

        input_ids = enc[:, begin:end].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        outputs = model(input_ids, labels=target_ids)
        nlls.append(outputs.loss.cpu() * stride)

        del input_ids, target_ids, outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return nlls, stride


@torch.no_grad()
def perplexity(model, tokenizer, device):
    """
    Evaluate perplexity with OOM fallback to CPU.
    MAX_EVAL_TOKENS caps evaluation length; set to None for the full test set.
    """
    model.eval()

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids

    if MAX_EVAL_TOKENS and enc.size(1) > MAX_EVAL_TOKENS:
        print(f"[Eval] Capping eval tokens: {enc.size(1)} → {MAX_EVAL_TOKENS}")
        enc = enc[:, :MAX_EVAL_TOKENS]

    eval_device = device if (device.type == "cuda" and can_fit_on_gpu(model)) else torch.device("cpu")
    print(f"[Eval] Moving model to {eval_device} for perplexity...")
    model.to(eval_device)

    try:
        nlls, stride = _eval_nll(model, enc, eval_device)
    except Exception as e:
        if not _is_oom(e):
            raise
        print("[OOM] CUDA OOM during eval — falling back to CPU.")
        torch.cuda.empty_cache()
        gc.collect()
        model.to("cpu")
        torch.cuda.empty_cache()
        nlls, stride = _eval_nll(model, enc, torch.device("cpu"))
    finally:
        # Always return to CPU to keep VRAM free and avoid device mismatch
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.exp(torch.stack(nlls).sum() / (len(nlls) * stride)).item()


# =========================
# MODEL LOADING
# =========================

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
    # Load to CPU — moved to GPU only during calibration and eval
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return tokenizer, model


# =========================
# TESTS (Consolidated from test_billm.py)
# =========================

class TestCoreMath(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.w = torch.randn(64, 128)
        self.h = torch.rand(128).abs() + EPSILON

    def test_saliency_shape(self):
        s = compute_saliency(self.w, self.h)
        self.assertEqual(s.shape, self.w.shape)

    def test_saliency_nonnegative(self):
        s = compute_saliency(self.w, self.h)
        self.assertTrue((s >= 0).all())

    def test_mask_keep_ratio(self):
        s = compute_saliency(self.w, self.h)
        mask = apply_mask(s, 0.10)
        actual = mask.float().mean().item()
        self.assertAlmostEqual(actual, 0.10, delta=0.02)

    def test_mask_at_least_one(self):
        s = compute_saliency(self.w, self.h)
        mask = apply_mask(s, 0.0)
        self.assertGreaterEqual(mask.sum().item(), 1)

    def test_alpha_shape(self):
        alpha = compute_alpha(self.w)
        self.assertEqual(alpha.shape, (64, 1))

    def test_alpha_nonneg(self):
        alpha = compute_alpha(self.w)
        self.assertTrue((alpha >= 0).all())

    def test_binarize_shape(self):
        s = compute_saliency(self.w, self.h)
        mask = apply_mask(s, 0.05)
        b = billm_binarize(self.w, mask)
        self.assertEqual(b.shape, self.w.shape)

    def test_binarize_non_salient_are_binary(self):
        s = compute_saliency(self.w, self.h)
        mask = apply_mask(s, 0.05)
        b = billm_binarize(self.w, mask)
        alpha = compute_alpha(self.w)

        rows, cols = torch.where(~mask)
        expected = alpha[rows, 0] * torch.sign(self.w[rows, cols])
        self.assertTrue(
            torch.allclose(b[rows, cols], expected, atol=1e-6),
            "Non-salient weights are not correctly binarized"
        )

    def test_residual_does_not_increase_error(self):
        s = compute_saliency(self.w, self.h)
        mask = apply_mask(s, 0.05)
        b_no_res = billm_binarize(self.w, mask)

        global RESIDUAL_STEPS
        orig = RESIDUAL_STEPS
        RESIDUAL_STEPS = 1
        b_res = apply_residual(self.w, b_no_res.clone(), mask)
        RESIDUAL_STEPS = orig

        err_pre  = torch.norm(self.w - b_no_res).item()
        err_post = torch.norm(self.w - b_res).item()
        self.assertLessEqual(err_post, err_pre + 1e-4)

    def test_get_keep_ratio_known_layers(self):
        self.assertAlmostEqual(get_keep_ratio("q_proj"),    0.06)
        self.assertAlmostEqual(get_keep_ratio("k_proj"),    0.06)
        self.assertAlmostEqual(get_keep_ratio("down_proj"), 0.02)
        self.assertAlmostEqual(get_keep_ratio("gate_proj"), 0.02)

    def test_get_keep_ratio_default(self):
        self.assertAlmostEqual(get_keep_ratio("embed_tokens"), 0.02)


class TestDeviceLogic(unittest.TestCase):

    def test_get_device_returns_device_object(self):
        dev = get_device()
        self.assertIsInstance(dev, torch.device)

    def test_estimate_model_vram_positive(self):
        model = nn.Linear(256, 256, bias=False)
        gb = estimate_model_vram_gb(model)
        self.assertGreater(gb, 0)
        self.assertLess(gb, 1.0)

    def test_estimate_model_vram_scales_with_size(self):
        small = nn.Linear(64,  64,  bias=False)
        large = nn.Linear(256, 256, bias=False)
        self.assertLess(
            estimate_model_vram_gb(small),
            estimate_model_vram_gb(large)
        )

    def test_detect_cpu_threads_positive(self):
        n = detect_cpu_threads()
        self.assertGreaterEqual(n, 1)

    def test_is_oom_detects_runtime_error(self):
        e = RuntimeError("CUDA out of memory. Tried to allocate ...")
        self.assertTrue(_is_oom(e))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj   = nn.Linear(32, 32, bias=False)
        self.k_proj   = nn.Linear(32, 32, bias=False)
        self.v_proj   = nn.Linear(32, 32, bias=False)
        self.o_proj   = nn.Linear(32, 32, bias=False)
        self.up_proj  = nn.Linear(32, 64, bias=False)
        self.gate_proj = nn.Linear(32, 64, bias=False)
        self.down_proj = nn.Linear(64, 32, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x) * torch.sigmoid(self.gate_proj(x)))


def _make_fake_hessians(model):
    hess = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hess[name] = torch.rand(m.in_features).abs() + EPSILON
    return hess


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel()

    def test_quantize_runs_without_error(self):
        quantize(self.model, _make_fake_hessians(self.model))

    def test_billm_principle_most_weights_binarized(self):
        global RESIDUAL_STEPS
        orig_res = RESIDUAL_STEPS
        RESIDUAL_STEPS = 0
        hess = _make_fake_hessians(self.model)
        orig_weights = {n: m.weight.data.clone() for n, m in self.model.named_modules() if isinstance(m, nn.Linear)}
        try:
            quantize(self.model, hess)
        finally:
            RESIDUAL_STEPS = orig_res

        for name, m in self.model.named_modules():
            if not isinstance(m, nn.Linear): continue
            w = m.weight.data
            w_orig = orig_weights[name]
            h = hess[name]
            keep_ratio = get_keep_ratio(name)
            saliency = compute_saliency(w_orig, h)
            mask = apply_mask(saliency, keep_ratio)
            for i in range(w.shape[0]):
                row_w = w[i]
                row_mask = mask[i]
                non_salient_abs = row_w[~row_mask].abs()
                if non_salient_abs.numel() > 0:
                    alpha = non_salient_abs.mean()
                    self.assertTrue(torch.allclose(non_salient_abs, alpha, atol=1e-5))


class TestHookIsolation(unittest.TestCase):

    def test_hooks_removed_after_normal_run(self):
        model = nn.Linear(16, 16, bias=False)
        samples = [torch.randint(0, 100, (1, 16)) for _ in range(2)]
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = model
                self.config = type("cfg", (), {"use_cache": False})()
            def forward(self, input_ids, use_cache=False):
                return self.linear(input_ids.float())
        wrapped = Wrapper()
        before = len(wrapped.linear._forward_hooks)
        _run_calibration_on_device(wrapped, samples, torch.device("cpu"))
        after = len(wrapped.linear._forward_hooks)
        self.assertEqual(before, after)


class TestRealModelSmoke(unittest.TestCase):
    """
    Skipped unless MODEL_PATH env var points to a local Qwen3.5-0.8B directory.
    """

    @classmethod
    def setUpClass(cls):
        model_path = os.environ.get("MODEL_PATH", "")
        if not model_path or not os.path.isdir(model_path):
            raise unittest.SkipTest("Set MODEL_PATH=<local-dir> to run real model smoke tests")
        print(f"\n[Smoke] Loading model from {model_path}...")
        cls.tokenizer = AutoTokenizer.from_pretrained(model_path)
        cls.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        cls.model.config.use_cache = False

    def test_calibration_produces_positive_hessians(self):
        samples = get_data(self.tokenizer)
        hess = collect_hessian(self.model, samples[:4], torch.device("cpu"))
        self.assertGreater(len(hess), 0)
        for k, v in hess.items():
            self.assertGreater(v.min().item(), 0)

    def test_quantize_completes(self):
        samples = get_data(self.tokenizer)
        hess = collect_hessian(self.model, samples[:4], torch.device("cpu"))
        quantize(self.model, hess)

    def test_perplexity_is_finite(self):
        device = get_device()
        ppl = perplexity(self.model, self.tokenizer, device)
        self.assertTrue(torch.isfinite(torch.tensor(ppl)))
        print(f"\n[Smoke] Perplexity: {ppl:.2f}")


# =========================
# MAIN
# =========================

def main():
    device = get_device()

    n_threads = CPU_THREADS if CPU_THREADS is not None else detect_cpu_threads()
    torch.set_num_threads(n_threads)

    if device.type == "cuda":
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] Total VRAM: {total_vram:.1f} GB | Overhead reserve: {VRAM_OVERHEAD_RESERVE_GB:.1f} GB")

    tokenizer, model = download_and_load_model(MODEL_NAME, "models")
    model.config.use_cache = False

    samples = get_data(tokenizer)
    hess = collect_hessian(model, samples, device)

    print("Quantizing on CPU (weight-only, no activations needed)...")
    gc.collect()

    quantize(model, hess)

    ppl = perplexity(model, tokenizer, device)
    print("Perplexity:", ppl)

    if OUTPUT_DIR:
        print(f"Saving quantized model and tokenizer to '{OUTPUT_DIR}'...")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    if "--test" in sys.argv:
        # Remove --test from argv so unittest doesn't complain
        sys.argv.remove("--test")
        print("=" * 60)
        print("BiLLM Test Suite — Consolidated Mode")
        print("=" * 60)
        unittest.main(verbosity=2)
    else:
        main()
