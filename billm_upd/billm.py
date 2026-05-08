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
EPSILON                    = 1e-8
SEQ_LEN                    = 512
N_CALIBRATION_SAMPLES      = 128
CALIBRATION_BATCH_SIZE     = 6   # MI300X: larger batches fill HBM bandwidth
QUANTIZE_BLOCK_SIZE        = 128 # Must be 128 because intra-block error is not compensated
MODEL_NAME                 = "google/gemma-4-31B-it"
OUTPUT_DIR                 = "gemma-4-E4B-billm"

# VRAM safety: reserve this many GB as overhead buffer before moving to GPU
VRAM_OVERHEAD_RESERVE_GB   = 0.5
# Cap perplexity eval tokens; None = no cap (full test set)
MAX_EVAL_TOKENS            = 1024

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

def billm_quantize_block(Wb, C_diag_b, keep_ratio):
    """
    True BiLLM block binarization:
    1. Structural (column) salient selection
    2. Binary Residual Approximation for salient
    3. Bell-shaped splitting for non-salient
    """
    # Saliency = w_i^2 / [H^-1]_ii^2
    saliency = Wb ** 2 / (C_diag_b.unsqueeze(0) ** 2 + EPSILON)
    col_saliency = saliency.sum(dim=0)
    
    n_cols = Wb.shape[1]
    k = max(1, int(n_cols * keep_ratio))
    
    if k >= n_cols:
        salient_cols = torch.arange(n_cols, device=Wb.device)
        non_salient_cols = torch.tensor([], dtype=torch.long, device=Wb.device)
    else:
        salient_cols = torch.topk(col_saliency, k).indices
        non_salient_mask = torch.ones(n_cols, dtype=torch.bool, device=Wb.device)
        non_salient_mask[salient_cols] = False
        non_salient_cols = torch.where(non_salient_mask)[0]
        
    Qb = torch.zeros_like(Wb)
    
    # 1. Salient Weights: Binary Residual Approximation
    if len(salient_cols) > 0:
        W_sal = Wb[:, salient_cols]
        alpha_o = W_sal.abs().mean(dim=1, keepdim=True)
        B_o = torch.sign(W_sal)
        Residual = W_sal - alpha_o * B_o
        
        alpha_r = Residual.abs().mean(dim=1, keepdim=True)
        B_r = torch.sign(Residual)
        
        Qb[:, salient_cols] = alpha_o * B_o + alpha_r * B_r
        
    # 2. Non-Salient Weights: Bell-shaped Splitting Search (fully vectorized)
    if len(non_salient_cols) > 0:
        W_uns = Wb[:, non_salient_cols]          # [R, C_uns]
        W_uns_abs = W_uns.abs()
        W_uns_sign = torch.sign(W_uns)

        percentiles = torch.linspace(0.1, 0.9, 9, device=Wb.device)
        if W_uns_abs.numel() > 16777216:
            sample_size = min(1_000_000, W_uns_abs.numel())
            idx = torch.randint(0, W_uns_abs.numel(), (sample_size,), device=Wb.device)
            p_cands = torch.quantile(
                W_uns_abs.flatten()[idx].float(), percentiles.float()
            ).to(W_uns_abs.dtype)               # [9]
        else:
            p_cands = torch.quantile(
                W_uns_abs.float(), percentiles.float()
            ).to(W_uns_abs.dtype)               # [9]

        # Vectorize over all 9 thresholds simultaneously
        # W_uns_abs: [R, C]  p_cands: [9]  -> broadcast to [9, R, C]
        p = p_cands[:, None, None]              # [9, 1, 1]
        a = W_uns_abs.unsqueeze(0)              # [1, R, C]
        sg = W_uns_sign.unsqueeze(0)            # [1, R, C]

        mask_c = a <= p                         # [9, R, C]  ("center" region)
        mask_s = ~mask_c                        # [9, R, C]  ("spike" region)

        # Per-row, per-candidate alpha values
        alpha_c = (a * mask_c).sum(-1, keepdim=True) / mask_c.sum(-1, keepdim=True).clamp(min=1)  # [9, R, 1]
        alpha_s = (a * mask_s).sum(-1, keepdim=True) / mask_s.sum(-1, keepdim=True).clamp(min=1)  # [9, R, 1]

        Q_all = alpha_c * sg * mask_c + alpha_s * sg * mask_s  # [9, R, C]

        # Squared Frobenius error per candidate
        errs = ((a - Q_all.abs()) ** 2).sum(dim=(-2, -1))      # [9]
        best_idx = errs.argmin()
        Qb[:, non_salient_cols] = Q_all[best_idx]

    return Qb


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
            # BiLLM / GPTQ uses full Hessian: H = X^T X / N
            # Must move to CPU: for 31B models, keeping all Hessians in VRAM causes OOM!
            h = (torch.matmul(x.float().t(), x.float()) / x.shape[0]).cpu()
            del x
            if name not in hess:
                hess[name] = h
            else:
                hess[name].add_(h)
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook(name)))

    # Batch samples to improve GPU utilization
    batch_size = CALIBRATION_BATCH_SIZE if device.type == "cuda" else 1
    try:
        for i in tqdm(range(0, len(samples), batch_size), desc=f"Calibrating [{device.type.upper()}]"):
            batch = torch.cat(samples[i:i+batch_size], dim=0).to(device)
            with torch.inference_mode():
                model(batch, use_cache=False)
            del batch

            if device.type == "cuda":
                torch.cuda.synchronize()
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
    layer_order = [n for n, m in modules.items() if isinstance(m, nn.Linear)]

    # Determine compute device — with 200 GB VRAM everything stays on GPU
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in tqdm(layer_order, desc="Quantizing layers"):
        if name not in hessians:
            continue

        m = modules[name]
        # Move to fast compute device
        W = m.weight.data.clone().float().to(dev)
        H = hessians[name].to(dev).float()
        
        # GPTQ-style damping
        damp = 0.01
        diag = torch.diag(H)
        H[torch.arange(H.shape[0]), torch.arange(H.shape[0])] += damp * diag.mean()
        
        for attempt in range(100):
            try:
                H_inv = torch.linalg.inv(H)
                U = torch.linalg.cholesky(H_inv, upper=True)
                break
            except Exception:
                if attempt == 0:
                    print(f"[{name}] Cholesky failed, adding extra damping...")
                H[torch.arange(H.shape[0]), torch.arange(H.shape[0])] += 0.05 * diag.mean()
        else:
            raise RuntimeError(f"[{name}] Cholesky failed even after maximum damping attempts.")
            
        C_diag = torch.diag(H_inv)
        U_diag = torch.diag(U)
        
        keep_ratio = get_keep_ratio(name)
        block_size = QUANTIZE_BLOCK_SIZE
        in_features = W.shape[1]
        
        for i1 in range(0, in_features, block_size):
            i2 = min(i1 + block_size, in_features)
            
            Wb = W[:, i1:i2]
            C_diag_b = C_diag[i1:i2]
            
            Qb = billm_quantize_block(Wb, C_diag_b, keep_ratio)
            
            # Exact Block-wise OBC error update without approximation
            Err = (Wb - Qb) @ torch.linalg.inv(U[i1:i2, i1:i2])
            if i2 < in_features:
                W[:, i2:] -= Err.matmul(U[i1:i2, i2:])
                
            # Update quantized weights
            W[:, i1:i2] = Qb
            
        # Copy back to the original parameter
        m.weight.data.copy_(W.to(dtype=m.weight.data.dtype, device=m.weight.data.device))
        
        del H, H_inv, U, W
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"{name} | keep={keep_ratio:.4f}")


# =========================
# PERPLEXITY
# =========================

@torch.no_grad()
def _eval_nll(model, enc, device):
    """Inner NLL loop — caller handles device placement."""
    stride = SEQ_LEN
    # Cap window to 2048 to prevent OOM on long-context models during eval
    max_len = min(getattr(model.config, "max_position_embeddings", 8192), 2048)
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
        self.Wb = torch.randn(64, 128)
        self.C_diag_b = torch.rand(128).abs() + EPSILON

    def test_billm_quantize_block_shape(self):
        Qb = billm_quantize_block(self.Wb, self.C_diag_b, 0.1)
        self.assertEqual(Qb.shape, self.Wb.shape)

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
            H = torch.randn(m.in_features, m.in_features)
            H = H.t().matmul(H) + torch.eye(m.in_features)
            hess[name] = H
    return hess


class TestEndToEnd(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyModel()

    def test_quantize_runs_without_error(self):
        quantize(self.model, _make_fake_hessians(self.model))


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

    # quantize() moves each layer's W/H to GPU and writes back — model weights on CPU
    print("Quantizing (GPU-accelerated)...")
    quantize(model, hess)
    del hess  # free GPU Hessian memory before eval

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
