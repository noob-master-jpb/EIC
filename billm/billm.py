import os
import gc
import sys
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import unittest.mock as mock

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

from datasets import load_dataset
from tqdm import tqdm
import random

from huggingface_hub import snapshot_download

# Must be set before torch initializes the CUDA allocator
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True"
)

# ============================================================
# CONFIG
# ============================================================

KEEP_RATIO = {
    # Attention projections are extremely sensitive
    # These are now PRESERVED for salient columns.
    "q_proj": 0.99,
    "k_proj": 0.99,

    # Moderate protection
    "v_proj": 0.99,
    "o_proj": 0.99,

    # MLP can tolerate more ternary
    "up_proj": 0.4,
    "down_proj": 0.4,
    "gate_proj": 0.4,

    "default": 0.5
}

EPSILON = 1e-8

SEQ_LEN = 256
# Better Hessian estimation
N_CALIBRATION_SAMPLES = 512*75

# Keep your GPU behavior unchanged
CALIBRATION_BATCH_SIZE = 135

# Stable local quantization
QUANTIZE_BLOCK_SIZE = 64

MODEL_NAME = "/root/EIC/gemma-4-31B-it-merged"
OUTPUT_DIR = "/root/EIC/gemma-4-31B-it-merged-billm"

# MODEL_NAME = "google/gemma-4-E4B-it"
# OUTPUT_DIR = "gemma-4-E4B-billm"

# Calibration datasets — local paths or HF names
DATASETS = [
    "/root/EIC/Datasets/openhermes.parquet",
    "/root/EIC/Datasets/oss-ins-75k.parquet",
    "/root/EIC/Datasets/nvidia_compute_eval.parquet",
    "/root/EIC/Datasets/cass_diverse_selected.parquet",
    "/root/EIC/Datasets/cass_part1.parquet",
    "/root/EIC/Datasets/codefeedback_part1.parquet",
    "/root/EIC/Datasets/cuda_to_rocm_distill_glm5.jsonl",
    "/root/EIC/Datasets/open_thoughts_reasoning.parquet",
    "/root/EIC/Datasets/smoltalk_chat.parquet",
    "/root/EIC/Datasets/numina_math_reasoning.parquet",
    # "wikitext",
]

# Maps each dataset path → [prompt_col, response_col] to concatenate.
# If a dataset is NOT listed here, get_data() falls back to auto-detection.
COLUMN_MAPPING = {
    "/root/EIC/Datasets/openhermes.parquet":            ["problem", "solution"],
    "/root/EIC/Datasets/oss-ins-75k.parquet":           ["problem", "solution"],
    "/root/EIC/Datasets/nvidia_compute_eval.parquet":   ["problem", "solution"],
    "/root/EIC/Datasets/cass_diverse_selected.parquet": ["problem", "answer"],
    "/root/EIC/Datasets/cass_part1.parquet":            ["problem", "answer"],
    "/root/EIC/Datasets/codefeedback_part1.parquet":    ["query", "answer"],
    "/root/EIC/Datasets/cuda_to_rocm_distill_glm5.jsonl": ["problem", "response"],
    "/root/EIC/Datasets/open_thoughts_reasoning.parquet": ["problem", "solution"],
    "/root/EIC/Datasets/smoltalk_chat.parquet":           ["problem", "solution"],
    "/root/EIC/Datasets/numina_math_reasoning.parquet":   ["problem", "solution"],
}

VRAM_OVERHEAD_RESERVE_GB = 0.5

MAX_EVAL_TOKENS = 4096//4

FORCE_CPU = (
    os.environ.get("FORCE_CPU", "False")
    .lower() == "true"
)

CPU_THREADS = (
    int(os.environ.get("CPU_THREADS"))
    if os.environ.get("CPU_THREADS")
    else None
)

# Stability protections
SKIP_KEYWORDS = [
    "lm_head",
    "embed_tokens",
]

PROTECT_FIRST_LAYERS = 2
PROTECT_LAST_LAYERS = 2

# Groupwise alpha
GROUP_SIZE = 32

torch.manual_seed(42)
random.seed(42)

# ============================================================
# CPU SPEC DETECTION
# ============================================================

def detect_cpu_threads():

    logical = os.cpu_count() or 4

    try:
        import psutil

        physical = (
            psutil.cpu_count(logical=False)
            or logical // 2
        )

        ram_gb = (
            psutil.virtual_memory().total
            / 1024**3
        )

        safe = max(1, physical - 1)

        print(
            f"[CPU] {physical} physical / "
            f"{logical} logical cores | "
            f"{ram_gb:.1f} GB RAM"
        )

    except ImportError:

        safe = max(1, logical - 2)

        print(
            f"[CPU] {logical} logical cores "
            f"detected"
        )

    print(f"[CPU] Using {safe} threads")

    return safe

# ============================================================
# DEVICE DETECTION
# ============================================================

def get_device():

    if FORCE_CPU:
        return torch.device("cpu")

    if torch.cuda.is_available():

        dev = torch.device("cuda")

        name = torch.cuda.get_device_name(0)

        props = torch.cuda.get_device_properties(0)

        total_gb = props.total_memory / 1024**3

        free_gb = (
            props.total_memory
            - torch.cuda.memory_allocated(0)
        ) / 1024**3

        alloc_gb = (
            torch.cuda.memory_allocated(0)
            / 1024**3
        )

        hip_ver = getattr(
            torch.version,
            "hip",
            None
        )

        runtime = (
            f"ROCm {hip_ver}"
            if hip_ver
            else f"CUDA {torch.version.cuda}"
        )

        print(f"[Device] {runtime} — {name}")

        print(
            f"[VRAM] total={total_gb:.1f} GB | "
            f"allocated={alloc_gb:.1f} GB | "
            f"free={free_gb:.1f} GB"
        )

        return dev

    return torch.device("cpu")

# ============================================================
# VRAM ESTIMATION
# ============================================================

def estimate_model_vram_gb(model):

    total_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    return (total_bytes * 1.15) / 1024**3

def can_fit_on_gpu(
    model,
    reserve_gb=VRAM_OVERHEAD_RESERVE_GB
):

    if not torch.cuda.is_available():
        return False

    free_gb = (
        torch.cuda.get_device_properties(0).total_memory
        - torch.cuda.memory_allocated(0)
    ) / 1024**3

    needed_gb = (
        estimate_model_vram_gb(model)
        + reserve_gb
    )

    if free_gb < needed_gb:

        print(
            f"[VRAM] Need ~{needed_gb:.1f} GB "
            f"(incl reserve), "
            f"only {free_gb:.1f} GB free."
        )

        return False

    return True

# ============================================================
# OOM DETECTION
# ============================================================

def _is_oom(exc):

    oom_type = getattr(
        torch.cuda,
        "OutOfMemoryError",
        None
    )

    if oom_type and isinstance(exc, oom_type):
        return True

    return (
        isinstance(exc, RuntimeError)
        and "out of memory" in str(exc).lower()
    )

# ============================================================
# CORE QUANTIZATION
# ============================================================

def billm_quantize_block(
    Wb,
    C_diag_b,
    keep_ratio
):
    """
    Mixed-precision BiLLM-style quantization.

    IMPORTANT:
    ----------
    - Salient columns preserved in FP16/BF16.
    - Only non-salient columns ternarized.
    - Activation-aware scaling.
    - Groupwise alpha.
    """

    # ========================================================
    # Activation-aware saliency
    # ========================================================

    act_scale = 1.0 / (
        torch.sqrt(C_diag_b + EPSILON)
    )

    saliency = (
        Wb.abs()
        * act_scale.unsqueeze(0)
    )

    col_saliency = saliency.sum(dim=0)

    n_cols = Wb.shape[1]

    k = max(
        1,
        int(n_cols * keep_ratio)
    )

    # ========================================================
    # Column split
    # ========================================================

    if k >= n_cols:

        salient_cols = torch.arange(
            n_cols,
            device=Wb.device
        )

        non_salient_cols = torch.tensor(
            [],
            dtype=torch.long,
            device=Wb.device
        )

    else:

        salient_cols = torch.topk(
            col_saliency,
            k
        ).indices

        mask = torch.ones(
            n_cols,
            dtype=torch.bool,
            device=Wb.device
        )

        mask[salient_cols] = False

        non_salient_cols = torch.where(mask)[0]

    Qb = torch.empty_like(Wb)

    # ========================================================
    # PRESERVE salient columns
    # ========================================================

    if len(salient_cols) > 0:

        Qb[:, salient_cols] = (
            Wb[:, salient_cols]
        )

    # ========================================================
    # TERNARIZE only non-salient columns
    # ========================================================

    if len(non_salient_cols) > 0:

        W_uns = Wb[:, non_salient_cols]

        W_abs = W_uns.abs()

        W_sign = torch.sign(W_uns)

        # ====================================================
        # Activation-aware normalization
        # ====================================================

        scale = (
            act_scale[non_salient_cols]
            .unsqueeze(0)
        )

        W_scaled = W_abs * scale

        # ====================================================
        # Gentler threshold candidates
        # ====================================================

        pcts = torch.tensor(
            [
                0.60,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.92,
                0.95
            ],
            device=Wb.device
        )

        if W_scaled.numel() > 2_000_000:

            idx = torch.randint(
                0,
                W_scaled.numel(),
                (
                    min(
                        500_000,
                        W_scaled.numel()
                    ),
                ),
                device=Wb.device
            )

            t_cands = torch.quantile(
                W_scaled.flatten()[idx].float(),
                pcts.float()
            ).to(W_scaled.dtype)

        else:

            t_cands = torch.quantile(
                W_scaled.float(),
                pcts.float()
            ).to(W_scaled.dtype)

        # ====================================================
        # Vectorized threshold search
        # ====================================================

        t = t_cands[:, None, None]

        wa = W_scaled.unsqueeze(0)

        ws = W_sign.unsqueeze(0)

        nz = wa >= t

        # ====================================================
        # GROUPWISE alpha
        # ====================================================

        group_size = GROUP_SIZE

        R, C = W_uns.shape

        pad = (
            group_size
            - (C % group_size)
        ) % group_size

        if pad > 0:

            wa_pad = F.pad(
                wa,
                (0, pad)
            )

            nz_pad = F.pad(
                nz,
                (0, pad)
            )

        else:

            wa_pad = wa
            nz_pad = nz

        G = (
            wa_pad.shape[-1]
            // group_size
        )

        wa_g = wa_pad.view(
            wa.shape[0],
            R,
            G,
            group_size
        )

        nz_g = nz_pad.view(
            nz.shape[0],
            R,
            G,
            group_size
        )

        alpha = (
            (wa_g * nz_g).sum(-1, keepdim=True)
            / nz_g.sum(-1, keepdim=True)
            .clamp(min=1)
        )

        alpha = alpha.repeat_interleave(
            group_size,
            dim=-1
        )

        alpha = alpha.view(
            nz_pad.shape
        )[..., :C]

        # ====================================================
        # Reconstruction
        # ====================================================

        Q_all = alpha * ws * nz

        errs = (
            (wa - Q_all.abs()) ** 2
        ).sum(dim=(-2, -1))

        best_idx = errs.argmin()

        Q_best = Q_all[best_idx]

        # Undo activation scaling
        Q_best = Q_best / scale

        Qb[:, non_salient_cols] = Q_best

    return Qb

# ============================================================
# CALIBRATION DATA
# ============================================================

@torch.no_grad()
def get_data(tokenizer):
    dataset_texts = []

    for ds_path in DATASETS:
        print(f"[Data] Loading {ds_path}...")
        ds_text = []
        try:
            if os.path.exists(ds_path):
                if ds_path.endswith(".parquet"):
                    ds = load_dataset("parquet", data_files=ds_path, split="train")
                elif ds_path.endswith(".jsonl"):
                    ds = load_dataset("json", data_files=ds_path, split="train")
                else:
                    print(f"[Warn] Unsupported local format: {ds_path}")
                    continue
            elif ds_path == "wikitext":
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            else:
                # Try loading as a standard HF dataset
                ds = load_dataset(ds_path, split="train")

            # Check if this dataset has a specific mapping
            cols_to_extract = COLUMN_MAPPING.get(ds_path)

            subset = ds.select(range(min(5000, len(ds))))
            
            if cols_to_extract:
                # Gemma-4 thinking format (verified from chat_template.jinja):
                #
                # INPUT trigger:  <|think|>  — injected at top of system turn
                #                             via enable_thinking=True in apply_chat_template
                #
                # OUTPUT format:  <|channel>thought\n{reasoning}\n<channel|>{answer}<turn|>
                #
                # apply_chat_template with enable_thinking=True handles the input side.
                # For the assistant turn we build the wire format manually because
                # strip_thinking() in the template would erase the channel blocks
                # if passed inside 'content'.
                prompt_col, response_col = cols_to_extract[0], cols_to_extract[1]
                for row in subset:
                    user_text = str(row.get(prompt_col, "") or "").strip()
                    assistant_text = str(row.get(response_col, "") or "").strip()
                    if not user_text or not assistant_text:
                        continue
                    try:
                        # User turn: enable_thinking=True injects <|think|> into the
                        # system block and adds <|turn>model\n (no suppressor channel)
                        user_turn = tokenizer.apply_chat_template(
                            [{"role": "user", "content": user_text}],
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                        # Strip <think>...</think> wrapper that some datasets include
                        # (e.g. open_thoughts) — we provide the channel wrapper ourselves.
                        thinking_text = ""
                        answer_text = assistant_text
                        think_match = re.match(r"<think>(.*?)</think>\s*(.*)", assistant_text, re.DOTALL)
                        if think_match:
                            thinking_text = think_match.group(1).strip()
                            answer_text   = think_match.group(2).strip()

                        model_turn = f"<|channel>thought\n{thinking_text}\n<channel|>{answer_text}<turn|>\n"
                        ds_text.append(user_turn + model_turn)
                    except Exception:
                        # Fallback: raw concat if template fails
                        ds_text.append(f"{user_text}\n\n{assistant_text}")


            else:
                # Auto-detect a text column
                possible_fields = ["text", "content", "instruction", "output", "code"]
                text_field = next((f for f in possible_fields if f in ds.column_names), None)

                if text_field:
                    ds_text.extend([str(x) for x in subset[text_field] if x])
                else:
                    print(f"[Warn] No text field found in {ds_path}. Columns: {ds.column_names}")

            if ds_text:
                dataset_texts.append(ds_text)

        except Exception as e:
            print(f"[Error] Failed to load {ds_path}: {e}")

    if not dataset_texts:
        raise RuntimeError("No calibration data could be loaded!")

    samples = []
    
    samples_per_dataset = N_CALIBRATION_SAMPLES // len(dataset_texts)
    remainder = N_CALIBRATION_SAMPLES % len(dataset_texts)

    for idx, ds_lines in enumerate(dataset_texts):
        n_samples = samples_per_dataset + (remainder if idx == len(dataset_texts) - 1 else 0)
        
        # We only need enough tokens to fulfill n_samples sequences of SEQ_LEN
        target_tokens = n_samples * SEQ_LEN + 1024
        
        # Randomize line order
        random.shuffle(ds_lines)
        
        all_ids = []
        total_len = 0
        
        for line in ds_lines:
            if not isinstance(line, str) or not line.strip():
                continue
                
            enc = tokenizer(line, add_special_tokens=False, return_tensors="pt").input_ids[0]
            if enc.size(0) > 0:
                all_ids.append(enc)
                total_len += enc.size(0)
                
            if total_len >= target_tokens:
                break
                
        if not all_ids:
            continue
            
        concat_ids = torch.cat(all_ids).unsqueeze(0)
        
        max_start_idx = max(0, concat_ids.shape[1] - SEQ_LEN - 1)
        
        for _ in range(n_samples):
            i = random.randint(0, max_start_idx)
            samples.append(concat_ids[:, i:i + SEQ_LEN])

    random.shuffle(samples)
    return samples

# ============================================================
# HESSIAN COLLECTION
# ============================================================

@torch.no_grad()
def _run_calibration_on_device(
    model,
    samples,
    device
):

    hess = {}

    hooks = []

    def hook(name):

        def fn(module, inp, out):

            x = inp[0].detach()

            if x.dim() == 3:
                x = x.reshape(
                    -1,
                    x.shape[-1]
                )

            h = (
                torch.matmul(
                    x.float().t(),
                    x.float()
                ) / x.shape[0]
            ).cpu()

            del x

            if name not in hess:
                hess[name] = h
            else:
                hess[name].add_(h)

        return fn

    for name, m in model.named_modules():

        if isinstance(m, nn.Linear):

            hooks.append(
                m.register_forward_hook(
                    hook(name)
                )
            )

    batch_size = (
        CALIBRATION_BATCH_SIZE
        if device.type == "cuda"
        else 1
    )

    try:

        for i in tqdm(
            range(
                0,
                len(samples),
                batch_size
            ),
            desc=f"Calibrating [{device.type.upper()}]"
        ):

            batch = torch.cat(
                samples[i:i + batch_size],
                dim=0
            ).to(device)

            with torch.inference_mode():

                model(
                    batch,
                    use_cache=False
                )

            del batch

            if device.type == "cuda":
                torch.cuda.synchronize()

    finally:

        for h in hooks:
            h.remove()

    n = len(samples)

    for k in hess:
        hess[k] = hess[k] / n

    return hess

@torch.no_grad()
def collect_hessian(
    model,
    samples,
    device
):

    if (
        device.type == "cuda"
        and can_fit_on_gpu(model)
    ):

        print(
            "[Calibration] Moving model to GPU..."
        )

        model.to(device)

        torch.cuda.synchronize()

        alloc = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        print(
            f"[Calibration] "
            f"Model on GPU — "
            f"{alloc:.2f} GB allocated"
        )

        try:

            hess = _run_calibration_on_device(
                model,
                samples,
                device
            )

        except Exception as e:

            if not _is_oom(e):
                raise

            print(
                "[OOM] CUDA OOM during "
                "calibration — CPU fallback."
            )

            torch.cuda.empty_cache()

            gc.collect()

            model.to("cpu")

            torch.cuda.empty_cache()

            hess = _run_calibration_on_device(
                model,
                samples,
                torch.device("cpu")
            )

        else:

            print(
                "[Calibration] "
                "Moving model back to CPU..."
            )

            model.to("cpu")

            torch.cuda.synchronize()

            torch.cuda.empty_cache()

    else:

        print(
            "[Calibration] Running on CPU..."
        )

        hess = _run_calibration_on_device(
            model,
            samples,
            torch.device("cpu")
        )

    gc.collect()

    return hess

# ============================================================
# QUANTIZATION
# ============================================================

def get_keep_ratio(name):

    for k, v in KEEP_RATIO.items():

        if k in name:
            return v

    return KEEP_RATIO["default"]

def should_skip_layer(
    name,
    total_layers
):

    if any(
        k in name
        for k in SKIP_KEYWORDS
    ):
        return True

    for i in range(PROTECT_FIRST_LAYERS):

        if f".{i}." in name:
            return True

    for i in range(
        total_layers - PROTECT_LAST_LAYERS,
        total_layers
    ):

        if f".{i}." in name:
            return True

    return False

@torch.no_grad()
def quantize(
    model,
    hessians
):

    modules = dict(
        model.named_modules()
    )

    layer_order = [
        n
        for n, m in modules.items()
        if isinstance(m, nn.Linear)
    ]

    transformer_layers = []

    for n in layer_order:

        parts = n.split(".")

        for p in parts:

            if p.isdigit():
                transformer_layers.append(
                    int(p)
                )

    total_layers = (
        max(transformer_layers) + 1
        if transformer_layers
        else 0
    )

    dev = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    for name in tqdm(
        layer_order,
        desc="Quantizing layers"
    ):

        if name not in hessians:
            continue

        if should_skip_layer(
            name,
            total_layers
        ):

            print(f"[SKIP] {name}")

            continue

        m = modules[name]

        try:

            W = (
                m.weight.data
                .clone()
                .float()
                .to(dev)
            )

            H = (
                hessians[name]
                .to(dev)
                .float()
            )

            diag = torch.diag(H)

            damp = (
                0.05 * diag.mean()
            )

            H[
                torch.arange(H.shape[0]),
                torch.arange(H.shape[0])
            ] += damp

            success = False

            for attempt in range(10):

                try:

                    H_inv = torch.linalg.inv(H)

                    H_inv = (
                        0.5
                        * (H_inv + H_inv.T)
                    )

                    C_diag = torch.diag(H_inv)

                    success = True

                    break

                except Exception:

                    extra = (
                        0.05
                        * (attempt + 1)
                    ) * diag.mean()

                    H[
                        torch.arange(H.shape[0]),
                        torch.arange(H.shape[0])
                    ] += extra

            if not success:

                print(
                    f"[WARN] Failed inversion "
                    f"for {name}"
                )

                continue

            keep_ratio = get_keep_ratio(name)

            block_size = QUANTIZE_BLOCK_SIZE

            in_features = W.shape[1]

            for i1 in range(
                0,
                in_features,
                block_size
            ):

                i2 = min(
                    i1 + block_size,
                    in_features
                )

                Wb = W[:, i1:i2]

                C_diag_b = C_diag[i1:i2]

                Qb = billm_quantize_block(
                    Wb,
                    C_diag_b,
                    keep_ratio
                )

                W[:, i1:i2] = Qb

            m.weight.data.copy_(
                W.to(
                    dtype=m.weight.data.dtype,
                    device=m.weight.data.device
                )
            )

            print(
                f"{name} | "
                f"keep={keep_ratio:.2f} | "
                f"block={block_size}"
            )

            del W
            del H
            del H_inv
            del C_diag

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:

            print(
                f"[WARN] Quantization failed "
                f"for {name}: {e}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()

            continue

# ============================================================
# PERPLEXITY
# ============================================================

@torch.no_grad()
def _eval_nll(
    model,
    enc,
    device
):

    stride = SEQ_LEN

    max_len = min(
        getattr(
            model.config,
            "max_position_embeddings",
            8192
        ),
        2048
    )

    nlls = []

    for i in range(
        0,
        enc.size(1),
        stride
    ):

        begin = max(
            i + stride - max_len,
            0
        )

        end = i + stride

        input_ids = (
            enc[:, begin:end]
            .to(device)
        )

        target_ids = input_ids.clone()

        target_ids[:, :-stride] = -100

        outputs = model(
            input_ids,
            labels=target_ids
        )

        nlls.append(
            outputs.loss.cpu() * stride
        )

        del input_ids
        del target_ids
        del outputs

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return nlls, stride

@torch.no_grad()
def perplexity(
    model,
    tokenizer,
    device
):

    model.eval()

    data = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test"
    )

    enc = tokenizer(
        "\n\n".join(data["text"]),
        return_tensors="pt"
    ).input_ids

    if (
        MAX_EVAL_TOKENS
        and enc.size(1)
        > MAX_EVAL_TOKENS
    ):

        print(
            f"[Eval] Capping eval tokens: "
            f"{enc.size(1)} → "
            f"{MAX_EVAL_TOKENS}"
        )

        enc = enc[:, :MAX_EVAL_TOKENS]

    eval_device = (
        device
        if (
            device.type == "cuda"
            and can_fit_on_gpu(model)
        )
        else torch.device("cpu")
    )

    print(
        f"[Eval] Moving model to "
        f"{eval_device}..."
    )

    model.to(eval_device)

    try:

        nlls, stride = _eval_nll(
            model,
            enc,
            eval_device
        )

    except Exception as e:

        if not _is_oom(e):
            raise

        print(
            "[OOM] CUDA OOM during eval "
            "— CPU fallback."
        )

        torch.cuda.empty_cache()

        gc.collect()

        model.to("cpu")

        torch.cuda.empty_cache()

        nlls, stride = _eval_nll(
            model,
            enc,
            torch.device("cpu")
        )

    finally:

        model.to("cpu")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.exp(
        torch.stack(nlls).sum()
        / (len(nlls) * stride)
    ).item()

# ============================================================
# MODEL LOADING
# ============================================================

def download_and_load_model(
    model_name,
    local_dir="models"
):
    # --- Fallback: local path provided directly ---
    if os.path.isabs(model_name) or os.path.exists(model_name):
        model_path = model_name
        print(
            f"Loading local model from "
            f"{model_path}"
        )

    else:
        model_path = os.path.join(
            local_dir,
            model_name.split("/")[-1]
        )

        if (
            not os.path.exists(model_path)
            or not os.listdir(model_path)
        ):

            print(
                f"Downloading model "
                f"{model_name}..."
            )

            os.makedirs(
                model_path,
                exist_ok=True
            )

            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                local_dir_use_symlinks=False
            )

        else:

            print(
                f"Loading local model from "
                f"{model_path}"
            )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    return tokenizer, model

# ============================================================
# MAIN
# ============================================================

def main():

    device = get_device()

    n_threads = (
        CPU_THREADS
        if CPU_THREADS is not None
        else detect_cpu_threads()
    )

    torch.set_num_threads(n_threads)

    if device.type == "cuda":

        total_vram = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3
        )

        print(
            f"[GPU] Total VRAM: "
            f"{total_vram:.1f} GB"
        )

    tokenizer, model = download_and_load_model(
        MODEL_NAME,
        "models"
    )

    model.config.use_cache = False

    samples = get_data(tokenizer)

    hess = collect_hessian(
        model,
        samples,
        device
    )

    print(
        "Quantizing "
        "(GPU-accelerated)..."
    )

    quantize(model, hess)

    del hess

    ppl = perplexity(
        model,
        tokenizer,
        device
    )

    print(f"Perplexity: {ppl}")

    if OUTPUT_DIR:

        print(
            f"Saving quantized model "
            f"to '{OUTPUT_DIR}'..."
        )

        model.save_pretrained(
            OUTPUT_DIR
        )

        tokenizer.save_pretrained(
            OUTPUT_DIR
        )

if __name__ == "__main__":

    main()