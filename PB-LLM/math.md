# The Deep Mathematics of PB-LLM Quantization

*Note: The `pb_llm_pipeline.py` script features a dual-mode engine. It supports both **BitsAndBytes 4-bit QLoRA** (hardware-specific for NVIDIA) and **PB-LLM 1-Bit** (pure PyTorch, compatible natively with AMD ROCm/Cloud). This document exclusively details the underlying mathematics of the custom 1-Bit PB-LLM algorithm.*

This document provides an exhaustive, step-by-step mathematical breakdown of the **Partially Binarized Large Language Models (PB-LLM)** algorithm. It explains exactly *what* calculations are happening, *how* the formulas work, and *why* they are mathematically necessary to compress models like Qwen and Gemma to 1-bit without destroying their intelligence.

Crucially, this document also shows **exactly where** each mathematical formula is executed in your `pb_llm_pipeline.py` script.

---

## 1. The Fundamental Problem: Why Pure 1-Bit Fails

In standard neural networks, a weight matrix $W \in \mathbb{R}^{d_{out} \times d_{in}}$ contains floating-point numbers (FP16 or BF16). 
A pure Binarized Neural Network (BNN) tries to force every weight into $\{-1, +1\}$ using the Sign function:

$$
B = \text{sign}(W)
$$


**Why does this fail for LLMs?**
LLMs (like Qwen and Gemma) learn through **"Outlier Features"**. A tiny fraction of their weights (often less than 1%) have massive magnitudes. These specific weights act as critical routing mechanisms for syntax, grammar, and factual recall. 

If you apply pure binarization, a weight of `0.001` becomes `+1`, and an outlier weight of `45.0` also becomes `+1`. The relative importance of the outlier is completely destroyed. The mathematical variance of the layer collapses, and the model outputs random noise (gibberish).

**The PB-LLM Solution:** Protect the massive outlier weights (keep them in FP16), and only binarize the small, distributed weights.

---

## 2. Step-by-Step Mathematical Operations in PB-LLM

The script applies the following operations to every Linear layer (Attention Q,K,V,O and MLP Up,Down,Gate matrices) in the model.

Let $W$ be a high-precision weight matrix.
Let $S$ be the `salient_ratio` (e.g., $S = 0.01$, meaning 1%).

### Step A: Salience Threshold Calculation (Finding the Outliers)
**What type of calculation?** Magnitude sorting and thresholding.

First, we need to mathematically define what an "outlier" is for this specific matrix. We do this by looking at the absolute magnitudes of the weights.
1. Calculate the number of weights to protect, $k$:
   
$$
k = \lfloor \text{TotalElements}(W) \times S \rfloor
$$

2. Flatten the matrix into a 1D vector and take the absolute values: $|W|$.
3. Sort $|W|$ in descending order to find the $k$-th largest value. This value becomes our threshold, $\tau$.

**The Formula:**

$$
\tau = \text{TopK_Value}(|W|, k)
$$


**Where this happens in `pb_llm_pipeline.py`:**
```python
# Calculated once inside PBLLMLinear.__init__
k = max(int(self.weight.numel() * salient_ratio), 1)
threshold, _ = torch.topk(self.weight.abs().flatten(), k)
threshold_val = threshold[-1].item()
```

**Why it happens:** We cannot use a hardcoded threshold (like "protect anything > 2.0") because every layer in a Transformer has a different distribution of weights. Dynamically calculating $\tau$ using `TopK` ensures we always capture the top 1% of outliers, regardless of the layer's overall variance.

### Step B: Mask Generation (Separating the Matrix)
**What type of calculation?** Boolean tensor logic.

We divide the matrix into two distinct sets using boolean masks.
* **Salient Mask ($M_{salient}$):** 
  
$$
M_{salient} = |W| \ge \tau
$$

  (This mask is `True` for the 1% of massive weights).
* **Non-Salient Mask ($M_{non\_salient}$):** 
  
$$
M_{non\_salient} = |W| < \tau
$$

  (This mask is `True` for the 99% of normal/small weights).

**Where this happens in `pb_llm_pipeline.py`:**
```python
# To save VRAM, we only store the salient mask as a persistent buffer
self.register_buffer('salient_mask', self.weight.abs() >= threshold_val)

# Inside forward(), we dynamically extract the non-salient weights 
# using the inverse of the salient mask
non_salient_weights = self.weight[~self.salient_mask]
```

### Step C: The Scaling Factor ($\alpha$) Calculation
**What type of calculation?** L1-Norm / Mean Absolute Value.

For the 99% of weights that are non-salient, we will turn them into 1-bit. However, if we just turn them into `+1` and `-1`, we change the overall "energy" or variance of the matrix. 

To fix this, we calculate a scalar value $\alpha$ that represents the average magnitude of the non-salient weights.

**The Formula:**

$$
\alpha = \frac{1}{|M_{non\_salient}|} \sum_{W_{ij} \in M_{non\_salient}} |W_{ij}|
$$


**Where this happens in `pb_llm_pipeline.py`:**
```python
# We calculate alpha in float32 to prevent float16 overflow, 
# and detach it to stabilize gradients during QAT
alpha = (non_salient_weights.to(torch.float32).abs().mean().to(self.weight.dtype) + 1e-9).detach()
```

**Why it happens:** Multiplying our binarized weights by $\alpha$ ensures that the expected value (the mathematical expectation) of the matrix operations remains identical to the original high-precision matrix. It minimizes the **quantization error** mathematically defined as $|| W - \alpha B ||^2$.

### Step D: Binarization of Non-Salient Weights
**What type of calculation?** Sign function and scalar multiplication.

Now we binarize the 99% of normal weights.

**The Formula:**

$$
B = \text{sign}(W) \times \alpha
$$


Where:

$$
\text{sign}(x) = \begin{cases} +1 & \text{if } x > 0 \\ -1 & \text{if } x \le 0 \end{cases}
$$


**Where this happens in `pb_llm_pipeline.py`:**
```python
# SignSTE allows backpropagation to pass through the otherwise non-differentiable sign function
binarized_non_salient = SignSTE.apply(self.weight) * alpha
```

**Why it happens:** This is the core compression step. By restricting these weights to exactly two states ($+\alpha$ and $-\alpha$), they can be stored in computer memory using a single bit (`0` or `1`), with $\alpha$ stored once as a float for the whole layer. This achieves the ~99% compression rate for this portion of the matrix.

### Step E: Matrix Reconstruction (The Hybrid Matrix)
**What type of calculation?** Conditional Tensor Merging (`torch.where`).

Finally, we reconstruct the weight matrix $W'$ that will actually be used by the LLM for inference.

**The Formula:**

$$
W'_{ij} = \begin{cases} 
      W_{ij} & \text{if } M_{salient} \text{ is True (keep original FP16)} \\
      B_{ij} & \text{if } M_{non\_salient} \text{ is True (use } \pm\alpha \text{)}
   \end{cases}
$$


**Where this happens in `pb_llm_pipeline.py`:**
```python
# Splice the high-precision outliers back into the 1-bit matrix
quantized_weight = torch.where(self.salient_mask, self.weight, binarized_non_salient)
```

---

## 3. How the Model Calculates Text Generation Post-Quantization

Once the PB-LLM script finishes the math above, the original Qwen or Gemma model's layers have been entirely replaced by these $W'$ hybrid matrices.

**Where this happens in `pb_llm_pipeline.py`:**
```python
# The loop that recursively replaces standard nn.Linear layers with our PBLLMLinear layers
setattr(model, name, PBLLMLinear(module, salient_ratio))
```

**How does text generation mathematically work now?**

In a Transformer, the core calculation inside every layer is a linear transformation (a matrix multiplication) of the input tokens $X$ against the weight matrix:

$$
Y = X W^T
$$


With PB-LLM, this becomes:

$$
Y = X (W')^T
$$


Because matrix multiplication is distributive, PyTorch is implicitly calculating:

$$
Y = X (W_{salient})^T + X (B_{non\_salient})^T
$$


**Where this happens in `pb_llm_pipeline.py`:**
```python
# The generation loop natively processes the new W' matrix
outputs = quantized_model.generate(**inputs, max_new_tokens=100)
```

**Why does this result in coherent English?**
1. **The $X (W_{salient})^T$ part:** The input tokens multiply against the exact, high-precision outlier weights. This calculation preserves the critical attention routing and factual data retrieval perfectly.
2. **The $X (B_{non\_salient})^T$ part:** The input tokens multiply against the sea of $+\alpha$ and $-\alpha$. Because this is a massive sum of binary numbers, it acts as a robust, distributed associative memory. Small rounding errors cancel each other out across the 4096+ dimensions.

The output vector $Y$ remains mathematically close enough to the original unquantized $Y$ that the final Softmax probability distribution still heavily favors the correct next word in the sentence.

### Summary
The math works because it is a surgically targeted operation. It uses **Magnitude Thresholding** to identify structural load-bearing weights (protecting them), and uses **Mean Absolute Value Scaling** alongside the **Sign Function** to compress the redundant distributed memory into 1-bit without altering the overall variance of the neural network's calculations.
