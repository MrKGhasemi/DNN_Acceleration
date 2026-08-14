# Why the acceleration techniques work

This document connects the implementation to the mathematics and the GPU
workload. The techniques are complementary: precision controls arithmetic,
compilation controls execution, and the common input pipeline attempts to keep
the GPU fed.

## 1. The workload being accelerated

### Patch embedding

For a 384 x 384 RGB image and 16 x 16 patches, the number of image tokens is

```math
N = \frac{384}{16}\frac{384}{16} = 24^2 = 576.
```

After prepending the learned class token, the Transformer sees 577 tokens. A
convolution with kernel and stride 16 performs the patch projection into the
embedding dimension `D = 768`.

### Self-attention

For a token matrix `X` and learned projection matrices, one attention head is

```math
Q=XW_Q,\quad K=XW_K,\quad V=XW_V,
```

```math
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.
```

The `QK^T` and probability-times-`V` operations are large matrix
multiplications. Attention's token interaction cost is quadratic in sequence
length, `O(N^2 D)`, while the linear projections and MLP blocks are dominated
by dense GEMMs. Those matrix-heavy operations are excellent Tensor Core
candidates and explain why lower precision can help ViT substantially.

### Partial fine-tuning

Let `theta_f` denote frozen parameters and `theta_t` trainable parameters. The
forward pass still computes

```math
\hat y=f(x;\theta_f,\theta_t),
```

but optimization computes and applies only

```math
\nabla_{\theta_t}\mathcal{L}.
```

The derivative with respect to `theta_f` can exist mathematically, but
autograd neither stores nor applies that parameter gradient because those
parameters have `requires_grad=False`.

The final two encoder blocks, final layer normalization, and classifier head
are trainable. This is 14.25M of 86.17M parameters. Frozen earlier blocks still
cost forward time, but avoid their backward activation/gradient and optimizer
state costs.

## 2. Float32 matmul precision and TF32

The baseline calls

```python
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

The accelerated profile calls

```python
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

FP32 uses one sign bit, eight exponent bits, and 23 explicitly stored fraction
bits. NVIDIA TF32 keeps FP32's exponent range but uses a 10-bit fraction for
the multiply; accumulation remains FP32. A simplified dot product is therefore

```math
s = \sum_i \mathrm{round}_{TF32}(a_i)
               \mathrm{round}_{TF32}(b_i),
```

with the sum accumulated at FP32 precision. The result tensor remains FP32.
On supported Tensor Core GPUs this exchanges mantissa precision for much
higher matmul throughput.

PyTorch's `"high"` policy may use TF32 or a bfloat16-based decomposition for
float32 matrix products when a fast implementation is available. It affects
internal matmul computation, not the public tensor dtype. Convolutions are
controlled separately through the cuDNN TF32 setting.

Performance effect:

- faster FP32 linear algebra on Ampere-or-newer NVIDIA GPUs;
- no expected benefit on unsupported hardware or operations that are not
  eligible matrix multiplications/convolutions;
- small numerical differences because inputs are rounded more aggressively.

Because BF16 autocast already moves many eligible operations away from FP32,
TF32 and BF16 benefits are not simply additive. TF32 still helps FP32 regions
left by autocast.

## 3. Automatic mixed precision

Automatic mixed precision (AMP) chooses dtypes operation by operation:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(images)
    loss = cross_entropy(logits, targets)
```

Eligible matrix multiplications and convolutions use BF16; numerically
sensitive operations can remain FP32. Parameters are not permanently converted
to BF16 by this context.

### Why BF16?

| Format | Sign | Exponent | Fraction | Approx. dynamic range |
|---|---:|---:|---:|---:|
| FP32 | 1 | 8 | 23 | large |
| TF32 multiply | 1 | 8 | 10 | FP32-like |
| BF16 | 1 | 8 | 7 | FP32-like |
| FP16 | 1 | 5 | 10 | much smaller |

BF16 has coarse precision but essentially FP32's exponent range, making
gradient underflow much less likely than FP16. For that reason the default BF16
path does not use gradient scaling. The optional `--amp-dtype float16` path
uses `torch.amp.GradScaler`.

For FP16, scaling replaces the loss with

```math
\widetilde{\mathcal L}=S\mathcal L,
```

so the computed gradient is `S` times larger and less likely to underflow.
Before the optimizer update, gradients are divided by `S`. The scaler changes
`S` dynamically when overflow is detected.

Performance effect:

- BF16/FP16 operands use roughly half the storage of FP32;
- lower memory traffic and activation footprint can improve throughput;
- Tensor Cores can execute eligible lower-precision matrix operations much
  faster;
- larger batches may fit, although this benchmark deliberately keeps batch
  size fixed so batch size is not a confounder;
- casts and unsupported operations limit the maximum gain.

## 4. `torch.compile`

Eager PyTorch executes Python and dispatches operators one at a time.
`torch.compile` captures regions of the program, specializes them using runtime
guards, and sends them to a backend (Inductor by default). The backend can fuse
compatible operations and emit optimized kernels.

Conceptually, eager execution might launch

```text
kernel(A) -> write memory -> kernel(B) -> read memory -> kernel(C)
```

whereas a fused graph can execute

```text
kernel(A + B + C) -> one final write
```

Potential savings include:

- fewer Python-to-dispatch transitions;
- fewer GPU kernel launches;
- fewer intermediate reads/writes to global memory;
- layout and scheduling choices made with a larger graph view.

Compilation is not free. The first input triggers graph capture, code
generation, and compilation. Therefore

```math
T_{cold}=T_{compile}+T_{first\ execution},
```

while later compatible executions approach

```math
T_{steady}\approx T_{compiled\ execution}.
```

The benchmark reports both. Fixed 384 x 384 images and a fixed batch size reduce
shape-driven recompilation. The final partial batch can still have a different
shape; PyTorch may compile another specialization for it.

The exploratory notebook enabled `torch._dynamo.config.suppress_errors`, which
can silently fall back to eager execution. This repository deliberately does
not suppress compiler errors: a failed compiled run must fail visibly rather
than being mislabeled.

## 5. Data loading and memory transfer

Both profiles use the same four workers, pinned host memory, two-batch worker
prefetch, persistent workers, and non-blocking host-to-device copies. These are
held constant and are **not** counted as accelerated-profile treatments.

This matters because an epoch is a pipeline:

```math
T_{epoch}=T_{input}+T_{host\to device}+T_{GPU}+T_{synchronization}.
```

Even an infinitely fast GPU cannot remove `T_input`. Amdahl's law expresses the
limit. If fraction `p` of runtime is accelerated by factor `s`, total speedup is

```math
S=\frac{1}{(1-p)+p/s}.
```

If data loading consumes 30% of the epoch, accelerating GPU work by 4x yields
only `1 / (0.30 + 0.70/4) = 2.11x` overall.

The original slow notebook set `CUDA_LAUNCH_BLOCKING=1`, which synchronizes
normally asynchronous launches and is intended for debugging. The repository
does not set it and records any externally inherited value in `result.json`.
Performance measurements assume that it is absent.

## 6. Optimization mathematics

### Label-smoothed cross-entropy

For `K=101`, true class `y`, smoothing `epsilon=0.2`, and predicted
probabilities `p`, the target distribution is a mixture of the one-hot target
and a uniform distribution:

```math
q_k=(1-\epsilon)I(k=y)+\frac{\epsilon}{K}.
```

Here, `I(k=y)` is one for the true class and zero otherwise.

The loss is

```math
\mathcal L=-\sum_{k=1}^{K}q_k\log p_k.
```

It discourages extreme confidence and often improves transfer-learning
generalization. Both profiles use the same loss, so it cannot explain their
timing difference.

### Adagrad

For gradient `g_t`, Adagrad accumulates squared gradients elementwise:

```math
G_t=G_{t-1}+g_t^2,
```

then applies

```math
\theta_{t+1}=\theta_t-
\frac{\eta}{\sqrt{G_t}+\delta}g_t.
```

Frequently updated coordinates receive smaller effective steps. The repo uses
the notebook's initial learning rate `1e-3` and weight decay `5e-4`.

### Cosine annealing

Across `T` epochs, the learning rate follows approximately

```math
\eta_t=\eta_{min}+\frac{1}{2}(\eta_{max}-\eta_{min})
\left(1+\cos\frac{\pi t}{T}\right).
```

It starts near the configured rate and smoothly approaches the minimum. Both
profiles step the same scheduler after each epoch.

## 7. Accuracy and numerical equivalence

Top-1 accuracy is

```math
\frac{1}{M}\sum_{i=1}^{M}I
[\mathrm{argmax}_k z_{ik}=y_i],
```

and Top-5 counts a prediction correct when `y_i` appears among the five largest
logits. Metrics are stored as fractions in JSON and displayed as percentages.
This corrects the original notebook, which printed values such as `0.9020%`
even though that value means 90.20%.

Reduced precision is not bitwise equivalent to FP32. The fair question is
whether speed improves while final accuracy remains within a predeclared
tolerance over multiple seeds—not whether every weight is identical.

## 8. Why a 2x result is plausible but not guaranteed

The ViT is matmul-heavy, input shapes are fixed, BF16 is Tensor Core-friendly,
and compilation can fuse non-matmul work. Those conditions favor acceleration.
The actual ratio still depends on:

- GPU architecture and Tensor Core generation;
- memory bandwidth and available VRAM;
- compiler/driver/PyTorch versions;
- CPU, storage, worker count, and augmentation cost;
- whether compilation breaks/recompiles;
- batch size and the fraction of time spent in eligible kernels.

## Official technical references

- [`torch.set_float32_matmul_precision`, PyTorch 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.set_float32_matmul_precision.html)
- [`torch.compile`, PyTorch 2.9](https://docs.pytorch.org/docs/2.9/generated/torch.compile.html)
- [Automatic mixed precision, PyTorch 2.9](https://docs.pytorch.org/docs/2.9/amp.html)
- [ViT-B/16 SWAG E2E metadata, torchvision 0.24](https://docs.pytorch.org/vision/0.24/models/generated/torchvision.models.vit_b_16.html)
