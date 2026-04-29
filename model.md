# TabICL Model Architecture

Source: [`src/tabicl/_model/`](src/tabicl/_model/)

---

## Overview

TabICL is a transformer-based foundation model for **in-context learning on tabular data**. Given a labelled training set and an unlabelled test set, it produces predictions in a single forward pass without any gradient updates.

The model processes data through three sequential stages, each implemented as a dedicated `nn.Module`:

```
X (B, T, H)  +  y_train (B, train_size)
        │
        ▼
┌──────────────────────┐
│  Stage 1: ColEmbedding  │  (embedding.py)
│  Set-transformer per    │
│  feature column         │
└──────────┬───────────┘
           │ (B, T, G+C, E)
           ▼
┌──────────────────────┐
│  Stage 2: RowInteraction│  (interaction.py)
│  Transformer encoder    │
│  with RoPE + CLS tokens │
└──────────┬───────────┘
           │ (B, T, C·E)   ← test_representations
           ▼
┌──────────────────────┐
│  Stage 3: ICLearning    │  (learning.py)
│  Full-sequence ICL      │
│  transformer + decoder  │
└──────────┬───────────┘
           │                ← test_icl_representations (pre-decoder)
           ▼
   predictions (B, test_size, out_dim)
```

`B` = tables in batch, `T` = total rows (train + test), `H` = features,
`G` = feature groups, `C` = CLS tokens, `E` = `embed_dim`.

---

## Top-level module: `TabICL` ([tabicl.py](src/tabicl/_model/tabicl.py))

```python
class TabICL(nn.Module)
```

**Key constructor parameters** (defaults match the released checkpoint):

| Parameter | Default | Meaning |
|---|---|---|
| `max_classes` | 10 | 0 = regression; >0 = classification (native capacity) |
| `num_quantiles` | 999 | Quantiles predicted for regression |
| `embed_dim` | 128 | Dimension `E` used in Stage 1 & 2 |
| `col_num_blocks` | 3 | ISAB blocks in Stage 1 set-transformer |
| `col_nhead` | 8 | Attention heads in Stage 1 |
| `col_num_inds` | 128 | Inducing-point count in Stage 1 ISAB |
| `col_affine` | False | If True, Stage 1 outputs affine weights/biases |
| `col_feature_group` | `"same"` | Feature grouping mode (circular permutation) |
| `col_feature_group_size` | 3 | Features per group |
| `col_target_aware` | True | Inject `y_train` labels into Stage 1 |
| `row_num_blocks` | 3 | Attention blocks in Stage 2 encoder |
| `row_nhead` | 8 | Attention heads in Stage 2 |
| `row_num_cls` | 4 | CLS tokens (`C`); Stage 2 output = `C·E = 512` |
| `row_rope_base` | 100000 | RoPE base in Stage 2 |
| `icl_num_blocks` | 12 | Attention blocks in Stage 3 ICL transformer |
| `icl_nhead` | 8 | Attention heads in Stage 3 |
| `col_ssmax` / `icl_ssmax` | `"qassmax-mlp-elementwise"` | Scalable softmax variant |
| `ff_factor` | 2 | FFN hidden = `d_model × ff_factor` |
| `norm_first` | True | Pre-norm architecture |

**Derived dimension** — Stage 3 model dimension:
```
icl_dim = embed_dim × row_num_cls = 128 × 4 = 512
```

### Entry points

| Method | Purpose |
|---|---|
| `forward(X, y_train, ...)` | Dispatch to `_train_forward` or `_inference_forward` |
| `forward_with_cache(...)` | Reuse KV/repr cache across repeated queries |
| `predict_stats(X, y_train, ...)` | Regression only — returns mean/median/quantiles |

#### Optional embedding returns

Pass any combination of these flags to `forward` / `predict_proba`:

```python
return_col_embedding_sample=True    # column-identity embedding  shape (G+C, E)
return_test_representations=True    # row representations after Stage 2  shape (B, test_size, C·E)
return_test_icl_representations=True # pre-decoder activations after Stage 3  shape (B, test_size, icl_dim)
```

When set, the return value becomes `(predictions, [col_emb,] [row_repr,] [icl_repr])` in that order.

---

### `return_col_embedding_sample` — column-identity embeddings

**What it is:** a single matrix, shape `(G+C, E)`, where each row is a 128-dim vector summarising the statistical distribution of one feature column (plus reserved CLS slots).

#### Extraction point

The full `ColEmbedding` output has shape `(B, T, G+C, E)`. The sample is extracted from the **first table** in the batch (`[0]`), with different aggregations depending on mode:

```python
# inference path (tabicl.py:480) — row-mean, stable column fingerprint
col_embedding_sample = torch.mean(col_embeddings[0], dim=0)
# col_embeddings[0]: (T, G+C, E)  →  mean over dim 0 (rows)  →  (G+C, E)

# training path (tabicl.py:358) — snapshot of row 0 only
col_embedding_sample = col_embeddings[0, 0]
# col_embeddings[0, 0]: (G+C, E)  — first table, first row
```

The inference path is more representative: averaging over all T rows yields a stable column-level fingerprint that does not depend on which specific row is sampled.

#### Dimension breakdown

| Dim | Size | Meaning |
|---|---|---|
| 0 | `G+C` | One entry per feature group plus reserved CLS slots |
| 1 | `E = 128` | Learned embedding from the Set Transformer |

**Layout of dim 0 (`G+C` axis):**

The first `C = row_num_cls = 4` positions are CLS placeholders — padded with `skip_value = -100.0` in `ColEmbedding` and never given real embeddings. Only the trailing `G` positions carry meaningful feature information:

```python
# embedding.py:484 — CLS slots padded to the LEFT of the H dimension
X = F.pad(X, (reserve_cls_tokens, 0), value=-100.0)
# layout after pad: [CLS_0, CLS_1, CLS_2, CLS_3, feat_0, feat_1, ..., feat_{H-1}]
```

| Index range | Content |
|---|---|
| `0 .. C-1`   (first 4) | All `-100.0` — CLS placeholder, not meaningful |
| `C .. G+C-1` (last G)  | One distribution embedding per feature group/column |

**Concrete sizes:**

| Dataset | H features | G groups | C CLS | Shape |
|---|---|---|---|---|
| Two-moon | 2 | 2 (`"same"` mode: G=H) | 4 | `(6, 128)` |
| SKAB 8-sensor | 8 | 8 | 4 | `(12, 128)` |

For SKAB: `col_embedding_sample[4]` = Accelerometer1RMS, `col_embedding_sample[5]` = Accelerometer2RMS, …, `col_embedding_sample[11]` = Volume Flow RateRMS.

**What the 128-dim vector encodes:**

The Set Transformer processes all T rows of a column through ISAB blocks where the `num_inds = 128` inducing points attend only to training rows. The output for each cell therefore encodes the value of that cell *in the context of the full training column distribution*. After averaging over rows, the 128 dimensions represent a holistic distribution fingerprint — features with similar scale, spread, or shape cluster together in this embedding space. The dimensions are not individually interpretable; they are learned via the contrastive pressure of predicting labels across many datasets during pretraining.

---

## Stage 1 — Column Embedding: `ColEmbedding` ([embedding.py](src/tabicl/_model/embedding.py))

**Goal:** map each scalar cell `X[b, t, h]` to a distribution-aware vector in `ℝ^E`, capturing the statistical context of the entire column.

### Data flow

```
X (B, T, H)
  │  feature_grouping  →  (B, T, G, group_size)
  │  pad CLS slots     →  (B, T, G+C, group_size)
  │  transpose         →  (B, G+C, T, group_size)
  │
  │  in_linear (SkippableLinear)
  │     group_size → E
  │                      (B, G+C, T, E)  ← src
  │
  │  [if target_aware]  src[..., :train_size, :] += y_encoder(y_train)
  │
  │  tf_col (SetTransformer, num_blocks ISAB blocks)
  │     inducing pts attend to train rows only (train_size guard)
  │                      (B, G+C, T, E)
  │
  │  [if affine=True]
  │     W = ln_w(out_w(src))  →  per-cell weights
  │     b = ln_b(out_b(src))  →  per-cell biases
  │     output = features * W + b
  │  [else]
  │     output = src
  │
  └─ transpose  →  (B, T, G+C, E)
```

### Feature grouping

`col_feature_group="same"` (default): circular permutation with shifts `2^0, 2^1, ..., 2^(g-1)`, so each group of size 3 covers the feature itself plus two shifted neighbours. This gives `G = H` groups.

`col_feature_group="valid"`: pad to multiple of `group_size`, reshape → `G = ⌈H/group_size⌉`.

### Target-aware embedding

When `target_aware=True`, training-row vectors in `src` are additively offset by the label embedding before the set-transformer run. For classification: `y_encoder = OneHotAndLinear(max_classes, E)`. For regression: `y_encoder = Linear(1, E)`.

### Mixed-radix ensemble (many-class classification)

When `num_classes > max_classes`, labels are decomposed into `D` digits with balanced bases `[k₀, …, k_{D-1}]` where `∏ kᵢ ≥ num_classes` and each `kᵢ ≤ max_classes`. The set-transformer is run once per digit, outputs are averaged:

```
src = mean( tf_col(src + y_encoder(digit_i)) for i in 0..D-1 )
```

### Sub-modules

| Sub-module | Type | Role |
|---|---|---|
| `in_linear` | `SkippableLinear(group_size, E)` | Project grouped scalars into embedding space; outputs `skip_value` for padded CLS slots |
| `tf_col` | `SetTransformer` | Distribution-aware context via ISAB blocks |
| `y_encoder` | `OneHotAndLinear` or `Linear` | Encode training labels for target-aware mode |
| `out_w`, `out_b` | `SkippableLinear(E, E)` | Produce affine weights/biases (only when `affine=True`) |
| `ln_w`, `ln_b` | `LayerNorm(E)` | Normalize before affine transform (pre-norm mode) |

---

## Stage 2 — Row Interaction: `RowInteraction` ([interaction.py](src/tabicl/_model/interaction.py))

**Goal:** capture interactions between features within each row and compress the variable-length `(G+C, E)` token sequence into a fixed-size row representation.

### Data flow

```
embeddings (B, T, G+C, E)
  │  fill CLS slots with learned cls_tokens  (num_cls, E)
  │
  │  Blocks 0 … num_blocks-2  (MultiheadAttentionBlock, full self-attention + RoPE)
  │     q = k = v = full sequence  (G+C tokens)
  │
  │  Last block  (cross-attention shortcut)
  │     q = CLS tokens only  (num_cls tokens)
  │     k = v = full sequence  (G+C tokens)
  │                                     → cls_outputs  (B, T, num_cls, E)
  │
  │  out_ln (LayerNorm)
  │  flatten last two dims
  └─ representations  (B, T, num_cls · E)  =  (B, T, 4·128) = (B, T, 512)
```

The last-block trick means the final output only needs to compute attention for the `C` CLS query positions against the full `G+C` key/value sequence, saving cost.

### Sub-modules

| Sub-module | Type | Role |
|---|---|---|
| `tf_row` | `Encoder(num_blocks, E, nhead, ...)` | Standard multi-head attention blocks with RoPE |
| `cls_tokens` | `Parameter(num_cls, E)` | Learnable CLS tokens; initialised `trunc_normal(std=0.02)` |
| `out_ln` | `LayerNorm(E)` | Applied to CLS outputs (pre-norm mode only) |

Embedding flag `return_test_representations=True` captures `representations[:, train_size:]` — shape `(B, test_size, C·E)` — after this stage.

### `return_test_representations` — row-level feature representations

**What it is:** for every test sample, the 512-dim vector produced by Stage 2 after the feature-interaction transformer. This encodes how all feature columns relate *within that row*, but contains **no label information** — `y_train` has not been injected yet at this point.

#### Extraction point

```python
# tabicl.py:489-491
representations = self.row_interactor(col_embeddings, ...)  # (B, T, C·E)
if return_test_representations:
    test_representations = representations[:, train_size:].detach().clone()
    # slices off the test portion: (B, test_size, C·E)
```

`representations` covers both training and test rows (indices `0..train_size-1` and `train_size..T-1`). Only the test slice is returned.

#### Dimension breakdown

| Dim | Size | Meaning |
|---|---|---|
| 0 | `B` (or `n_estimators` in sklearn) | One matrix per forward pass / ensemble member |
| 1 | `test_size` | One vector per test sample |
| 2 | `C·E = 4·128 = 512` | Concatenated CLS-token outputs from Stage 2 |

**Composition of dim 2 (512 = 4 × 128):**

The 512 values are the *concatenated* outputs of the 4 CLS tokens from the last attention block of `RowInteraction`. Each CLS token independently aggregates the full `(G+C)` feature-group sequence via cross-attention; concatenating them gives the model four complementary views of the same row:

```
[CLS_0 output (128) | CLS_1 output (128) | CLS_2 output (128) | CLS_3 output (128)]
```

The 4 CLS tokens are shared learned parameters — they are the same for every row and every dataset. What differentiates them per row are the key/value sequences they attend to (the column embeddings from Stage 1 for that specific row).

**What this captures:**

Because Stage 1 already encoded each feature column in the context of its training distribution, the Stage 2 output encodes *cross-feature interactions within a row* conditioned on those distributions. Two test samples with structurally similar feature patterns (relative scales, correlations) will have similar representations here, regardless of their absolute values or class labels.

**When to use:** clustering / anomaly detection based purely on feature structure, before any label information is injected.

---

## Stage 3 — In-Context Learning: `ICLearning` ([learning.py](src/tabicl/_model/learning.py))

**Goal:** given the full sequence of row representations (train + test), use the labelled training rows as context to predict test labels.

### Data flow

```
R (B, T, D)   where D = icl_dim = C·E = 512
  │  R[:, :train_size] += y_encoder(y_train)   ← bake labels into train rows
  │
  │  tf_icl (Encoder, num_blocks=12, SSMax)
  │     each block attends to train tokens only (train_size mask)
  │                                          → src  (B, T, D)
  │  ln (LayerNorm, pre-norm only)           → test_icl_representations
  │
  │  decoder: Linear(D, 2D) → GELU → Linear(2D, out_dim)
  │
  └─ out[:, train_size:]    (B, test_size, out_dim)
```

`test_icl_representations = src[:, train_size:]` — the post-LayerNorm activations just before the decoder, shape `(B, test_size, D)`.

### `return_test_icl_representations` — in-context-learning representations

**What it is:** for every test sample, the 512-dim activation that directly feeds the final decoder. Stage 3 has already attended each test row to all labelled training rows, so this representation is **fully label-aware** — it encodes the in-context learning signal alongside the feature structure.

#### Extraction point

```python
# learning.py:280-288  (_icl_predictions)
R[:, :train_size] += y_encoder(y_train)       # bake labels into training rows
src = self.tf_icl(R, train_size=train_size)   # 12-block ICL transformer
if self.norm_first:
    src = self.ln(src)                         # final LayerNorm
# src shape: (B, T, D=512)

if return_test_icl_representations:
    test_icl_representations = src[:, train_size:].detach().clone()
    # (B, test_size, D)
```

The decoder that follows (`Linear(D, 2D) → GELU → Linear(2D, out_dim)`) is the only remaining operation after this point.

#### Dimension breakdown

| Dim | Size | Meaning |
|---|---|---|
| 0 | `B` (or `n_estimators` in sklearn) | One matrix per forward pass / ensemble member |
| 1 | `test_size` | One vector per test sample |
| 2 | `D = icl_dim = C·E = 512` | Post-LayerNorm ICL activations, input to the decoder |

**What this captures:**

The 12-block `tf_icl` transformer processes the full `(B, T, D)` sequence where training tokens carry both their row representation and their class label embedding. Every test token attends to all `train_size` training tokens at every layer. By the time `src[:, train_size:]` is extracted, each vector encodes:
- The feature structure of that test row (inherited from Stage 2)
- The alignment of that row with the labeled training distribution (learned by 12 ICL attention layers)
- The contrastive signal that directly predicts the class

The geometry of this space is closest to the final decision boundary: test samples that the model is confident about will sit near the centroids of their predicted class, while uncertain samples will sit in ambiguous regions between classes.

**Comparison with `test_representations`:**

| Property | `test_representations` | `test_icl_representations` |
|---|---|---|
| Stage | After Stage 2 (RowInteraction) | After Stage 3 (ICLearning), before decoder |
| Label-aware | No | Yes — y_train baked into training rows |
| Content | Cross-feature interactions within a row | Feature structure + ICL alignment to labelled training data |
| Next operation | ICL transformer (12 blocks) | Decoder (2-layer MLP) |
| Shape | `(n_estimators, test_size, 512)` | `(n_estimators, test_size, 512)` |
| Best for | Feature-structure analysis, label-free clustering | Visualising class separability as the model sees it |

**When to use:** t-SNE / UMAP visualisation of decision structure, probing whether the model separates classes, understanding prediction confidence by inspecting how far a test vector sits from class clusters.

### Many-class classification: hierarchical tree

When `num_classes > max_classes`, a balanced classification tree is built at inference time:

1. **`_fit_hierarchical`** — partitions classes into `≤ max_classes` groups recursively, storing per-node training representations.
2. **`_predict_hierarchical`** — traverses the tree; leaf nodes call standard ICL, internal nodes multiply group probabilities bottom-up: `P(class) = P(group) × P(class | group)`.

### Sub-modules

| Sub-module | Type | Role |
|---|---|---|
| `tf_icl` | `Encoder(num_blocks=12, D, nhead, SSMax)` | 12-block causal-style ICL transformer |
| `ln` | `LayerNorm(D)` | Post-transformer normalisation (pre-norm mode) |
| `y_encoder` | `OneHotAndLinear(max_classes, D)` or `Linear(1, D)` | Encode training labels to add to row representations |
| `decoder` | `Linear(D, 2D) → GELU → Linear(2D, out_dim)` | Map ICL activations to logits/quantiles |

---

## Supporting Components

### `Encoder` ([encoders.py](src/tabicl/_model/encoders.py))

A stack of `MultiheadAttentionBlock` layers. Used by both `RowInteraction` (`tf_row`, with RoPE) and `ICLearning` (`tf_icl`, with SSMax). The `train_size` argument passed to `forward` restricts key/value to training rows in the ICL transformer.

### `SetTransformer` ([encoders.py](src/tabicl/_model/encoders.py))

A stack of `InducedSelfAttentionBlock` layers. Used by `ColEmbedding` (`tf_col`). Each ISAB block runs O(n) attention via `num_inds` inducing points:

```
Stage 1: hidden = Attn1(ind_vectors, src[:train_size], src[:train_size])
Stage 2: out    = Attn2(src, hidden, hidden)
```

SSMax applies only to `Attn1` (the larger attention over the data sequence).

### `MultiheadAttentionBlock` ([layers.py](src/tabicl/_model/layers.py))

`nn.TransformerEncoderLayer` subclass with:
- `MultiheadAttention` replacing the default `self_attn` (adds RoPE + SSMax + KV-cache support)
- Zero-init on output projections (`out_proj` and `linear2`) for stable training
- Supports asymmetric `q ≠ k = v` (cross-attention) — used for the CLS-only last block in `RowInteraction`
- `train_size` argument restricts k/v to training positions in the ICL transformer

### `MultiheadAttention` ([layers.py](src/tabicl/_model/layers.py))

`nn.MultiheadAttention` subclass. Delegates to `multi_head_attention_forward` ([attention.py](src/tabicl/_model/attention.py)) which:
- Applies rotary position embeddings (`RotaryEmbedding`) to Q and K
- Applies scalable softmax scaling to Q before softmax
- Uses Flash Attention 3 when available, falls back to PyTorch SDPA

### `RotaryEmbedding` ([rope.py](src/tabicl/_model/rope.py))

Implements RoPE (Su et al. 2021). Two modes:
- **Interleaved** (`rope_interleaved=True`): pairs `(dim 0, dim 1)`, `(dim 2, dim 3)`, …
- **Non-interleaved** (`rope_interleaved=False`, default in RowInteraction): splits embedding into `[0:d/2]` and `[d/2:d]`

XPOS (extrapolatable) and NTK-aware scaling variants are also implemented. Frequencies are cached for efficiency.

### Scalable Softmax — `SSMax` ([ssmax.py](src/tabicl/_model/ssmax.py))

Prevents attention from collapsing on long sequences. Three families:

| Type | Formula |
|---|---|
| `ssmax` | `q_scaled = q · (s · log n)`, learnable per-head scalar `s` |
| `ssmax-mlp` | `q_scaled = q · MLP(log n)` |
| `ssmax-mlp-elementwise` | elementwise scaling per head dimension |
| `qassmax-mlp` | `scale = base_mlp(log n) · (1 + tanh(query_mlp(q)))` |
| `qassmax-mlp-elementwise` | elementwise query-aware (default for Stage 1 & 3) |

### `SkippableLinear` / `OneHotAndLinear` ([layers.py](src/tabicl/_model/layers.py))

- `SkippableLinear`: regular `nn.Linear` that propagates `skip_value = -100.0` for padded CLS slots without corrupting embeddings.
- `OneHotAndLinear`: fused one-hot + linear projection — converts integer class indices to dense embeddings in a single matrix multiply.

---

## Inference Infrastructure

### `InferenceManager` ([inference.py](src/tabicl/_model/inference.py))

Wraps the forward functions of each stage to handle:
- **Automatic row batching** — splits large test sets into sub-batches that fit in GPU memory
- **GPU/CPU/disk offloading** — `MemoryEstimator` profiles memory usage; `AsyncCopyManager` overlaps GPU-CPU transfers with computation; `DiskTensor` memory-maps tensors to disk when neither GPU nor CPU RAM suffices
- **Output re-assembly** — concatenates sub-batch outputs along the sample axis

Each stage registers its own `InferenceManager`:
```python
col_embedder.inference_mgr   # enc_name="tf_col"
row_interactor.inference_mgr # enc_name="tf_row",  out_no_seq=True
icl_predictor.inference_mgr  # enc_name="tf_icl"
```

### `TabICLCache` / `KVCache` ([kv_cache.py](src/tabicl/_model/kv_cache.py))

Two caching strategies exposed via `forward_with_cache(cache_mode=...)`:

| Mode | What is cached | Memory vs. speed trade-off |
|---|---|---|
| `"kv"` | K/V projections in Stage 1 ISAB (col cache) and all Stage 3 blocks (icl cache) | More memory, fastest repeated inference |
| `"repr"` | Stage 1 col cache + Stage 2 row representations with `y_train` baked in | ~24× less ICL-cache memory, must re-run Stage 3 |

`TabICLCache` aggregates three `KVCache` objects (`col_cache`, `icl_cache`, `row_repr`) plus metadata (`train_shape`, `num_classes`, `cache_type`).

### `QuantileToDistribution` ([quantile_dist.py](src/tabicl/_model/quantile_dist.py))

Used for regression (`max_classes=0`). Converts raw `num_quantiles=999` model outputs into a proper monotone distribution with:
- Monotonicity enforcement (PAVA isotonic regression or sorting)
- GPD / exponential tail extrapolation beyond the observed range
- Analytical CDF, PDF, ICDF, CRPS, mean, variance methods

---

## Tensor Shape Cheat-Sheet

| Tensor | Shape | Notes |
|---|---|---|
| Input `X` | `(B, T, H)` | T = train_size + test_size |
| `y_train` | `(B, train_size)` | Integer labels or floats |
| After `feature_grouping` | `(B, T, G, group_size)` | `G = H` for "same" mode |
| After `ColEmbedding` | `(B, T, G+C, E)` | `C = row_num_cls` CLS slots |
| After `RowInteraction` | `(B, T, C·E)` | = `(B, T, 512)` by default |
| `test_representations` | `(B, test_size, C·E)` | Row-level embedding output |
| After Stage 3 `tf_icl` + `ln` | `(B, T, icl_dim)` | `icl_dim = C·E = 512` |
| `test_icl_representations` | `(B, test_size, icl_dim)` | Pre-decoder activations |
| Classification output | `(B, test_size, num_classes)` | Logits or probabilities |
| Regression output | `(B, test_size, num_quantiles)` | 999 quantile values |

When accessed through `TabICLClassifier.predict_proba` (which runs `n_estimators` forward passes and concatenates along axis 0):

| Attribute | Shape |
|---|---|
| `col_embedding_sample_` | `(G+C, E)` — first 4 rows are CLS placeholders (-100); rows `[4:]` are feature-group distribution embeddings averaged over all rows |
| `test_representations_` | `(n_estimators, test_size, 512)` — Stage 2 output; label-free cross-feature interactions; to get one vector per sample: `mean(axis=0)` |
| `test_icl_representations_` | `(n_estimators, test_size, 512)` — Stage 3 post-LayerNorm, pre-decoder; label-aware; directly drives predictions; to get one vector per sample: `mean(axis=0)` |

---

## File Map

| File | Primary class(es) | Stage |
|---|---|---|
| [tabicl.py](src/tabicl/_model/tabicl.py) | `TabICL` | Orchestrator |
| [embedding.py](src/tabicl/_model/embedding.py) | `ColEmbedding` | Stage 1 |
| [interaction.py](src/tabicl/_model/interaction.py) | `RowInteraction` | Stage 2 |
| [learning.py](src/tabicl/_model/learning.py) | `ICLearning`, `ClassNode` | Stage 3 |
| [encoders.py](src/tabicl/_model/encoders.py) | `Encoder`, `SetTransformer` | Shared |
| [layers.py](src/tabicl/_model/layers.py) | `MultiheadAttentionBlock`, `InducedSelfAttentionBlock`, `SkippableLinear`, `OneHotAndLinear` | Shared |
| [attention.py](src/tabicl/_model/attention.py) | `multi_head_attention_forward` | Shared |
| [rope.py](src/tabicl/_model/rope.py) | `RotaryEmbedding` | Positional encoding |
| [ssmax.py](src/tabicl/_model/ssmax.py) | `SSMax`, `SSMaxMLP`, `QASSMaxMLP` | Attention scaling |
| [kv_cache.py](src/tabicl/_model/kv_cache.py) | `TabICLCache`, `KVCache`, `KVCacheEntry` | Inference caching |
| [inference.py](src/tabicl/_model/inference.py) | `InferenceManager`, `MemoryEstimator`, `DiskTensor` | Memory management |
| [inference_config.py](src/tabicl/_model/inference_config.py) | `InferenceConfig`, `MgrConfig` | Configuration |
| [quantile_dist.py](src/tabicl/_model/quantile_dist.py) | `QuantileToDistribution` | Regression post-processing |
