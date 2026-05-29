# src/tabicl 技术文档

本文档面向当前仓库的 `src/tabicl` 代码，目标是把 TabICL 的包结构、用户入口、模型主干、推理/缓存、训练 prior、训练入口以及外围工具串成一条可读的工程调用链。

需要先明确一个边界：`tabicl.sklearn.TabICLClassifier.fit()` 和 `tabicl.sklearn.TabICLRegressor.fit()` 在当前代码中不是微调训练，它们只完成 checkpoint 加载、输入/标签预处理、ensemble 配置和可选 cache 构建。模型权重更新发生在 `src/tabicl/train/run.py` 的预训练训练入口，或你自己在 TTT 脚本里显式打开梯度、构造 loss、调用 optimizer。

## 1. 总体架构

`src/tabicl` 可以分成三层：

```text
用户接口层
  tabicl.__init__
  tabicl.sklearn.TabICLClassifier
  tabicl.sklearn.TabICLRegressor
  tabicl.forecast.TabICLForecaster
  tabicl.unsupervised.TabICLUnsupervised
  tabicl.shap helpers

模型与推理层
  tabicl.model.TabICL
    ColEmbedding
    RowInteraction
    ICLearning
    QuantileToDistribution
  InferenceConfig / InferenceManager
  KVCache / TabICLCache

训练与数据层
  tabicl.prior.PriorDataset
  tabicl.prior.LoadPriorDataset / SavePriorDataset
  tabicl.train.Trainer
  tabicl.train.train_config
```

主干模型的核心前向是：

```text
X, y_train
  -> ColEmbedding
       将列/特征分布编码成 cell-level feature embeddings
  -> RowInteraction
       在每一行内部做 feature interaction，并用 CLS tokens 汇聚成 row representation
  -> ICLearning
       在样本维度上做 dataset-level in-context learning
  -> logits / probabilities / regression quantiles / regression statistics
```

核心定位：

| 模块 | 主要文件 | 关键对象 |
|---|---|---|
| 顶层导出 | `src/tabicl/__init__.py` | `TabICLClassifier`, `TabICLRegressor`, `TabICLCache`, `InferenceConfig` |
| 原始 PyTorch 模型 | `src/tabicl/model/tabicl.py` | `TabICL` |
| 列嵌入 | `src/tabicl/model/embedding.py` | `ColEmbedding` |
| 行内交互 | `src/tabicl/model/interaction.py` | `RowInteraction` |
| in-context learning | `src/tabicl/model/learning.py` | `ICLearning` |
| sklearn 分类 | `src/tabicl/sklearn/classifier.py` | `TabICLClassifier` |
| sklearn 回归 | `src/tabicl/sklearn/regressor.py` | `TabICLRegressor` |
| 预处理/ensemble | `src/tabicl/sklearn/preprocessing.py` | `TransformToNumerical`, `PreprocessingPipeline`, `EnsembleGenerator` |
| 推理配置 | `src/tabicl/model/inference_config.py` | `MgrConfig`, `InferenceConfig` |
| 推理执行/内存管理 | `src/tabicl/model/inference.py` | `InferenceManager` |
| cache 数据结构 | `src/tabicl/model/kv_cache.py` | `KVCache`, `TabICLCache` |
| synthetic prior | `src/tabicl/prior/dataset.py` | `PriorDataset`, `SCMPrior` |
| 预生成 prior 加载/保存 | `src/tabicl/prior/genload.py` | `LoadPriorDataset`, `SavePriorDataset` |
| 训练入口 | `src/tabicl/train/run.py` | `Trainer` |
| 时间序列封装 | `src/tabicl/forecast` | `TabICLForecaster`, `ForecastEngine` |

## 2. 用户入口与 sklearn API

### 2.1 顶层导出

`src/tabicl/__init__.py` 直接导出常用对象：

- `TabICLClassifier`
- `TabICLRegressor`
- `InferenceConfig`
- `TabICLCache`

因此用户侧一般这样使用：

```python
from tabicl import TabICLClassifier, TabICLRegressor
```

而不是直接实例化 `tabicl.model.TabICL`。直接使用 `TabICL` 需要自己处理数值化、ensemble、标签编码、target scaling、checkpoint 加载、cache 等细节。

### 2.2 `TabICLClassifier`

`TabICLClassifier` 定义在 `src/tabicl/sklearn/classifier.py`，核心方法如下：

- `__init__()`：保存用户配置，如 `n_estimators`、`norm_methods`、`feat_shuffle_method`、`class_shuffle_method`、`batch_size`、`kv_cache`、`checkpoint_version`、`device`、`use_amp`、`offload_mode` 等。
- `_load_model()`：从本地路径或 Hugging Face Hub 加载 checkpoint，然后用 checkpoint 中的 `config` 构造 `TabICL(**config)`，再加载 `state_dict`。
- `fit(X, y)`：准备推理所需状态，不训练权重。
- `predict_proba(X)`：生成 ensemble 视角，将训练集和测试集拼成 in-context 输入，调用模型前向，汇总概率。
- `predict(X)`：调用 `predict_proba()` 后取最大概率类别，并通过 `LabelEncoder` 还原原始标签。

`fit()` 的实际流程是：

```text
validate_data(X, y)
  -> check_classification_targets(y)
  -> resolve device
  -> build InferenceConfig
  -> _load_model()
  -> model_.to(device_)
  -> LabelEncoder 编码 y
  -> 检查 n_classes_ 和 model_.max_classes
  -> TransformToNumerical.fit_transform(X)
  -> EnsembleGenerator.fit(X, y)
  -> 可选 _build_kv_cache()
```

这里没有 `loss.backward()`、`optimizer.step()`，也不会改 `model_.state_dict()`。`fit()` 只是 sklearn 语义中的“拟合估计器状态”，不是深度学习 fine-tuning。

分类任务的多类边界：

- checkpoint 中的 `model_.max_classes` 是模型原生支持的最大类别数。
- 如果 `n_classes_ <= max_classes`，走普通分类。
- 如果 `n_classes_ > max_classes` 且 `support_many_classes=True`，代码启用 many-class 策略。
- many-class 场景下不支持 `kv_cache`，因为 cache 逻辑当前按原生类别前向组织。

### 2.3 `TabICLRegressor`

`TabICLRegressor` 定义在 `src/tabicl/sklearn/regressor.py`。它和分类器结构相似，但有两个关键差异：

- `fit()` 会用 `StandardScaler` 对 `y` 做 target scaling。
- 模型输出不是类别 logits，而是 regression quantiles，再由 `QuantileToDistribution` 转成 mean、median、variance、指定 quantiles 等统计量。

`fit()` 的实际流程是：

```text
validate_data(X, y)
  -> y 转 float32，必要时 flatten column vector
  -> resolve device
  -> build InferenceConfig
  -> _load_model()
  -> model_.to(device_)
  -> y_scaler_.fit_transform(y)
  -> TransformToNumerical.fit_transform(X)
  -> EnsembleGenerator.fit(X, y_scaled)
  -> 可选 _build_kv_cache()
```

`predict(X, output_type="mean", alphas=None)` 支持：

- `"mean"`
- `"median"`
- `"variance"`
- `"quantiles"`
- `"raw_quantiles"`
- 或上述字段的列表

最终输出会经过 `y_scaler_` 逆变换，回到原始 target 尺度。

### 2.4 `TabICLBaseEstimator`

`TabICLBaseEstimator` 定义在 `src/tabicl/sklearn/base.py`，提供分类器和回归器共享逻辑：

- `_resolve_device()`：默认 CUDA 优先，否则 CPU；用户也可以指定 `cuda`、`cpu`、`mps`。
- `_resolve_amp_fa3()`：根据数据规模和用户配置决定 AMP / FA3。
- `_build_inference_config()`：构造三段模型各自的 `COL_CONFIG`、`ROW_CONFIG`、`ICL_CONFIG`。
- `_move_cache_to_device()`：加载持久化 estimator 后，把 cache 移到当前设备，必要时升精度到 float32。
- `__getstate__()` / `__setstate__()`：控制 pickle 序列化，避免直接保存设备绑定的 `nn.Module`。
- `save()` / `load()`：面向用户的持久化接口。

持久化策略非常关键：默认不会把 `model_` 这个大 `nn.Module` 直接 pickle 进去，而是保存足够恢复 estimator 的配置、预处理器、ensemble 状态、可选模型权重和可选 KV cache。

## 3. 模型主干：`TabICL`

`TabICL` 定义在 `src/tabicl/model/tabicl.py`。它是纯 PyTorch 模块，负责真正的三段式前向：

```text
TabICL
  col_embedder: ColEmbedding
  row_interactor: RowInteraction
  icl_predictor: ICLearning
  quantile_dist: QuantileToDistribution   # only regression
```

初始化时，`max_classes` 决定任务类型：

- `max_classes == 0`：回归，输出维度是 `num_quantiles`，并创建 `QuantileToDistribution`。
- `max_classes > 0`：分类，输出维度是 `max_classes`。

`row_num_cls` 会影响 ICL 维度：

```text
icl_dim = embed_dim * row_num_cls
```

因为 `RowInteraction` 会把每行的多个 CLS token 拼接成一个 row representation。

### 3.1 输入输出约定

模型主干大部分函数使用统一张量约定：

```text
X:       (B, T, H)
y_train: (B, train_size)
d:       optional, (B,)
```

含义：

- `B`：batch 内的表格数，通常也是 ensemble member 数。
- `T`：总样本数，前 `train_size` 行是 context/train，后面是 query/test。
- `H`：特征数。
- `y_train`：只提供 context/train 部分标签。
- `d`：训练阶段可表示每个 synthetic dataset 的有效特征数，用于处理 padding 后的 feature mask。

输出：

- 分类：`(B, test_size, max_classes)`，可为 logits 或 probabilities。
- 回归：`(B, test_size, num_quantiles)`，或由 `predict_stats()` 转成统计量。

### 3.2 训练模式与推理模式

`TabICL.forward()` 根据 `self.training` 分两条路：

```text
model.train() -> _train_forward()
model.eval()  -> _inference_forward()
```

`_train_forward()` 用在预训练训练循环中：

```text
ColEmbedding(X, y_train, d, embed_with_test)
  -> RowInteraction(embeddings, d)
  -> ICLearning(representations, y_train)
```

`_inference_forward()` 用在 sklearn predictor 中：

```text
ColEmbedding(..., feature_shuffles, mgr_config=COL_CONFIG)
  -> RowInteraction(..., mgr_config=ROW_CONFIG)
  -> ICLearning(..., return_logits, softmax_temperature, mgr_config=ICL_CONFIG)
```

训练模式支持 `d` 和 padding mask；推理模式支持 `feature_shuffles`、`InferenceConfig` 和 logits/probabilities 控制。

## 4. 第一段：`ColEmbedding`

`ColEmbedding` 定义在 `src/tabicl/model/embedding.py`，作用是把原始表格 cell 值变成 distribution-aware column embeddings。

输入大致是：

```text
X: (B, T, H)
y_train: (B, train_size)
```

输出大致是：

```text
embeddings: (B, T, H + C, E)
```

其中：

- `H` 是特征数。
- `C` 是为后续 row CLS token 预留的位置，来自 `reserve_cls_tokens`。
- `E` 是 `embed_dim`。

核心机制：

- `in_linear` 将 scalar cell 或 feature group 投影到 embedding 维度。
- `tf_col` 是 `SetTransformer`，用 column set 的方式建模每个特征列的分布。
- `target_aware=True` 时，会把训练标签信息注入列嵌入。
- `feature_group` 可以将相邻或循环排列的特征组合成 group。
- `feature_shuffles` 用于推理时复用列嵌入并映射不同 ensemble 的特征排列。
- many-class 分类时，`mixed_radix_ensemble` 会把超出 `max_classes` 的标签拆成多个低基数视角，以便列嵌入阶段仍能使用有限类别空间。

`ColEmbedding.forward_with_cache()` 是 cache 路径的一部分。它可以在训练 context 上保存列嵌入相关 KV，后续只对测试部分继续计算。

## 5. 第二段：`RowInteraction`

`RowInteraction` 定义在 `src/tabicl/model/interaction.py`，作用是在每一行内部建模 feature-feature interaction。

输入：

```text
embeddings: (B, T, H + C, E)
```

输出：

```text
representations: (B, T, C * E)
```

核心机制：

- `tf_row` 是带 RoPE 的 `Encoder`。
- `cls_tokens` 是 learnable parameters，作为每行的聚合 token。
- `_aggregate_embeddings()` 先让 CLS token 和特征 token 共同经过 transformer blocks。
- 最后一层用 CLS token 作为 query，全序列作为 key/value，只取 CLS outputs。
- 多个 CLS token flatten 后得到每一行的 representation。

这里的注意点是：`RowInteraction` 不是跨样本学习，它只在每一行内部看特征。跨样本的 in-context learning 在下一段 `ICLearning` 中完成。

## 6. 第三段：`ICLearning`

`ICLearning` 定义在 `src/tabicl/model/learning.py`，作用是在样本维度上做 dataset-wise in-context learning。

输入：

```text
representations: (B, T, C * E)
y_train:        (B, train_size)
```

输出：

```text
classification: (B, test_size, max_classes)
regression:     (B, test_size, num_quantiles)
```

核心机制：

- `tf_icl` 是样本维度 transformer encoder。
- `y_encoder` 把 context 标签编码到和 row representation 相同的维度。
- 对 context/train 部分，模型可以利用 `X_train, y_train`；对 query/test 部分，只提供 `X_test`，模型预测其标签/目标。
- `decoder` 将 ICL 输出映射到分类 logits 或回归 quantiles。

分类分两类路径：

- `_predict_standard()`：类别数不超过 `max_classes` 的普通分类。
- `_predict_hierarchical()`：类别数超过 `max_classes` 时，使用层次分类树，把大类别集合拆成多个小分类问题。

回归路径输出 raw quantiles，不直接输出单点值。`TabICL.predict_stats()` 会调用 `QuantileToDistribution`，保证 quantiles 单调后再计算 mean、median、variance 或指定分位数。

## 7. 分类推理完整调用链

分类器从用户调用到模型前向的链路是：

```text
TabICLClassifier.fit(X_train, y_train)
  -> validate_data
  -> LabelEncoder
  -> TransformToNumerical
  -> EnsembleGenerator.fit
  -> _load_model
  -> optional _build_kv_cache

TabICLClassifier.predict_proba(X_test)
  -> check_is_fitted
  -> validate_data(reset=False)
  -> X_encoder_.transform
  -> ensemble_generator_.transform(mode="test")
  -> for each norm_method:
       if model_kv_cache_ exists:
           _batch_forward_with_cache
             -> model_.forward_with_cache
       else:
           _batch_forward
             -> model_(X, y_train, feature_shuffles, return_logits=...)
  -> average logits or probabilities across ensemble members
  -> inverse class shuffle / probability aggregation

TabICLClassifier.predict(X_test)
  -> predict_proba
  -> argmax
  -> y_encoder_.inverse_transform
```

分类 ensemble 的主要来源：

- normalization method 多视角。
- feature shuffle 多视角。
- class label shuffle 多视角。

`average_logits=True` 时先平均 logits 再 softmax；`average_logits=False` 时平均 probabilities。默认平均 logits。

## 8. 回归推理完整调用链

回归器从用户调用到模型前向的链路是：

```text
TabICLRegressor.fit(X_train, y_train)
  -> validate_data
  -> y 转 float32
  -> y_scaler_.fit_transform
  -> TransformToNumerical
  -> EnsembleGenerator.fit
  -> _load_model
  -> optional _build_kv_cache

TabICLRegressor.predict(X_test, output_type, alphas)
  -> check_is_fitted
  -> validate_data(reset=False)
  -> X_encoder_.transform
  -> ensemble_generator_.transform(mode="test")
  -> for each norm_method:
       if model_kv_cache_ exists:
           _batch_forward_with_cache
             -> model_.predict_stats_with_cache
       else:
           _batch_forward
             -> model_.predict_stats
  -> average ensemble outputs
  -> y_scaler_.inverse_transform for target-scale outputs
```

回归的输出类型由 `TabICL.predict_stats()` 负责：

- `"mean"`：raw quantiles 的均值。
- `"variance"`：raw quantiles 的方差。
- `"median"`：通过 distribution `icdf(0.5)`。
- `"quantiles"`：通过 distribution `icdf(alphas)`。
- `"raw_quantiles"`：返回单调修正后的 quantile tensor。

## 9. 预处理与 ensemble

预处理代码集中在 `src/tabicl/sklearn/preprocessing.py`。

### 9.1 `TransformToNumerical`

`TransformToNumerical` 将用户输入表格转成纯数值矩阵。它处理的核心问题是：

- pandas / numpy 输入统一。
- 非数值列转成数值表示。
- 缺失值处理。
- 保持 sklearn transformer 语义，支持 `fit_transform()` 和 `transform()`。

这是进入 TabICL 之前的第一层输入标准化。模型主干只吃 float tensor，不直接理解字符串类别列。

### 9.2 `PreprocessingPipeline`

`PreprocessingPipeline` 的顺序是：

```text
CustomStandardScaler
  -> normalization method
       none / power / quantile / quantile_rtdl / robust
  -> OutlierRemover
```

注意：

- `normalization_method="none"` 仍然会做 standard scaling 和 outlier handling。
- `quantile` 类方法在极端测试 outlier 下可能失败，代码会用训练阶段的 min/max clip 后重试。
- `OutlierRemover` 是为了防止异常值主导表格分布。

### 9.3 `EnsembleGenerator`

`EnsembleGenerator` 同时管理 preprocessing 和 ensemble 配置：

```text
fit(X, y)
  -> UniqueFeatureFilter 删除常数特征
  -> 生成 norm_methods
  -> 生成 feature shuffle patterns
  -> 分类任务生成 class shuffle patterns
  -> 为每个 normalization method fit PreprocessingPipeline
```

`transform(mode="train")` 会返回训练 context 的多视角版本；`transform(mode="test")` 会把训练集状态和测试集输入合并成模型需要的 in-context 格式。

feature shuffle 的方法：

- `"none"`：不打乱。
- `"shift"`：循环平移。
- `"random"`：随机排列。
- `"latin"`：Latin square 排列。

class shuffle 只用于分类任务，方法同样支持 `"none"`、`"shift"`、`"random"`、`"latin"`。

## 10. many-class 与 quantile regression

### 10.1 many-class classification

TabICL checkpoint 有原生 `max_classes` 限制。超过时，代码采用两层策略：

- `ColEmbedding` 阶段：mixed-radix ensembling，把大类别编号拆成多个不超过 `max_classes` 的 digit/group 视角。
- `ICLearning` 阶段：hierarchical classification，把类别集合递归拆成若干子组，每个节点都是一个不超过 `max_classes` 的局部分类问题。

这能让模型在类别数超过 checkpoint 原生输出维度时仍可运行，但它不是普通 logits 维度扩展，也不是训练新分类头。

### 10.2 quantile regression

回归 checkpoint 的 `max_classes=0`。这时：

- `TabICL` 的输出维度是 `num_quantiles`。
- `ICLearning.decoder` 输出 raw quantiles。
- `QuantileToDistribution` 将 raw quantiles 变成可查询的 distribution。
- `predict_stats()` 提供 mean、median、variance、quantiles 等输出。

因此回归不是直接学一个标量，而是通过 quantile distribution 表达不确定性。

## 11. `InferenceConfig`、内存管理与加速

推理配置定义在 `src/tabicl/model/inference_config.py`。

`InferenceConfig` 包含三个配置块：

```text
COL_CONFIG
ROW_CONFIG
ICL_CONFIG
```

这对应模型三段：

- `COL_CONFIG` 给 `ColEmbedding` / `tf_col`。
- `ROW_CONFIG` 给 `RowInteraction` / `tf_row`。
- `ICL_CONFIG` 给 `ICLearning` / `tf_icl`。

每个 `MgrConfig` 可配置：

- `device`
- `use_amp`
- `use_fa3`
- `verbose`
- `min_batch_size`
- `safety_factor`
- `offload`
- `auto_offload_threshold`
- `cpu_safety_factor`
- `max_pinned_memory_mb`
- `disk_offload_dir`
- `disk_min_free_mb`
- `disk_flush_mb`
- `disk_cleanup`
- `disk_file_prefix`
- `disk_dtype`
- `disk_safety_factor`
- `use_async`
- `async_depth`

`TabICLBaseEstimator._build_inference_config()` 会把用户在 sklearn estimator 上设置的 `device`、`use_amp`、`use_fa3`、`offload_mode`、`disk_offload_dir` 等参数映射到这些配置块。

### 11.1 `InferenceManager`

`InferenceManager` 定义在 `src/tabicl/model/inference.py`，负责大张量推理时的实际调度：

- 根据显存估计选择 batch size。
- 在 GPU / CPU / disk 间 offload 中间输出。
- 使用 pinned memory 和 async copy 优化 GPU 到 CPU 的传输。
- 在支持时启用 Flash Attention 3。
- 管理临时 disk tensor 的清理。

最重要的内存瓶颈通常在列嵌入阶段，因为 `ColEmbedding` 会产生类似 `(batch, rows, columns, embed_dim)` 的大张量。`offload_mode` 主要就是为这个问题服务。

### 11.2 AMP 与 FA3

`use_amp="auto"` 的默认逻辑在 `TabICLBaseEstimator._resolve_amp_fa3()` 中：

- 小数据：关闭 AMP 和 FA3，避免额外开销。
- 中等数据：开启 AMP。
- 大数据：开启 AMP，并可能开启 FA3。

如果用户显式关闭 AMP，但数据不小，代码可能把 FA3 当作 attention 加速补偿。

## 12. KV cache 与 repr cache

cache 数据结构定义在 `src/tabicl/model/kv_cache.py`：

- `KVCacheEntry`：单层 attention 的 key/value。
- `KVCache`：一个 encoder 内多层 KV 的集合。
- `TabICLCache`：整个 TabICL 三段 cache 的容器，包含 `col_cache`、`row_cache`、`icl_cache` 或 row representations。

sklearn estimator 的 `kv_cache` 参数支持：

- `False`：不缓存。
- `True` 或 `"kv"`：缓存 column embedding 和 ICL transformer 的 KV projections。
- `"repr"`：缓存 column embedding KV 和 row interaction outputs / representations。

cache 的使用方式：

```text
fit(..., kv_cache=True/"kv"/"repr")
  -> _build_kv_cache()
       -> ensemble_generator_.transform(mode="train")
       -> model_.forward_with_cache(...) 或 model_.predict_stats_with_cache(...)
       -> 保存每个 norm_method 对应的 TabICLCache

predict/predict_proba(...)
  -> 如果 model_kv_cache_ 存在：
       对 cache 按 batch slice
       只处理 X_test 相关计算
```

需要特别注意：

- KV cache 是推理加速机制，不是训练机制。
- cache 保存的是训练 context 的中间激活/KV，不会更新模型参数。
- cache tensor 的 dtype 可能来自 AMP。加载到 CPU/MPS 或 CUDA 但 AMP 关闭时，代码会自动 upcast 到 float32。
- many-class 分类当前不支持 `kv_cache`。

## 13. synthetic prior 与数据生成

训练数据生成代码集中在 `src/tabicl/prior`。

### 13.1 `PriorDataset`

`PriorDataset` 定义在 `src/tabicl/prior/dataset.py`，是训练用的 infinite `IterableDataset`。它每次迭代生成一批 synthetic tabular datasets：

```text
X, y, d, seq_lens, train_sizes = PriorDataset.get_batch()
```

返回含义：

- `X`：synthetic 表格特征。
- `y`：synthetic target。
- `d`：每个 dataset 的有效特征数。
- `seq_lens`：每个 dataset 的样本数。
- `train_sizes`：context/query 切分位置。

支持的 prior 类型：

- `"mlp_scm"`：MLP-based structural causal model。
- `"tree_scm"`：tree-based structural causal model。
- `"mix_scm"`：混合 MLP SCM 和 tree SCM。
- `"dummy"`：随机数据，主要用于调试。

### 13.2 `SCMPrior`

`SCMPrior` 是主要 synthetic data 生成器，负责：

- 按 group / subgroup 生成相似或共享结构的数据。
- 采样特征数、样本数、训练切分位置。
- 调用 `MLPSCM` 或 `TreeSCM` 生成底层变量。
- 调用 `Reg2Cls` 把连续目标转换成分类任务。
- 支持 `seq_len_per_gp` 下的 variable-length / nested tensor 场景。

这部分代码的定位是预训练数据分布设计，不属于 sklearn inference API。

### 13.3 `LoadPriorDataset` 与 `SavePriorDataset`

`src/tabicl/prior/genload.py` 提供预生成 prior 数据的保存与加载：

- `SavePriorDataset`：把在线生成的 batch 保存成磁盘文件。
- `LoadPriorDataset`：从磁盘加载预生成 batch，支持 DDP rank 切分、从指定位置开始、加载后删除等。
- `dense2sparse()` / `sparse2dense()`：用于更紧凑地存储 padding 后的表格 batch。

训练时可以选择：

- 在线生成 prior：`prior_dir is None`。
- 从预生成目录加载：`prior_dir` 指向已保存数据。

## 14. 训练入口

训练入口在 `src/tabicl/train/run.py`，核心类是 `Trainer`。

`Trainer.__init__()` 依次执行：

```text
configure_ddp()
configure_wandb()
build_model()
configure_prior()
configure_optimizer()
configure_amp()
load_checkpoint()
```

### 14.1 DDP 与设备

`configure_ddp()` 通过环境变量判断是否处于 distributed run：

- 如果存在 `RANK`，使用 `torch.distributed.init_process_group(backend="nccl")`。
- 设置 `ddp_rank`、`ddp_local_rank`、`ddp_world_size`。
- 将 `config.device` 改成当前 local rank 对应的 CUDA。
- 按 world size 调整每 GPU batch size。
- 设置 numpy / torch seed。

非 DDP 时，单进程训练。

### 14.2 模型构建

`build_model()` 从 `config` 组装 `model_config`，创建 `TabICL(**model_config)`，并支持：

- `freeze_col`
- `freeze_row`
- `freeze_icl`
- `torch.compile`
- DDP wrapper

这里才是预训练权重更新路径之一。optimizer 会拿 `raw_model.parameters()`，训练 loop 会进行 backward 和 step。

### 14.3 prior 与 dataloader

`configure_prior()` 决定训练数据来源：

- `prior_dir is None`：构造 `PriorDataset` 在线生成。
- `prior_dir` 非空：构造 `LoadPriorDataset` 加载预生成数据。

随后用 `DataLoader(batch_size=None)` 包装，因为 `PriorDataset` 自己已经生成 batch。

### 14.4 optimizer、scheduler、AMP、checkpoint

`configure_optimizer()` 使用 `AdamW`，scheduler 来自 `src/tabicl/train/optim.py`。

`configure_amp()` 根据 `config.amp` 和 CUDA 设备创建 autocast context 和 `GradScaler`。

`load_checkpoint()` 支持从指定 checkpoint 或 `checkpoint_dir` 最新 `step-*.ckpt` 恢复：

- 加载 `state_dict`。
- 如果不是 `only_load_model`，同时恢复 optimizer、scheduler、`curr_step`。

`save_checkpoint()` 保存模型权重、optimizer、scheduler、当前 step 和 config。

## 15. 时间序列 forecast 封装

forecast 相关代码在 `src/tabicl/forecast`。它不是新的时间序列 foundation model，而是把时间序列问题转成带时间特征的 tabular regression，然后调用 `TabICLRegressor`。

主链路：

```text
TabICLForecaster
  -> TimeTransformChain
       IndexEncoder
       DatetimeEncoder
       AutoPeriodicEncoder
  -> ForecastEngine
       -> SeriesDispatcher
       -> TabICLRegressor.fit(train_X, train_y)
       -> TabICLRegressor.predict(test_X, output_type=[point_estimate, "quantiles"])
```

### 15.1 `TabICLForecaster`

`TabICLForecaster` 定义在 `src/tabicl/forecast/forecaster.py`，负责：

- 缺失 target 处理。
- context 截断到 `max_context_length`。
- future horizon 检查。
- covariate 对齐。
- 时间特征构造。
- 调用 `ForecastEngine.predict()`。

它支持 `predict()` 输入 `TimeSeriesDataFrame`，也支持 `predict_df()` 处理普通 pandas DataFrame。

### 15.2 `ForecastEngine`

`ForecastEngine` 定义在 `src/tabicl/forecast/engine.py`。它对每条时间序列执行：

```text
model = TabICLRegressor(**tabicl_config)
model.fit(train_X, train_y)
raw = model.predict(test_X, output_type=[point_estimate, "quantiles"], alphas=quantiles)
```

返回结果包含：

- `"target"`：mean 或 median 点预测。
- 每个 quantile 的预测列。

## 16. SHAP 与 unsupervised 外围功能

### 16.1 `tabicl.shap`

`src/tabicl/shap` 提供解释工具：

- `get_shap_values()`：基于 `shap` 包计算 SHAP values。
- `get_shap_explainer()`：构造 SHAP explainer。
- `plot_shap()` / `plot_shap_feature()`：绘图。
- `get_shapiq_explainer()`：基于 `shapiq` 的 interaction explainer。

这些函数面向已训练/已 fit 的 sklearn estimator，不改变 TabICL 模型结构。

### 16.2 `tabicl.unsupervised`

`TabICLUnsupervised` 定义在 `src/tabicl/unsupervised/_unsupervised.py`。它把 TabICL 分类器/回归器用于无监督特征分析或异常/分布相关任务，内部仍然复用：

- `TabICLClassifier`
- `TabICLRegressor`
- `Shuffler`
- `QuantileDistribution`

这是外围应用层，不是主训练入口。

## 17. TTT / ensemble 改造时的关键边界

当前你在 `ttt_continus.py`、`TTT_ensemble_eva.py`、`Ensemble_search_block_TTT.py` 等脚本里做 TTT/ensemble 改造时，最需要守住这些边界：

1. sklearn wrapper 的 `fit()` 不会更新模型权重  
   它只是准备 ICL 推理上下文。若要 TTT，必须显式打开目标参数的 `requires_grad`，构造 loss，执行 `backward()` 和 `optimizer.step()`。

2. `model.eval()` 加 `torch.inference_mode()` 是普通推理路径  
   该路径会计算 activations/QKV，但不会保留梯度，也不会更新参数。

3. KV/repr cache 是推理加速，不是训练  
   cache 保存 context 的中间状态，帮助重复预测时少算一部分训练集前向。它不能替代 TTT，也不能表示模型已经适配。

4. 如果要做模块选择，主干模块边界是清楚的  
   - `model.col_embedder`
   - `model.row_interactor`
   - `model.icl_predictor`
   - 回归额外有 `model.quantile_dist`，但这是 quantile 后处理模块，不是分类 TTT 重点。

5. 当前 checkpoint 差异分析时，应按参数名映射到模块  
   例如 `icl_predictor.tf_icl.blocks.*` 属于 dataset-level ICL transformer；`col_embedder.tf_col.*` 属于列分布编码；`row_interactor.tf_row.*` 属于行内特征交互。

6. many-class 和 cache 有冲突限制  
   当前分类器在 `n_classes_ > model_.max_classes` 时不允许 `kv_cache`，所以 benchmark/TTT 脚本如果强行组合这两者，需要先改 cache 语义。

7. 回归输出不是单标量 head  
   如果把分类 TTT 逻辑迁移到回归，需要考虑 quantile loss / distribution 统计，而不是只替换 CE loss。

8. `InferenceConfig` 会影响推理内存和 dtype  
   TTT 训练路径如果复用 inference config，需要确认没有被 `torch.inference_mode()`、offload tensor 或 dtype upcast/downcast 破坏梯度路径。

## 18. 快速阅读路线

如果只想最快理解当前 TabICL 怎么跑分类：

1. `src/tabicl/sklearn/classifier.py`
   - 看 `fit()`、`predict_proba()`、`_batch_forward()`、`_batch_forward_with_cache()`。
2. `src/tabicl/sklearn/preprocessing.py`
   - 看 `TransformToNumerical`、`PreprocessingPipeline`、`EnsembleGenerator`。
3. `src/tabicl/model/tabicl.py`
   - 看 `forward()`、`_inference_forward()`、`forward_with_cache()`。
4. `src/tabicl/model/embedding.py`
   - 看 `ColEmbedding.forward()`。
5. `src/tabicl/model/interaction.py`
   - 看 `RowInteraction.forward()`。
6. `src/tabicl/model/learning.py`
   - 看 `ICLearning.forward()` 和 classification prediction 分支。

如果想理解预训练：

1. `src/tabicl/train/run.py`
   - 看 `Trainer.__init__()`、`build_model()`、`configure_prior()`、`train()`。
2. `src/tabicl/prior/dataset.py`
   - 看 `PriorDataset`、`SCMPrior`。
3. `src/tabicl/prior/mlp_scm.py` 和 `src/tabicl/prior/tree_scm.py`
   - 看 synthetic table 的生成机制。
4. `src/tabicl/prior/reg2cls.py`
   - 看连续目标如何转换成分类标签。

## 19. 结论

当前 `src/tabicl` 的核心思想是：用 synthetic tabular prior 预训练一个三段式 transformer，让它在推理时把训练集样本作为 context，通过 in-context learning 直接预测测试样本。sklearn API 的 `fit()` 只是准备 context 和推理状态；真正的深度学习训练在 `train/run.py`，而 TTT 需要在你自己的脚本中显式建立梯度更新路径。

对当前 TTT/ensemble 实验最有用的模块划分是：

```text
col_embedder   -> 学列分布和 target-aware column embedding
row_interactor -> 学每行内部 feature interaction
icl_predictor  -> 学样本维度的 context/query 映射，是当前 TTT 最直接的适配空间
```

因此，分析 checkpoint diff、LoRA 注入、block search、TTT 训练参数时，应优先把参数名映射回这三个模块，再判断改动是在列分布、行内交互，还是 dataset-level ICL 上生效。
