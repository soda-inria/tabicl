# ensmble_ttt_step20_ckpt_test vs iclv1.1_baseline

## 结论

`ensmble_ttt_step20_ckpt_test` 在成功交集上相对 `iclv1.1_baseline` 有轻微正收益，但不能解释为稳定整体提升。

- baseline: `178/178 ok`
- step20 ckpt test: `144/178 ok`
- 成功交集: `144` 个 dataset
- 成功交集 mean accuracy: baseline `0.844898`，step20 `0.846597`，平均提升 `+0.1699 pp`
- 成功交集 median delta: `0.0000 pp`
- dataset 级别: `69 胜 / 55 负 / 20 平`
- sample 加权 accuracy: baseline `0.860831`，step20 `0.863778`，加权提升 `+0.2946 pp`
- sample 加权后约多对 `572 / 194,131` 个测试样本
- sign test: non-tie `124`，wins `69`，p ~= `0.243`，不能认为 dataset 级别有显著优势

这个 run 的核心特点是：它不是现场执行 TTT update，而是加载 `step_20.ckpt` 后直接推理。因此 wall time 很短，`472.514s`，但失败全部来自 checkpoint 缺失，而不是模型推理本身失败。

## 评测口径

交集定义为：

1. `result/ensmble_TTT_result/ensmble_ttt_step20_ckpt_test/all_classification_results.csv` 中 `status == ok`
2. `result/ensmble_TTT_result/iclv1.1_baseline/tabiclv1_1_classification_3gpu/all_classification_results.csv` 中同名 dataset `status == ok`

baseline 所有 `178` 个 dataset 都成功；step20 只有 `144` 个成功。因此当前 accuracy 对比只覆盖有 `step_20.ckpt` 的 subset，不能代表完整 178 benchmark。

## 分层结果

### 按任务类型

| task_type | n | baseline mean | step20 mean | mean delta | median delta | 胜/负/平 | weighted delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| binclass | 86 | 0.860757 | 0.859885 | -0.0872 pp | 0.0000 pp | 33 / 40 / 13 | +0.1729 pp |
| multiclass | 58 | 0.821384 | 0.826894 | +0.5511 pp | +0.1844 pp | 36 / 15 / 7 | +0.5791 pp |

主要收益来自 multiclass。binary 的 dataset-level 平均略低于 baseline，但 sample 加权后略正，说明少数大样本 binary 任务有收益，不能简单说 binary 全面变差。

### 按类别数

| class_group | n | baseline mean | step20 mean | mean delta | median delta | 胜/负/平 |
|---|---:|---:|---:|---:|---:|---:|
| binary | 86 | 0.860757 | 0.859885 | -0.0872 pp | 0.0000 pp | 33 / 40 / 13 |
| multi_3-5 | 33 | 0.796906 | 0.801673 | +0.4767 pp | +0.1852 pp | 21 / 11 / 1 |
| multi_6-10 | 25 | 0.853694 | 0.860187 | +0.6493 pp | +0.1836 pp | 15 / 4 / 6 |

step20 checkpoint 对 `3-10` 类 multiclass 最有利，尤其 `6-10` 类任务胜负比明显更好。

### 按训练集规模

| n_train | n | baseline mean | step20 mean | mean delta | median delta | 胜/负/平 |
|---|---:|---:|---:|---:|---:|---:|
| <=1k | 16 | 0.848346 | 0.851225 | +0.2879 pp | 0.0000 pp | 7 / 5 / 4 |
| 1k-5k | 82 | 0.827784 | 0.827024 | -0.0760 pp | 0.0000 pp | 34 / 36 / 12 |
| 5k-20k | 39 | 0.875844 | 0.882357 | +0.6513 pp | +0.1117 pp | 22 / 13 / 4 |
| >20k | 7 | 0.865088 | 0.866075 | +0.0986 pp | +0.0626 pp | 6 / 1 / 0 |

最明确的收益区间是 `5k-20k`。`1k-5k` 是最大的不稳定区域，胜负接近但均值略负。

### 按 baseline accuracy

| baseline acc bin | n | baseline mean | step20 mean | mean delta | median delta | 胜/负/平 |
|---|---:|---:|---:|---:|---:|---:|
| <=.5 | 3 | 0.387909 | 0.380091 | -0.7818 pp | 0.0000 pp | 1 / 1 / 1 |
| .5-.7 | 19 | 0.627124 | 0.628784 | +0.1660 pp | +0.1250 pp | 10 / 9 / 0 |
| .7-.85 | 36 | 0.778377 | 0.779919 | +0.1542 pp | +0.0500 pp | 19 / 16 / 1 |
| .85-.95 | 45 | 0.897454 | 0.901423 | +0.3969 pp | +0.1000 pp | 25 / 14 / 6 |
| >.95 | 41 | 0.979982 | 0.980041 | +0.0060 pp | 0.0000 pp | 14 / 15 / 12 |

最强收益来自 baseline 已经较强但仍有空间的 `.85-.95` 区间。`>.95` 基本饱和，几乎没有改动空间。

## 最大收益任务

| dataset | task | baseline | step20 | delta |
|---|---|---:|---:|---:|
| eye_movements_bin | binclass | 0.625493 | 0.693824 | +6.8331 pp |
| mfeat-zernike | multiclass | 0.852500 | 0.915000 | +6.2500 pp |
| compass | binclass | 0.826675 | 0.886452 | +5.9778 pp |
| eye_movements | multiclass | 0.714808 | 0.773309 | +5.8501 pp |
| artificial-characters | multiclass | 0.863992 | 0.908513 | +4.4521 pp |
| hill-valley | binclass | 0.930041 | 0.971193 | +4.1152 pp |
| autoUniv-au4-2500 | multiclass | 0.522000 | 0.562000 | +4.0000 pp |
| vehicle | multiclass | 0.858824 | 0.894118 | +3.5294 pp |
| GesturePhaseSegmentationProcessed | multiclass | 0.786329 | 0.811646 | +2.5316 pp |
| mfeat-morphological | multiclass | 0.760000 | 0.782500 | +2.2500 pp |

加权贡献最大的收益来自：

| dataset | n_test | delta | correct_delta |
|---|---:|---:|---:|
| compass | 3329 | +5.9778 pp | +199 |
| eye_movements | 2188 | +5.8501 pp | +128 |
| eye_movements_bin | 1522 | +6.8331 pp | +104 |
| artificial-characters | 2044 | +4.4521 pp | +91 |
| GesturePhaseSegmentationProcessed | 1975 | +2.5316 pp | +50 |

这些任务解释了 sample 加权结果为什么比 dataset 平均更乐观。

## 最大回退任务

| dataset | task | baseline | step20 | delta |
|---|---|---:|---:|---:|
| Water_Quality_and_Potability | binclass | 0.641768 | 0.596037 | -4.5732 pp |
| FOREX_audjpy-day-High | binclass | 0.765668 | 0.724796 | -4.0872 pp |
| FOREX_audchf-day-High | binclass | 0.754768 | 0.716621 | -3.8147 pp |
| FOREX_cadjpy-day-High | binclass | 0.716621 | 0.686649 | -2.9973 pp |
| waveform_database_generator | multiclass | 0.356000 | 0.328000 | -2.8000 pp |
| abalone | multiclass | 0.651914 | 0.630383 | -2.1531 pp |
| MIC | binclass | 0.909091 | 0.887879 | -2.1212 pp |
| statlog | binclass | 0.745000 | 0.725000 | -2.0000 pp |
| FOREX_audcad-day-High | binclass | 0.743869 | 0.724796 | -1.9074 pp |
| qsar | binclass | 0.905213 | 0.886256 | -1.8957 pp |

加权损失最大的任务：

| dataset | n_test | delta | correct_delta |
|---|---:|---:|---:|
| E-CommereShippingData | 2200 | -1.5909 pp | -35 |
| Water_Quality_and_Potability | 656 | -4.5732 pp | -30 |
| waveform_database_generator | 1000 | -2.8000 pp | -28 |
| house_16H | 2698 | -0.6672 pp | -18 |
| abalone | 836 | -2.1531 pp | -18 |

FOREX day 任务仍然是明显负例，说明 checkpoint 在时间序列/金融二分类类任务上可能过拟合了 holdout 或被伪标签/适配目标带偏。

## 失败覆盖率

step20 失败 `34` 个，全部是：

`FileNotFoundError: Missing checkpoint for dataset ... step=20`

这些失败不是低价值任务。它们在 baseline 上：

- mean accuracy: `0.807502`
- median accuracy: `0.850613`
- min/max: `0.440154 / 1.000000`
- task 分布: `21 multiclass / 13 binclass`

失败列表包含大量高分或大样本任务：

| dataset | baseline acc | task | n_train | n_test | n_classes |
|---|---:|---|---:|---:|---:|
| texture | 1.000000 | multiclass | 4400 | 1100 | 11 |
| shuttle | 0.999483 | multiclass | 46400 | 11600 | 7 |
| mobile_c36_oversampling | 0.990533 | binclass | 41408 | 10352 | 2 |
| letter | 0.986250 | multiclass | 16000 | 4000 | 26 |
| SDSS17 | 0.976200 | multiclass | 80000 | 20000 | 3 |
| naticusdroid+android+permissions+dataset | 0.971024 | binclass | 23465 | 5867 | 2 |
| Indian_pines | 0.962274 | multiclass | 7315 | 1829 | 8 |
| customer_satisfaction_in_airline | 0.961541 | binclass | 103904 | 25976 | 2 |
| internet_firewall | 0.931029 | multiclass | 52425 | 13107 | 4 |
| gina_agnostic | 0.927954 | binclass | 2774 | 694 | 2 |

所以 step20 当前最大问题不是速度或推理稳定性，而是 checkpoint 覆盖率。只看 144 成功交集会高估完整 benchmark 表现。

## 机制解释

1. step20 比 step40 更像是一个较早停止的 TTT checkpoint，降低了过适配风险。成功交集上它的负向尾部比 step40 小，最大负 delta 约 `-4.57 pp`，不是 step40 那种接近 `-9 pp` 的回退。
2. multiclass 尤其 `3-10` 类任务收益更明确，说明 TTT checkpoint 可能学到了有用的类别结构或特征重标定。
3. binary 任务总体没有稳定提升，FOREX、Water Quality、E-CommereShipping 等仍然是主要负例。这类任务可能存在 label noise、时间分布偏移或适配目标与真实 test label 不一致。
4. sample 加权收益比 dataset 平均更好，主要由 `compass`、`eye_movements`、`artificial-characters`、`GesturePhaseSegmentationProcessed` 等中等测试集任务贡献。
5. 当前结果没有现场 update 成本，因此适合评估 checkpoint 本身的推理效果；但不包含生成 checkpoint 的训练成本。

## 下一轮建议

1. 先补齐或显式 fallback missing checkpoint。当前 `34/178` missing ckpt 会让完整 benchmark 不可比。
2. 对 binary 任务加 gate。至少对 FOREX、Water_Quality、E-CommereShipping、MIC、qsar 这类已知负例先 fallback baseline。
3. 对 multiclass `3-10` 类保留 step20 checkpoint 路线，这是当前最清晰的正收益分层。
4. 做 step sweep 时优先比较 `step10 / step20 / step40` 的同一成功交集，避免 checkpoint 覆盖率差异造成选择偏差。
5. 对每个 dataset 记录 checkpoint validation/holdout 指标，并用 holdout delta 做是否启用 TTT checkpoint 的 gating；不要只按 test accuracy 事后挑选。

## 简短 summary

`step20_ckpt_test` 在 144 个成功交集上比 ICLv1.1 baseline 略好：mean accuracy `+0.1699 pp`，sample-weighted `+0.2946 pp`，多对约 `572` 个样本。收益主要来自 multiclass，尤其 `3-10` 类和 `5k-20k` 训练规模任务；binary 任务整体不稳，FOREX/Water Quality/E-CommereShipping 是主要负例。当前最大风险是覆盖率：`34` 个 dataset 缺少 `step_20.ckpt`，其中包含不少高 baseline accuracy 和大样本任务，因此这个 run 只能作为 checkpoint subset 的正向信号，不能直接作为完整 178 benchmark 的整体提升结论。
