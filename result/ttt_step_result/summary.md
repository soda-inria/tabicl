# ttt_step_result 汇总分析

## 输入目录
- 根目录: `result/ttt_step_result`
- baseline: `result/ttt_step_result/iclv1.1_baseline/tabiclv1_1_classification_3gpu/all_classification_results.csv`
- steps: `iclv1.1_ttt_step1/2/3/4/5/10/20/30/40/50`
- 图像: `ttt_step_vs_acc.png`, `ttt_step_down_trend.png`
- 汇总表: `step_summary.csv`, `dataset_best_step.csv`, `dataset_instability.csv`, `dataset_up_summary.csv`, `dataset_down_summary.csv`, `dataset_down_trend.csv`

## 主要结论
- 以 `avg_accuracy_ok` 看，当前最优 step 是 `40`，平均 accuracy 为 `0.838830`，相对 baseline `0.837755` 提升 `+0.1075 pp`。
- 以两边都成功的 172 个数据集做公平对比，最优 step 仍是 `40`，平均 accuracy `0.838830`，比 baseline 的 `0.838128` 提升 `+0.0701 pp`。
- 收益不是单调增长: `step20` 已明显优于 `step10`，`step40` 达到峰值，`step50` 虽然在 72 个数据集上优于 baseline，但平均 accuracy 略低于 `step40`，说明更大 step 带来了更强的收益和更重的回归尾部。
- 所有 step 的失败集完全一致，都是 6 个数据集: `Credit_c, Rain_in_Australia, SDSS17, accelerometer, customer_satisfaction_in_airline, dabetes_130-us_hospitals`。这说明 step 数变化没有改变运行稳定性，主要差异集中在成功数据集上的 accuracy 变化。
- 每个 step 都只有 160 个数据集真正执行了 TTT，另外 12 个被稳定跳过；跳过原因主要是 `n_classes > 10`，另有 `volkert` 因 OOM 跳过。因此 step 曲线的收益只来自固定 160 个可适配数据集。

## Step 对比
| step | ok | fail | avg_accuracy_ok | common_ok_avg | vs baseline | common delta | 提升 | 下降 | 持平 | wall_seconds | runtime_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 172 | 6 | 0.838201 | 0.838201 | +0.0446 pp | +0.0073 pp | 25 | 18 | 129 | 2004.549 | 2.11x |
| 2 | 172 | 6 | 0.838189 | 0.838189 | +0.0434 pp | +0.0061 pp | 27 | 26 | 119 | 936.109 | 0.98x |
| 3 | 172 | 6 | 0.838295 | 0.838295 | +0.0540 pp | +0.0166 pp | 33 | 31 | 108 | 1037.233 | 1.09x |
| 4 | 172 | 6 | 0.838243 | 0.838243 | +0.0488 pp | +0.0115 pp | 34 | 33 | 105 | 1179.360 | 1.24x |
| 5 | 172 | 6 | 0.838305 | 0.838305 | +0.0550 pp | +0.0176 pp | 40 | 35 | 97 | 1294.936 | 1.36x |
| 10 | 172 | 6 | 0.838531 | 0.838531 | +0.0776 pp | +0.0402 pp | 53 | 36 | 83 | 1763.958 | 1.85x |
| 20 | 172 | 6 | 0.838791 | 0.838791 | +0.1036 pp | +0.0663 pp | 58 | 37 | 77 | 2747.490 | 2.89x |
| 30 | 172 | 6 | 0.838591 | 0.838591 | +0.0836 pp | +0.0463 pp | 61 | 48 | 63 | 3724.954 | 3.91x |
| 40 | 172 | 6 | 0.838830 | 0.838830 | +0.1075 pp | +0.0701 pp | 66 | 44 | 62 | 4791.617 | 5.04x |
| 50 | 172 | 6 | 0.838709 | 0.838709 | +0.0954 pp | +0.0580 pp | 72 | 47 | 53 | 5885.859 | 6.19x |

## 数据集最优 Step 分布
仅统计相对 baseline 有正收益的数据集；若多个 step 并列最优，取最小 step。
| best_step | dataset_count |
| --- | --- |
| 1 | 2 |
| 2 | 3 |
| 3 | 5 |
| 4 | 1 |
| 5 | 6 |
| 10 | 8 |
| 20 | 19 |
| 30 | 16 |
| 40 | 13 |
| 50 | 24 |

## 全局收益最大的 Dataset
| dataset | task | best_step | baseline_acc | best_acc | delta |
| --- | --- | --- | --- | --- | --- |
| artificial-characters | multiclass | 50 | 0.863992 | 0.882583 | +1.8591 pp |
| autoUniv-au7-1100 | multiclass | 50 | 0.404545 | 0.422727 | +1.8182 pp |
| mfeat-zernike | multiclass | 50 | 0.852500 | 0.870000 | +1.7500 pp |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 50 | 0.862450 | 0.879630 | +1.7180 pp |
| autoUniv-au4-2500 | multiclass | 50 | 0.522000 | 0.538000 | +1.6000 pp |
| FOREX_audsgd-hour-High | binclass | 40 | 0.683856 | 0.699030 | +1.5174 pp |
| satimage | multiclass | 30 | 0.928460 | 0.942457 | +1.3997 pp |
| FOREX_audusd-hour-High | binclass | 30 | 0.681689 | 0.695608 | +1.3919 pp |
| turiye_student_evaluation | multiclass | 50 | 0.522337 | 0.535223 | +1.2887 pp |
| compass | binclass | 50 | 0.826675 | 0.839291 | +1.2616 pp |
| mfeat-fourier | multiclass | 50 | 0.885000 | 0.897500 | +1.2500 pp |
| eye_movements_bin | binclass | 50 | 0.625493 | 0.637976 | +1.2484 pp |

## 跨 Step 波动最大的 Dataset
| dataset | task | baseline_acc | best_acc | worst_acc | swing |
| --- | --- | --- | --- | --- | --- |
| banknote_authentication | binclass | 0.549091 | 0.556364 | 0.516364 | +4.0000 pp |
| autoUniv-au7-1100 | multiclass | 0.404545 | 0.422727 | 0.390909 | +3.1818 pp |
| FOREX_audchf-day-High | binclass | 0.754768 | 0.757493 | 0.730245 | +2.7248 pp |
| FOREX_cadjpy-day-High | binclass | 0.716621 | 0.719346 | 0.694823 | +2.4523 pp |
| abalone | multiclass | 0.651914 | 0.653110 | 0.629187 | +2.3923 pp |
| sports_articles_for_objectivity_analysis | binclass | 0.855000 | 0.855000 | 0.835000 | +2.0000 pp |
| artificial-characters | multiclass | 0.863992 | 0.882583 | 0.864481 | +1.8102 pp |
| autoUniv-au4-2500 | multiclass | 0.522000 | 0.538000 | 0.520000 | +1.8000 pp |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 0.862450 | 0.879630 | 0.862115 | +1.7515 pp |
| mfeat-zernike | multiclass | 0.852500 | 0.870000 | 0.855000 | +1.5000 pp |
| turiye_student_evaluation | multiclass | 0.522337 | 0.535223 | 0.520619 | +1.4605 pp |
| yeast | multiclass | 0.629630 | 0.629630 | 0.616162 | +1.3468 pp |

## 固定失败数据集
| dataset | baseline_acc | error_type | error_first_line |
| --- | --- | --- | --- |
| Credit_c | 0.792000 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| Rain_in_Australia | 0.850371 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| SDSS17 | 0.976200 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| accelerometer | 0.741054 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| customer_satisfaction_in_airline | 0.961541 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| dabetes_130-us_hospitals | 0.641152 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |

## TTT 跳过原因
| reason | count |
| --- | --- |
| stratified split \| TTT training skipped because n_classes=100 exceeds model max_classes=10 | 3 |
| stratified split \| TTT training skipped because n_classes=11 exceeds model max_classes=10 | 2 |
| stratified split \| TTT training skipped because n_classes=18 exceeds model max_classes=10 | 2 |
| stratified split \| TTT training skipped because n_classes=35 exceeds model max_classes=10 | 1 |
| stratified split \| TTT training skipped because n_classes=46 exceeds model max_classes=10 | 1 |
| stratified split \| TTT training skipped because n_classes=26 exceeds model max_classes=10 | 1 |
| TTT skipped for dataset=volkert to avoid OOM | 1 |
| stratified split \| TTT training skipped because n_classes=22 exceeds model max_classes=10 | 1 |

## 解读
- `step1` 到 `step5` 的提升很小，说明少量 TTT 更新不足以稳定改变平均表现。
- `step10` 到 `step20` 是收益最明显的一段，公平对比平均 delta 从 `+0.0402 pp` 提升到 `+0.0663 pp`。
- `step40` 的均值最好，但 `step50` 拿到更多单点收益数据集，说明不同数据集的最优步数分布较散，不适合一刀切固定大步数。
- 从波动最大的 dataset 看，`banknote_authentication`、`autoUniv-au7-1100`、多组 `FOREX` 数据对 step 很敏感，后续更适合加 holdout gate、early stopping 或 loss-based stopping，而不是盲目继续增大 step。
- 由于失败集和跳过集都固定，下一轮如果目标是提升总体均值，优先方向应该是控制高 step 下的回归尾部，而不是继续增加 step 上限。
