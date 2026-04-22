# TTT / baseline_ttt / Baseline 对比总结

## 输入文件
- baseline: `TTT_compare_baseline/baseline/tabiclv1_1_classification_3gpu/all_classification_results.csv`
- baseline_ttt: `TTT_compare_baseline/baseline_ttt/all_classification_results.csv`
- TTT: `TTT_compare_baseline/ensmble_TTT_result/tabiclv1_1_ttt2/all_classification_results.csv`
- 明细文件: `TTT_compare_baseline/detail.csv`

## 主要结论
- `baseline_ttt` 在和 baseline 都成功的 172 个数据集上，平均 accuracy 只提升 +0.0073 pp：baseline 0.838128 -> baseline_ttt 0.838201；提升 / 下降 / 持平为 25 / 18 / 129。
- `TTT` 在和 baseline 都成功的 171 个数据集上，平均 accuracy 提升 +0.0336 pp：baseline 0.838739 -> TTT 0.839075；提升 / 下降 / 持平为 48 / 29 / 94。
- 在三方都成功的 171 个数据集上，TTT 比 baseline_ttt 平均高 +0.0263 pp，TTT 更好 / 更差 / 持平为 47 / 28 / 96。
- 可靠性上 baseline 最稳：baseline 0 失败，baseline_ttt 6 失败，TTT 7 失败；TTT 额外失败的是 `volkert`。
- 耗时上 baseline_ttt 约为 baseline 的 2.11x，TTT 约为 baseline 的 5.08x；TTT 的收益更明显，但运行成本也更高。

## 运行状态
| 方法 | processed | ok | fail | ok 数据集平均 accuracy | 总耗时秒 | 相对 baseline 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 178 | 178 | 0 | 0.837755 | 951.494 | 1.00x |
| baseline_ttt | 178 | 172 | 6 | 0.838201 | 2004.549 | 2.11x |
| TTT | 178 | 171 | 7 | 0.839075 | 4837.584 | 5.08x |

## 相对 baseline 的 Accuracy 对比
| 对比 | 数量 | baseline 平均 accuracy | 对比方法平均 accuracy | 平均 delta | delta 中位数 | 提升 | 下降 | 持平 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_ttt vs baseline | 172 | 0.838128 | 0.838201 | +0.0073 pp | 0.0000 pp | 25 | 18 | 129 |
| TTT vs baseline | 171 | 0.838739 | 0.839075 | +0.0336 pp | 0.0000 pp | 48 | 29 | 94 |

## 三方都成功的数据集
| 方法 | 数量 | 平均 accuracy | 相对 baseline 平均 delta | 提升/下降/持平 |
| --- | --- | --- | --- | --- |
| baseline | 171 | 0.838739 | 0.0000 pp | - |
| baseline_ttt | 171 | 0.838812 | +0.0073 pp | 25/18/128 |
| TTT | 171 | 0.839075 | +0.0336 pp | 48/29/94 |

## TTT 与 baseline_ttt 直接对比
| 范围 | 数量 | baseline_ttt 平均 accuracy | TTT 平均 accuracy | 平均 delta | delta 中位数 | TTT 更好 | TTT 更差 | 持平 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTT vs baseline_ttt（三方都成功） | 171 | 0.838812 | 0.839075 | +0.0263 pp | 0.0000 pp | 47 | 28 | 96 |

## 不同提升阈值下的数量
| 方法 | 阈值 | >= baseline + 阈值 | <= baseline - 阈值 |
| --- | --- | --- | --- |
| baseline_ttt | +0.1000 pp | 16 | 4 |
| baseline_ttt | +0.5000 pp | 0 | 1 |
| baseline_ttt | +1.0000 pp | 0 | 1 |
| baseline_ttt | +2.0000 pp | 0 | 0 |
| baseline_ttt | +5.0000 pp | 0 | 0 |
| TTT | +0.1000 pp | 28 | 16 |
| TTT | +0.5000 pp | 9 | 2 |
| TTT | +1.0000 pp | 2 | 1 |
| TTT | +2.0000 pp | 0 | 0 |
| TTT | +5.0000 pp | 0 | 0 |

## 相对 baseline 提升最大的数据集
| 方法 | 数据集 | 任务 | baseline acc | 方法 acc | delta |
| --- | --- | --- | --- | --- | --- |
| baseline_ttt | waveform_database_generator | multiclass | 0.356000 | 0.360000 | +0.4000 pp |
| baseline_ttt | GesturePhaseSegmentationProcessed | multiclass | 0.786329 | 0.789873 | +0.3544 pp |
| baseline_ttt | satimage | multiclass | 0.928460 | 0.931571 | +0.3110 pp |
| baseline_ttt | car-evaluation | multiclass | 0.973988 | 0.976879 | +0.2890 pp |
| baseline_ttt | in_vehicle_coupon_recommendation | binclass | 0.790698 | 0.793457 | +0.2759 pp |
| baseline_ttt | FOREX_audchf-day-High | binclass | 0.754768 | 0.757493 | +0.2725 pp |
| baseline_ttt | mfeat-fourier | multiclass | 0.885000 | 0.887500 | +0.2500 pp |
| baseline_ttt | airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass | 0.602500 | 0.605000 | +0.2500 pp |
| TTT | FOREX_audcad-day-High | binclass | 0.743869 | 0.754768 | +1.0899 pp |
| TTT | mfeat-zernike | multiclass | 0.852500 | 0.862500 | +1.0000 pp |
| TTT | compass | binclass | 0.826675 | 0.835086 | +0.8411 pp |
| TTT | GesturePhaseSegmentationProcessed | multiclass | 0.786329 | 0.793924 | +0.7595 pp |
| TTT | vehicle | multiclass | 0.858824 | 0.864706 | +0.5882 pp |
| TTT | FOREX_audjpy-day-High | binclass | 0.765668 | 0.771117 | +0.5450 pp |
| TTT | artificial-characters | multiclass | 0.863992 | 0.869374 | +0.5382 pp |
| TTT | FOREX_audsgd-hour-High | binclass | 0.683856 | 0.688876 | +0.5020 pp |

## 相对 baseline 下降最大的数据集
| 方法 | 数据集 | 任务 | baseline acc | 方法 acc | delta |
| --- | --- | --- | --- | --- | --- |
| baseline_ttt | yeast | multiclass | 0.629630 | 0.619529 | -1.0101 pp |
| baseline_ttt | PieChart3 | binclass | 0.875000 | 0.870370 | -0.4630 pp |
| baseline_ttt | wine-quality-red | multiclass | 0.668750 | 0.665625 | -0.3125 pp |
| baseline_ttt | autoUniv-au4-2500 | multiclass | 0.522000 | 0.520000 | -0.2000 pp |
| baseline_ttt | phoneme | binclass | 0.903793 | 0.902868 | -0.0925 pp |
| baseline_ttt | turiye_student_evaluation | multiclass | 0.522337 | 0.521478 | -0.0859 pp |
| baseline_ttt | first-order-theorem-proving | multiclass | 0.637255 | 0.636438 | -0.0817 pp |
| baseline_ttt | eye_movements_bin | binclass | 0.625493 | 0.624836 | -0.0657 pp |
| TTT | waveform_database_generator | multiclass | 0.356000 | 0.339000 | -1.7000 pp |
| TTT | autoUniv-au4-2500 | multiclass | 0.522000 | 0.516000 | -0.6000 pp |
| TTT | autoUniv-au7-1100 | multiclass | 0.404545 | 0.400000 | -0.4545 pp |
| TTT | yeast | multiclass | 0.629630 | 0.626263 | -0.3367 pp |
| TTT | Water_Quality_and_Potability | binclass | 0.641768 | 0.638720 | -0.3049 pp |
| TTT | rice_cammeo_and_osmancik | binclass | 0.929134 | 0.926509 | -0.2625 pp |
| TTT | mfeat-morphological | multiclass | 0.760000 | 0.757500 | -0.2500 pp |
| TTT | mfeat-factors | multiclass | 0.975000 | 0.972500 | -0.2500 pp |

## 失败数据集
| 方法 | 数据集 | baseline acc | 错误类型 | 错误首行 |
| --- | --- | --- | --- | --- |
| baseline_ttt | Credit_c | 0.792000 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| baseline_ttt | Rain_in_Australia | 0.850371 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| baseline_ttt | SDSS17 | 0.976200 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| baseline_ttt | accelerometer | 0.741054 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| baseline_ttt | customer_satisfaction_in_airline | 0.961541 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| baseline_ttt | dabetes_130-us_hospitals | 0.641152 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| TTT | Credit_c | 0.792000 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | Rain_in_Australia | 0.850371 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | SDSS17 | 0.976200 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | accelerometer | 0.741054 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | customer_satisfaction_in_airline | 0.961541 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | dabetes_130-us_hospitals | 0.641152 | RuntimeError | RuntimeError: Caught RuntimeError in replica 0 on device 0. |
| TTT | volkert | 0.733751 | OutOfMemoryError | OutOfMemoryError: Caught OutOfMemoryError in replica 0 on device 0. |

## TTT 跳过原因
| 方法 | 原因 | 数量 |
| --- | --- | --- |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=100 exceeds model max_classes=10 | 3 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=11 exceeds model max_classes=10 | 2 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=18 exceeds model max_classes=10 | 2 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=35 exceeds model max_classes=10 | 1 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=46 exceeds model max_classes=10 | 1 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=26 exceeds model max_classes=10 | 1 |
| baseline_ttt | TTT skipped for dataset=volkert to avoid OOM | 1 |
| baseline_ttt | stratified split \| TTT training skipped because n_classes=22 exceeds model max_classes=10 | 1 |
| TTT | stratified split \| TTT training skipped because n_classes=100 exceeds model max_classes=10 | 3 |
| TTT | stratified split \| TTT training skipped because n_classes=11 exceeds model max_classes=10 | 2 |
| TTT | stratified split \| TTT training skipped because n_classes=18 exceeds model max_classes=10 | 2 |
| TTT | stratified split \| TTT training skipped because n_classes=35 exceeds model max_classes=10 | 1 |
| TTT | stratified split \| TTT training skipped because n_classes=46 exceeds model max_classes=10 | 1 |
| TTT | stratified split \| TTT training skipped because n_classes=26 exceeds model max_classes=10 | 1 |
| TTT | stratified split \| TTT training skipped because n_classes=22 exceeds model max_classes=10 | 1 |

## 解读备注
- `avg_accuracy_ok` 都只统计成功数据集；有失败的 run 不能只看这个均值，需要同时看 failed_count。
- `baseline_ttt` 的变化更保守：多数数据集持平，平均收益接近 0，但失败数比 TTT 少 1 个，耗时也明显低于 TTT。
- `TTT` 的平均收益和提升数据集数量更高，但它在 `volkert` 上额外失败，并且总耗时约为 baseline_ttt 的 2.41x。
- 当前结果更像是 accuracy 小幅收益与运行可靠性/成本之间的 trade-off；如果用于主实验结论，建议单独处理失败数据集，尤其是 `Credit_c`、`Rain_in_Australia`、`SDSS17`、`accelerometer`、`customer_satisfaction_in_airline`、`dabetes_130-us_hospitals` 和 `volkert`。
