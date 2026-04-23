# result_ttt: ttt_step10 与 baseline 对比总结

## 输入文件
- baseline: `result_ttt/baseline/tabiclv1_1_classification_3gpu/all_classification_results.csv`
- ttt_step10: `result_ttt/iclv1.1_ttt_step10/all_classification_results.csv`
- 明细文件: `result_ttt/detail.csv`

## 主要结论
- `ttt_step10` 在两边都成功的 172 个数据集上，平均 accuracy 提升 +0.0402 pp：baseline 0.838128 -> ttt_step10 0.838531。
- 提升 / 下降 / 持平数量为 53 / 36 / 83；delta 中位数为 0.0000 pp。
- `ttt_step10` 有 6 个失败数据集，baseline 为 0 个失败；失败数据集是 accelerometer, customer_satisfaction_in_airline, Rain_in_Australia, dabetes_130-us_hospitals, Credit_c, SDSS17.
- 总耗时 ttt_step10 为 1763.958s，baseline 为 951.494s，约为 baseline 的 1.85x。
- 相比前面 step=1/ensemble 类结果，step10 的平均收益更高、提升数据集更多，但下降数据集也更多，说明 step 数增加后收益和过适配风险同时放大。

## 运行状态
| 方法 | processed | ok | fail | ok 数据集平均 accuracy | 总耗时秒 | 相对 baseline 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 178 | 178 | 0 | 0.837755 | 951.494 | 1.00x |
| ttt_step10 | 178 | 172 | 6 | 0.838531 | 1763.958 | 1.85x |

## Accuracy 对比
| 范围 | 数量 | baseline 平均 accuracy | ttt_step10 平均 accuracy | 平均 delta | delta 中位数 | 提升 | 下降 | 持平 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 两边都成功的数据集 | 172 | 0.838128 | 0.838531 | +0.0402 pp | 0.0000 pp | 53 | 36 | 83 |

## TTT 执行与跳过
| 范围 | 数量 | baseline 平均 accuracy | ttt_step10 平均 accuracy | 平均 delta | 提升 | 下降 | 持平 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 执行了 TTT | 160 | 0.843314 | 0.843746 | +0.0432 pp | 53 | 36 | 71 |
| 跳过了 TTT | 12 | 0.768991 | 0.768991 | 0.0000 pp | 0 | 0 | 12 |

## 按任务类型统计
| 任务类型 | 数量 | baseline 平均 accuracy | ttt_step10 平均 accuracy | 平均 delta | 提升 | 下降 | 持平 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| binclass | 97 | 0.855139 | 0.855355 | +0.0216 pp | 34 | 22 | 41 |
| multiclass | 75 | 0.816129 | 0.816772 | +0.0643 pp | 19 | 14 | 42 |

## 不同提升阈值下的数量
| 阈值 | ttt_step10 >= baseline + 阈值 | ttt_step10 <= baseline - 阈值 |
| --- | --- | --- |
| +0.1000 pp | 35 | 28 |
| +0.5000 pp | 13 | 5 |
| +1.0000 pp | 2 | 2 |
| +2.0000 pp | 0 | 0 |
| +5.0000 pp | 0 | 0 |

## 提升最大的数据集
| 数据集 | 任务 | baseline acc | ttt_step10 acc | delta |
| --- | --- | --- | --- | --- |
| mfeat-fourier | multiclass | 0.885000 | 0.895000 | +1.0000 pp |
| mfeat-zernike | multiclass | 0.852500 | 0.862500 | +1.0000 pp |
| satimage | multiclass | 0.928460 | 0.937792 | +0.9331 pp |
| car-evaluation | multiclass | 0.973988 | 0.982659 | +0.8671 pp |
| FOREX_audsgd-hour-High | binclass | 0.683856 | 0.692299 | +0.8443 pp |
| compass | binclass | 0.826675 | 0.834785 | +0.8111 pp |
| FOREX_audusd-hour-High | binclass | 0.681689 | 0.689561 | +0.7872 pp |
| GesturePhaseSegmentationProcessed | multiclass | 0.786329 | 0.793924 | +0.7595 pp |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass | 0.602500 | 0.610000 | +0.7500 pp |
| artificial-characters | multiclass | 0.863992 | 0.870841 | +0.6849 pp |

## 下降最大的数据集
| 数据集 | 任务 | baseline acc | ttt_step10 acc | delta |
| --- | --- | --- | --- | --- |
| FOREX_audchf-day-High | binclass | 0.754768 | 0.738420 | -1.6349 pp |
| statlog | binclass | 0.745000 | 0.735000 | -1.0000 pp |
| FOREX_cadjpy-day-High | binclass | 0.716621 | 0.708447 | -0.8174 pp |
| yeast | multiclass | 0.629630 | 0.622896 | -0.6734 pp |
| mfeat-morphological | multiclass | 0.760000 | 0.755000 | -0.5000 pp |
| Basketball_c | binclass | 0.712687 | 0.708955 | -0.3731 pp |
| cmc | multiclass | 0.589831 | 0.586441 | -0.3390 pp |
| contraceptive_method_choice | multiclass | 0.627119 | 0.623729 | -0.3390 pp |
| waveform_database_generator | multiclass | 0.356000 | 0.353000 | -0.3000 pp |
| FOREX_audcad-day-High | binclass | 0.743869 | 0.741144 | -0.2725 pp |

## ttt_step10 失败数据集
| 数据集 | baseline acc | 错误类型 | 错误首行 |
| --- | --- | --- | --- |
| Credit_c | 0.792000 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| Rain_in_Australia | 0.850371 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| SDSS17 | 0.976200 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| accelerometer | 0.741054 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| customer_satisfaction_in_airline | 0.961541 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |
| dabetes_130-us_hospitals | 0.641152 | RuntimeError | RuntimeError: CUDA error: invalid configuration argument |

## TTT 跳过原因
| 原因 | 数量 |
| --- | --- |
| stratified split \| TTT training skipped because n_classes=100 exceeds model max_classes=10 | 3 |
| stratified split \| TTT training skipped because n_classes=11 exceeds model max_classes=10 | 2 |
| stratified split \| TTT training skipped because n_classes=18 exceeds model max_classes=10 | 2 |
| stratified split \| TTT training skipped because n_classes=35 exceeds model max_classes=10 | 1 |
| stratified split \| TTT training skipped because n_classes=46 exceeds model max_classes=10 | 1 |
| stratified split \| TTT training skipped because n_classes=26 exceeds model max_classes=10 | 1 |
| TTT skipped for dataset=volkert to avoid OOM | 1 |
| stratified split \| TTT training skipped because n_classes=22 exceeds model max_classes=10 | 1 |

## 解读备注
- 公平 accuracy 对比只使用两边都成功的 172 个数据集；`avg_accuracy_ok` 不包含 6 个失败数据集。
- step10 的收益比 step1 更明显，但 top regressions 中已经出现超过 1pp 的下降，建议后续加 holdout gate 或 early stopping，而不是无条件固定 10 步。
- 失败集中在同一批大/复杂数据集，错误首行为 CUDA invalid configuration argument；这更像运行路径或 kernel 配置问题，不应和 accuracy 收益混在一个结论里。
