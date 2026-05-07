# ensmble_ttt_step9_lr5e-6 与 baseline 对比总结

## 输入文件
- baseline: `1b_result/v1.1_baseline/all_classification_results.csv`
- ensmble_ttt_step9_lr5e-6: `1b_result/ensmble_ttt_step9_lr5e-6/all_classification_results.csv`
- 明细文件: `1b_result/ensmble_ttt_step9_lr5e-6/detail.csv`

## 主要结论
- 失败的 6 个数据集已按 baseline fallback 处理，accuracy 替换为 `v1.1_baseline` 对应值，delta 记为 0。
- 在 178 个数据集上，平均 accuracy 提升 +0.3351 pp：baseline 0.837755 -> ensmble_ttt/fallback 0.841106。
- 提升 / 下降 / 持平数量为 85 / 48 / 45；delta 中位数为 +0.0000 pp。
- baseline fallback 数据集：Credit_c, Rain_in_Australia, SDSS17, accelerometer, customer_satisfaction_in_airline, dabetes_130-us_hospitals。

## 运行状态
| 方法 | processed | ok | fail | ok 数据集平均 accuracy | 总耗时秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 178 | 178 | 0 | 0.837755 | 951.494 |
| ensmble_ttt_step9_lr5e-6/fallback | 178 | 178 | 0 | 0.841106 | 31414.179 |

## Accuracy 对比
| 范围 | 数量 | baseline 平均 accuracy | ensmble_ttt/fallback 平均 accuracy | 平均 delta | delta 中位数 | 提升 | 下降 | 持平 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全部数据集，含 baseline fallback | 178 | 0.837755 | 0.841106 | +0.3351 pp | +0.0000 pp | 85 | 48 | 45 |
