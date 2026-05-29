# 1C ICL TTT epoch30 chunk vs ICLv2 ensemble32 成功交集分析

## 结论
- 当前成功交集为 178 个数据集；`1c_icl_ttt_epoch30_chunk` ok=178，`iclv2_ensmble32` ok=178。
- 全 178 交集：1C accuracy=0.851702，ICLv2=0.847162，delta=0.004541（0.4541 pp）；weighted delta=0.005899（0.5899 pp），delta_correct=3191.00。
- 胜/负/平：1C 提升 48，下降 30，持平 100。
- 其中 baseline-fill 行 13 个，全部来自 `iclv2_ensmble32` 补齐，基本应视为覆盖率修复而不是 TTT 收益；排除这些行后，真实非 fallback 子集 n=165，delta=0.004898（0.4898 pp），weighted delta=0.007971（0.7971 pp），胜/负/平=48/30/87。
- `ttt_applied=True` 行 153 个，delta=0.005283；`true_ttt_accepted`（非 fallback 且 best_epoch>0）103 个，delta=0.007848，说明有效收益主要来自验证集接受的更新。

## 主指标
| subset | n | iclv2 mean acc | 1c mean acc | delta | weighted delta | delta_correct | win/loss/tie |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_success_intersection | 178 | 0.847162 | 0.851702 | 0.004541 | 0.005899 | 3191.00 | 48/30/100 |
| non_baseline_fallback | 165 | 0.845005 | 0.849903 | 0.004898 | 0.007971 | 3191.00 | 48/30/87 |
| baseline_fallback_only | 13 | 0.874539 | 0.874539 | 0.000000 | 0.000000 | 0.00 | 0/0/13 |
| ttt_applied=True | 153 | 0.849169 | 0.854451 | 0.005283 | 0.009414 | 3191.00 | 48/30/75 |
| true_ttt_accepted | 103 | 0.844037 | 0.851885 | 0.007848 | 0.011661 | 3192.00 | 48/29/26 |
| applied_but_best_epoch0 | 50 | 0.859740 | 0.859737 | -0.000003 | -0.000015 | -1.00 | 0/1/49 |

## 提升/下降数据集规模特征
- 提升组：n=48, 样本数均值/中位数=18998.6/9928.5, 特征数均值/中位数=22.1/16.0
- 下降组：n=30, 样本数均值/中位数=10206.9/5075.0, 特征数均值/中位数=20.7/12.5
- 持平组：n=100, 样本数均值/中位数=14862.5/3772.0, 特征数均值/中位数=52.0/20.0

### 样本量分桶
| sample_size_bin | 提升 | 下降 | 持平 |
|---|---:|---:|---:|
| <=1k | 0 | 0 | 5 |
| 1k-5k | 19 | 15 | 55 |
| 5k-20k | 18 | 11 | 23 |
| 20k-100k | 10 | 4 | 13 |
| >100k | 1 | 0 | 4 |

### 特征数分桶
| feature_count_bin | 提升 | 下降 | 持平 |
|---|---:|---:|---:|
| <=10 | 18 | 14 | 24 |
| 11-30 | 19 | 12 | 42 |
| 31-100 | 11 | 4 | 23 |
| 101-300 | 0 | 0 | 10 |
| >300 | 0 | 0 | 1 |

## 提升最多的数据集
| dataset | iclv2_acc | 1c_acc | delta | rows | features | ttt_steps | best_epoch | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| eye_movements_bin | 0.655059 | 0.950723 | 0.295664 | 7608 | 20 | 30 | 30 | False |
| eye_movements | 0.835466 | 0.965722 | 0.130256 | 10936 | 27 | 30 | 30 | False |
| Click_prediction_small | 0.717897 | 0.833417 | 0.115519 | 39948 | 3 | 28 | 6 | False |
| jungle_chess_2pcs_raw_endgame_complete | 0.892459 | 0.980366 | 0.087907 | 44819 | 6 | 60 | 30 | False |
| compass | 0.830580 | 0.898768 | 0.068189 | 16644 | 17 | 30 | 30 | False |
| BLE_RSSI_dataset_for_Indoor_localization | 0.733600 | 0.776665 | 0.043065 | 9984 | 3 | 30 | 26 | False |
| artificial-characters | 0.940802 | 0.974070 | 0.033268 | 10218 | 7 | 30 | 30 | False |
| electricity | 0.926183 | 0.941741 | 0.015558 | 45312 | 8 | 60 | 30 | False |
| Basketball_c | 0.701493 | 0.716418 | 0.014925 | 1340 | 11 | 19 | 11 | False |
| GesturePhaseSegmentationProcessed | 0.827342 | 0.842025 | 0.014684 | 9873 | 32 | 25 | 17 | False |
| mfeat-zernike | 0.897500 | 0.907500 | 0.010000 | 2000 | 47 | 17 | 9 | False |
| Firm-Teacher_Clave-Direction_Classification | 0.879167 | 0.887037 | 0.007870 | 10800 | 16 | 23 | 15 | False |
| steel_plates_faults | 0.848329 | 0.856041 | 0.007712 | 1941 | 27 | 11 | 3 | False |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | 0.595000 | 0.602500 | 0.007500 | 2000 | 7 | 12 | 4 | False |
| kdd_ipums_la_97-small | 0.885356 | 0.890173 | 0.004817 | 5188 | 20 | 23 | 15 | False |

## 下降最多的数据集
| dataset | iclv2_acc | 1c_acc | delta | rows | features | ttt_steps | best_epoch | fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| FOREX_cadjpy-day-High | 0.727520 | 0.711172 | -0.016349 | 1834 | 10 | 24 | 16 | False |
| autoUniv-au4-2500 | 0.728000 | 0.714000 | -0.014000 | 2500 | 100 | 12 | 4 | False |
| contraceptive_method_choice | 0.637288 | 0.627119 | -0.010169 | 1473 | 9 | 16 | 8 | False |
| led24 | 0.734375 | 0.725000 | -0.009375 | 3200 | 24 | 26 | 18 | False |
| cmc | 0.603390 | 0.596610 | -0.006780 | 1473 | 9 | 13 | 5 | False |
| yeast | 0.636364 | 0.629630 | -0.006734 | 1484 | 8 | 27 | 19 | False |
| wine-quality-red | 0.681250 | 0.675000 | -0.006250 | 1599 | 4 | 9 | 1 | False |
| QSAR_biodegradation | 0.890995 | 0.886256 | -0.004739 | 1054 | 41 | 9 | 1 | False |
| Water_Quality_and_Potability | 0.657012 | 0.652439 | -0.004573 | 3276 | 8 | 14 | 6 | False |
| heloc | 0.733000 | 0.729000 | -0.004000 | 10000 | 22 | 17 | 9 | False |
| Fitness_Club_c | 0.803333 | 0.800000 | -0.003333 | 1500 | 6 | 10 | 2 | False |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 | 0.684375 | 0.681250 | -0.003125 | 1600 | 20 | 17 | 9 | False |
| FOREX_audjpy-day-High | 0.787466 | 0.784741 | -0.002725 | 1832 | 10 | 13 | 5 | False |
| FOREX_audcad-hour-High | 0.717627 | 0.715117 | -0.002510 | 43825 | 10 | 30 | 7 | False |
| PhishingWebsites | 0.980552 | 0.978290 | -0.002261 | 11055 | 30 | 21 | 13 | False |

## baseline-fill 行
- baseline-fill 数据集：customer_satisfaction_in_airline, gina_agnostic, BNG(cmc), INNHotelsGroup, Rain_in_Australia, dabetes_130-us_hospitals, madeline, Credit_c, Indian_pines, bank, default_of_credit_card_clients, gas-drift, naticusdroid+android+permissions+dataset

## 解释
- 这个结果说明 `1c_icl_ttt_epoch30_chunk` 相比 `iclv2_ensmble32` 是弱正收益：全量成功交集平均 +0.4541pp，weighted +0.5899pp。
- 但 100/178 个数据集持平，其中 13 个是 baseline-fill，另有一批是 TTT 跳过或 best_epoch=0 恢复原始状态；所以收益不是普遍发生，而是集中在少数被验证集接受的更新上。
- 从规模上看，提升组样本量均值/中位数高于下降组，说明 1C TTT 更像在中大样本数据集上有选择性收益，而不是由高特征数驱动。

明细文件：`1b_result_v2/1c_icl_ttt_epoch30_chunk/compare_1c_icl_ttt_epoch30_chunk_vs_iclv2_ensmble32_detail.csv`
