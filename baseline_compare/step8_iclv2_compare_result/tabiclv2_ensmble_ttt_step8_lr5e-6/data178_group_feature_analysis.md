# TTT step8 vs ICLv2 的 data178 分组特征分析

## 结论先行
- 成功交集 `170` 个数据集全部找到 `data178/<dataset>/info.json`；TTT step8 相对 ICLv2 平均提升 `+0.0554 pp`，总体正收益很小但比 LoRA step20 更明显。
- 三组分布为：`TTT step8更高` `64` 个、`ICLv2更高` `59` 个、`持平` `47` 个。
- TTT step8 的收益和风险都比 LoRA step20 更强：有少数明显提升，也有更明显下降，所以不能无保护地固定使用。
- 不能把下降数据集简单叫噪声；需要同时看 headroom、类别数、特征维度、数据规模和是否真的执行 TTT。

## 分组概览
| group | n | icl_avg | ttt_avg | mean_delta | median_delta | median_train | median_test | median_features | median_classes | ttt_applied_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTT step8更高 | 64 | 0.853398 | 0.859164 | +0.5766 pp | +0.2184 pp | 3919.5 | 1225.0 | 14.0 | 2.0 | 64 |
| ICLv2更高 | 59 | 0.822247 | 0.817588 | -0.4659 pp | -0.3080 pp | 3036.0 | 950.0 | 20.0 | 2.0 | 59 |
| 持平 | 47 | 0.872394 | 0.872394 | +0.0000 pp | +0.0000 pp | 1350.0 | 423.0 | 29.0 | 4.0 | 35 |

## 训练行数分桶
| train_size_bin | large_5k-20k | medium_1.5k-5k | small_<1.5k | very_large_>=20k |
| --- | --- | --- | --- | --- |
| TTT step8更高 | 16 | 24 | 13 | 11 |
| ICLv2更高 | 14 | 19 | 21 | 5 |
| 持平 | 8 | 12 | 25 | 2 |

## 特征列数分桶
| feature_count_bin | high_dim_51-100 | low_dim_<=10 | mid_dim_11-50 | very_high_dim_>100 |
| --- | --- | --- | --- | --- |
| TTT step8更高 | 5 | 26 | 32 | 1 |
| ICLv2更高 | 6 | 21 | 30 | 2 |
| 持平 | 9 | 7 | 23 | 8 |

## 数值/类别特征模态
| feature_modality | categorical_only | mixed_num_cat | numeric_only |
| --- | --- | --- | --- |
| TTT step8更高 | 5 | 19 | 40 |
| ICLv2更高 | 3 | 19 | 37 |
| 持平 | 9 | 14 | 24 |

## 类别数分桶
| class_count_bin | binary | class_gt10_ttt_skip_risk | multi_3-5 | multi_6-10 |
| --- | --- | --- | --- | --- |
| TTT step8更高 | 42 | 0 | 14 | 8 |
| ICLv2更高 | 39 | 0 | 10 | 10 |
| 持平 | 15 | 11 | 11 | 10 |

## 类别不平衡分桶
| imbalance_bin | balanced_<=1.5 | extreme_>10 | high_3-10 | mild_1.5-3 | missing |
| --- | --- | --- | --- | --- | --- |
| TTT step8更高 | 23 | 7 | 5 | 10 | 19 |
| ICLv2更高 | 17 | 5 | 9 | 7 | 21 |
| 持平 | 14 | 10 | 5 | 2 | 16 |

## 数据来源
| source_family | kaggle | openml | uci |
| --- | --- | --- | --- |
| TTT step8更高 | 6 | 48 | 10 |
| ICLv2更高 | 7 | 39 | 13 |
| 持平 | 5 | 33 | 9 |

## TTT 执行状态
| ttt_execution_group | TTT执行 | TTT未执行_OOM_or_skip | TTT未执行_class_count_gt10 |
| --- | --- | --- | --- |
| TTT step8更高 | 64 | 0 | 0 |
| ICLv2更高 | 59 | 0 | 0 |
| 持平 | 35 | 1 | 11 |

## TTT step8 更高的数据集特征
- 这组提升更集中在真正执行 TTT 的数据集，代表数据集包括 `jungle_chess_2pcs_raw_endgame_complete`、`compass`、`eye_movements`。
- 从行列规模看，收益覆盖大表和中等维度数据，也包括一些低基线但仍有可适配信号的数据集；更稳定的前提是 `n_classes <= 10` 且 TTT 实际执行。
- 与 LoRA step20 相比，step8 的最大收益更大，说明 full/ensemble TTT 的适配能力更强，但也带来更高风险。
| dataset | task | icl_acc | ttt_acc | delta | train/test | features | class_bin | modality | ttt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 0.892459 | 0.933066 | +4.0607 pp | 28684 / 8964 | 6 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| compass | binclass | 0.830580 | 0.861820 | +3.1241 pp | 10652 / 3329 | 8 num + 9 cat | binary | mixed_num_cat | TTT执行 |
| eye_movements | multiclass | 0.835466 | 0.865631 | +3.0165 pp | 6998 / 2188 | 24 num + 3 cat | multi_3-5 | mixed_num_cat | TTT执行 |
| mfeat-morphological | multiclass | 0.770000 | 0.795000 | +2.5000 pp | 1280 / 400 | 6 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| eye_movements_bin | binclass | 0.655059 | 0.678055 | +2.2996 pp | 4868 / 1522 | 20 num + 0 cat | binary | numeric_only | TTT执行 |
| Pima_Indians_Diabetes_Database | binclass | 0.727273 | 0.746753 | +1.9481 pp | 491 / 154 | 8 num + 0 cat | binary | numeric_only | TTT执行 |
| banknote_authentication | binclass | 0.538182 | 0.556364 | +1.8182 pp | 877 / 275 | 4 num + 0 cat | binary | numeric_only | TTT执行 |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass | 0.595000 | 0.610000 | +1.5000 pp | 1280 / 400 | 3 num + 4 cat | binary | mixed_num_cat | TTT执行 |
| hill-valley | binclass | 0.971193 | 0.983539 | +1.2346 pp | 775 / 243 | 100 num + 0 cat | binary | numeric_only | TTT执行 |
| kc1 | binclass | 0.872038 | 0.883886 | +1.1848 pp | 1349 / 422 | 21 num + 0 cat | binary | numeric_only | TTT执行 |
| FOREX_audcad-day-High | binclass | 0.746594 | 0.757493 | +1.0899 pp | 1173 / 367 | 10 num + 0 cat | binary | numeric_only | TTT执行 |
| mfeat-fourier | multiclass | 0.922500 | 0.932500 | +1.0000 pp | 1280 / 400 | 76 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| ada_prior | binclass | 0.855422 | 0.861993 | +0.6572 pp | 2919 / 913 | 6 num + 8 cat | binary | mixed_num_cat | TTT执行 |
| waveform-5000 | multiclass | 0.865000 | 0.871000 | +0.6000 pp | 3200 / 1000 | 40 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| Firm-Teacher_Clave-Direction_Classification | multiclass | 0.879167 | 0.884722 | +0.5556 pp | 6912 / 2160 | 0 num + 16 cat | multi_3-5 | categorical_only | TTT执行 |

## ICLv2 更高的数据集特征
- 下降组也大多执行了 TTT，说明不是单纯未执行造成；这些数据集更像是 step8 对当前任务边界做了不可靠移动。
- 代表性下降包括 `waveform_database_generator`、`steel_plates_faults`、`autoUniv-au4-2500`、`first-order-theorem-proving`。
- 对低基线/高难度、多类别边界复杂、时间/生成式结构明显或小样本高维数据集，应优先使用 protected selector 或 baseline fallback。
| dataset | task | icl_acc | ttt_acc | delta | train/test | features | class_bin | modality | ttt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| waveform_database_generator | multiclass | 0.354000 | 0.332000 | -2.2000 pp | 3199 / 1000 | 21 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| steel_plates_faults | multiclass | 0.848329 | 0.827763 | -2.0566 pp | 1241 / 389 | 25 num + 2 cat | multi_6-10 | mixed_num_cat | TTT执行 |
| autoUniv-au4-2500 | multiclass | 0.728000 | 0.712000 | -1.6000 pp | 1600 / 500 | 58 num + 42 cat | multi_3-5 | mixed_num_cat | TTT执行 |
| first-order-theorem-proving | multiclass | 0.651144 | 0.637255 | -1.3889 pp | 3915 / 1224 | 51 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| Diabetic_Retinopathy_Debrecen | binclass | 0.740260 | 0.727273 | -1.2987 pp | 736 / 231 | 16 num + 3 cat | binary | mixed_num_cat | TTT执行 |
| wine-quality-red | multiclass | 0.681250 | 0.668750 | -1.2500 pp | 1023 / 320 | 4 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| Gender_Gap_in_Spanish_WP | multiclass | 0.614737 | 0.603158 | -1.1579 pp | 3036 / 950 | 13 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| segment | multiclass | 0.954545 | 0.943723 | -1.0823 pp | 1478 / 462 | 17 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| statlog | binclass | 0.730000 | 0.720000 | -1.0000 pp | 640 / 200 | 7 num + 13 cat | binary | mixed_num_cat | TTT执行 |
| pc3 | binclass | 0.900958 | 0.891374 | -0.9585 pp | 1000 / 313 | 37 num + 0 cat | binary | numeric_only | TTT执行 |
| golf_play_dataset_extended | binclass | 0.945205 | 0.936073 | -0.9132 pp | 700 / 219 | 3 num + 6 cat | binary | mixed_num_cat | TTT执行 |
| contraceptive_method_choice | multiclass | 0.637288 | 0.630508 | -0.6780 pp | 942 / 295 | 5 num + 4 cat | multi_3-5 | mixed_num_cat | TTT执行 |
| Fitness_Club_c | binclass | 0.803333 | 0.796667 | -0.6667 pp | 960 / 300 | 3 num + 3 cat | binary | mixed_num_cat | TTT执行 |
| led24 | multiclass | 0.734375 | 0.728125 | -0.6250 pp | 2048 / 640 | 0 num + 24 cat | multi_6-10 | categorical_only | TTT执行 |
| FOREX_audchf-day-High | binclass | 0.749319 | 0.743869 | -0.5450 pp | 1172 / 367 | 10 num + 0 cat | binary | numeric_only | TTT执行 |

## 持平的数据集特征
- 持平组需要拆开看：一部分是 `TTT未执行`，例如类别数超过模型上限或 OOM/skip 时会直接沿用 ICL 推理；另一部分是 `TTT执行但预测未变`。
- 因此持平不是单一含义：它既可能表示 TTT 没机会更新，也可能表示 TTT 更新幅度不足以改变最终类别。
| dataset | task | acc | train/test | features | class_bin | modality | ttt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ibm-employee-performance | binclass | 1.000000 | 940 / 294 | 23 num + 7 cat | binary | mixed_num_cat | TTT执行 |
| mice_protein_expression | multiclass | 1.000000 | 691 / 216 | 75 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| national-longitudinal-survey-binary | binclass | 1.000000 | 3140 / 982 | 9 num + 7 cat | binary | mixed_num_cat | TTT执行 |
| thyroid | multiclass | 0.995833 | 4608 / 1440 | 21 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| analcatdata_authorship | multiclass | 0.994083 | 537 / 169 | 69 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| wall-robot-navigation | multiclass | 0.993590 | 3491 / 1092 | 24 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| Satellite | binclass | 0.993137 | 3264 / 1020 | 36 num + 0 cat | binary | numeric_only | TTT执行 |
| thyroid-ann | multiclass | 0.992053 | 2413 / 755 | 21 num + 0 cat | multi_3-5 | numeric_only | TTT执行 |
| dis | binclass | 0.990728 | 2413 / 755 | 6 num + 23 cat | binary | mixed_num_cat | TTT执行 |
| estimation_of_obesity_levels | multiclass | 0.990544 | 1350 / 423 | 8 num + 8 cat | multi_6-10 | mixed_num_cat | TTT执行 |
| Indian_pines | multiclass | 0.987972 | 5852 / 1829 | 220 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| car-evaluation | multiclass | 0.985549 | 1105 / 346 | 0 num + 21 cat | multi_3-5 | categorical_only | TTT执行 |
| mfeat-factors | multiclass | 0.982500 | 1280 / 400 | 216 num + 0 cat | multi_6-10 | numeric_only | TTT执行 |
| PhishingWebsites | binclass | 0.980552 | 7075 / 2211 | 0 num + 30 cat | binary | categorical_only | TTT执行 |
| allbp | multiclass | 0.977483 | 2413 / 755 | 6 num + 23 cat | multi_3-5 | mixed_num_cat | TTT执行 |

## 数据质量与一致性检查
- `info.json` 缺失：`0` / `170`。
- JSON 特征数与 CSV `n_features` 不一致：`0` / `170`。
- JSON 类别数与 CSV `n_classes` 不一致：`0` / `170`。
- 若存在不一致，明细中已用 `feature_count_mismatch` / `class_count_mismatch` 标出；分组统计优先使用 JSON 归一化字段。

## 结论和下一步
- TTT step8 的平均收益小，但强赢家比 LoRA step20 更明显；同时下降幅度也更明显。
- 当前结果更适合做“候选适配步”而不是直接替代 ICLv2：在验证集或 held-out split 上做 protected selector，只有 TTT 提升时才采用。
- 后续若要继续实验，优先比较 `step0/step8/step20/LoRA step20` 的 protected selection，而不是只看固定 step8 的均值。
