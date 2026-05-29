# iclv2_ensmble32 vs v2_ensmble_ttt_step8_lr5e-6 成功交集对比

## 输入与口径
- ICLv2 ensemble32: `1b_result_v2/iclv2_ensmble32/all_classification_results.csv`
- TTT step8: `1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/all_classification_results.csv`
- 对比口径：只保留两个文件中 `status=ok` 的共同数据集；本次共同成功交集为 `170` 个。
- 明细文件：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/detail.csv`

## 核心结论
- 在成功交集 `170` 个数据集上，TTT step8 平均 acc 为 `0.848392`，ICLv2 ensemble32 平均 acc 为 `0.847838`，TTT step8 平均变化 `+0.0554 pp`。
- 逐数据集胜负：TTT step8 更高 `64` 个，ICLv2 ensemble32 更高 `59` 个，持平 `47` 个。
- 按 test 样本数加权后，TTT step8 为 `0.835096`，ICLv2 ensemble32 为 `0.833459`，净变化约 `+602.0` 个预测。
- 交集总耗时：TTT step8 `25402.361s`，ICLv2 ensemble32 `1561.482s`，TTT 约为 `16.27x`。
- 被排除的是 TTT step8 的 `8` 个失败数据集；ICLv2 ensemble32 没有失败数据集。

## 原始运行概况
| method | rows | ok | fail | accuracy_nan | avg_accuracy_ok | wall_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| iclv2_ensmble32 | 178 | 178 | 0 | 0 | 0.847162 | 1010.495 |
| v2_ensmble_ttt_step8_lr5e-6 | 178 | 170 | 8 | 8 | 0.848392 | 25447.246 |

## 成功交集汇总
| common_ok | icl_avg | ttt_avg | mean_delta_pp | median_delta_pp | ttt_win | icl_win | tie | weighted_icl | weighted_ttt | net_correct_est |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 170 | 0.847838 | 0.848392 | +0.0554 | +0.0000 | 64 | 59 | 47 | 0.833459 | 0.835096 | +602.0 |

## TTT step8 提升最大的数据集
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

## TTT step8 下降最大的数据集
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

## 排除的数据集
- TTT step8 失败而未进入交集 `8` 个：`Cardiovascular-Disease-dataset, Credit_c, Rain_in_Australia, SDSS17, accelerometer, customer_satisfaction_in_airline, dabetes_130-us_hospitals, internet_firewall`。
- ICLv2 ensemble32 失败而未进入交集 `0` 个：`(none)`。
