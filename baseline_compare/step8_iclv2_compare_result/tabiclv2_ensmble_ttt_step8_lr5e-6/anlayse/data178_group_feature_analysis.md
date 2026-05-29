# data178 特征富集：TTT step8 vs ICLv2 ensemble32 分组分析

## 口径
- 输入明细：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/detail.csv`。这是上一轮两个结果文件的成功交集，共 `170` 个数据集。
- 数据属性来自 `data178/<dataset>/info.json`，已读取 `178` 个 info 文件，当前交集全部匹配。
- `domain_tag` 是基于 dataset name/source/task_intro 的启发式标签，只用于聚合观察，不作为精确定义。
- 分组：`TTT更高` 表示 `acc_ttt_step8 > acc_iclv2_ensmble32`；`ICLv2更高` 表示 step8 下降；`持平` 表示两者 acc 完全相等。
- 富集明细 CSV：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/data178_group_feature_detail.csv`。

## 先给结论
- 成功交集里，TTT step8 更高 `64` 个，ICLv2 ensemble32 更高 `59` 个，持平 `47` 个；平均变化 `+0.0554 pp`，中位数 `+0.0000 pp`，按 test 数估计净多对 `+602.0` 个样本。
- 真正的大收益集中在少数数据集：`strong` 提升 `5` 个，贡献主要来自 `jungle_chess_2pcs_raw_endgame_complete`、`compass`、`eye_movements`；大下降 `3` 个，主要是 `waveform_database_generator`、`first-order-theorem-proving`、`Gender_Gap_in_Spanish_WP` 等。
- 持平组不是一种单一现象：其中 `12` 个没有执行 TTT，主要因为类别数超过模型 `max_classes=10` 或主动 OOM skip；另外 `35` 个执行了 TTT 但最终预测不变，通常是 ceiling 高准确率或边界未被 step8 推动。
- 从 data178 特征看，TTT 胜负没有被单一来源或单一领域完全决定；更可靠的判断信号是：是否真的执行 TTT、ICLv2 原始 acc/headroom、类别数是否超过上限、test 样本数导致的离散波动、以及少数领域如 FOREX/规则合成数据的方向不稳定。

## 分组总体统计
| group | n | icl_avg | ttt_avg | mean_delta_pp | median_delta_pp | net_correct_est | median_train | median_test | median_features | median_classes | ttt_applied_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTT更高 | 64 | 0.853398 | 0.859164 | +0.5766 | +0.2184 | +844.0 | 4900 | 1225 | 14 | 2 | 100.0% |
| ICLv2更高 | 59 | 0.822247 | 0.817588 | -0.4659 | -0.3080 | -242.0 | 3796 | 950 | 20 | 2 | 100.0% |
| 持平 | 47 | 0.872394 | 0.872394 | +0.0000 | +0.0000 | +0.0 | 1688 | 423 | 29 | 4 | 74.5% |

### 强/中/弱变化
| bucket | n | mean_delta_pp | net_correct_est |
| --- | --- | --- | --- |
| TTT strong gain | 5 | +3.0002 | +579.0 |
| TTT medium gain | 17 | +0.8047 | +155.0 |
| TTT weak gain | 42 | +0.1958 | +110.0 |
| ICLv2 strong loss | 3 | -1.5823 | -50.0 |
| ICLv2 medium loss | 15 | -0.9078 | -88.0 |
| ICLv2 weak loss | 41 | -0.2225 | -104.0 |
| tie | 47 | +0.0000 | +0.0 |

## 分层特征

### 任务类型
| task_type | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binclass | 96 | 42 | 43.8% | 39 | 40.6% | 15 | 15.6% | +0.0847 |
| multiclass | 74 | 22 | 29.7% | 20 | 27.0% | 32 | 43.2% | +0.0174 |

解释：二分类和多分类在胜负数量上都接近均衡；多分类持平更多，主要被 `>10` 类的 TTT skip 拉高。

### ICLv2 原始 acc / headroom
| icl_acc_bin | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <=0.6 | 10 | 3 | 30.0% | 3 | 30.0% | 4 | 40.0% | +0.0614 |
| 0.6-0.7 | 16 | 4 | 25.0% | 7 | 43.8% | 5 | 31.2% | -0.1370 |
| 0.7-0.8 | 27 | 11 | 40.7% | 13 | 48.1% | 3 | 11.1% | -0.0047 |
| 0.8-0.9 | 37 | 17 | 45.9% | 15 | 40.5% | 5 | 13.5% | +0.2699 |
| 0.9-0.95 | 29 | 13 | 44.8% | 9 | 31.0% | 7 | 24.1% | -0.0063 |
| >0.95 | 51 | 16 | 31.4% | 12 | 23.5% | 23 | 45.1% | +0.0259 |

解释：TTT 更高的数据集并不只出现在低 acc 区间；`0.8-0.95` 的中高准确率区间更像有效适配区，既有结构信号又仍有可调整空间。`>0.95` 持平很多，是 ceiling 效应。

### 样本规模
| train_bin | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <=1k | 16 | 5 | 31.2% | 8 | 50.0% | 3 | 18.8% | -0.0585 |
| 1k-5k | 89 | 27 | 30.3% | 30 | 33.7% | 32 | 36.0% | -0.0482 |
| 5k-20k | 42 | 20 | 47.6% | 14 | 33.3% | 8 | 19.0% | +0.2365 |
| >20k | 23 | 12 | 52.2% | 7 | 30.4% | 4 | 17.4% | +0.2047 |

解释：`5k-20k` 和 `>20k` 的 TTT 胜率略高，说明足够规模的数据在 step8 上更容易给出可用适配信号；但大表仍可能失败或耗时高，失败集已被交集口径排除。小样本组的 pp 变化需要看 `delta_correct_est`，不能只看百分比。

### 特征数与类别数
| feature_bin | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <=10 | 54 | 26 | 48.1% | 21 | 38.9% | 7 | 13.0% | +0.1692 |
| 11-30 | 67 | 23 | 34.3% | 25 | 37.3% | 19 | 28.4% | +0.0124 |
| 31-80 | 33 | 12 | 36.4% | 9 | 27.3% | 12 | 36.4% | -0.0040 |
| >80 | 16 | 3 | 18.8% | 4 | 25.0% | 9 | 56.2% | -0.0264 |

| class_bin | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 96 | 42 | 43.8% | 39 | 40.6% | 15 | 15.6% | +0.0847 |
| 3-5 | 35 | 14 | 40.0% | 10 | 28.6% | 11 | 31.4% | +0.0978 |
| 6-10 | 28 | 8 | 28.6% | 10 | 35.7% | 10 | 35.7% | -0.0763 |
| >10 | 11 | 0 | 0.0% | 0 | 0.0% | 11 | 100.0% | +0.0000 |

解释：`>10` 类全部持平，不是因为 TTT 稳定，而是因为 TTT 被跳过。`<=10` 类里才是真正可比较的 TTT 行为；其中 `3-5` 类平均 delta 最正，`6-10` 更接近混合。高特征数 `>80` 持平多，往往是高 acc ceiling 或高类别/高维模式数据。

### 数值/类别特征形态
| feature_modality | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| numeric_only | 101 | 40 | 39.6% | 37 | 36.6% | 24 | 23.8% | +0.0876 |
| mixed | 52 | 19 | 36.5% | 19 | 36.5% | 14 | 26.9% | +0.0090 |
| categorical_only | 17 | 5 | 29.4% | 3 | 17.6% | 9 | 52.9% | +0.0060 |

解释：纯数值和混合特征的 TTT/ICLv2 胜负接近；纯类别特征持平偏多，说明离散组合类数据上 step8 未必总能改变最终决策。

### 不平衡度
| imbalance_bin | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <=1.2 | 48 | 18 | 37.5% | 17 | 35.4% | 13 | 27.1% | +0.1836 |
| 1.2-2 | 16 | 10 | 62.5% | 4 | 25.0% | 2 | 12.5% | +0.3216 |
| 2-5 | 12 | 6 | 50.0% | 5 | 41.7% | 1 | 8.3% | +0.0386 |
| >5 | 38 | 11 | 28.9% | 12 | 31.6% | 15 | 39.5% | -0.0296 |
| unknown | 56 | 19 | 33.9% | 21 | 37.5% | 16 | 28.6% | -0.0693 |

解释：`1.2-2` 的中等不平衡组 TTT 胜率高；`>5` 极不平衡组持平偏多，accuracy 对多数类边界敏感，不能仅按 acc 判定 TTT 机制。`unknown` 来自部分 info.json 缺少 imbalance_ratio。

### 来源与领域标签
| source_family | n | TTT更高_n | TTT更高_rate | ICLv2更高_n | ICLv2更高_rate | 持平_n | 持平_rate | mean_delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| openml | 120 | 48 | 40.0% | 39 | 32.5% | 33 | 27.5% | +0.1160 |
| uci | 32 | 10 | 31.2% | 13 | 40.6% | 9 | 28.1% | -0.1608 |
| kaggle | 18 | 6 | 33.3% | 7 | 38.9% | 5 | 27.8% | +0.0357 |

| domain | n_label_hits | TTT更高_n | ICLv2更高_n | 持平_n | mean_delta_pp |
| --- | --- | --- | --- | --- | --- |
| other | 54 | 24 | 17 | 13 | +0.0157 |
| vision/sensor/pattern | 33 | 12 | 8 | 13 | +0.2351 |
| business/customer/finance | 32 | 10 | 14 | 8 | +0.0479 |
| scientific/health/environment | 26 | 9 | 9 | 8 | +0.0099 |
| text/sequence/symbolic | 25 | 10 | 8 | 7 | +0.0945 |
| synthetic/rule-generated | 18 | 7 | 7 | 4 | +0.0279 |
| financial/FOREX | 9 | 4 | 5 | 0 | -0.0013 |

解释：OpenML 占样本主体，TTT 胜负都多，不能把来源本身当因果解释。领域上，规则/合成、视觉/传感器、科学/健康都有正负两面；FOREX 数量少但方向混合，提示时间粒度类任务需要单独规则。

## 代表性数据集

### TTT step8 提升贡献最大
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jungle_chess_2pcs_raw_endgame_complete | TTT更高 | 0.892459 | 0.933066 | +4.0607 | +364.0 | 35855 | 8964 | 6 | 3 | numeric_only | 5.320 | synthetic/rule-generated | yes |
| compass | TTT更高 | 0.830580 | 0.861820 | +3.1241 | +104.0 | 13315 | 3329 | 17 | 2 | mixed | 1.000 | business/customer/finance | yes |
| eye_movements | TTT更高 | 0.835466 | 0.865631 | +3.0165 | +66.0 | 8748 | 2188 | 27 | 3 | mixed | 1.485 | text/sequence/symbolic; vision/sensor/pattern | yes |
| eye_movements_bin | TTT更高 | 0.655059 | 0.678055 | +2.2996 | +35.0 | 6086 | 1522 | 20 | 2 | numeric_only | 1.000 | text/sequence/symbolic; vision/sensor/pattern | yes |
| electricity | TTT更高 | 0.926183 | 0.929383 | +0.3200 | +29.0 | 36249 | 9063 | 8 | 2 | mixed | 1.355 | scientific/health/environment; synthetic/rule-generated | yes |
| bank | TTT更高 | 0.908990 | 0.911202 | +0.2212 | +20.0 | 36168 | 9043 | 16 | 2 | mixed | NA | business/customer/finance | yes |
| FOREX_audusd-hour-High | TTT更高 | 0.707017 | 0.709184 | +0.2168 | +19.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.126 | financial/FOREX | yes |
| FOREX_audsgd-hour-High | TTT更高 | 0.710553 | 0.711922 | +0.1369 | +12.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.061 | financial/FOREX | yes |
| Firm-Teacher_Clave-Direction_Classification | TTT更高 | 0.879167 | 0.884722 | +0.5556 | +12.0 | 8640 | 2160 | 16 | 4 | categorical_only | NA | other | yes |
| BNG(tic-tac-toe) | TTT更高 | 0.814834 | 0.816358 | +0.1524 | +12.0 | 31492 | 7874 | 9 | 2 | categorical_only | 1.881 | other | yes |
| mfeat-morphological | TTT更高 | 0.770000 | 0.795000 | +2.5000 | +10.0 | 1600 | 400 | 6 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| GesturePhaseSegmentationProcessed | TTT更高 | 0.827342 | 0.830886 | +0.3544 | +7.0 | 7898 | 1975 | 32 | 5 | numeric_only | 2.956 | vision/sensor/pattern; synthetic/rule-generated | yes |
| okcupid_stem | TTT更高 | 0.750000 | 0.751312 | +0.1312 | +7.0 | 21341 | 5336 | 13 | 3 | mixed | 6.826 | text/sequence/symbolic | yes |
| satimage | TTT更高 | 0.941680 | 0.947123 | +0.5443 | +7.0 | 5144 | 1286 | 36 | 6 | numeric_only | 2.450 | text/sequence/symbolic; vision/sensor/pattern | yes |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | TTT更高 | 0.595000 | 0.610000 | +1.5000 | +6.0 | 1600 | 400 | 7 | 2 | mixed | 1.245 | other | yes |
| waveform-5000 | TTT更高 | 0.865000 | 0.871000 | +0.6000 | +6.0 | 4000 | 1000 | 40 | 3 | numeric_only | 1.024 | synthetic/rule-generated | yes |
| ada_prior | TTT更高 | 0.855422 | 0.861993 | +0.6572 | +6.0 | 3649 | 913 | 14 | 2 | mixed | 3.030 | business/customer/finance | yes |
| MagicTelescope | TTT更高 | 0.894059 | 0.895636 | +0.1577 | +6.0 | 15216 | 3804 | 9 | 2 | numeric_only | 1.844 | vision/sensor/pattern | yes |
| eeg-eye-state | TTT更高 | 0.993992 | 0.995995 | +0.2003 | +6.0 | 11984 | 2996 | 14 | 2 | numeric_only | 1.228 | vision/sensor/pattern | yes |
| INNHotelsGroup | TTT更高 | 0.904618 | 0.905445 | +0.0827 | +6.0 | 29020 | 7255 | 17 | 2 | mixed | NA | other | yes |

### ICLv2 ensemble32 优势最大 / step8 风险样本
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FOREX_audjpy-hour-High | ICLv2更高 | 0.722076 | 0.718996 | -0.3080 | -27.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.062 | financial/FOREX | yes |
| waveform_database_generator | ICLv2更高 | 0.354000 | 0.332000 | -2.2000 | -22.0 | 3999 | 1000 | 21 | 3 | numeric_only | NA | synthetic/rule-generated | yes |
| first-order-theorem-proving | ICLv2更高 | 0.651144 | 0.637255 | -1.3889 | -17.0 | 4894 | 1224 | 51 | 6 | numeric_only | 5.255 | text/sequence/symbolic | yes |
| BNG(cmc) | ICLv2更高 | 0.587432 | 0.586076 | -0.1356 | -15.0 | 44236 | 11060 | 9 | 3 | mixed | 1.893 | other | yes |
| Gender_Gap_in_Spanish_WP | ICLv2更高 | 0.614737 | 0.603158 | -1.1579 | -11.0 | 3796 | 950 | 13 | 3 | numeric_only | NA | other | yes |
| FOREX_audcad-hour-High | ICLv2更高 | 0.717627 | 0.716486 | -0.1141 | -10.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.061 | financial/FOREX | yes |
| autoUniv-au4-2500 | ICLv2更高 | 0.728000 | 0.712000 | -1.6000 | -8.0 | 2000 | 500 | 100 | 3 | mixed | 5.985 | text/sequence/symbolic; synthetic/rule-generated | yes |
| heloc | ICLv2更高 | 0.733000 | 0.729000 | -0.4000 | -8.0 | 8000 | 2000 | 22 | 2 | numeric_only | 1.000 | business/customer/finance | yes |
| steel_plates_faults | ICLv2更高 | 0.848329 | 0.827763 | -2.0566 | -8.0 | 1552 | 389 | 27 | 7 | mixed | NA | other | yes |
| Click_prediction_small | ICLv2更高 | 0.717897 | 0.716896 | -0.1001 | -8.0 | 31958 | 7990 | 3 | 2 | numeric_only | NA | business/customer/finance | yes |
| California-Housing-Classification | ICLv2更高 | 0.921754 | 0.920058 | -0.1696 | -7.0 | 16512 | 4128 | 8 | 2 | numeric_only | 1.001 | business/customer/finance | yes |
| segment | ICLv2更高 | 0.954545 | 0.943723 | -1.0823 | -5.0 | 1848 | 462 | 17 | 7 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| phoneme | ICLv2更高 | 0.903793 | 0.899167 | -0.4625 | -5.0 | 4323 | 1081 | 5 | 2 | numeric_only | 2.407 | business/customer/finance; scientific/health/environment; synthetic/rule-generated | yes |
| htru | ICLv2更高 | 0.980726 | 0.979609 | -0.1117 | -4.0 | 14318 | 3580 | 8 | 2 | numeric_only | NA | other | yes |
| Employee | ICLv2更高 | 0.860365 | 0.856069 | -0.4296 | -4.0 | 3722 | 931 | 8 | 2 | mixed | NA | business/customer/finance | yes |
| wine-quality-white | ICLv2更高 | 0.689796 | 0.685714 | -0.4082 | -4.0 | 3918 | 980 | 11 | 7 | numeric_only | 439.600 | scientific/health/environment | yes |
| wine-quality-red | ICLv2更高 | 0.681250 | 0.668750 | -1.2500 | -4.0 | 1279 | 320 | 4 | 6 | numeric_only | 68.100 | scientific/health/environment | yes |
| led24 | ICLv2更高 | 0.734375 | 0.728125 | -0.6250 | -4.0 | 2560 | 640 | 24 | 10 | categorical_only | 1.139 | other | yes |
| telco-customer-churn | ICLv2更高 | 0.805536 | 0.802697 | -0.2839 | -4.0 | 5634 | 1409 | 18 | 2 | mixed | 2.768 | business/customer/finance | yes |
| ringnorm | ICLv2更高 | 0.979730 | 0.977027 | -0.2703 | -4.0 | 5920 | 1480 | 20 | 2 | numeric_only | 1.020 | synthetic/rule-generated | yes |

### 持平组中的 TTT skip 样本
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one-hundred-plants-margin | 持平 | 0.903125 | 0.903125 | +0.0000 | +0.0 | 1280 | 320 | 64 | 100 | numeric_only | 1.000 | vision/sensor/pattern | no |
| one-hundred-plants-shape | 持平 | 0.840625 | 0.840625 | +0.0000 | +0.0 | 1280 | 320 | 64 | 100 | numeric_only | 1.000 | vision/sensor/pattern | no |
| one-hundred-plants-texture | 持平 | 0.928125 | 0.928125 | +0.0000 | +0.0 | 1279 | 320 | 64 | 100 | numeric_only | 1.067 | vision/sensor/pattern | no |
| internet_usage | 持平 | 0.549951 | 0.549951 | +0.0000 | +0.0 | 8086 | 2022 | 70 | 46 | categorical_only | NA | business/customer/finance | no |
| UJI_Pen_Characters | 持平 | 0.538462 | 0.538462 | +0.0000 | +0.0 | 1091 | 273 | 80 | 35 | numeric_only | NA | vision/sensor/pattern | no |
| letter | 持平 | 0.990000 | 0.990000 | +0.0000 | +0.0 | 16000 | 4000 | 15 | 26 | numeric_only | 1.108 | text/sequence/symbolic; vision/sensor/pattern | no |
| walking-activity | 持平 | 0.675093 | 0.675093 | +0.0000 | +0.0 | 119465 | 29867 | 4 | 22 | numeric_only | NA | other | no |
| kr-vs-k | 持平 | 0.919636 | 0.919636 | +0.0000 | +0.0 | 22444 | 5612 | 6 | 18 | mixed | 168.630 | synthetic/rule-generated | no |
| kropt | 持平 | 0.916429 | 0.916429 | +0.0000 | +0.0 | 22444 | 5612 | 6 | 18 | categorical_only | 168.630 | synthetic/rule-generated | no |
| ASP-POTASSCO-classification | 持平 | 0.490347 | 0.490347 | +0.0000 | +0.0 | 1035 | 259 | 141 | 11 | mixed | 12.286 | other | no |
| texture | 持平 | 1.000000 | 1.000000 | +0.0000 | +0.0 | 4400 | 1100 | 40 | 11 | numeric_only | 1.000 | vision/sensor/pattern | no |
| volkert | 持平 | 0.751158 | 0.751158 | +0.0000 | +0.0 | 46648 | 11662 | 180 | 10 | numeric_only | 9.409 | business/customer/finance; scientific/health/environment; text/sequence/symbolic | no |

## 全量分组索引

### TTT更高 64 个
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jungle_chess_2pcs_raw_endgame_complete | TTT更高 | 0.892459 | 0.933066 | +4.0607 | +364.0 | 35855 | 8964 | 6 | 3 | numeric_only | 5.320 | synthetic/rule-generated | yes |
| compass | TTT更高 | 0.830580 | 0.861820 | +3.1241 | +104.0 | 13315 | 3329 | 17 | 2 | mixed | 1.000 | business/customer/finance | yes |
| eye_movements | TTT更高 | 0.835466 | 0.865631 | +3.0165 | +66.0 | 8748 | 2188 | 27 | 3 | mixed | 1.485 | text/sequence/symbolic; vision/sensor/pattern | yes |
| mfeat-morphological | TTT更高 | 0.770000 | 0.795000 | +2.5000 | +10.0 | 1600 | 400 | 6 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| eye_movements_bin | TTT更高 | 0.655059 | 0.678055 | +2.2996 | +35.0 | 6086 | 1522 | 20 | 2 | numeric_only | 1.000 | text/sequence/symbolic; vision/sensor/pattern | yes |
| Pima_Indians_Diabetes_Database | TTT更高 | 0.727273 | 0.746753 | +1.9481 | +3.0 | 614 | 154 | 8 | 2 | numeric_only | NA | scientific/health/environment | yes |
| banknote_authentication | TTT更高 | 0.538182 | 0.556364 | +1.8182 | +5.0 | 1097 | 275 | 4 | 2 | numeric_only | NA | other | yes |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | TTT更高 | 0.595000 | 0.610000 | +1.5000 | +6.0 | 1600 | 400 | 7 | 2 | mixed | 1.245 | other | yes |
| hill-valley | TTT更高 | 0.971193 | 0.983539 | +1.2346 | +3.0 | 969 | 243 | 100 | 2 | numeric_only | 1.000 | other | yes |
| kc1 | TTT更高 | 0.872038 | 0.883886 | +1.1848 | +5.0 | 1687 | 422 | 21 | 2 | numeric_only | 5.469 | other | yes |
| FOREX_audcad-day-High | TTT更高 | 0.746594 | 0.757493 | +1.0899 | +4.0 | 1467 | 367 | 10 | 2 | numeric_only | 1.031 | financial/FOREX | yes |
| mfeat-fourier | TTT更高 | 0.922500 | 0.932500 | +1.0000 | +4.0 | 1600 | 400 | 76 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| ada_prior | TTT更高 | 0.855422 | 0.861993 | +0.6572 | +6.0 | 3649 | 913 | 14 | 2 | mixed | 3.030 | business/customer/finance | yes |
| waveform-5000 | TTT更高 | 0.865000 | 0.871000 | +0.6000 | +6.0 | 4000 | 1000 | 40 | 3 | numeric_only | 1.024 | synthetic/rule-generated | yes |
| Firm-Teacher_Clave-Direction_Classification | TTT更高 | 0.879167 | 0.884722 | +0.5556 | +12.0 | 8640 | 2160 | 16 | 4 | categorical_only | NA | other | yes |
| satimage | TTT更高 | 0.941680 | 0.947123 | +0.5443 | +7.0 | 5144 | 1286 | 36 | 6 | numeric_only | 2.450 | text/sequence/symbolic; vision/sensor/pattern | yes |
| mfeat-zernike | TTT更高 | 0.897500 | 0.902500 | +0.5000 | +2.0 | 1600 | 400 | 47 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| maternal_health_risk | TTT更高 | 0.871921 | 0.876847 | +0.4926 | +1.0 | 811 | 203 | 6 | 3 | numeric_only | NA | scientific/health/environment | yes |
| abalone | TTT更高 | 0.639952 | 0.644737 | +0.4785 | +4.0 | 3341 | 836 | 8 | 3 | mixed | 1.094 | other | yes |
| QSAR_biodegradation | TTT更高 | 0.890995 | 0.895735 | +0.4739 | +1.0 | 843 | 211 | 41 | 2 | numeric_only | NA | scientific/health/environment | yes |
| qsar | TTT更高 | 0.900474 | 0.905213 | +0.4739 | +1.0 | 844 | 211 | 40 | 2 | mixed | 1.963 | scientific/health/environment | yes |
| Telecom_Churn_Dataset | TTT更高 | 0.961019 | 0.965517 | +0.4498 | +3.0 | 2666 | 667 | 17 | 2 | mixed | NA | business/customer/finance | yes |
| Pumpkin_Seeds | TTT更高 | 0.882000 | 0.886000 | +0.4000 | +2.0 | 2000 | 500 | 12 | 2 | numeric_only | NA | other | yes |
| baseball | TTT更高 | 0.944030 | 0.947761 | +0.3731 | +1.0 | 1072 | 268 | 16 | 3 | mixed | 21.316 | other | yes |
| GesturePhaseSegmentationProcessed | TTT更高 | 0.827342 | 0.830886 | +0.3544 | +7.0 | 7898 | 1975 | 32 | 5 | numeric_only | 2.956 | vision/sensor/pattern; synthetic/rule-generated | yes |
| ada_agnostic | TTT更高 | 0.851041 | 0.854326 | +0.3286 | +3.0 | 3649 | 913 | 48 | 2 | numeric_only | NA | other | yes |
| electricity | TTT更高 | 0.926183 | 0.929383 | +0.3200 | +29.0 | 36249 | 9063 | 8 | 2 | mixed | 1.355 | scientific/health/environment; synthetic/rule-generated | yes |
| Water_Quality_and_Potability | TTT更高 | 0.657012 | 0.660061 | +0.3049 | +2.0 | 2620 | 656 | 8 | 2 | numeric_only | NA | scientific/health/environment | yes |
| gina_agnostic | TTT更高 | 0.974063 | 0.976945 | +0.2882 | +2.0 | 2774 | 694 | 970 | 2 | numeric_only | NA | other | yes |
| allrep | TTT更高 | 0.984106 | 0.986755 | +0.2649 | +2.0 | 3017 | 755 | 29 | 4 | mixed | 107.294 | other | yes |
| bank | TTT更高 | 0.908990 | 0.911202 | +0.2212 | +20.0 | 36168 | 9043 | 16 | 2 | mixed | NA | business/customer/finance | yes |
| taiwanese_bankruptcy_prediction | TTT更高 | 0.971408 | 0.973607 | +0.2199 | +3.0 | 5455 | 1364 | 95 | 2 | numeric_only | NA | business/customer/finance | yes |
| FOREX_audusd-hour-High | TTT更高 | 0.707017 | 0.709184 | +0.2168 | +19.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.126 | financial/FOREX | yes |
| twonorm | TTT更高 | 0.977703 | 0.979730 | +0.2027 | +3.0 | 5920 | 1480 | 20 | 2 | numeric_only | 1.002 | synthetic/rule-generated | yes |
| eeg-eye-state | TTT更高 | 0.993992 | 0.995995 | +0.2003 | +6.0 | 11984 | 2996 | 14 | 2 | numeric_only | 1.228 | vision/sensor/pattern | yes |
| ozone-level-8hr | TTT更高 | 0.954635 | 0.956607 | +0.1972 | +1.0 | 2027 | 507 | 72 | 2 | numeric_only | 14.838 | scientific/health/environment | yes |
| in_vehicle_coupon_recommendation | TTT更高 | 0.787150 | 0.789121 | +0.1971 | +5.0 | 10147 | 2537 | 21 | 2 | categorical_only | NA | business/customer/finance; vision/sensor/pattern | yes |
| artificial-characters | TTT更高 | 0.940802 | 0.942759 | +0.1957 | +4.0 | 8174 | 2044 | 7 | 10 | numeric_only | 2.360 | text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated | yes |
| sylvine | TTT更高 | 0.977561 | 0.979512 | +0.1951 | +2.0 | 4099 | 1025 | 20 | 2 | numeric_only | 1.000 | business/customer/finance; scientific/health/environment; text/sequence/symbolic | yes |
| online_shoppers | TTT更高 | 0.905921 | 0.907543 | +0.1622 | +4.0 | 9864 | 2466 | 14 | 2 | mixed | 5.462 | other | yes |
| MagicTelescope | TTT更高 | 0.894059 | 0.895636 | +0.1577 | +6.0 | 15216 | 3804 | 9 | 2 | numeric_only | 1.844 | vision/sensor/pattern | yes |
| led7 | TTT更高 | 0.739062 | 0.740625 | +0.1563 | +1.0 | 2560 | 640 | 7 | 10 | categorical_only | 1.263 | other | yes |
| BNG(tic-tac-toe) | TTT更高 | 0.814834 | 0.816358 | +0.1524 | +12.0 | 31492 | 7874 | 9 | 2 | categorical_only | 1.881 | other | yes |
| BLE_RSSI_dataset_for_Indoor_localization | TTT更高 | 0.733600 | 0.735103 | +0.1502 | +3.0 | 7987 | 1997 | 3 | 3 | numeric_only | NA | other | yes |
| FOREX_audsgd-hour-High | TTT更高 | 0.710553 | 0.711922 | +0.1369 | +12.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.061 | financial/FOREX | yes |
| rice_cammeo_and_osmancik | TTT更高 | 0.931759 | 0.933071 | +0.1312 | +1.0 | 3048 | 762 | 7 | 2 | numeric_only | NA | scientific/health/environment | yes |
| okcupid_stem | TTT更高 | 0.750000 | 0.751312 | +0.1312 | +7.0 | 21341 | 5336 | 13 | 3 | mixed | 6.826 | text/sequence/symbolic | yes |
| mozilla4 | TTT更高 | 0.944033 | 0.945320 | +0.1287 | +4.0 | 12436 | 3109 | 4 | 2 | numeric_only | 2.043 | text/sequence/symbolic | yes |
| microaggregation2 | TTT更高 | 0.644250 | 0.645500 | +0.1250 | +5.0 | 16000 | 4000 | 20 | 5 | numeric_only | 15.023 | other | yes |
| spambase | TTT更高 | 0.958740 | 0.959826 | +0.1086 | +1.0 | 3680 | 921 | 57 | 2 | numeric_only | 1.538 | text/sequence/symbolic; vision/sensor/pattern | yes |
| Wilt | TTT更高 | 0.992746 | 0.993782 | +0.1036 | +1.0 | 3856 | 965 | 5 | 2 | numeric_only | NA | other | yes |
| rl | TTT更高 | 0.905433 | 0.906439 | +0.1006 | +1.0 | 3976 | 994 | 12 | 2 | mixed | 1.000 | other | yes |
| KDDCup09_upselling | TTT更高 | 0.810916 | 0.811891 | +0.0975 | +1.0 | 4102 | 1026 | 49 | 2 | mixed | 1.000 | business/customer/finance | yes |
| page-blocks | TTT更高 | 0.977169 | 0.978082 | +0.0913 | +1.0 | 4378 | 1095 | 10 | 5 | numeric_only | 175.464 | text/sequence/symbolic; synthetic/rule-generated | yes |
| turiye_student_evaluation | TTT更高 | 0.519759 | 0.520619 | +0.0859 | +1.0 | 4656 | 1164 | 32 | 5 | mixed | NA | other | yes |
| INNHotelsGroup | TTT更高 | 0.904618 | 0.905445 | +0.0827 | +6.0 | 29020 | 7255 | 17 | 2 | mixed | NA | other | yes |
| delta_ailerons | TTT更高 | 0.953015 | 0.953717 | +0.0701 | +1.0 | 5703 | 1426 | 5 | 2 | numeric_only | 1.131 | other | yes |
| FOREX_cadjpy-hour-High | TTT更高 | 0.715916 | 0.716486 | +0.0570 | +5.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.074 | financial/FOREX | yes |
| FICO-HELOC-cleaned | TTT更高 | 0.754937 | 0.755443 | +0.0506 | +1.0 | 7896 | 1975 | 23 | 2 | mixed | 1.085 | text/sequence/symbolic | yes |
| JapaneseVowels | TTT更高 | 0.999498 | 1.000000 | +0.0502 | +1.0 | 7968 | 1993 | 14 | 9 | numeric_only | 2.064 | other | yes |
| Amazon_employee_access | TTT更高 | 0.944156 | 0.944461 | +0.0305 | +2.0 | 26215 | 6554 | 7 | 2 | categorical_only | 16.274 | business/customer/finance | yes |
| HR_Analytics_Job_Change_of_Data_Scientists | TTT更高 | 0.800626 | 0.800887 | +0.0261 | +1.0 | 15326 | 3832 | 13 | 2 | mixed | NA | business/customer/finance | yes |
| BNG(breast-w) | TTT更高 | 0.988062 | 0.988316 | +0.0254 | +2.0 | 31492 | 7874 | 9 | 2 | numeric_only | 1.906 | other | yes |
| shuttle | TTT更高 | 0.999310 | 0.999397 | +0.0086 | +1.0 | 46400 | 11600 | 9 | 7 | numeric_only | 4558.600 | other | yes |

### ICLv2更高 59 个
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| waveform_database_generator | ICLv2更高 | 0.354000 | 0.332000 | -2.2000 | -22.0 | 3999 | 1000 | 21 | 3 | numeric_only | NA | synthetic/rule-generated | yes |
| steel_plates_faults | ICLv2更高 | 0.848329 | 0.827763 | -2.0566 | -8.0 | 1552 | 389 | 27 | 7 | mixed | NA | other | yes |
| autoUniv-au4-2500 | ICLv2更高 | 0.728000 | 0.712000 | -1.6000 | -8.0 | 2000 | 500 | 100 | 3 | mixed | 5.985 | text/sequence/symbolic; synthetic/rule-generated | yes |
| first-order-theorem-proving | ICLv2更高 | 0.651144 | 0.637255 | -1.3889 | -17.0 | 4894 | 1224 | 51 | 6 | numeric_only | 5.255 | text/sequence/symbolic | yes |
| Diabetic_Retinopathy_Debrecen | ICLv2更高 | 0.740260 | 0.727273 | -1.2987 | -3.0 | 920 | 231 | 19 | 2 | mixed | NA | scientific/health/environment | yes |
| wine-quality-red | ICLv2更高 | 0.681250 | 0.668750 | -1.2500 | -4.0 | 1279 | 320 | 4 | 6 | numeric_only | 68.100 | scientific/health/environment | yes |
| Gender_Gap_in_Spanish_WP | ICLv2更高 | 0.614737 | 0.603158 | -1.1579 | -11.0 | 3796 | 950 | 13 | 3 | numeric_only | NA | other | yes |
| segment | ICLv2更高 | 0.954545 | 0.943723 | -1.0823 | -5.0 | 1848 | 462 | 17 | 7 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| statlog | ICLv2更高 | 0.730000 | 0.720000 | -1.0000 | -2.0 | 800 | 200 | 20 | 2 | mixed | NA | business/customer/finance | yes |
| pc3 | ICLv2更高 | 0.900958 | 0.891374 | -0.9585 | -3.0 | 1250 | 313 | 37 | 2 | numeric_only | 8.769 | vision/sensor/pattern | yes |
| golf_play_dataset_extended | ICLv2更高 | 0.945205 | 0.936073 | -0.9132 | -2.0 | 876 | 219 | 9 | 2 | mixed | NA | other | yes |
| contraceptive_method_choice | ICLv2更高 | 0.637288 | 0.630508 | -0.6780 | -2.0 | 1178 | 295 | 9 | 3 | mixed | NA | other | yes |
| Fitness_Club_c | ICLv2更高 | 0.803333 | 0.796667 | -0.6667 | -2.0 | 1200 | 300 | 6 | 2 | mixed | NA | other | yes |
| led24 | ICLv2更高 | 0.734375 | 0.728125 | -0.6250 | -4.0 | 2560 | 640 | 24 | 10 | categorical_only | 1.139 | other | yes |
| FOREX_audchf-day-High | ICLv2更高 | 0.749319 | 0.743869 | -0.5450 | -2.0 | 1466 | 367 | 10 | 2 | numeric_only | 1.028 | financial/FOREX | yes |
| sports_articles_for_objectivity_analysis | ICLv2更高 | 0.845000 | 0.840000 | -0.5000 | -1.0 | 800 | 200 | 59 | 2 | mixed | NA | text/sequence/symbolic | yes |
| PizzaCutter3 | ICLv2更高 | 0.885167 | 0.880383 | -0.4785 | -1.0 | 834 | 209 | 37 | 2 | numeric_only | 7.213 | other | yes |
| madeline | ICLv2更高 | 0.923567 | 0.918790 | -0.4777 | -3.0 | 2512 | 628 | 259 | 2 | numeric_only | 1.012 | other | yes |
| PieChart3 | ICLv2更高 | 0.884259 | 0.879630 | -0.4630 | -1.0 | 861 | 216 | 37 | 2 | numeric_only | 7.037 | other | yes |
| phoneme | ICLv2更高 | 0.903793 | 0.899167 | -0.4625 | -5.0 | 4323 | 1081 | 5 | 2 | numeric_only | 2.407 | business/customer/finance; scientific/health/environment; synthetic/rule-generated | yes |
| autoUniv-au7-1100 | ICLv2更高 | 0.409091 | 0.404545 | -0.4545 | -1.0 | 880 | 220 | 12 | 5 | mixed | 1.993 | text/sequence/symbolic; synthetic/rule-generated | yes |
| pc1 | ICLv2更高 | 0.945946 | 0.941441 | -0.4505 | -1.0 | 887 | 222 | 21 | 2 | numeric_only | 13.403 | vision/sensor/pattern | yes |
| Employee | ICLv2更高 | 0.860365 | 0.856069 | -0.4296 | -4.0 | 3722 | 931 | 8 | 2 | mixed | NA | business/customer/finance | yes |
| wine-quality-white | ICLv2更高 | 0.689796 | 0.685714 | -0.4082 | -4.0 | 3918 | 980 | 11 | 7 | numeric_only | 439.600 | scientific/health/environment | yes |
| heloc | ICLv2更高 | 0.733000 | 0.729000 | -0.4000 | -8.0 | 8000 | 2000 | 22 | 2 | numeric_only | 1.000 | business/customer/finance | yes |
| ada | ICLv2更高 | 0.862651 | 0.859036 | -0.3614 | -3.0 | 3317 | 830 | 48 | 2 | numeric_only | 3.030 | other | yes |
| pc4 | ICLv2更高 | 0.910959 | 0.907534 | -0.3425 | -1.0 | 1166 | 292 | 37 | 2 | numeric_only | 7.191 | vision/sensor/pattern | yes |
| cmc | ICLv2更高 | 0.603390 | 0.600000 | -0.3390 | -1.0 | 1178 | 295 | 9 | 3 | mixed | 1.889 | other | yes |
| splice | ICLv2更高 | 0.960815 | 0.957680 | -0.3135 | -2.0 | 2552 | 638 | 60 | 3 | categorical_only | 2.158 | scientific/health/environment; text/sequence/symbolic; vision/sensor/pattern | yes |
| FOREX_audjpy-hour-High | ICLv2更高 | 0.722076 | 0.718996 | -0.3080 | -27.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.062 | financial/FOREX | yes |
| waveform_database_generator_version_1 | ICLv2更高 | 0.871000 | 0.868000 | -0.3000 | -3.0 | 4000 | 1000 | 21 | 3 | numeric_only | NA | synthetic/rule-generated | yes |
| telco-customer-churn | ICLv2更高 | 0.805536 | 0.802697 | -0.2839 | -4.0 | 5634 | 1409 | 18 | 2 | mixed | 2.768 | business/customer/finance | yes |
| FOREX_cadjpy-day-High | ICLv2更高 | 0.727520 | 0.724796 | -0.2725 | -1.0 | 1467 | 367 | 10 | 2 | numeric_only | 1.007 | financial/FOREX | yes |
| FOREX_audjpy-day-High | ICLv2更高 | 0.787466 | 0.784741 | -0.2725 | -1.0 | 1465 | 367 | 10 | 2 | numeric_only | 1.056 | financial/FOREX | yes |
| ringnorm | ICLv2更高 | 0.979730 | 0.977027 | -0.2703 | -4.0 | 5920 | 1480 | 20 | 2 | numeric_only | 1.020 | synthetic/rule-generated | yes |
| Customer_Personality_Analysis | ICLv2更高 | 0.890625 | 0.888393 | -0.2232 | -1.0 | 1792 | 448 | 24 | 2 | mixed | NA | business/customer/finance | yes |
| Marketing_Campaign | ICLv2更高 | 0.883929 | 0.881696 | -0.2232 | -1.0 | 1792 | 448 | 27 | 2 | mixed | NA | business/customer/finance | yes |
| wine | ICLv2更高 | 0.786693 | 0.784736 | -0.1957 | -1.0 | 2043 | 511 | 4 | 2 | numeric_only | 1.000 | business/customer/finance; scientific/health/environment | yes |
| thyroid-dis | ICLv2更高 | 0.698214 | 0.696429 | -0.1786 | -1.0 | 2240 | 560 | 26 | 5 | mixed | 52.645 | scientific/health/environment | yes |
| California-Housing-Classification | ICLv2更高 | 0.921754 | 0.920058 | -0.1696 | -7.0 | 16512 | 4128 | 8 | 2 | numeric_only | 1.001 | business/customer/finance | yes |
| pol | ICLv2更高 | 0.994051 | 0.992563 | -0.1487 | -3.0 | 8065 | 2017 | 26 | 2 | numeric_only | 1.000 | other | yes |
| BNG(cmc) | ICLv2更高 | 0.587432 | 0.586076 | -0.1356 | -15.0 | 44236 | 11060 | 9 | 3 | mixed | 1.893 | other | yes |
| mammography | ICLv2更高 | 0.989718 | 0.988377 | -0.1341 | -3.0 | 8946 | 2237 | 6 | 2 | numeric_only | 42.012 | other | yes |
| FOREX_audcad-hour-High | ICLv2更高 | 0.717627 | 0.716486 | -0.1141 | -10.0 | 35060 | 8765 | 10 | 2 | numeric_only | 1.061 | financial/FOREX | yes |
| htru | ICLv2更高 | 0.980726 | 0.979609 | -0.1117 | -4.0 | 14318 | 3580 | 8 | 2 | numeric_only | NA | other | yes |
| dry_bean_dataset | ICLv2更高 | 0.928755 | 0.927653 | -0.1102 | -3.0 | 10888 | 2723 | 16 | 7 | numeric_only | NA | scientific/health/environment | yes |
| Click_prediction_small | ICLv2更高 | 0.717897 | 0.716896 | -0.1001 | -8.0 | 31958 | 7990 | 3 | 2 | numeric_only | NA | business/customer/finance | yes |
| churn | ICLv2更高 | 0.967000 | 0.966000 | -0.1000 | -1.0 | 4000 | 1000 | 20 | 2 | mixed | 6.072 | business/customer/finance | yes |
| kdd_ipums_la_97-small | ICLv2更高 | 0.885356 | 0.884393 | -0.0963 | -1.0 | 4150 | 1038 | 20 | 2 | numeric_only | 1.000 | other | yes |
| optdigits | ICLv2更高 | 0.997331 | 0.996441 | -0.0890 | -1.0 | 4496 | 1124 | 64 | 10 | numeric_only | 1.032 | vision/sensor/pattern | yes |
| water_quality | ICLv2更高 | 0.911875 | 0.911250 | -0.0625 | -1.0 | 6396 | 1600 | 20 | 2 | numeric_only | NA | scientific/health/environment | yes |
| naticusdroid+android+permissions+dataset | ICLv2更高 | 0.973070 | 0.972558 | -0.0511 | -3.0 | 23465 | 5867 | 86 | 2 | categorical_only | NA | text/sequence/symbolic | yes |
| Bank_Customer_Churn_Dataset | ICLv2更高 | 0.875000 | 0.874500 | -0.0500 | -1.0 | 8000 | 2000 | 10 | 2 | mixed | NA | business/customer/finance | yes |
| jm1 | ICLv2更高 | 0.819936 | 0.819476 | -0.0459 | -1.0 | 8708 | 2177 | 21 | 2 | numeric_only | 4.169 | text/sequence/symbolic | yes |
| pendigits | ICLv2更高 | 0.997271 | 0.996817 | -0.0455 | -1.0 | 8793 | 2199 | 16 | 10 | numeric_only | 1.084 | text/sequence/symbolic; vision/sensor/pattern | yes |
| gas-drift | ICLv2更高 | 0.997124 | 0.996765 | -0.0359 | -1.0 | 11128 | 2782 | 128 | 6 | numeric_only | 1.834 | vision/sensor/pattern; synthetic/rule-generated | yes |
| credit | ICLv2更高 | 0.785821 | 0.785522 | -0.0299 | -1.0 | 13371 | 3343 | 10 | 2 | numeric_only | 1.000 | business/customer/finance | yes |
| default_of_credit_card_clients | ICLv2更高 | 0.827167 | 0.827000 | -0.0167 | -1.0 | 24000 | 6000 | 23 | 2 | mixed | NA | business/customer/finance | yes |
| mobile_c36_oversampling | ICLv2更高 | 0.994301 | 0.994204 | -0.0097 | -1.0 | 41408 | 10352 | 6 | 2 | numeric_only | NA | other | yes |

### 持平 47 个
| dataset | group | icl_acc | ttt_acc | delta_pp | delta_correct | train | test | features | classes | modality | imbalance | domain | ttt_applied |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| one-hundred-plants-margin | 持平 | 0.903125 | 0.903125 | +0.0000 | +0.0 | 1280 | 320 | 64 | 100 | numeric_only | 1.000 | vision/sensor/pattern | no |
| one-hundred-plants-shape | 持平 | 0.840625 | 0.840625 | +0.0000 | +0.0 | 1280 | 320 | 64 | 100 | numeric_only | 1.000 | vision/sensor/pattern | no |
| one-hundred-plants-texture | 持平 | 0.928125 | 0.928125 | +0.0000 | +0.0 | 1279 | 320 | 64 | 100 | numeric_only | 1.067 | vision/sensor/pattern | no |
| internet_usage | 持平 | 0.549951 | 0.549951 | +0.0000 | +0.0 | 8086 | 2022 | 70 | 46 | categorical_only | NA | business/customer/finance | no |
| UJI_Pen_Characters | 持平 | 0.538462 | 0.538462 | +0.0000 | +0.0 | 1091 | 273 | 80 | 35 | numeric_only | NA | vision/sensor/pattern | no |
| letter | 持平 | 0.990000 | 0.990000 | +0.0000 | +0.0 | 16000 | 4000 | 15 | 26 | numeric_only | 1.108 | text/sequence/symbolic; vision/sensor/pattern | no |
| walking-activity | 持平 | 0.675093 | 0.675093 | +0.0000 | +0.0 | 119465 | 29867 | 4 | 22 | numeric_only | NA | other | no |
| kr-vs-k | 持平 | 0.919636 | 0.919636 | +0.0000 | +0.0 | 22444 | 5612 | 6 | 18 | mixed | 168.630 | synthetic/rule-generated | no |
| kropt | 持平 | 0.916429 | 0.916429 | +0.0000 | +0.0 | 22444 | 5612 | 6 | 18 | categorical_only | 168.630 | synthetic/rule-generated | no |
| ASP-POTASSCO-classification | 持平 | 0.490347 | 0.490347 | +0.0000 | +0.0 | 1035 | 259 | 141 | 11 | mixed | 12.286 | other | no |
| texture | 持平 | 1.000000 | 1.000000 | +0.0000 | +0.0 | 4400 | 1100 | 40 | 11 | numeric_only | 1.000 | vision/sensor/pattern | no |
| volkert | 持平 | 0.751158 | 0.751158 | +0.0000 | +0.0 | 46648 | 11662 | 180 | 10 | numeric_only | 9.409 | business/customer/finance; scientific/health/environment; text/sequence/symbolic | no |
| mfeat-factors | 持平 | 0.982500 | 0.982500 | +0.0000 | +0.0 | 1600 | 400 | 216 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| mfeat-karhunen | 持平 | 0.975000 | 0.975000 | +0.0000 | +0.0 | 1600 | 400 | 64 | 10 | numeric_only | 1.000 | vision/sensor/pattern | yes |
| mfeat-pixel | 持平 | 0.975000 | 0.975000 | +0.0000 | +0.0 | 1600 | 400 | 240 | 10 | categorical_only | 1.000 | vision/sensor/pattern | yes |
| semeion | 持平 | 0.968652 | 0.968652 | +0.0000 | +0.0 | 1274 | 319 | 256 | 10 | numeric_only | 1.045 | vision/sensor/pattern | yes |
| yeast | 持平 | 0.636364 | 0.636364 | +0.0000 | +0.0 | 1187 | 297 | 8 | 10 | numeric_only | 92.600 | scientific/health/environment | yes |
| Indian_pines | 持平 | 0.987972 | 0.987972 | +0.0000 | +0.0 | 7315 | 1829 | 220 | 8 | numeric_only | 202.500 | other | yes |
| mice_protein_expression | 持平 | 1.000000 | 1.000000 | +0.0000 | +0.0 | 864 | 216 | 75 | 8 | numeric_only | NA | scientific/health/environment | yes |
| drug_consumption | 持平 | 0.403183 | 0.403183 | +0.0000 | +0.0 | 1507 | 377 | 12 | 7 | numeric_only | NA | other | yes |
| estimation_of_obesity_levels | 持平 | 0.990544 | 0.990544 | +0.0000 | +0.0 | 1688 | 423 | 16 | 7 | mixed | NA | other | yes |
| Mobile_Price_Classification | 持平 | 0.965000 | 0.965000 | +0.0000 | +0.0 | 1600 | 400 | 20 | 4 | mixed | NA | other | yes |
| analcatdata_authorship | 持平 | 0.994083 | 0.994083 | +0.0000 | +0.0 | 672 | 169 | 69 | 4 | numeric_only | 5.764 | text/sequence/symbolic | yes |
| car-evaluation | 持平 | 0.985549 | 0.985549 | +0.0000 | +0.0 | 1382 | 346 | 21 | 4 | categorical_only | 18.615 | other | yes |
| vehicle | 持平 | 0.876471 | 0.876471 | +0.0000 | +0.0 | 676 | 170 | 18 | 4 | numeric_only | 1.095 | vision/sensor/pattern | yes |
| wall-robot-navigation | 持平 | 0.993590 | 0.993590 | +0.0000 | +0.0 | 4364 | 1092 | 24 | 4 | numeric_only | 6.723 | vision/sensor/pattern | yes |
| allbp | 持平 | 0.977483 | 0.977483 | +0.0000 | +0.0 | 3017 | 755 | 29 | 3 | mixed | 257.786 | other | yes |
| dna | 持平 | 0.971787 | 0.971787 | +0.0000 | +0.0 | 2548 | 638 | 180 | 3 | categorical_only | 2.162 | scientific/health/environment; text/sequence/symbolic | yes |
| predict_students_dropout_and_academic_success | 持平 | 0.783051 | 0.783051 | +0.0000 | +0.0 | 3539 | 885 | 34 | 3 | mixed | NA | other | yes |
| thyroid | 持平 | 0.995833 | 0.995833 | +0.0000 | +0.0 | 5760 | 1440 | 21 | 3 | numeric_only | NA | scientific/health/environment | yes |
| thyroid-ann | 持平 | 0.992053 | 0.992053 | +0.0000 | +0.0 | 3017 | 755 | 21 | 3 | numeric_only | 37.505 | scientific/health/environment; text/sequence/symbolic | yes |
| website_phishing | 持平 | 0.926199 | 0.926199 | +0.0000 | +0.0 | 1082 | 271 | 9 | 3 | categorical_only | NA | business/customer/finance; text/sequence/symbolic | yes |
| Basketball_c | 持平 | 0.701493 | 0.701493 | +0.0000 | +0.0 | 1072 | 268 | 11 | 2 | numeric_only | NA | business/customer/finance | yes |
| E-CommereShippingData | 持平 | 0.676364 | 0.676364 | +0.0000 | +0.0 | 8799 | 2200 | 10 | 2 | mixed | NA | business/customer/finance | yes |
| GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1 | 持平 | 0.678125 | 0.678125 | +0.0000 | +0.0 | 1280 | 320 | 20 | 2 | categorical_only | 1.000 | synthetic/rule-generated | yes |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 | 持平 | 0.684375 | 0.684375 | +0.0000 | +0.0 | 1280 | 320 | 20 | 2 | categorical_only | 1.000 | synthetic/rule-generated | yes |
| IBM_HR_Analytics_Employee_Attrition_and_Performance | 持平 | 0.877551 | 0.877551 | +0.0000 | +0.0 | 1176 | 294 | 31 | 2 | mixed | NA | business/customer/finance | yes |
| MIC | 持平 | 0.909091 | 0.909091 | +0.0000 | +0.0 | 1319 | 330 | 104 | 2 | mixed | 5.416 | scientific/health/environment | yes |
| National_Health_and_Nutrition_Health_Survey | 持平 | 0.833333 | 0.833333 | +0.0000 | +0.0 | 1822 | 456 | 7 | 2 | numeric_only | NA | scientific/health/environment | yes |
| PhishingWebsites | 持平 | 0.980552 | 0.980552 | +0.0000 | +0.0 | 8844 | 2211 | 30 | 2 | categorical_only | 1.257 | text/sequence/symbolic | yes |
| Satellite | 持平 | 0.993137 | 0.993137 | +0.0000 | +0.0 | 4080 | 1020 | 36 | 2 | numeric_only | 67.000 | vision/sensor/pattern | yes |
| company_bankruptcy_prediction | 持平 | 0.968475 | 0.968475 | +0.0000 | +0.0 | 5455 | 1364 | 95 | 2 | mixed | NA | business/customer/finance | yes |
| dis | 持平 | 0.990728 | 0.990728 | +0.0000 | +0.0 | 3017 | 755 | 29 | 2 | mixed | 64.034 | other | yes |
| house_16H | 持平 | 0.891772 | 0.891772 | +0.0000 | +0.0 | 10790 | 2698 | 16 | 2 | numeric_only | 1.000 | other | yes |
| ibm-employee-performance | 持平 | 1.000000 | 1.000000 | +0.0000 | +0.0 | 1176 | 294 | 30 | 2 | mixed | 5.504 | business/customer/finance | yes |
| national-longitudinal-survey-binary | 持平 | 1.000000 | 1.000000 | +0.0000 | +0.0 | 3926 | 982 | 16 | 2 | mixed | 1.649 | other | yes |
| seismic+bumps | 持平 | 0.934236 | 0.934236 | +0.0000 | +0.0 | 2067 | 517 | 18 | 2 | mixed | NA | other | yes |
