# iclv2_ensmble8 vs tabpfnv3_results 准确率对比

比较口径：两个目录都筛选 `status=ok`，再按 `dataset_name` 取共同成功交集；本次两个目录均为 178/178 成功，因此交集为 178 个数据集。

## 总体结果

- 共同成功数据集数：178
- tabpfnv3_results 胜：74 / 178，全量胜率 41.57%，排除平局胜率 48.68%
- iclv2_ensmble8 胜：78 / 178，全量胜率 43.82%，排除平局胜率 51.32%
- 平局：26 / 178，平局率 14.61%
- 平均准确率：iclv2_ensmble8 = 0.846386，tabpfnv3_results = 0.847407
- 平均差值 tabpfnv3 - iclv2_ensmble8：+0.001022，即 +0.1022 个百分点
- 中位数差值：+0.000000
- 按测试集样本数估算净多正确数：+2131

## tabpfnv3_results 优势最大的 10 个数据集

| dataset_name                                                   |   accuracy_iclv2_ensmble8 |   accuracy_tabpfnv3 |   delta_tabpfnv3_minus_iclv2_ensmble8 |
|:---------------------------------------------------------------|--------------------------:|--------------------:|--------------------------------------:|
| Click_prediction_small                                         |                  0.717897 |            0.833417 |                              0.115519 |
| UJI_Pen_Characters                                             |                  0.534799 |            0.622711 |                              0.087912 |
| jungle_chess_2pcs_raw_endgame_complete                         |                  0.891789 |            0.960174 |                              0.068385 |
| compass                                                        |                  0.825774 |            0.863923 |                              0.038150 |
| autoUniv-au7-1100                                              |                  0.404545 |            0.436364 |                              0.031818 |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True |                  0.597500 |            0.622500 |                              0.025000 |
| steel_plates_faults                                            |                  0.848329 |            0.868895 |                              0.020566 |
| Pima_Indians_Diabetes_Database                                 |                  0.727273 |            0.746753 |                              0.019481 |
| eye_movements                                                  |                  0.836837 |            0.853291 |                              0.016453 |
| Marketing_Campaign                                             |                  0.883929 |            0.899554 |                              0.015625 |

## iclv2_ensmble8 优势最大的 10 个数据集

| dataset_name                |   accuracy_iclv2_ensmble8 |   accuracy_tabpfnv3 |   delta_tabpfnv3_minus_iclv2_ensmble8 |
|:----------------------------|--------------------------:|--------------------:|--------------------------------------:|
| mfeat-zernike               |                  0.890000 |            0.837500 |                             -0.052500 |
| hill-valley                 |                  0.975309 |            0.942387 |                             -0.032922 |
| artificial-characters       |                  0.944227 |            0.915362 |                             -0.028865 |
| ASP-POTASSCO-classification |                  0.471042 |            0.444015 |                             -0.027027 |
| PieChart3                   |                  0.884259 |            0.861111 |                             -0.023148 |
| one-hundred-plants-shape    |                  0.837500 |            0.815625 |                             -0.021875 |
| electricity                 |                  0.926514 |            0.907536 |                             -0.018978 |
| FOREX_audusd-hour-High      |                  0.707929 |            0.689218 |                             -0.018711 |
| Indian_pines                |                  0.987972 |            0.970476 |                             -0.017496 |
| gina_agnostic               |                  0.975504 |            0.958213 |                             -0.017291 |

完整逐数据集结果见 `compare_iclv2_ensmble8_vs_tabpfnv3_detail.csv`。
