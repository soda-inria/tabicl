# 1d_taar_sample_ttt_chunk3000 vs iclv2_ensemble32

比较口径：只比较两个结果目录里 `status=ok` 的数据集交集，主表使用普通未加权平均。
- `baseline/1d_taar_sample_ttt_chunk3000`: ok=40
- `baseline/iclv2_ensemble32`: ok=178
- 成功交集: 40
- 仅 `iclv2_ensemble32` 成功但不在 TTT 结果中的数据集: 138，不纳入均值比较

## 成功交集均值

| metric | iclv2_ensemble32 | 1d_taar_sample_ttt_chunk3000 | delta (TTT - ICLv2) |
|---|---:|---:|---:|
| accuracy | 0.787587 | 0.790382 | 0.002795 |
| f1 | 0.783867 | 0.784273 | 0.000406 |
| balanced_accuracy | 0.743199 | 0.742408 | -0.000792 |
| roc_auc | 0.848695 | 0.848332 | -0.000363 |
| log_loss | 0.452473 | 0.452316 | -0.000157 |

按 accuracy 计数：上升=16，下降=13，持平=11。

结论：TTT 在这 40 个共同成功数据集上的 mean accuracy 为 0.790382，比 iclv2_ensemble32 的 0.787587 高 0.002795；但 balanced_accuracy 和 roc_auc 分别小幅下降 0.000792 和 0.000363。log_loss 的 delta 为 -0.000157，按越低越好的口径是微弱改善。

备注：`Indian_pines` 在 TTT 中 `status=ok`，但 `ttt_oom_fallback=True`，因此包含在成功交集均值里，同时在这里单独标记。

## 分组规模和特征数

样本量和特征数优先读取 `data178/<dataset>/info.json`，不是直接使用结果 CSV 的 split 字段。

Accuracy 上升数据集：
- n=16, mean delta accuracy=0.010088, median delta accuracy=0.001059, mean samples=23269.2, median samples=15009.5, mean features=14.7, median features=10.0

Accuracy 下降数据集：
- n=13, mean delta accuracy=-0.003815, median delta accuracy=-0.002396, mean samples=25314.1, median samples=4746.0, mean features=28.5, median features=10.0

Accuracy 持平数据集：
- n=11, mean delta accuracy=0.000000, median delta accuracy=0.000000, mean samples=9628.0, median samples=9144.0, mean features=34.1, median features=19.0

## Accuracy 上升数据集

- Click_prediction_small: 0.717897 -> 0.833542 (delta 0.115645), samples=39948, features=3
- Basketball_c: 0.701493 -> 0.712687 (delta 0.011194), samples=1340, features=11
- GesturePhaseSegmentationProcessed: 0.827342 -> 0.838481 (delta 0.011139), samples=9873, features=32
- FOREX_audcad-day-High: 0.746594 -> 0.754768 (delta 0.008174), samples=1834, features=10
- California-Housing-Classification: 0.921754 -> 0.926599 (delta 0.004845), samples=20640, features=8
- IBM_HR_Analytics_Employee_Attrition_and_Performance: 0.877551 -> 0.880952 (delta 0.003401), samples=1470, features=31
- Firm-Teacher_Clave-Direction_Classification: 0.879167 -> 0.881019 (delta 0.001852), samples=10800, features=16
- Cardiovascular-Disease-dataset: 0.732214 -> 0.733357 (delta 0.001143), samples=70000, features=11
- KDDCup09_upselling: 0.810916 -> 0.811891 (delta 0.000975), samples=5128, features=49
- E-CommereShippingData: 0.676364 -> 0.677273 (delta 0.000909), samples=10999, features=10
- Bank_Customer_Churn_Dataset: 0.875000 -> 0.875500 (delta 0.000500), samples=10000, features=10
- Amazon_employee_access: 0.944156 -> 0.944614 (delta 0.000458), samples=32769, features=7
- BNG(tic-tac-toe): 0.814834 -> 0.815215 (delta 0.000381), samples=39366, features=9
- FOREX_cadjpy-hour-High: 0.715916 -> 0.716258 (delta 0.000342), samples=43825, features=10
- MagicTelescope: 0.894059 -> 0.894322 (delta 0.000263), samples=19020, features=9
- BNG(cmc): 0.587432 -> 0.587613 (delta 0.000181), samples=55296, features=9

## Accuracy 下降数据集

- Fitness_Club_c: 0.803333 -> 0.790000 (delta -0.013333), samples=1500, features=6
- Gender_Gap_in_Spanish_WP: 0.614737 -> 0.603158 (delta -0.011579), samples=4746, features=13
- MIC: 0.909091 -> 0.903030 (delta -0.006061), samples=1649, features=104
- FOREX_audjpy-day-High: 0.787466 -> 0.782016 (delta -0.005450), samples=1832, features=10
- ASP-POTASSCO-classification: 0.490347 -> 0.486486 (delta -0.003861), samples=1294, features=141
- FOREX_cadjpy-day-High: 0.727520 -> 0.724796 (delta -0.002725), samples=1834, features=10
- FOREX_audsgd-hour-High: 0.710553 -> 0.708157 (delta -0.002396), samples=43825, features=10
- Employee: 0.860365 -> 0.859291 (delta -0.001074), samples=4653, features=8
- FOREX_audjpy-hour-High: 0.722076 -> 0.721050 (delta -0.001027), samples=43825, features=10
- INNHotelsGroup: 0.904618 -> 0.903653 (delta -0.000965), samples=36275, features=17
- FOREX_audcad-hour-High: 0.717627 -> 0.717056 (delta -0.000570), samples=43825, features=10
- FOREX_audusd-hour-High: 0.707017 -> 0.706560 (delta -0.000456), samples=43825, features=10
- Credit_c: 0.818650 -> 0.818550 (delta -0.000100), samples=100000, features=22

## Accuracy 持平数据集

- BLE_RSSI_dataset_for_Indoor_localization: 0.733600 -> 0.733600 (delta 0.000000), samples=9984, features=3
- BNG(breast-w): 0.988062 -> 0.988062 (delta 0.000000), samples=39366, features=9
- Customer_Personality_Analysis: 0.890625 -> 0.890625 (delta 0.000000), samples=2240, features=24
- Diabetic_Retinopathy_Debrecen: 0.740260 -> 0.740260 (delta 0.000000), samples=1151, features=19
- FICO-HELOC-cleaned: 0.754937 -> 0.754937 (delta 0.000000), samples=9871, features=23
- FOREX_audchf-day-High: 0.749319 -> 0.749319 (delta 0.000000), samples=1833, features=10
- GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1: 0.678125 -> 0.678125 (delta 0.000000), samples=1600, features=20
- GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001: 0.684375 -> 0.684375 (delta 0.000000), samples=1600, features=20
- HR_Analytics_Job_Change_of_Data_Scientists: 0.800626 -> 0.800626 (delta 0.000000), samples=19158, features=13
- Indian_pines: 0.987972 -> 0.987972 (delta 0.000000), samples=9144, features=220
- JapaneseVowels: 0.999498 -> 0.999498 (delta 0.000000), samples=9961, features=14

## 输出文件

- 明细 CSV：`compare_success_with_iclv2_ensemble32.csv`
- 本汇总：`compare_success_with_iclv2_ensemble32.md`
