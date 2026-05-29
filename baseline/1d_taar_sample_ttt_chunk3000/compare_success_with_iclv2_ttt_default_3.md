# 1d_taar_sample_ttt_chunk3000 vs iclv2_ttt_default 3

比较口径：只比较两个结果目录里 `status=ok` 的数据集交集，主表使用普通未加权平均。delta 定义为 `chunk3000 - default3`。
- `baseline/1d_taar_sample_ttt_chunk3000`: ok=40, failed=0
- `baseline/iclv2_ttt_default 3`: ok=176, failed=2
- 成功交集: 40
- 仅 `1d_taar_sample_ttt_chunk3000` ok 但不在 default3 ok 中: 0
- 仅 `iclv2_ttt_default 3` ok 但不在 chunk3000 ok 中: 136，不纳入均值比较
- default3 failed datasets: accelerometer, shuttle

## 成功交集均值

| metric | iclv2_ttt_default 3 | 1d_taar_sample_ttt_chunk3000 | delta (chunk3000 - default3) |
|---|---:|---:|---:|
| accuracy | 0.792240 | 0.790382 | -0.001858 |
| f1 | 0.786313 | 0.784273 | -0.002040 |
| balanced_accuracy | 0.743984 | 0.742408 | -0.001577 |
| roc_auc | 0.850566 | 0.848332 | -0.002234 |
| log_loss | 0.447559 | 0.452316 | 0.004757 |

按 accuracy 计数：chunk3000 上升=13，下降=19，持平=8。

结论：在 40 个共同成功数据集上，chunk3000 mean accuracy=0.790382，default3 mean accuracy=0.792240，delta=-0.001858。balanced_accuracy delta=-0.001577，roc_auc delta=-0.002234，log_loss delta=0.004757。

chunk3000 交集内 `ttt_oom_fallback=True`: 1，Indian_pines。
default3 交集内 `ttt_oom_fallback=True`: 1，Indian_pines。

## 分组规模和特征数

样本量和特征数优先读取 `data178/<dataset>/info.json`。

Accuracy 上升数据集：
- n=13, mean delta accuracy=0.003615, median delta accuracy=0.002148, mean samples=16366.2, median samples=9871.0, mean features=16.0, median features=10.0

Accuracy 下降数据集：
- n=19, mean delta accuracy=-0.006384, median delta accuracy=-0.002725, mean samples=24500.3, median samples=10800.0, mean features=24.8, median features=11.0

Accuracy 持平数据集：
- n=8, mean delta accuracy=0.000000, median delta accuracy=0.000000, mean samples=16129.2, median samples=9552.5, mean features=37.8, median features=10.0

## Accuracy 上升数据集

- Diabetic_Retinopathy_Debrecen: 0.727273 -> 0.740260 (delta 0.012987), samples=1151, features=19
- Basketball_c: 0.705224 -> 0.712687 (delta 0.007463), samples=1340, features=11
- FICO-HELOC-cleaned: 0.747848 -> 0.754937 (delta 0.007089), samples=9871, features=23
- KDDCup09_upselling: 0.806043 -> 0.811891 (delta 0.005848), samples=5128, features=49
- IBM_HR_Analytics_Employee_Attrition_and_Performance: 0.877551 -> 0.880952 (delta 0.003401), samples=1470, features=31
- GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001: 0.681250 -> 0.684375 (delta 0.003125), samples=1600, features=20
- Employee: 0.857143 -> 0.859291 (delta 0.002148), samples=4653, features=8
- FOREX_audcad-hour-High: 0.715345 -> 0.717056 (delta 0.001711), samples=43825, features=10
- E-CommereShippingData: 0.676364 -> 0.677273 (delta 0.000909), samples=10999, features=10
- Click_prediction_small: 0.832666 -> 0.833542 (delta 0.000876), samples=39948, features=3
- California-Housing-Classification: 0.925872 -> 0.926599 (delta 0.000727), samples=20640, features=8
- Amazon_employee_access: 0.944156 -> 0.944614 (delta 0.000458), samples=32769, features=7
- BNG(tic-tac-toe): 0.814961 -> 0.815215 (delta 0.000254), samples=39366, features=9

## Accuracy 下降数据集

- BLE_RSSI_dataset_for_Indoor_localization: 0.771657 -> 0.733600 (delta -0.038057), samples=9984, features=3
- FOREX_cadjpy-day-High: 0.741144 -> 0.724796 (delta -0.016349), samples=1834, features=10
- Fitness_Club_c: 0.803333 -> 0.790000 (delta -0.013333), samples=1500, features=6
- Gender_Gap_in_Spanish_WP: 0.612632 -> 0.603158 (delta -0.009474), samples=4746, features=13
- GesturePhaseSegmentationProcessed: 0.847089 -> 0.838481 (delta -0.008608), samples=9873, features=32
- Firm-Teacher_Clave-Direction_Classification: 0.885185 -> 0.881019 (delta -0.004167), samples=10800, features=16
- ASP-POTASSCO-classification: 0.490347 -> 0.486486 (delta -0.003861), samples=1294, features=141
- FOREX_audsgd-hour-High: 0.711694 -> 0.708157 (delta -0.003537), samples=43825, features=10
- MIC: 0.906061 -> 0.903030 (delta -0.003030), samples=1649, features=104
- FOREX_audjpy-day-High: 0.784741 -> 0.782016 (delta -0.002725), samples=1832, features=10
- FOREX_cadjpy-hour-High: 0.718768 -> 0.716258 (delta -0.002510), samples=43825, features=10
- Credit_c: 0.820950 -> 0.818550 (delta -0.002400), samples=100000, features=22
- MagicTelescope: 0.896688 -> 0.894322 (delta -0.002366), samples=19020, features=9
- FOREX_audusd-hour-High: 0.708842 -> 0.706560 (delta -0.002282), samples=43825, features=10
- Customer_Personality_Analysis: 0.892857 -> 0.890625 (delta -0.002232), samples=2240, features=24
- INNHotelsGroup: 0.905858 -> 0.903653 (delta -0.002205), samples=36275, features=17
- FOREX_audjpy-hour-High: 0.722647 -> 0.721050 (delta -0.001597), samples=43825, features=10
- HR_Analytics_Job_Change_of_Data_Scientists: 0.802192 -> 0.800626 (delta -0.001566), samples=19158, features=13
- Cardiovascular-Disease-dataset: 0.734357 -> 0.733357 (delta -0.001000), samples=70000, features=11

## Accuracy 持平数据集

- BNG(breast-w): 0.988062 -> 0.988062 (delta 0.000000), samples=39366, features=9
- BNG(cmc): 0.587613 -> 0.587613 (delta 0.000000), samples=55296, features=9
- Bank_Customer_Churn_Dataset: 0.875500 -> 0.875500 (delta 0.000000), samples=10000, features=10
- FOREX_audcad-day-High: 0.754768 -> 0.754768 (delta 0.000000), samples=1834, features=10
- FOREX_audchf-day-High: 0.749319 -> 0.749319 (delta 0.000000), samples=1833, features=10
- GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1: 0.678125 -> 0.678125 (delta 0.000000), samples=1600, features=20
- Indian_pines: 0.987972 -> 0.987972 (delta 0.000000), samples=9144, features=220
- JapaneseVowels: 0.999498 -> 0.999498 (delta 0.000000), samples=9961, features=14

## 输出文件

- 明细 CSV：`compare_success_with_iclv2_ttt_default_3.csv`
- 本汇总：`compare_success_with_iclv2_ttt_default_3.md`
