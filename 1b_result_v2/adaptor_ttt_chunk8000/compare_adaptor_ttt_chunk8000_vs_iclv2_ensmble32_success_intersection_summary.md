# adaptor_ttt_chunk8000 vs iclv2_ensmble32 成功交集对比

## 口径

- 比较文件：`1b_result_v2/adaptor_ttt_chunk8000/all_classification_results.csv` vs `1b_result_v2/iclv2_ensmble32/all_classification_results.csv`。
- 主比较只取两边 `status=ok` 的 `dataset_name` 交集。
- adaptor 当前 `ok=83/90`，baseline 当前 `ok=178/178`，成功交集 `n=83`。
- adaptor 失败数据集未进入均值：Bank_Customer_Churn_Dataset, Click_prediction_small, E-CommereShippingData, JapaneseVowels, Satellite, accelerometer, artificial-characters。

## 总体结果

| metric | adaptor_ttt_chunk8000 | iclv2_ensmble32 | delta |
|---|---:|---:|---:|
| mean accuracy | 0.814751 | 0.815029 | -0.0278 pp |
| median accuracy | 0.836380 | 0.835466 | +0.0914 pp |
| mean per-dataset delta |  |  | -0.0278 pp |
| median per-dataset delta |  |  | +0.0000 pp |
| test-size weighted accuracy | 0.829158 | 0.829194 | -0.0036 pp |
| weighted delta_correct |  |  | -11.00 test examples |

- Win/Loss/Tie（adaptor 相对 iclv2）：`14/18/51`。
- adaptor 交集内 `ttt_adaptor_enabled=True` 为 `81/83`，`ttt_applied=True` 为 `81/83`。
- 仅 adaptor enabled 子集：n=81，mean delta=-0.0192 pp，W/L/T=14/16/51。
- adaptor 未启用/跳过子集：n=2，mean delta=-0.3762 pp，W/L/T=0/2/0。

## 上升/下降/持平数据集

- 上升 14 个：ada_prior (+0.110pp, feat=14, n=4562), Gender_Gap_in_Spanish_WP (+0.105pp, feat=13, n=4746), KDDCup09_upselling (+0.097pp, feat=49, n=5128), eye_movements (+0.091pp, feat=27, n=10936), FOREX_cadjpy-hour-High (+0.080pp, feat=10, n=43825), California-Housing-Classification (+0.048pp, feat=8, n=20640), Amazon_employee_access (+0.046pp, feat=7, n=32769), electricity (+0.044pp, feat=8, n=45312), Credit_c (+0.035pp, feat=22, n=100000), FOREX_audusd-hour-High (+0.034pp, feat=10, n=43825), MagicTelescope (+0.026pp, feat=9, n=19020), BNG(cmc) (+0.018pp, feat=9, n=55296), Cardiovascular-Disease-dataset (+0.014pp, feat=11, n=70000), BNG(tic-tac-toe) (+0.013pp, feat=9, n=39366)
- 下降 18 个：FOREX_audjpy-day-High (-0.545pp, feat=10, n=1832), Diabetic_Retinopathy_Debrecen (-0.433pp, feat=19, n=1151), autoUniv-au4-2500 (-0.400pp, feat=100, n=2500), ASP-POTASSCO-classification (-0.386pp, feat=141, n=1294), UJI_Pen_Characters (-0.366pp, feat=80, n=1364), ada_agnostic (-0.219pp, feat=48, n=4562), FOREX_audcad-hour-High (-0.137pp, feat=10, n=43825), ada (-0.120pp, feat=48, n=4147), Employee (-0.107pp, feat=8, n=4653), bank (-0.077pp, feat=16, n=45211), credit (-0.060pp, feat=10, n=16714), FOREX_audsgd-hour-High (-0.057pp, feat=10, n=43825), GesturePhaseSegmentationProcessed (-0.051pp, feat=32, n=9873), Firm-Teacher_Clave-Direction_Classification (-0.046pp, feat=16, n=10800), FOREX_audjpy-hour-High (-0.034pp, feat=10, n=43825), INNHotelsGroup (-0.014pp, feat=17, n=36275), Rain_in_Australia (-0.010pp, feat=18, n=145460), customer_satisfaction_in_airline (-0.008pp, feat=21, n=129880)
- 持平 51 个：BLE_RSSI_dataset_for_Indoor_localization (+0.000pp, feat=3, n=9984), BNG(breast-w) (+0.000pp, feat=9, n=39366), Basketball_c (+0.000pp, feat=11, n=1340), Customer_Personality_Analysis (+0.000pp, feat=24, n=2240), FICO-HELOC-cleaned (+0.000pp, feat=23, n=9871), FOREX_audcad-day-High (+0.000pp, feat=10, n=1834), FOREX_audchf-day-High (+0.000pp, feat=10, n=1833), FOREX_cadjpy-day-High (+0.000pp, feat=10, n=1834), Fitness_Club_c (+0.000pp, feat=6, n=1500), GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1 (+0.000pp, feat=20, n=1600), GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 (+0.000pp, feat=20, n=1600), HR_Analytics_Job_Change_of_Data_Scientists (+0.000pp, feat=13, n=19158), IBM_HR_Analytics_Employee_Attrition_and_Performance (+0.000pp, feat=31, n=1470), Indian_pines (+0.000pp, feat=220, n=9144), MIC (+0.000pp, feat=104, n=1649), Marketing_Campaign (+0.000pp, feat=27, n=2240), Mobile_Price_Classification (+0.000pp, feat=20, n=2000), National_Health_and_Nutrition_Health_Survey (+0.000pp, feat=7, n=2278), PhishingWebsites (+0.000pp, feat=30, n=11055), PieChart3 (+0.000pp, feat=37, n=1077), Pima_Indians_Diabetes_Database (+0.000pp, feat=8, n=768), PizzaCutter3 (+0.000pp, feat=37, n=1043), Pumpkin_Seeds (+0.000pp, feat=12, n=2500), QSAR_biodegradation (+0.000pp, feat=41, n=1054), SDSS17 (+0.000pp, feat=12, n=100000), Telecom_Churn_Dataset (+0.000pp, feat=17, n=3333), Water_Quality_and_Potability (+0.000pp, feat=8, n=3276), Wilt (+0.000pp, feat=5, n=4821), abalone (+0.000pp, feat=8, n=4177), airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True (+0.000pp, feat=7, n=2000), allbp (+0.000pp, feat=29, n=3772), allrep (+0.000pp, feat=29, n=3772), analcatdata_authorship (+0.000pp, feat=69, n=841), autoUniv-au7-1100 (+0.000pp, feat=12, n=1100), banknote_authentication (+0.000pp, feat=4, n=1372), baseball (+0.000pp, feat=16, n=1340), car-evaluation (+0.000pp, feat=21, n=1728), churn (+0.000pp, feat=20, n=5000), cmc (+0.000pp, feat=9, n=1473), company_bankruptcy_prediction (+0.000pp, feat=95, n=6819), compass (+0.000pp, feat=17, n=16644), contraceptive_method_choice (+0.000pp, feat=9, n=1473), dabetes_130-us_hospitals (+0.000pp, feat=20, n=101766), default_of_credit_card_clients (+0.000pp, feat=23, n=30000), delta_ailerons (+0.000pp, feat=5, n=7129), dis (+0.000pp, feat=29, n=3772), dna (+0.000pp, feat=180, n=3186), drug_consumption (+0.000pp, feat=12, n=1884), dry_bean_dataset (+0.000pp, feat=16, n=13611), eeg-eye-state (+0.000pp, feat=14, n=14980), estimation_of_obesity_levels (+0.000pp, feat=16, n=2111)

## 上升/下降数据规模特征

| outcome | count | mean delta | median delta | mean features | median features | mean samples | median samples |
|---|---:|---:|---:|---:|---:|---:|---:|
| win | 14 | +0.0545 pp | +0.0450 pp | 14.7 | 10.0 | 35387.5 | 36067.5 |
| loss | 18 | -0.1706 pp | -0.0924 pp | 34.1 | 17.5 | 30399.5 | 10336.5 |
| tie | 51 | +0.0000 pp | +0.0000 pp | 28.1 | 16.0 | 9212.1 | 2240.0 |

### 按样本量分桶

| sample_bin | win | loss | tie | mean delta |
|---|---:|---:|---:|---:|
| <=2k | 0 | 4 | 23 | -0.0641 pp |
| 2k-5k | 2 | 4 | 14 | -0.0316 pp |
| 5k-20k | 3 | 3 | 10 | +0.0037 pp |
| 20k-100k | 9 | 5 | 3 | +0.0008 pp |
| >100k | 0 | 2 | 1 | -0.0060 pp |

### 按特征数分桶

| feature_bin | win | loss | tie | mean delta |
|---|---:|---:|---:|---:|
| <=10 | 8 | 6 | 16 | -0.0210 pp |
| 11-30 | 5 | 6 | 26 | -0.0063 pp |
| 31-100 | 1 | 5 | 6 | -0.0883 pp |
| 101-300 | 0 | 1 | 3 | -0.0965 pp |

逐数据集明细见 `compare_adaptor_ttt_chunk8000_vs_iclv2_ensmble32_success_intersection_detail.csv`。
