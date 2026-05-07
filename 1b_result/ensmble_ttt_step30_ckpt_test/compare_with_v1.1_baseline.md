# ensmble_ttt_step30_ckpt_test vs v1.1_baseline

Note: 34 failed datasets were filled with v1.1_baseline accuracy.

- baseline ok datasets: 178
- test ok datasets after fallback: 178
- baseline fallback datasets: 34
- baseline mean acc: 0.837755
- test mean acc: 0.836909
- mean delta: -0.0846 pp
- median delta: +0.0000 pp
- improved / regressed / tie: 62 / 66 / 50
- Pearson corr(baseline_acc, delta_pp): 0.0762
- Spearman corr(baseline_acc, delta_pp): 0.0524

## Baseline Accuracy Quartiles

| base_q     |   n |   base_min |   base_max |   base_mean |   test_mean |   delta_mean_pp |   delta_median_pp |   improved |   regressed |   tie |   fallback |
|:-----------|----:|-----------:|-----------:|------------:|------------:|----------------:|------------------:|-----------:|------------:|------:|-----------:|
| Q1 lowest  |  45 |   0.356    |   0.74481  |    0.638872 |    0.63549  |         -0.3383 |                 0 |         12 |          17 |    16 |         14 |
| Q2         |  44 |   0.745    |   0.872    |    0.816737 |    0.817919 |          0.1182 |                 0 |         20 |          17 |     7 |          6 |
| Q3         |  44 |   0.875    |   0.956113 |    0.914141 |    0.913141 |         -0.1    |                 0 |         17 |          17 |    10 |          6 |
| Q4 highest |  45 |   0.956607 |   1        |    0.9825   |    0.982358 |         -0.0142 |                 0 |         13 |          15 |    17 |          8 |

## Improved Datasets

| dataset                                                        | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:---------------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| eye_movements_bin                                              | binclass   |       0.625493 |   0.693824 |     6.8331 |        10.924 |      6086 |     1522 |           20 |           2 | False               |
| mfeat-zernike                                                  | multiclass |       0.8525   |   0.9175   |     6.5    |         7.625 |      1600 |      400 |           47 |          10 | False               |
| eye_movements                                                  | multiclass |       0.714808 |   0.77925  |     6.4442 |         9.015 |      8748 |     2188 |           27 |           3 | False               |
| compass                                                        | binclass   |       0.826675 |   0.890658 |     6.3983 |         7.74  |     13315 |     3329 |           17 |           2 | False               |
| artificial-characters                                          | multiclass |       0.863992 |   0.90998  |     4.5988 |         5.323 |      8174 |     2044 |            7 |          10 | False               |
| vehicle                                                        | multiclass |       0.858824 |   0.9      |     4.1176 |         4.795 |       676 |      170 |           18 |           4 | False               |
| autoUniv-au4-2500                                              | multiclass |       0.522    |   0.554    |     3.2    |         6.13  |      2000 |      500 |          100 |           3 | False               |
| hill-valley                                                    | binclass   |       0.930041 |   0.958848 |     2.8807 |         3.097 |       969 |      243 |          100 |           2 | False               |
| madeline                                                       | binclass   |       0.753185 |   0.781847 |     2.8662 |         3.805 |      2512 |      628 |          259 |           2 | False               |
| mfeat-morphological                                            | multiclass |       0.76     |   0.7875   |     2.75   |         3.618 |      1600 |      400 |            6 |          10 | False               |
| mfeat-fourier                                                  | multiclass |       0.885    |   0.91     |     2.5    |         2.825 |      1600 |      400 |           76 |          10 | False               |
| GesturePhaseSegmentationProcessed                              | multiclass |       0.786329 |   0.809114 |     2.2785 |         2.898 |      7898 |     1975 |           32 |           5 | False               |
| car-evaluation                                                 | multiclass |       0.973988 |   0.99422  |     2.0231 |         2.077 |      1382 |      346 |           21 |           4 | False               |
| PieChart3                                                      | binclass   |       0.875    |   0.893519 |     1.8519 |         2.116 |       861 |      216 |           37 |           2 | False               |
| Fitness_Club_c                                                 | binclass   |       0.786667 |   0.803333 |     1.6667 |         2.119 |      1200 |      300 |            6 |           2 | False               |
| steel_plates_faults                                            | multiclass |       0.838046 |   0.8509   |     1.2853 |         1.534 |      1552 |      389 |           27 |           7 | False               |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001     | binclass   |       0.68125  |   0.69375  |     1.25   |         1.835 |      1280 |      320 |           20 |           2 | False               |
| banknote_authentication                                        | binclass   |       0.549091 |   0.56     |     1.0909 |         1.987 |      1097 |      275 |            4 |           2 | False               |
| BLE_RSSI_dataset_for_Indoor_localization                       | multiclass |       0.740611 |   0.750626 |     1.0015 |         1.352 |      7987 |     1997 |            3 |           3 | False               |
| waveform-5000                                                  | multiclass |       0.866    |   0.876    |     1      |         1.155 |      4000 |     1000 |           40 |           3 | False               |
| led7                                                           | multiclass |       0.7375   |   0.745313 |     0.7812 |         1.059 |      2560 |      640 |            7 |          10 | False               |
| FICO-HELOC-cleaned                                             | binclass   |       0.74481  |   0.751899 |     0.7089 |         0.952 |      7896 |     1975 |           23 |           2 | False               |
| sylvine                                                        | binclass   |       0.970732 |   0.976585 |     0.5854 |         0.603 |      4099 |     1025 |           20 |           2 | False               |
| thyroid-dis                                                    | multiclass |       0.691071 |   0.696429 |     0.5357 |         0.775 |      2240 |      560 |           26 |           5 | False               |
| wine-quality-white                                             | multiclass |       0.682653 |   0.687755 |     0.5102 |         0.747 |      3918 |      980 |           11 |           7 | False               |
| rl                                                             | binclass   |       0.854125 |   0.859155 |     0.503  |         0.589 |      3976 |      994 |           12 |           2 | False               |
| water_quality                                                  | binclass   |       0.908125 |   0.913125 |     0.5    |         0.551 |      6396 |     1600 |           20 |           2 | False               |
| QSAR_biodegradation                                            | binclass   |       0.886256 |   0.890995 |     0.4739 |         0.535 |       843 |      211 |           41 |           2 | False               |
| Customer_Personality_Analysis                                  | binclass   |       0.883929 |   0.888393 |     0.4464 |         0.505 |      1792 |      448 |           24 |           2 | False               |
| satimage                                                       | multiclass |       0.92846  |   0.932348 |     0.3888 |         0.419 |      5144 |     1286 |           36 |           6 | False               |
| semeion                                                        | multiclass |       0.959248 |   0.962382 |     0.3135 |         0.327 |      1274 |      319 |          256 |          10 | False               |
| dry_bean_dataset                                               | multiclass |       0.927653 |   0.930591 |     0.2938 |         0.317 |     10888 |     2723 |           16 |           7 | False               |
| jm1                                                            | binclass   |       0.821314 |   0.82407  |     0.2756 |         0.336 |      8708 |     2177 |           21 |           2 | False               |
| thyroid-ann                                                    | multiclass |       0.988079 |   0.990728 |     0.2649 |         0.268 |      3017 |      755 |           21 |           3 | False               |
| dis                                                            | binclass   |       0.984106 |   0.986755 |     0.2649 |         0.269 |      3017 |      755 |           29 |           2 | False               |
| mfeat-pixel                                                    | multiclass |       0.975    |   0.9775   |     0.25   |         0.256 |      1600 |      400 |          240 |          10 | False               |
| Mobile_Price_Classification                                    | multiclass |       0.94     |   0.9425   |     0.25   |         0.266 |      1600 |      400 |           20 |           4 | False               |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass   |       0.6025   |   0.605    |     0.25   |         0.415 |      1600 |      400 |            7 |           2 | False               |
| Amazon_employee_access                                         | binclass   |       0.944309 |   0.94675  |     0.2441 |         0.259 |     26215 |     6554 |            7 |           2 | False               |
| okcupid_stem                                                   | multiclass |       0.747564 |   0.75     |     0.2436 |         0.326 |     21341 |     5336 |           13 |           3 | False               |
| Firm-Teacher_Clave-Direction_Classification                    | multiclass |       0.878241 |   0.880556 |     0.2315 |         0.264 |      8640 |     2160 |           16 |           4 | False               |
| default_of_credit_card_clients                                 | binclass   |       0.826167 |   0.828333 |     0.2167 |         0.262 |     24000 |     6000 |           23 |           2 | False               |
| BNG(tic-tac-toe)                                               | binclass   |       0.815088 |   0.81712  |     0.2032 |         0.249 |     31492 |     7874 |            9 |           2 | False               |
| kdd_ipums_la_97-small                                          | binclass   |       0.888247 |   0.890173 |     0.1927 |         0.217 |      4150 |     1038 |           20 |           2 | False               |
| credit                                                         | binclass   |       0.78283  |   0.784625 |     0.1795 |         0.229 |     13371 |     3343 |           10 |           2 | False               |
| eeg-eye-state                                                  | binclass   |       0.991322 |   0.992991 |     0.1669 |         0.168 |     11984 |     2996 |           14 |           2 | False               |
| microaggregation2                                              | multiclass |       0.6405   |   0.642    |     0.15   |         0.234 |     16000 |     4000 |           20 |           5 | False               |
| Telecom_Churn_Dataset                                          | binclass   |       0.955022 |   0.956522 |     0.1499 |         0.157 |      2666 |      667 |           17 |           2 | False               |
| online_shoppers                                                | binclass   |       0.904298 |   0.905515 |     0.1217 |         0.135 |      9864 |     2466 |           14 |           2 | False               |
| California-Housing-Classification                              | binclass   |       0.912791 |   0.914002 |     0.1211 |         0.133 |     16512 |     4128 |            8 |           2 | False               |
| ada_agnostic                                                   | binclass   |       0.840088 |   0.841183 |     0.1095 |         0.13  |      3649 |      913 |           48 |           2 | False               |
| churn                                                          | binclass   |       0.952    |   0.953    |     0.1    |         0.105 |      4000 |     1000 |           20 |           2 | False               |
| waveform_database_generator_version_1                          | multiclass |       0.866    |   0.867    |     0.1    |         0.115 |      4000 |     1000 |           21 |           3 | False               |
| company_bankruptcy_prediction                                  | binclass   |       0.970674 |   0.971408 |     0.0733 |         0.076 |      5455 |     1364 |           95 |           2 | False               |
| taiwanese_bankruptcy_prediction                                | binclass   |       0.970674 |   0.971408 |     0.0733 |         0.076 |      5455 |     1364 |           95 |           2 | False               |
| Click_prediction_small                                         | binclass   |       0.83204  |   0.832666 |     0.0626 |         0.075 |     31958 |     7990 |            3 |           2 | False               |
| mammography                                                    | binclass   |       0.988824 |   0.989271 |     0.0447 |         0.045 |      8946 |     2237 |            6 |           2 | False               |
| BNG(breast-w)                                                  | binclass   |       0.988316 |   0.988697 |     0.0381 |         0.039 |     31492 |     7874 |            9 |           2 | False               |
| gas-drift                                                      | multiclass |       0.996405 |   0.996765 |     0.0359 |         0.036 |     11128 |     2782 |          128 |           6 | False               |
| htru                                                           | binclass   |       0.97933  |   0.979609 |     0.0279 |         0.029 |     14318 |     3580 |            8 |           2 | False               |
| MagicTelescope                                                 | binclass   |       0.884858 |   0.885121 |     0.0263 |         0.03  |     15216 |     3804 |            9 |           2 | False               |
| HR_Analytics_Job_Change_of_Data_Scientists                     | binclass   |       0.799322 |   0.799582 |     0.0261 |         0.033 |     15326 |     3832 |           13 |           2 | False               |

## Regressed Datasets

| dataset                                             | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:----------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| FOREX_audjpy-day-High                               | binclass   |       0.765668 |   0.697548 |    -6.812  |        -8.897 |      1465 |      367 |           10 |           2 | False               |
| FOREX_audchf-day-High                               | binclass   |       0.754768 |   0.692098 |    -6.267  |        -8.303 |      1466 |      367 |           10 |           2 | False               |
| Basketball_c                                        | binclass   |       0.712687 |   0.664179 |    -4.8507 |        -6.806 |      1072 |      268 |           11 |           2 | False               |
| FOREX_cadjpy-day-High                               | binclass   |       0.716621 |   0.673025 |    -4.3597 |        -6.084 |      1467 |      367 |           10 |           2 | False               |
| Water_Quality_and_Potability                        | binclass   |       0.641768 |   0.60061  |    -4.1159 |        -6.413 |      2620 |      656 |            8 |           2 | False               |
| sports_articles_for_objectivity_analysis            | binclass   |       0.855    |   0.815    |    -4      |        -4.678 |       800 |      200 |           59 |           2 | False               |
| abalone                                             | multiclass |       0.651914 |   0.61244  |    -3.9474 |        -6.055 |      3341 |      836 |            8 |           3 | False               |
| FOREX_audcad-day-High                               | binclass   |       0.743869 |   0.708447 |    -3.5422 |        -4.762 |      1467 |      367 |           10 |           2 | False               |
| contraceptive_method_choice                         | multiclass |       0.627119 |   0.59322  |    -3.3898 |        -5.405 |      1178 |      295 |            9 |           3 | False               |
| MIC                                                 | binclass   |       0.909091 |   0.875758 |    -3.3333 |        -3.667 |      1319 |      330 |          104 |           2 | False               |
| qsar                                                | binclass   |       0.905213 |   0.876777 |    -2.8436 |        -3.141 |       844 |      211 |           40 |           2 | False               |
| E-CommereShippingData                               | binclass   |       0.678182 |   0.652727 |    -2.5455 |        -3.753 |      8799 |     2200 |           10 |           2 | False               |
| waveform_database_generator                         | multiclass |       0.356    |   0.331    |    -2.5    |        -7.022 |      3999 |     1000 |           21 |           3 | False               |
| cmc                                                 | multiclass |       0.589831 |   0.566102 |    -2.3729 |        -4.023 |      1178 |      295 |            9 |           3 | False               |
| wine                                                | binclass   |       0.774951 |   0.751468 |    -2.3483 |        -3.03  |      2043 |      511 |            4 |           2 | False               |
| golf_play_dataset_extended                          | binclass   |       0.936073 |   0.913242 |    -2.2831 |        -2.439 |       876 |      219 |            9 |           2 | False               |
| wine-quality-red                                    | multiclass |       0.66875  |   0.646875 |    -2.1875 |        -3.271 |      1279 |      320 |            4 |           6 | False               |
| maternal_health_risk                                | multiclass |       0.871921 |   0.852217 |    -1.9704 |        -2.26  |       811 |      203 |            6 |           3 | False               |
| Pima_Indians_Diabetes_Database                      | binclass   |       0.746753 |   0.727273 |    -1.9481 |        -2.609 |       614 |      154 |            8 |           2 | False               |
| Diabetic_Retinopathy_Debrecen                       | binclass   |       0.748918 |   0.735931 |    -1.2987 |        -1.734 |       920 |      231 |           19 |           2 | False               |
| ozone-level-8hr                                     | binclass   |       0.956607 |   0.944773 |    -1.1834 |        -1.237 |      2027 |      507 |           72 |           2 | False               |
| heloc                                               | binclass   |       0.7255   |   0.715    |    -1.05   |        -1.447 |      8000 |     2000 |           22 |           2 | False               |
| house_16H                                           | binclass   |       0.89066  |   0.881394 |    -0.9266 |        -1.04  |     10790 |     2698 |           16 |           2 | False               |
| Bank_Customer_Churn_Dataset                         | binclass   |       0.877    |   0.868    |    -0.9    |        -1.026 |      8000 |     2000 |           10 |           2 | False               |
| ada_prior                                           | binclass   |       0.841183 |   0.832421 |    -0.8762 |        -1.042 |      3649 |      913 |           14 |           2 | False               |
| turiye_student_evaluation                           | multiclass |       0.522337 |   0.513746 |    -0.8591 |        -1.645 |      4656 |     1164 |           32 |           5 | False               |
| first-order-theorem-proving                         | multiclass |       0.637255 |   0.629902 |    -0.7353 |        -1.154 |      4894 |     1224 |           51 |           6 | False               |
| delta_ailerons                                      | binclass   |       0.950912 |   0.943899 |    -0.7013 |        -0.737 |      5703 |     1426 |            5 |           2 | False               |
| IBM_HR_Analytics_Employee_Attrition_and_Performance | binclass   |       0.863946 |   0.857143 |    -0.6803 |        -0.787 |      1176 |      294 |           31 |           2 | False               |
| Marketing_Campaign                                  | binclass   |       0.890625 |   0.883929 |    -0.6696 |        -0.752 |      1792 |      448 |           27 |           2 | False               |
| National_Health_and_Nutrition_Health_Survey         | binclass   |       0.839912 |   0.833333 |    -0.6579 |        -0.783 |      1822 |      456 |            7 |           2 | False               |
| spambase                                            | binclass   |       0.959826 |   0.953312 |    -0.6515 |        -0.679 |      3680 |      921 |           57 |           2 | False               |
| Employee                                            | binclass   |       0.852846 |   0.846402 |    -0.6445 |        -0.756 |      3722 |      931 |            8 |           2 | False               |
| wall-robot-navigation                               | multiclass |       0.986264 |   0.979853 |    -0.641  |        -0.65  |      4364 |     1092 |           24 |           4 | False               |
| pc3                                                 | binclass   |       0.900958 |   0.894569 |    -0.639  |        -0.709 |      1250 |      313 |           37 |           2 | False               |
| led24                                               | multiclass |       0.734375 |   0.728125 |    -0.625  |        -0.851 |      2560 |      640 |           24 |          10 | False               |
| ada                                                 | binclass   |       0.848193 |   0.842169 |    -0.6024 |        -0.71  |      3317 |      830 |           48 |           2 | False               |
| seismic+bumps                                       | binclass   |       0.934236 |   0.928433 |    -0.5803 |        -0.621 |      2067 |      517 |           18 |           2 | False               |
| statlog                                             | binclass   |       0.745    |   0.74     |    -0.5    |        -0.671 |       800 |      200 |           20 |           2 | False               |
| KDDCup09_upselling                                  | binclass   |       0.811891 |   0.807018 |    -0.4873 |        -0.6   |      4102 |     1026 |           49 |           2 | False               |
| dna                                                 | multiclass |       0.965517 |   0.960815 |    -0.4702 |        -0.487 |      2548 |      638 |          180 |           3 | False               |
| splice                                              | multiclass |       0.956113 |   0.951411 |    -0.4702 |        -0.492 |      2552 |      638 |           60 |           3 | False               |
| autoUniv-au7-1100                                   | multiclass |       0.404545 |   0.4      |    -0.4545 |        -1.124 |       880 |      220 |           12 |           5 | False               |
| pc1                                                 | binclass   |       0.941441 |   0.936937 |    -0.4505 |        -0.478 |       887 |      222 |           21 |           2 | False               |
| in_vehicle_coupon_recommendation                    | binclass   |       0.790698 |   0.786756 |    -0.3942 |        -0.499 |     10147 |     2537 |           21 |           2 | False               |
| baseball                                            | multiclass |       0.947761 |   0.94403  |    -0.3731 |        -0.394 |      1072 |      268 |           16 |           3 | False               |
| telco-customer-churn                                | binclass   |       0.804826 |   0.801278 |    -0.3549 |        -0.441 |      5634 |     1409 |           18 |           2 | False               |
| predict_students_dropout_and_academic_success       | multiclass |       0.762712 |   0.759322 |    -0.339  |        -0.444 |      3539 |      885 |           34 |           3 | False               |
| ringnorm                                            | binclass   |       0.97973  |   0.976351 |    -0.3378 |        -0.345 |      5920 |     1480 |           20 |           2 | False               |
| yeast                                               | multiclass |       0.62963  |   0.626263 |    -0.3367 |        -0.535 |      1187 |      297 |            8 |          10 | False               |
| phoneme                                             | binclass   |       0.903793 |   0.901018 |    -0.2775 |        -0.307 |      4323 |     1081 |            5 |           2 | False               |
| mfeat-karhunen                                      | multiclass |       0.975    |   0.9725   |    -0.25   |        -0.256 |      1600 |      400 |           64 |          10 | False               |
| kc1                                                 | binclass   |       0.881517 |   0.879147 |    -0.237  |        -0.269 |      1687 |      422 |           21 |           2 | False               |
| estimation_of_obesity_levels                        | multiclass |       0.985816 |   0.983452 |    -0.2364 |        -0.24  |      1688 |      423 |           16 |           7 | False               |
| Wilt                                                | binclass   |       0.992746 |   0.990674 |    -0.2073 |        -0.209 |      3856 |      965 |            5 |           2 | False               |
| pol                                                 | binclass   |       0.983639 |   0.981656 |    -0.1983 |        -0.202 |      8065 |     2017 |           26 |           2 | False               |
| Satellite                                           | binclass   |       0.993137 |   0.991176 |    -0.1961 |        -0.197 |      4080 |     1020 |           36 |           2 | False               |
| mozilla4                                            | binclass   |       0.940174 |   0.938244 |    -0.193  |        -0.205 |     12436 |     3109 |            4 |           2 | False               |
| INNHotelsGroup                                      | binclass   |       0.902136 |   0.900482 |    -0.1654 |        -0.183 |     29020 |     7255 |           17 |           2 | False               |
| allrep                                              | multiclass |       0.98543  |   0.984106 |    -0.1325 |        -0.134 |      3017 |      755 |           29 |           4 | False               |
| rice_cammeo_and_osmancik                            | binclass   |       0.929134 |   0.927822 |    -0.1312 |        -0.141 |      3048 |      762 |            7 |           2 | False               |
| Gender_Gap_in_Spanish_WP                            | multiclass |       0.605263 |   0.604211 |    -0.1053 |        -0.174 |      3796 |      950 |           13 |           3 | False               |
| page-blocks                                         | multiclass |       0.978995 |   0.978082 |    -0.0913 |        -0.093 |      4378 |     1095 |           10 |           5 | False               |
| optdigits                                           | multiclass |       0.995552 |   0.994662 |    -0.089  |        -0.089 |      4496 |     1124 |           64 |          10 | False               |
| thyroid                                             | multiclass |       0.99375  |   0.993056 |    -0.0694 |        -0.07  |      5760 |     1440 |           21 |           3 | False               |
| PhishingWebsites                                    | binclass   |       0.979195 |   0.978743 |    -0.0452 |        -0.046 |      8844 |     2211 |           30 |           2 | False               |

## Tie Datasets

| dataset                                     | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:--------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| ASP-POTASSCO-classification                 | multiclass |       0.440154 |   0.440154 |          0 |             0 |      1035 |      259 |          141 |          11 | True                |
| BNG(cmc)                                    | multiclass |       0.586257 |   0.586257 |          0 |             0 |     44236 |    11060 |            9 |           3 | True                |
| Cardiovascular-Disease-dataset              | binclass   |       0.734143 |   0.734143 |          0 |             0 |     56000 |    14000 |           11 |           2 | True                |
| Credit_c                                    | multiclass |       0.792    |   0.792    |          0 |             0 |     80000 |    20000 |           22 |           3 | True                |
| FOREX_audcad-hour-High                      | binclass   |       0.712037 |   0.712037 |          0 |             0 |     35060 |     8765 |           10 |           2 | True                |
| FOREX_audjpy-hour-High                      | binclass   |       0.708614 |   0.708614 |          0 |             0 |     35060 |     8765 |           10 |           2 | True                |
| FOREX_audsgd-hour-High                      | binclass   |       0.683856 |   0.683856 |          0 |             0 |     35060 |     8765 |           10 |           2 | True                |
| FOREX_audusd-hour-High                      | binclass   |       0.681689 |   0.681689 |          0 |             0 |     35060 |     8765 |           10 |           2 | True                |
| FOREX_cadjpy-hour-High                      | binclass   |       0.700742 |   0.700742 |          0 |             0 |     35060 |     8765 |           10 |           2 | True                |
| Indian_pines                                | multiclass |       0.962274 |   0.962274 |          0 |             0 |      7315 |     1829 |          220 |           8 | True                |
| Rain_in_Australia                           | multiclass |       0.850371 |   0.850371 |          0 |             0 |    116368 |    29092 |           18 |           3 | True                |
| SDSS17                                      | multiclass |       0.9762   |   0.9762   |          0 |             0 |     80000 |    20000 |           12 |           3 | True                |
| UJI_Pen_Characters                          | multiclass |       0.542125 |   0.542125 |          0 |             0 |      1091 |      273 |           80 |          35 | True                |
| accelerometer                               | multiclass |       0.741054 |   0.741054 |          0 |             0 |    122403 |    30601 |            4 |           4 | True                |
| bank                                        | binclass   |       0.90899  |   0.90899  |          0 |             0 |     36168 |     9043 |           16 |           2 | True                |
| customer_satisfaction_in_airline            | binclass   |       0.961541 |   0.961541 |          0 |             0 |    103904 |    25976 |           21 |           2 | True                |
| dabetes_130-us_hospitals                    | binclass   |       0.641152 |   0.641152 |          0 |             0 |     81412 |    20354 |           20 |           2 | True                |
| electricity                                 | binclass   |       0.903785 |   0.903785 |          0 |             0 |     36249 |     9063 |            8 |           2 | True                |
| gina_agnostic                               | binclass   |       0.927954 |   0.927954 |          0 |             0 |      2774 |      694 |          970 |           2 | True                |
| internet_firewall                           | multiclass |       0.931029 |   0.931029 |          0 |             0 |     52425 |    13107 |            7 |           4 | True                |
| internet_usage                              | multiclass |       0.529674 |   0.529674 |          0 |             0 |      8086 |     2022 |           70 |          46 | True                |
| jungle_chess_2pcs_raw_endgame_complete      | multiclass |       0.86245  |   0.86245  |          0 |             0 |     35855 |     8964 |            6 |           3 | True                |
| kr-vs-k                                     | multiclass |       0.850855 |   0.850855 |          0 |             0 |     22444 |     5612 |            6 |          18 | True                |
| kropt                                       | multiclass |       0.852815 |   0.852815 |          0 |             0 |     22444 |     5612 |            6 |          18 | True                |
| letter                                      | multiclass |       0.98625  |   0.98625  |          0 |             0 |     16000 |     4000 |           15 |          26 | True                |
| mobile_c36_oversampling                     | binclass   |       0.990533 |   0.990533 |          0 |             0 |     41408 |    10352 |            6 |           2 | True                |
| naticusdroid+android+permissions+dataset    | binclass   |       0.971024 |   0.971024 |          0 |             0 |     23465 |     5867 |           86 |           2 | True                |
| one-hundred-plants-margin                   | multiclass |       0.909375 |   0.909375 |          0 |             0 |      1280 |      320 |           64 |         100 | True                |
| one-hundred-plants-shape                    | multiclass |       0.8      |   0.8      |          0 |             0 |      1280 |      320 |           64 |         100 | True                |
| one-hundred-plants-texture                  | multiclass |       0.91875  |   0.91875  |          0 |             0 |      1279 |      320 |           64 |         100 | True                |
| shuttle                                     | multiclass |       0.999483 |   0.999483 |          0 |             0 |     46400 |    11600 |            9 |           7 | True                |
| texture                                     | multiclass |       1        |   1        |          0 |             0 |      4400 |     1100 |           40 |          11 | True                |
| volkert                                     | multiclass |       0.733751 |   0.733751 |          0 |             0 |     46648 |    11662 |          180 |          10 | True                |
| walking-activity                            | multiclass |       0.664144 |   0.664144 |          0 |             0 |    119465 |    29867 |            4 |          22 | True                |
| GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1 | binclass   |       0.66875  |   0.66875  |          0 |             0 |      1280 |      320 |           20 |           2 | False               |
| JapaneseVowels                              | multiclass |       0.998996 |   0.998996 |          0 |             0 |      7968 |     1993 |           14 |           9 | False               |
| PizzaCutter3                                | binclass   |       0.880383 |   0.880383 |          0 |             0 |       834 |      209 |           37 |           2 | False               |
| Pumpkin_Seeds                               | binclass   |       0.872    |   0.872    |          0 |             0 |      2000 |      500 |           12 |           2 | False               |
| allbp                                       | multiclass |       0.978808 |   0.978808 |          0 |             0 |      3017 |      755 |           29 |           3 | False               |
| analcatdata_authorship                      | multiclass |       0.988166 |   0.988166 |          0 |             0 |       672 |      169 |           69 |           4 | False               |
| drug_consumption                            | multiclass |       0.403183 |   0.403183 |          0 |             0 |      1507 |      377 |           12 |           7 | False               |
| ibm-employee-performance                    | binclass   |       1        |   1        |          0 |             0 |      1176 |      294 |           30 |           2 | False               |
| mfeat-factors                               | multiclass |       0.975    |   0.975    |          0 |             0 |      1600 |      400 |          216 |          10 | False               |
| mice_protein_expression                     | multiclass |       1        |   1        |          0 |             0 |       864 |      216 |           75 |           8 | False               |
| national-longitudinal-survey-binary         | binclass   |       1        |   1        |          0 |             0 |      3926 |      982 |           16 |           2 | False               |
| pc4                                         | binclass   |       0.914384 |   0.914384 |          0 |             0 |      1166 |      292 |           37 |           2 | False               |
| pendigits                                   | multiclass |       0.997271 |   0.997271 |          0 |             0 |      8793 |     2199 |           16 |          10 | False               |
| segment                                     | multiclass |       0.935065 |   0.935065 |          0 |             0 |      1848 |      462 |           17 |           7 | False               |
| twonorm                                     | binclass   |       0.979054 |   0.979054 |          0 |             0 |      5920 |     1480 |           20 |           2 | False               |
| website_phishing                            | multiclass |       0.911439 |   0.911439 |          0 |             0 |      1082 |      271 |            9 |           3 | False               |
