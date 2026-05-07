# ensmble_ttt_step40_lr5e-6 vs v1.1_baseline

Note: 22 failed datasets were filled with v1.1_baseline accuracy.

- baseline ok datasets: 178
- test ok datasets after fallback: 178
- baseline fallback datasets: 22
- baseline mean acc: 0.837755
- test mean acc: 0.834620
- mean delta: -0.3135 pp
- median delta: +0.0000 pp
- improved / regressed / tie: 54 / 79 / 45
- Pearson corr(baseline_acc, delta_pp): 0.1416
- Spearman corr(baseline_acc, delta_pp): 0.0927

## Baseline Accuracy Quartiles

| base_q     |   n |   base_min |   base_max |   base_mean |   test_mean |   delta_mean_pp |   delta_median_pp |   improved |   regressed |   tie |   fallback |
|:-----------|----:|-----------:|-----------:|------------:|------------:|----------------:|------------------:|-----------:|------------:|------:|-----------:|
| Q1 lowest  |  45 |   0.356    |   0.74481  |    0.638872 |    0.631206 |         -0.7666 |            0      |         12 |          19 |    14 |          9 |
| Q2         |  44 |   0.745    |   0.872    |    0.816737 |    0.814826 |         -0.1911 |            0      |         16 |          21 |     7 |          3 |
| Q3         |  44 |   0.875    |   0.956113 |    0.914141 |    0.911632 |         -0.2509 |           -0.0484 |         14 |          22 |     8 |          4 |
| Q4 highest |  45 |   0.956607 |   1        |    0.9825   |    0.982087 |         -0.0413 |            0      |         12 |          17 |    16 |          6 |

## Improved Datasets

| dataset                                                    | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:-----------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| eye_movements                                              | multiclass |       0.714808 |   0.791133 |     7.6325 |        10.678 |      8748 |     2188 |           27 |           3 | False               |
| eye_movements_bin                                          | binclass   |       0.625493 |   0.697109 |     7.1616 |        11.45  |      6086 |     1522 |           20 |           2 | False               |
| compass                                                    | binclass   |       0.826675 |   0.891559 |     6.4884 |         7.849 |     13315 |     3329 |           17 |           2 | False               |
| mfeat-zernike                                              | multiclass |       0.8525   |   0.9125   |     6      |         7.038 |      1600 |      400 |           47 |          10 | False               |
| artificial-characters                                      | multiclass |       0.863992 |   0.908513 |     4.4521 |         5.153 |      8174 |     2044 |            7 |          10 | False               |
| vehicle                                                    | multiclass |       0.858824 |   0.9      |     4.1176 |         4.795 |       676 |      170 |           18 |           4 | False               |
| madeline                                                   | binclass   |       0.753185 |   0.791401 |     3.8217 |         5.074 |      2512 |      628 |          259 |           2 | False               |
| hill-valley                                                | binclass   |       0.930041 |   0.958848 |     2.8807 |         3.097 |       969 |      243 |          100 |           2 | False               |
| mfeat-fourier                                              | multiclass |       0.885    |   0.9125   |     2.75   |         3.107 |      1600 |      400 |           76 |          10 | False               |
| PieChart3                                                  | binclass   |       0.875    |   0.898148 |     2.3148 |         2.646 |       861 |      216 |           37 |           2 | False               |
| car-evaluation                                             | multiclass |       0.973988 |   0.99711  |     2.3121 |         2.374 |      1382 |      346 |           21 |           4 | False               |
| autoUniv-au4-2500                                          | multiclass |       0.522    |   0.544    |     2.2    |         4.215 |      2000 |      500 |          100 |           3 | False               |
| GesturePhaseSegmentationProcessed                          | multiclass |       0.786329 |   0.808101 |     2.1772 |         2.769 |      7898 |     1975 |           32 |           5 | False               |
| mfeat-morphological                                        | multiclass |       0.76     |   0.78     |     2      |         2.632 |      1600 |      400 |            6 |          10 | False               |
| steel_plates_faults                                        | multiclass |       0.838046 |   0.8509   |     1.2853 |         1.534 |      1552 |      389 |           27 |           7 | False               |
| BLE_RSSI_dataset_for_Indoor_localization                   | multiclass |       0.740611 |   0.752128 |     1.1517 |         1.555 |      7987 |     1997 |            3 |           3 | False               |
| QSAR_biodegradation                                        | binclass   |       0.886256 |   0.895735 |     0.9479 |         1.07  |       843 |      211 |           41 |           2 | False               |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 | binclass   |       0.68125  |   0.690625 |     0.9375 |         1.376 |      1280 |      320 |           20 |           2 | False               |
| thyroid-dis                                                | multiclass |       0.691071 |   0.7      |     0.8929 |         1.292 |      2240 |      560 |           26 |           5 | False               |
| waveform-5000                                              | multiclass |       0.866    |   0.874    |     0.8    |         0.924 |      4000 |     1000 |           40 |           3 | False               |
| dry_bean_dataset                                           | multiclass |       0.927653 |   0.933896 |     0.6243 |         0.673 |     10888 |     2723 |           16 |           7 | False               |
| Mobile_Price_Classification                                | multiclass |       0.94     |   0.945    |     0.5    |         0.532 |      1600 |      400 |           20 |           4 | False               |
| autoUniv-au7-1100                                          | multiclass |       0.404545 |   0.409091 |     0.4545 |         1.124 |       880 |      220 |           12 |           5 | False               |
| Telecom_Churn_Dataset                                      | binclass   |       0.955022 |   0.95952  |     0.4498 |         0.471 |      2666 |      667 |           17 |           2 | False               |
| sylvine                                                    | binclass   |       0.970732 |   0.974634 |     0.3902 |         0.402 |      4099 |     1025 |           20 |           2 | False               |
| baseball                                                   | multiclass |       0.947761 |   0.951493 |     0.3731 |         0.394 |      1072 |      268 |           16 |           3 | False               |
| banknote_authentication                                    | binclass   |       0.549091 |   0.552727 |     0.3636 |         0.662 |      1097 |      275 |            4 |           2 | False               |
| Fitness_Club_c                                             | binclass   |       0.786667 |   0.79     |     0.3333 |         0.424 |      1200 |      300 |            6 |           2 | False               |
| Gender_Gap_in_Spanish_WP                                   | multiclass |       0.605263 |   0.608421 |     0.3158 |         0.522 |      3796 |      950 |           13 |           3 | False               |
| led7                                                       | multiclass |       0.7375   |   0.740625 |     0.3125 |         0.424 |      2560 |      640 |            7 |          10 | False               |
| rl                                                         | binclass   |       0.854125 |   0.857143 |     0.3018 |         0.353 |      3976 |      994 |           12 |           2 | False               |
| okcupid_stem                                               | multiclass |       0.747564 |   0.750375 |     0.2811 |         0.376 |     21341 |     5336 |           13 |           3 | False               |
| credit                                                     | binclass   |       0.78283  |   0.785522 |     0.2692 |         0.344 |     13371 |     3343 |           10 |           2 | False               |
| eeg-eye-state                                              | binclass   |       0.991322 |   0.993992 |     0.267  |         0.269 |     11984 |     2996 |           14 |           2 | False               |
| mfeat-pixel                                                | multiclass |       0.975    |   0.9775   |     0.25   |         0.256 |      1600 |      400 |          240 |          10 | False               |
| online_shoppers                                            | binclass   |       0.904298 |   0.906732 |     0.2433 |         0.269 |      9864 |     2466 |           14 |           2 | False               |
| Customer_Personality_Analysis                              | binclass   |       0.883929 |   0.886161 |     0.2232 |         0.253 |      1792 |      448 |           24 |           2 | False               |
| segment                                                    | multiclass |       0.935065 |   0.937229 |     0.2165 |         0.231 |      1848 |      462 |           17 |           7 | False               |
| wine-quality-white                                         | multiclass |       0.682653 |   0.684694 |     0.2041 |         0.299 |      3918 |      980 |           11 |           7 | False               |
| satimage                                                   | multiclass |       0.92846  |   0.930016 |     0.1555 |         0.168 |      5144 |     1286 |           36 |           6 | False               |
| company_bankruptcy_prediction                              | binclass   |       0.970674 |   0.972141 |     0.1466 |         0.151 |      5455 |     1364 |           95 |           2 | False               |
| taiwanese_bankruptcy_prediction                            | binclass   |       0.970674 |   0.972141 |     0.1466 |         0.151 |      5455 |     1364 |           95 |           2 | False               |
| thyroid-ann                                                | multiclass |       0.988079 |   0.989404 |     0.1325 |         0.134 |      3017 |      755 |           21 |           3 | False               |
| dis                                                        | binclass   |       0.984106 |   0.98543  |     0.1325 |         0.135 |      3017 |      755 |           29 |           2 | False               |
| htru                                                       | binclass   |       0.97933  |   0.980447 |     0.1117 |         0.114 |     14318 |     3580 |            8 |           2 | False               |
| ada_agnostic                                               | binclass   |       0.840088 |   0.841183 |     0.1095 |         0.13  |      3649 |      913 |           48 |           2 | False               |
| FICO-HELOC-cleaned                                         | binclass   |       0.74481  |   0.745823 |     0.1013 |         0.136 |      7896 |     1975 |           23 |           2 | False               |
| Click_prediction_small                                     | binclass   |       0.83204  |   0.832916 |     0.0876 |         0.105 |     31958 |     7990 |            3 |           2 | False               |
| default_of_credit_card_clients                             | binclass   |       0.826167 |   0.827    |     0.0833 |         0.101 |     24000 |     6000 |           23 |           2 | False               |
| Amazon_employee_access                                     | binclass   |       0.944309 |   0.945072 |     0.0763 |         0.081 |     26215 |     6554 |            7 |           2 | False               |
| twonorm                                                    | binclass   |       0.979054 |   0.97973  |     0.0676 |         0.069 |      5920 |     1480 |           20 |           2 | False               |
| water_quality                                              | binclass   |       0.908125 |   0.90875  |     0.0625 |         0.069 |      6396 |     1600 |           20 |           2 | False               |
| BNG(breast-w)                                              | binclass   |       0.988316 |   0.988697 |     0.0381 |         0.039 |     31492 |     7874 |            9 |           2 | False               |
| gas-drift                                                  | multiclass |       0.996405 |   0.996765 |     0.0359 |         0.036 |     11128 |     2782 |          128 |           6 | False               |

## Regressed Datasets

| dataset                                                        | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:---------------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| Basketball_c                                                   | binclass   |       0.712687 |   0.623134 |    -8.9552 |       -12.565 |      1072 |      268 |           11 |           2 | False               |
| FOREX_audjpy-day-High                                          | binclass   |       0.765668 |   0.683924 |    -8.1744 |       -10.676 |      1465 |      367 |           10 |           2 | False               |
| Water_Quality_and_Potability                                   | binclass   |       0.641768 |   0.580793 |    -6.0976 |        -9.501 |      2620 |      656 |            8 |           2 | False               |
| abalone                                                        | multiclass |       0.651914 |   0.594498 |    -5.7416 |        -8.807 |      3341 |      836 |            8 |           3 | False               |
| FOREX_audchf-day-High                                          | binclass   |       0.754768 |   0.697548 |    -5.7221 |        -7.581 |      1466 |      367 |           10 |           2 | False               |
| contraceptive_method_choice                                    | multiclass |       0.627119 |   0.576271 |    -5.0847 |        -8.108 |      1178 |      295 |            9 |           3 | False               |
| FOREX_cadjpy-day-High                                          | binclass   |       0.716621 |   0.667575 |    -4.9046 |        -6.844 |      1467 |      367 |           10 |           2 | False               |
| Pima_Indians_Diabetes_Database                                 | binclass   |       0.746753 |   0.701299 |    -4.5455 |        -6.087 |       614 |      154 |            8 |           2 | False               |
| waveform_database_generator                                    | multiclass |       0.356    |   0.314    |    -4.2    |       -11.798 |      3999 |     1000 |           21 |           3 | False               |
| maternal_health_risk                                           | multiclass |       0.871921 |   0.832512 |    -3.9409 |        -4.52  |       811 |      203 |            6 |           3 | False               |
| MIC                                                            | binclass   |       0.909091 |   0.872727 |    -3.6364 |        -4     |      1319 |      330 |          104 |           2 | False               |
| FOREX_audcad-day-High                                          | binclass   |       0.743869 |   0.708447 |    -3.5422 |        -4.762 |      1467 |      367 |           10 |           2 | False               |
| sports_articles_for_objectivity_analysis                       | binclass   |       0.855    |   0.82     |    -3.5    |        -4.094 |       800 |      200 |           59 |           2 | False               |
| wine                                                           | binclass   |       0.774951 |   0.741683 |    -3.3268 |        -4.293 |      2043 |      511 |            4 |           2 | False               |
| E-CommereShippingData                                          | binclass   |       0.678182 |   0.645    |    -3.3182 |        -4.893 |      8799 |     2200 |           10 |           2 | False               |
| qsar                                                           | binclass   |       0.905213 |   0.872038 |    -3.3175 |        -3.665 |       844 |      211 |           40 |           2 | False               |
| wine-quality-red                                               | multiclass |       0.66875  |   0.6375   |    -3.125  |        -4.673 |      1279 |      320 |            4 |           6 | False               |
| cmc                                                            | multiclass |       0.589831 |   0.559322 |    -3.0508 |        -5.172 |      1178 |      295 |            9 |           3 | False               |
| Marketing_Campaign                                             | binclass   |       0.890625 |   0.870536 |    -2.0089 |        -2.256 |      1792 |      448 |           27 |           2 | False               |
| kc1                                                            | binclass   |       0.881517 |   0.862559 |    -1.8957 |        -2.151 |      1687 |      422 |           21 |           2 | False               |
| ada_prior                                                      | binclass   |       0.841183 |   0.822563 |    -1.862  |        -2.214 |      3649 |      913 |           14 |           2 | False               |
| golf_play_dataset_extended                                     | binclass   |       0.936073 |   0.917808 |    -1.8265 |        -1.951 |       876 |      219 |            9 |           2 | False               |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass   |       0.6025   |   0.585    |    -1.75   |        -2.905 |      1600 |      400 |            7 |           2 | False               |
| pc3                                                            | binclass   |       0.900958 |   0.884984 |    -1.5974 |        -1.773 |      1250 |      313 |           37 |           2 | False               |
| heloc                                                          | binclass   |       0.7255   |   0.71     |    -1.55   |        -2.136 |      8000 |     2000 |           22 |           2 | False               |
| seismic+bumps                                                  | binclass   |       0.934236 |   0.918762 |    -1.5474 |        -1.656 |      2067 |      517 |           18 |           2 | False               |
| National_Health_and_Nutrition_Health_Survey                    | binclass   |       0.839912 |   0.824561 |    -1.5351 |        -1.828 |      1822 |      456 |            7 |           2 | False               |
| statlog                                                        | binclass   |       0.745    |   0.73     |    -1.5    |        -2.013 |       800 |      200 |           20 |           2 | False               |
| ozone-level-8hr                                                | binclass   |       0.956607 |   0.942801 |    -1.3807 |        -1.443 |      2027 |      507 |           72 |           2 | False               |
| telco-customer-churn                                           | binclass   |       0.804826 |   0.791341 |    -1.3485 |        -1.675 |      5634 |     1409 |           18 |           2 | False               |
| yeast                                                          | multiclass |       0.62963  |   0.616162 |    -1.3468 |        -2.139 |      1187 |      297 |            8 |          10 | False               |
| delta_ailerons                                                 | binclass   |       0.950912 |   0.939691 |    -1.122  |        -1.18  |      5703 |     1426 |            5 |           2 | False               |
| house_16H                                                      | binclass   |       0.89066  |   0.880282 |    -1.0378 |        -1.165 |     10790 |     2698 |           16 |           2 | False               |
| turiye_student_evaluation                                      | multiclass |       0.522337 |   0.512027 |    -1.0309 |        -1.974 |      4656 |     1164 |           32 |           5 | False               |
| IBM_HR_Analytics_Employee_Attrition_and_Performance            | binclass   |       0.863946 |   0.853741 |    -1.0204 |        -1.181 |      1176 |      294 |           31 |           2 | False               |
| Employee                                                       | binclass   |       0.852846 |   0.843179 |    -0.9667 |        -1.134 |      3722 |      931 |            8 |           2 | False               |
| Bank_Customer_Churn_Dataset                                    | binclass   |       0.877    |   0.8675   |    -0.95   |        -1.083 |      8000 |     2000 |           10 |           2 | False               |
| predict_students_dropout_and_academic_success                  | multiclass |       0.762712 |   0.753672 |    -0.904  |        -1.185 |      3539 |      885 |           34 |           3 | False               |
| Diabetic_Retinopathy_Debrecen                                  | binclass   |       0.748918 |   0.74026  |    -0.8658 |        -1.156 |       920 |      231 |           19 |           2 | False               |
| drug_consumption                                               | multiclass |       0.403183 |   0.395225 |    -0.7958 |        -1.974 |      1507 |      377 |           12 |           7 | False               |
| dna                                                            | multiclass |       0.965517 |   0.95768  |    -0.7837 |        -0.812 |      2548 |      638 |          180 |           3 | False               |
| wall-robot-navigation                                          | multiclass |       0.986264 |   0.978938 |    -0.7326 |        -0.743 |      4364 |     1092 |           24 |           4 | False               |
| rice_cammeo_and_osmancik                                       | binclass   |       0.929134 |   0.922572 |    -0.6562 |        -0.706 |      3048 |      762 |            7 |           2 | False               |
| GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1                    | binclass   |       0.66875  |   0.6625   |    -0.625  |        -0.935 |      1280 |      320 |           20 |           2 | False               |
| led24                                                          | multiclass |       0.734375 |   0.728125 |    -0.625  |        -0.851 |      2560 |      640 |           24 |          10 | False               |
| in_vehicle_coupon_recommendation                               | binclass   |       0.790698 |   0.784785 |    -0.5912 |        -0.748 |     10147 |     2537 |           21 |           2 | False               |
| kdd_ipums_la_97-small                                          | binclass   |       0.888247 |   0.882466 |    -0.578  |        -0.651 |      4150 |     1038 |           20 |           2 | False               |
| spambase                                                       | binclass   |       0.959826 |   0.954397 |    -0.5429 |        -0.566 |      3680 |      921 |           57 |           2 | False               |
| splice                                                         | multiclass |       0.956113 |   0.951411 |    -0.4702 |        -0.492 |      2552 |      638 |           60 |           3 | False               |
| pc1                                                            | binclass   |       0.941441 |   0.936937 |    -0.4505 |        -0.478 |       887 |      222 |           21 |           2 | False               |
| first-order-theorem-proving                                    | multiclass |       0.637255 |   0.63317  |    -0.4085 |        -0.641 |      4894 |     1224 |           51 |           6 | False               |
| ringnorm                                                       | binclass   |       0.97973  |   0.975676 |    -0.4054 |        -0.414 |      5920 |     1480 |           20 |           2 | False               |
| website_phishing                                               | multiclass |       0.911439 |   0.907749 |    -0.369  |        -0.405 |      1082 |      271 |            9 |           3 | False               |
| MagicTelescope                                                 | binclass   |       0.884858 |   0.881441 |    -0.3417 |        -0.386 |     15216 |     3804 |            9 |           2 | False               |
| mozilla4                                                       | binclass   |       0.940174 |   0.936957 |    -0.3216 |        -0.342 |     12436 |     3109 |            4 |           2 | False               |
| waveform_database_generator_version_1                          | multiclass |       0.866    |   0.863    |    -0.3    |        -0.346 |      4000 |     1000 |           21 |           3 | False               |
| KDDCup09_upselling                                             | binclass   |       0.811891 |   0.808967 |    -0.2924 |        -0.36  |      4102 |     1026 |           49 |           2 | False               |
| page-blocks                                                    | multiclass |       0.978995 |   0.976256 |    -0.274  |        -0.28  |      4378 |     1095 |           10 |           5 | False               |
| allrep                                                         | multiclass |       0.98543  |   0.982781 |    -0.2649 |        -0.269 |      3017 |      755 |           29 |           4 | False               |
| mfeat-karhunen                                                 | multiclass |       0.975    |   0.9725   |    -0.25   |        -0.256 |      1600 |      400 |           64 |          10 | False               |
| estimation_of_obesity_levels                                   | multiclass |       0.985816 |   0.983452 |    -0.2364 |        -0.24  |      1688 |      423 |           16 |           7 | False               |
| Wilt                                                           | binclass   |       0.992746 |   0.990674 |    -0.2073 |        -0.209 |      3856 |      965 |            5 |           2 | False               |
| Pumpkin_Seeds                                                  | binclass   |       0.872    |   0.87     |    -0.2    |        -0.229 |      2000 |      500 |           12 |           2 | False               |
| Satellite                                                      | binclass   |       0.993137 |   0.991176 |    -0.1961 |        -0.197 |      4080 |     1020 |           36 |           2 | False               |
| Firm-Teacher_Clave-Direction_Classification                    | multiclass |       0.878241 |   0.876389 |    -0.1852 |        -0.211 |      8640 |     2160 |           16 |           4 | False               |
| phoneme                                                        | binclass   |       0.903793 |   0.901943 |    -0.185  |        -0.205 |      4323 |     1081 |            5 |           2 | False               |
| HR_Analytics_Job_Change_of_Data_Scientists                     | binclass   |       0.799322 |   0.797495 |    -0.1827 |        -0.229 |     15326 |     3832 |           13 |           2 | False               |
| optdigits                                                      | multiclass |       0.995552 |   0.993772 |    -0.1779 |        -0.179 |      4496 |     1124 |           64 |          10 | False               |
| INNHotelsGroup                                                 | binclass   |       0.902136 |   0.900482 |    -0.1654 |        -0.183 |     29020 |     7255 |           17 |           2 | False               |
| jm1                                                            | binclass   |       0.821314 |   0.819936 |    -0.1378 |        -0.168 |      8708 |     2177 |           21 |           2 | False               |
| allbp                                                          | multiclass |       0.978808 |   0.977483 |    -0.1325 |        -0.135 |      3017 |      755 |           29 |           3 | False               |
| BNG(tic-tac-toe)                                               | binclass   |       0.815088 |   0.814072 |    -0.1016 |        -0.125 |     31492 |     7874 |            9 |           2 | False               |
| churn                                                          | binclass   |       0.952    |   0.951    |    -0.1    |        -0.105 |      4000 |     1000 |           20 |           2 | False               |
| pol                                                            | binclass   |       0.983639 |   0.982647 |    -0.0992 |        -0.101 |      8065 |     2017 |           26 |           2 | False               |
| California-Housing-Classification                              | binclass   |       0.912791 |   0.911822 |    -0.0969 |        -0.106 |     16512 |     4128 |            8 |           2 | False               |
| PhishingWebsites                                               | binclass   |       0.979195 |   0.97829  |    -0.0905 |        -0.092 |      8844 |     2211 |           30 |           2 | False               |
| microaggregation2                                              | multiclass |       0.6405   |   0.63975  |    -0.075  |        -0.117 |     16000 |     4000 |           20 |           5 | False               |
| thyroid                                                        | multiclass |       0.99375  |   0.993056 |    -0.0694 |        -0.07  |      5760 |     1440 |           21 |           3 | False               |
| mammography                                                    | binclass   |       0.988824 |   0.988377 |    -0.0447 |        -0.045 |      8946 |     2237 |            6 |           2 | False               |

## Tie Datasets

| dataset                                  | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:-----------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| BNG(cmc)                                 | multiclass |       0.586257 |   0.586257 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| Cardiovascular-Disease-dataset           | binclass   |       0.734143 |   0.734143 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| Credit_c                                 | multiclass |       0.792    |   0.792    |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| FOREX_audcad-hour-High                   | binclass   |       0.712037 |   0.712037 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| FOREX_audjpy-hour-High                   | binclass   |       0.708614 |   0.708614 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| FOREX_audsgd-hour-High                   | binclass   |       0.683856 |   0.683856 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| FOREX_audusd-hour-High                   | binclass   |       0.681689 |   0.681689 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| FOREX_cadjpy-hour-High                   | binclass   |       0.700742 |   0.700742 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| Indian_pines                             | multiclass |       0.962274 |   0.962274 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| Rain_in_Australia                        | multiclass |       0.850371 |   0.850371 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| SDSS17                                   | multiclass |       0.9762   |   0.9762   |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| accelerometer                            | multiclass |       0.741054 |   0.741054 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| bank                                     | binclass   |       0.90899  |   0.90899  |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| customer_satisfaction_in_airline         | binclass   |       0.961541 |   0.961541 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| dabetes_130-us_hospitals                 | binclass   |       0.641152 |   0.641152 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| electricity                              | binclass   |       0.903785 |   0.903785 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| gina_agnostic                            | binclass   |       0.927954 |   0.927954 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| internet_firewall                        | multiclass |       0.931029 |   0.931029 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| jungle_chess_2pcs_raw_endgame_complete   | multiclass |       0.86245  |   0.86245  |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| mobile_c36_oversampling                  | binclass   |       0.990533 |   0.990533 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| naticusdroid+android+permissions+dataset | binclass   |       0.971024 |   0.971024 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| shuttle                                  | multiclass |       0.999483 |   0.999483 |          0 |             0 |         0 |        0 |            0 |           0 | True                |
| ASP-POTASSCO-classification              | multiclass |       0.440154 |   0.440154 |          0 |             0 |      1035 |      259 |          141 |          11 | False               |
| JapaneseVowels                           | multiclass |       0.998996 |   0.998996 |          0 |             0 |      7968 |     1993 |           14 |           9 | False               |
| PizzaCutter3                             | binclass   |       0.880383 |   0.880383 |          0 |             0 |       834 |      209 |           37 |           2 | False               |
| UJI_Pen_Characters                       | multiclass |       0.542125 |   0.542125 |          0 |             0 |      1091 |      273 |           80 |          35 | False               |
| ada                                      | binclass   |       0.848193 |   0.848193 |          0 |             0 |      3317 |      830 |           48 |           2 | False               |
| analcatdata_authorship                   | multiclass |       0.988166 |   0.988166 |          0 |             0 |       672 |      169 |           69 |           4 | False               |
| ibm-employee-performance                 | binclass   |       1        |   1        |          0 |             0 |      1176 |      294 |           30 |           2 | False               |
| internet_usage                           | multiclass |       0.529674 |   0.529674 |          0 |             0 |      8086 |     2022 |           70 |          46 | False               |
| kr-vs-k                                  | multiclass |       0.850855 |   0.850855 |          0 |             0 |     22444 |     5612 |            6 |          18 | False               |
| kropt                                    | multiclass |       0.852815 |   0.852815 |          0 |             0 |     22444 |     5612 |            6 |          18 | False               |
| letter                                   | multiclass |       0.98625  |   0.98625  |          0 |             0 |     16000 |     4000 |           15 |          26 | False               |
| mfeat-factors                            | multiclass |       0.975    |   0.975    |          0 |             0 |      1600 |      400 |          216 |          10 | False               |
| mice_protein_expression                  | multiclass |       1        |   1        |          0 |             0 |       864 |      216 |           75 |           8 | False               |
| national-longitudinal-survey-binary      | binclass   |       1        |   1        |          0 |             0 |      3926 |      982 |           16 |           2 | False               |
| one-hundred-plants-margin                | multiclass |       0.909375 |   0.909375 |          0 |             0 |      1280 |      320 |           64 |         100 | False               |
| one-hundred-plants-shape                 | multiclass |       0.8      |   0.8      |          0 |             0 |      1280 |      320 |           64 |         100 | False               |
| one-hundred-plants-texture               | multiclass |       0.91875  |   0.91875  |          0 |             0 |      1279 |      320 |           64 |         100 | False               |
| pc4                                      | binclass   |       0.914384 |   0.914384 |          0 |             0 |      1166 |      292 |           37 |           2 | False               |
| pendigits                                | multiclass |       0.997271 |   0.997271 |          0 |             0 |      8793 |     2199 |           16 |          10 | False               |
| semeion                                  | multiclass |       0.959248 |   0.959248 |          0 |             0 |      1274 |      319 |          256 |          10 | False               |
| texture                                  | multiclass |       1        |   1        |          0 |             0 |      4400 |     1100 |           40 |          11 | False               |
| volkert                                  | multiclass |       0.733751 |   0.733751 |          0 |             0 |     46648 |    11662 |          180 |          10 | False               |
| walking-activity                         | multiclass |       0.664144 |   0.664144 |          0 |             0 |    119465 |    29867 |            4 |          22 | False               |
