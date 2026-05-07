# ensmble_ttt_step13_ckpt_test vs v1.1_baseline

Note: 18 failed datasets were filled with v1.1_baseline accuracy.

- baseline ok datasets: 178
- test ok datasets after fallback: 178
- baseline fallback datasets: 18
- baseline mean acc: 0.837755
- test mean acc: 0.840709
- mean delta: +0.2954 pp
- median delta: +0.0000 pp
- improved / regressed / tie: 80 / 52 / 46
- Pearson corr(baseline_acc, delta_pp): -0.0090
- Spearman corr(baseline_acc, delta_pp): -0.0179

## Baseline Accuracy Quartiles

| base_q     |   n |   base_min |   base_max |   base_mean |   test_mean |   delta_mean_pp |   delta_median_pp |   improved |   regressed |   tie |   fallback |
|:-----------|----:|-----------:|-----------:|------------:|------------:|----------------:|------------------:|-----------:|------------:|------:|-----------:|
| Q1 lowest  |  45 |   0.356    |   0.74481  |    0.638872 |    0.641283 |          0.2411 |            0      |         17 |          17 |    11 |          7 |
| Q2         |  44 |   0.745    |   0.872    |    0.816737 |    0.824199 |          0.7462 |            0.1481 |         27 |          10 |     7 |          5 |
| Q3         |  44 |   0.875    |   0.956113 |    0.914141 |    0.915718 |          0.1577 |            0      |         19 |          12 |    13 |          2 |
| Q4 highest |  45 |   0.956607 |   1        |    0.9825   |    0.982938 |          0.0438 |            0      |         17 |          13 |    15 |          4 |

## Improved Datasets

| dataset                                                        | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:---------------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| jungle_chess_2pcs_raw_endgame_complete                         | multiclass |       0.86245  |   0.946229 |     8.378  |         9.714 |     35855 |     8964 |            6 |           3 | False               |
| eye_movements_bin                                              | binclass   |       0.625493 |   0.682654 |     5.7162 |         9.139 |      6086 |     1522 |           20 |           2 | False               |
| compass                                                        | binclass   |       0.826675 |   0.883749 |     5.7074 |         6.904 |     13315 |     3329 |           17 |           2 | False               |
| mfeat-zernike                                                  | multiclass |       0.8525   |   0.905    |     5.25   |         6.158 |      1600 |      400 |           47 |          10 | False               |
| eye_movements                                                  | multiclass |       0.714808 |   0.763254 |     4.8446 |         6.777 |      8748 |     2188 |           27 |           3 | False               |
| autoUniv-au4-2500                                              | multiclass |       0.522    |   0.566    |     4.4    |         8.429 |      2000 |      500 |          100 |           3 | False               |
| hill-valley                                                    | binclass   |       0.930041 |   0.967078 |     3.7037 |         3.982 |       969 |      243 |          100 |           2 | False               |
| artificial-characters                                          | multiclass |       0.863992 |   0.900685 |     3.6693 |         4.247 |      8174 |     2044 |            7 |          10 | False               |
| mfeat-morphological                                            | multiclass |       0.76     |   0.7925   |     3.25   |         4.276 |      1600 |      400 |            6 |          10 | False               |
| FOREX_audsgd-hour-High                                         | binclass   |       0.683856 |   0.707587 |     2.3731 |         3.47  |     35060 |     8765 |           10 |           2 | False               |
| vehicle                                                        | multiclass |       0.858824 |   0.882353 |     2.3529 |         2.74  |       676 |      170 |           18 |           4 | False               |
| FOREX_audusd-hour-High                                         | binclass   |       0.681689 |   0.704392 |     2.2704 |         3.331 |     35060 |     8765 |           10 |           2 | False               |
| GesturePhaseSegmentationProcessed                              | multiclass |       0.786329 |   0.808101 |     2.1772 |         2.769 |      7898 |     1975 |           32 |           5 | False               |
| FOREX_audjpy-hour-High                                         | binclass   |       0.708614 |   0.723902 |     1.5288 |         2.157 |     35060 |     8765 |           10 |           2 | False               |
| website_phishing                                               | multiclass |       0.911439 |   0.926199 |     1.476  |         1.619 |      1082 |      271 |            9 |           3 | False               |
| car-evaluation                                                 | multiclass |       0.973988 |   0.988439 |     1.4451 |         1.484 |      1382 |      346 |           21 |           4 | False               |
| Fitness_Club_c                                                 | binclass   |       0.786667 |   0.8      |     1.3333 |         1.695 |      1200 |      300 |            6 |           2 | False               |
| Pima_Indians_Diabetes_Database                                 | binclass   |       0.746753 |   0.75974  |     1.2987 |         1.739 |       614 |      154 |            8 |           2 | False               |
| FOREX_cadjpy-hour-High                                         | binclass   |       0.700742 |   0.713177 |     1.2436 |         1.775 |     35060 |     8765 |           10 |           2 | False               |
| gina_agnostic                                                  | binclass   |       0.927954 |   0.939481 |     1.1527 |         1.242 |      2774 |      694 |          970 |           2 | False               |
| madeline                                                       | binclass   |       0.753185 |   0.764331 |     1.1146 |         1.48  |      2512 |      628 |          259 |           2 | False               |
| waveform-5000                                                  | multiclass |       0.866    |   0.877    |     1.1    |         1.27  |      4000 |     1000 |           40 |           3 | False               |
| maternal_health_risk                                           | multiclass |       0.871921 |   0.881773 |     0.9852 |         1.13  |       811 |      203 |            6 |           3 | False               |
| GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1                    | binclass   |       0.66875  |   0.678125 |     0.9375 |         1.402 |      1280 |      320 |           20 |           2 | False               |
| electricity                                                    | binclass   |       0.903785 |   0.911508 |     0.7724 |         0.855 |     36249 |     9063 |            8 |           2 | False               |
| steel_plates_faults                                            | multiclass |       0.838046 |   0.845758 |     0.7712 |         0.92  |      1552 |      389 |           27 |           7 | False               |
| Mobile_Price_Classification                                    | multiclass |       0.94     |   0.9475   |     0.75   |         0.798 |      1600 |      400 |           20 |           4 | False               |
| mfeat-fourier                                                  | multiclass |       0.885    |   0.8925   |     0.75   |         0.847 |      1600 |      400 |           76 |          10 | False               |
| ada_prior                                                      | binclass   |       0.841183 |   0.847755 |     0.6572 |         0.781 |      3649 |      913 |           14 |           2 | False               |
| thyroid-ann                                                    | multiclass |       0.988079 |   0.993377 |     0.5298 |         0.536 |      3017 |      755 |           21 |           3 | False               |
| rl                                                             | binclass   |       0.854125 |   0.859155 |     0.503  |         0.589 |      3976 |      994 |           12 |           2 | False               |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass   |       0.6025   |   0.6075   |     0.5    |         0.83  |      1600 |      400 |            7 |           2 | False               |
| Indian_pines                                                   | multiclass |       0.962274 |   0.967195 |     0.4921 |         0.511 |      7315 |     1829 |          220 |           8 | False               |
| MagicTelescope                                                 | binclass   |       0.884858 |   0.88959  |     0.4732 |         0.535 |     15216 |     3804 |            9 |           2 | False               |
| predict_students_dropout_and_academic_success                  | multiclass |       0.762712 |   0.767232 |     0.452  |         0.593 |      3539 |      885 |           34 |           3 | False               |
| pc1                                                            | binclass   |       0.941441 |   0.945946 |     0.4505 |         0.478 |       887 |      222 |           21 |           2 | False               |
| Gender_Gap_in_Spanish_WP                                       | multiclass |       0.605263 |   0.609474 |     0.4211 |         0.696 |      3796 |      950 |           13 |           3 | False               |
| waveform_database_generator_version_1                          | multiclass |       0.866    |   0.87     |     0.4    |         0.462 |      4000 |     1000 |           21 |           3 | False               |
| satimage                                                       | multiclass |       0.92846  |   0.932348 |     0.3888 |         0.419 |      5144 |     1286 |           36 |           6 | False               |
| jm1                                                            | binclass   |       0.821314 |   0.824989 |     0.3675 |         0.447 |      8708 |     2177 |           21 |           2 | False               |
| banknote_authentication                                        | binclass   |       0.549091 |   0.552727 |     0.3636 |         0.662 |      1097 |      275 |            4 |           2 | False               |
| BLE_RSSI_dataset_for_Indoor_localization                       | multiclass |       0.740611 |   0.744116 |     0.3505 |         0.473 |      7987 |     1997 |            3 |           3 | False               |
| IBM_HR_Analytics_Employee_Attrition_and_Performance            | binclass   |       0.863946 |   0.867347 |     0.3401 |         0.394 |      1176 |      294 |           31 |           2 | False               |
| yeast                                                          | multiclass |       0.62963  |   0.632997 |     0.3367 |         0.535 |      1187 |      297 |            8 |          10 | False               |
| water_quality                                                  | binclass   |       0.908125 |   0.91125  |     0.3125 |         0.344 |      6396 |     1600 |           20 |           2 | False               |
| dis                                                            | binclass   |       0.984106 |   0.986755 |     0.2649 |         0.269 |      3017 |      755 |           29 |           2 | False               |
| dry_bean_dataset                                               | multiclass |       0.927653 |   0.930224 |     0.2571 |         0.277 |     10888 |     2723 |           16 |           7 | False               |
| mfeat-pixel                                                    | multiclass |       0.975    |   0.9775   |     0.25   |         0.256 |      1600 |      400 |          240 |          10 | False               |
| mobile_c36_oversampling                                        | binclass   |       0.990533 |   0.992948 |     0.2415 |         0.244 |     41408 |    10352 |            6 |           2 | False               |
| estimation_of_obesity_levels                                   | multiclass |       0.985816 |   0.98818  |     0.2364 |         0.24  |      1688 |      423 |           16 |           7 | False               |
| eeg-eye-state                                                  | binclass   |       0.991322 |   0.993658 |     0.2336 |         0.236 |     11984 |     2996 |           14 |           2 | False               |
| FOREX_audcad-hour-High                                         | binclass   |       0.712037 |   0.714318 |     0.2282 |         0.32  |     35060 |     8765 |           10 |           2 | False               |
| segment                                                        | multiclass |       0.935065 |   0.937229 |     0.2165 |         0.231 |      1848 |      462 |           17 |           7 | False               |
| default_of_credit_card_clients                                 | binclass   |       0.826167 |   0.828167 |     0.2    |         0.242 |     24000 |     6000 |           23 |           2 | False               |
| Firm-Teacher_Clave-Direction_Classification                    | multiclass |       0.878241 |   0.880093 |     0.1852 |         0.211 |      8640 |     2160 |           16 |           4 | False               |
| Amazon_employee_access                                         | binclass   |       0.944309 |   0.94614  |     0.1831 |         0.194 |     26215 |     6554 |            7 |           2 | False               |
| HR_Analytics_Job_Change_of_Data_Scientists                     | binclass   |       0.799322 |   0.801148 |     0.1827 |         0.229 |     15326 |     3832 |           13 |           2 | False               |
| thyroid-dis                                                    | multiclass |       0.691071 |   0.692857 |     0.1786 |         0.258 |      2240 |      560 |           26 |           5 | False               |
| htru                                                           | binclass   |       0.97933  |   0.981006 |     0.1676 |         0.171 |     14318 |     3580 |            8 |           2 | False               |
| BNG(tic-tac-toe)                                               | binclass   |       0.815088 |   0.816739 |     0.1651 |         0.203 |     31492 |     7874 |            9 |           2 | False               |
| led7                                                           | multiclass |       0.7375   |   0.739062 |     0.1562 |         0.212 |      2560 |      640 |            7 |          10 | False               |
| Telecom_Churn_Dataset                                          | binclass   |       0.955022 |   0.956522 |     0.1499 |         0.157 |      2666 |      667 |           17 |           2 | False               |
| taiwanese_bankruptcy_prediction                                | binclass   |       0.970674 |   0.972141 |     0.1466 |         0.151 |      5455 |     1364 |           95 |           2 | False               |
| allbp                                                          | multiclass |       0.978808 |   0.980132 |     0.1325 |         0.135 |      3017 |      755 |           29 |           3 | False               |
| okcupid_stem                                                   | multiclass |       0.747564 |   0.748876 |     0.1312 |         0.175 |     21341 |     5336 |           13 |           3 | False               |
| online_shoppers                                                | binclass   |       0.904298 |   0.905515 |     0.1217 |         0.135 |      9864 |     2466 |           14 |           2 | False               |
| credit                                                         | binclass   |       0.78283  |   0.784026 |     0.1197 |         0.153 |     13371 |     3343 |           10 |           2 | False               |
| in_vehicle_coupon_recommendation                               | binclass   |       0.790698 |   0.79188  |     0.1182 |         0.15  |     10147 |     2537 |           21 |           2 | False               |
| sylvine                                                        | binclass   |       0.970732 |   0.971707 |     0.0976 |         0.101 |      4099 |     1025 |           20 |           2 | False               |
| kdd_ipums_la_97-small                                          | binclass   |       0.888247 |   0.88921  |     0.0963 |         0.108 |      4150 |     1038 |           20 |           2 | False               |
| optdigits                                                      | multiclass |       0.995552 |   0.996441 |     0.089  |         0.089 |      4496 |     1124 |           64 |          10 | False               |
| naticusdroid+android+permissions+dataset                       | binclass   |       0.971024 |   0.971877 |     0.0852 |         0.088 |     23465 |     5867 |           86 |           2 | False               |
| telco-customer-churn                                           | binclass   |       0.804826 |   0.805536 |     0.071  |         0.088 |      5634 |     1409 |           18 |           2 | False               |
| twonorm                                                        | binclass   |       0.979054 |   0.97973  |     0.0676 |         0.069 |      5920 |     1480 |           20 |           2 | False               |
| mozilla4                                                       | binclass   |       0.940174 |   0.940817 |     0.0643 |         0.068 |     12436 |     3109 |            4 |           2 | False               |
| FICO-HELOC-cleaned                                             | binclass   |       0.74481  |   0.745316 |     0.0506 |         0.068 |      7896 |     1975 |           23 |           2 | False               |
| INNHotelsGroup                                                 | binclass   |       0.902136 |   0.90255  |     0.0414 |         0.046 |     29020 |     7255 |           17 |           2 | False               |
| Click_prediction_small                                         | binclass   |       0.83204  |   0.832416 |     0.0375 |         0.045 |     31958 |     7990 |            3 |           2 | False               |
| gas-drift                                                      | multiclass |       0.996405 |   0.996765 |     0.0359 |         0.036 |     11128 |     2782 |          128 |           6 | False               |
| BNG(breast-w)                                                  | binclass   |       0.988316 |   0.98857  |     0.0254 |         0.026 |     31492 |     7874 |            9 |           2 | False               |

## Regressed Datasets

| dataset                                     | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:--------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| FOREX_audchf-day-High                       | binclass   |       0.754768 |   0.730245 |    -2.4523 |        -3.249 |      1466 |      367 |           10 |           2 | False               |
| waveform_database_generator                 | multiclass |       0.356    |   0.333    |    -2.3    |        -6.461 |      3999 |     1000 |           21 |           3 | False               |
| Water_Quality_and_Potability                | binclass   |       0.641768 |   0.618902 |    -2.2866 |        -3.563 |      2620 |      656 |            8 |           2 | False               |
| FOREX_audjpy-day-High                       | binclass   |       0.765668 |   0.743869 |    -2.1798 |        -2.847 |      1465 |      367 |           10 |           2 | False               |
| statlog                                     | binclass   |       0.745    |   0.725    |    -2      |        -2.685 |       800 |      200 |           20 |           2 | False               |
| Basketball_c                                | binclass   |       0.712687 |   0.69403  |    -1.8657 |        -2.618 |      1072 |      268 |           11 |           2 | False               |
| cmc                                         | multiclass |       0.589831 |   0.572881 |    -1.6949 |        -2.874 |      1178 |      295 |            9 |           3 | False               |
| golf_play_dataset_extended                  | binclass   |       0.936073 |   0.922374 |    -1.3699 |        -1.463 |       876 |      219 |            9 |           2 | False               |
| FOREX_cadjpy-day-High                       | binclass   |       0.716621 |   0.705722 |    -1.0899 |        -1.521 |      1467 |      367 |           10 |           2 | False               |
| abalone                                     | multiclass |       0.651914 |   0.641148 |    -1.0766 |        -1.651 |      3341 |      836 |            8 |           3 | False               |
| E-CommereShippingData                       | binclass   |       0.678182 |   0.668182 |    -1      |        -1.475 |      8799 |     2200 |           10 |           2 | False               |
| turiye_student_evaluation                   | multiclass |       0.522337 |   0.512887 |    -0.945  |        -1.809 |      4656 |     1164 |           32 |           5 | False               |
| FOREX_audcad-day-High                       | binclass   |       0.743869 |   0.735695 |    -0.8174 |        -1.099 |      1467 |      367 |           10 |           2 | False               |
| rice_cammeo_and_osmancik                    | binclass   |       0.929134 |   0.922572 |    -0.6562 |        -0.706 |      3048 |      762 |            7 |           2 | False               |
| dna                                         | multiclass |       0.965517 |   0.959248 |    -0.627  |        -0.649 |      2548 |      638 |          180 |           3 | False               |
| wine-quality-red                            | multiclass |       0.66875  |   0.6625   |    -0.625  |        -0.935 |      1279 |      320 |            4 |           6 | False               |
| MIC                                         | binclass   |       0.909091 |   0.90303  |    -0.6061 |        -0.667 |      1319 |      330 |          104 |           2 | False               |
| sports_articles_for_objectivity_analysis    | binclass   |       0.855    |   0.85     |    -0.5    |        -0.585 |       800 |      200 |           59 |           2 | False               |
| splice                                      | multiclass |       0.956113 |   0.951411 |    -0.4702 |        -0.492 |      2552 |      638 |           60 |           3 | False               |
| led24                                       | multiclass |       0.734375 |   0.729688 |    -0.4687 |        -0.638 |      2560 |      640 |           24 |          10 | False               |
| autoUniv-au7-1100                           | multiclass |       0.404545 |   0.4      |    -0.4545 |        -1.124 |       880 |      220 |           12 |           5 | False               |
| Bank_Customer_Churn_Dataset                 | binclass   |       0.877    |   0.8725   |    -0.45   |        -0.513 |      8000 |     2000 |           10 |           2 | False               |
| ozone-level-8hr                             | binclass   |       0.956607 |   0.952663 |    -0.3945 |        -0.412 |      2027 |      507 |           72 |           2 | False               |
| house_16H                                   | binclass   |       0.89066  |   0.887695 |    -0.2965 |        -0.333 |     10790 |     2698 |           16 |           2 | False               |
| mfeat-karhunen                              | multiclass |       0.975    |   0.9725   |    -0.25   |        -0.256 |      1600 |      400 |           64 |          10 | False               |
| pol                                         | binclass   |       0.983639 |   0.98116  |    -0.2479 |        -0.252 |      8065 |     2017 |           26 |           2 | False               |
| ada                                         | binclass   |       0.848193 |   0.845783 |    -0.241  |        -0.284 |      3317 |      830 |           48 |           2 | False               |
| Marketing_Campaign                          | binclass   |       0.890625 |   0.888393 |    -0.2232 |        -0.251 |      1792 |      448 |           27 |           2 | False               |
| Customer_Personality_Analysis               | binclass   |       0.883929 |   0.881696 |    -0.2232 |        -0.253 |      1792 |      448 |           24 |           2 | False               |
| National_Health_and_Nutrition_Health_Survey | binclass   |       0.839912 |   0.837719 |    -0.2193 |        -0.261 |      1822 |      456 |            7 |           2 | False               |
| spambase                                    | binclass   |       0.959826 |   0.957655 |    -0.2172 |        -0.226 |      3680 |      921 |           57 |           2 | False               |
| Employee                                    | binclass   |       0.852846 |   0.850698 |    -0.2148 |        -0.252 |      3722 |      931 |            8 |           2 | False               |
| Pumpkin_Seeds                               | binclass   |       0.872    |   0.87     |    -0.2    |        -0.229 |      2000 |      500 |           12 |           2 | False               |
| wine                                        | binclass   |       0.774951 |   0.772994 |    -0.1957 |        -0.253 |      2043 |      511 |            4 |           2 | False               |
| phoneme                                     | binclass   |       0.903793 |   0.901943 |    -0.185  |        -0.205 |      4323 |     1081 |            5 |           2 | False               |
| wall-robot-navigation                       | multiclass |       0.986264 |   0.984432 |    -0.1832 |        -0.186 |      4364 |     1092 |           24 |           4 | False               |
| first-order-theorem-proving                 | multiclass |       0.637255 |   0.635621 |    -0.1634 |        -0.256 |      4894 |     1224 |           51 |           6 | False               |
| heloc                                       | binclass   |       0.7255   |   0.724    |    -0.15   |        -0.207 |      8000 |     2000 |           22 |           2 | False               |
| ringnorm                                    | binclass   |       0.97973  |   0.978378 |    -0.1351 |        -0.138 |      5920 |     1480 |           20 |           2 | False               |
| allrep                                      | multiclass |       0.98543  |   0.984106 |    -0.1325 |        -0.134 |      3017 |      755 |           29 |           4 | False               |
| Wilt                                        | binclass   |       0.992746 |   0.99171  |    -0.1036 |        -0.104 |      3856 |      965 |            5 |           2 | False               |
| Satellite                                   | binclass   |       0.993137 |   0.992157 |    -0.098  |        -0.099 |      4080 |     1020 |           36 |           2 | False               |
| KDDCup09_upselling                          | binclass   |       0.811891 |   0.810916 |    -0.0975 |        -0.12  |      4102 |     1026 |           49 |           2 | False               |
| page-blocks                                 | multiclass |       0.978995 |   0.978082 |    -0.0913 |        -0.093 |      4378 |     1095 |           10 |           5 | False               |
| company_bankruptcy_prediction               | binclass   |       0.970674 |   0.969941 |    -0.0733 |        -0.076 |      5455 |     1364 |           95 |           2 | False               |
| Cardiovascular-Disease-dataset              | binclass   |       0.734143 |   0.733429 |    -0.0714 |        -0.097 |     56000 |    14000 |           11 |           2 | False               |
| delta_ailerons                              | binclass   |       0.950912 |   0.95021  |    -0.0701 |        -0.074 |      5703 |     1426 |            5 |           2 | False               |
| bank                                        | binclass   |       0.90899  |   0.908659 |    -0.0332 |        -0.036 |     36168 |     9043 |           16 |           2 | False               |
| microaggregation2                           | multiclass |       0.6405   |   0.64025  |    -0.025  |        -0.039 |     16000 |     4000 |           20 |           5 | False               |
| California-Housing-Classification           | binclass   |       0.912791 |   0.912548 |    -0.0242 |        -0.027 |     16512 |     4128 |            8 |           2 | False               |
| BNG(cmc)                                    | multiclass |       0.586257 |   0.586076 |    -0.0181 |        -0.031 |     44236 |    11060 |            9 |           3 | False               |
| shuttle                                     | multiclass |       0.999483 |   0.99931  |    -0.0172 |        -0.017 |     46400 |    11600 |            9 |           7 | False               |

## Tie Datasets

| dataset                                                    | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:-----------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| ASP-POTASSCO-classification                                | multiclass |       0.440154 |   0.440154 |          0 |             0 |      1035 |      259 |          141 |          11 | True                |
| Credit_c                                                   | multiclass |       0.792    |   0.792    |          0 |             0 |     80000 |    20000 |           22 |           3 | True                |
| Rain_in_Australia                                          | multiclass |       0.850371 |   0.850371 |          0 |             0 |    116368 |    29092 |           18 |           3 | True                |
| SDSS17                                                     | multiclass |       0.9762   |   0.9762   |          0 |             0 |     80000 |    20000 |           12 |           3 | True                |
| UJI_Pen_Characters                                         | multiclass |       0.542125 |   0.542125 |          0 |             0 |      1091 |      273 |           80 |          35 | True                |
| accelerometer                                              | multiclass |       0.741054 |   0.741054 |          0 |             0 |    122403 |    30601 |            4 |           4 | True                |
| customer_satisfaction_in_airline                           | binclass   |       0.961541 |   0.961541 |          0 |             0 |    103904 |    25976 |           21 |           2 | True                |
| dabetes_130-us_hospitals                                   | binclass   |       0.641152 |   0.641152 |          0 |             0 |     81412 |    20354 |           20 |           2 | True                |
| internet_usage                                             | multiclass |       0.529674 |   0.529674 |          0 |             0 |      8086 |     2022 |           70 |          46 | True                |
| kr-vs-k                                                    | multiclass |       0.850855 |   0.850855 |          0 |             0 |     22444 |     5612 |            6 |          18 | True                |
| kropt                                                      | multiclass |       0.852815 |   0.852815 |          0 |             0 |     22444 |     5612 |            6 |          18 | True                |
| letter                                                     | multiclass |       0.98625  |   0.98625  |          0 |             0 |     16000 |     4000 |           15 |          26 | True                |
| one-hundred-plants-margin                                  | multiclass |       0.909375 |   0.909375 |          0 |             0 |      1280 |      320 |           64 |         100 | True                |
| one-hundred-plants-shape                                   | multiclass |       0.8      |   0.8      |          0 |             0 |      1280 |      320 |           64 |         100 | True                |
| one-hundred-plants-texture                                 | multiclass |       0.91875  |   0.91875  |          0 |             0 |      1279 |      320 |           64 |         100 | True                |
| texture                                                    | multiclass |       1        |   1        |          0 |             0 |      4400 |     1100 |           40 |          11 | True                |
| volkert                                                    | multiclass |       0.733751 |   0.733751 |          0 |             0 |     46648 |    11662 |          180 |          10 | True                |
| walking-activity                                           | multiclass |       0.664144 |   0.664144 |          0 |             0 |    119465 |    29867 |            4 |          22 | True                |
| Diabetic_Retinopathy_Debrecen                              | binclass   |       0.748918 |   0.748918 |          0 |             0 |       920 |      231 |           19 |           2 | False               |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 | binclass   |       0.68125  |   0.68125  |          0 |             0 |      1280 |      320 |           20 |           2 | False               |
| JapaneseVowels                                             | multiclass |       0.998996 |   0.998996 |          0 |             0 |      7968 |     1993 |           14 |           9 | False               |
| PhishingWebsites                                           | binclass   |       0.979195 |   0.979195 |          0 |             0 |      8844 |     2211 |           30 |           2 | False               |
| PieChart3                                                  | binclass   |       0.875    |   0.875    |          0 |             0 |       861 |      216 |           37 |           2 | False               |
| PizzaCutter3                                               | binclass   |       0.880383 |   0.880383 |          0 |             0 |       834 |      209 |           37 |           2 | False               |
| QSAR_biodegradation                                        | binclass   |       0.886256 |   0.886256 |          0 |             0 |       843 |      211 |           41 |           2 | False               |
| ada_agnostic                                               | binclass   |       0.840088 |   0.840088 |          0 |             0 |      3649 |      913 |           48 |           2 | False               |
| analcatdata_authorship                                     | multiclass |       0.988166 |   0.988166 |          0 |             0 |       672 |      169 |           69 |           4 | False               |
| baseball                                                   | multiclass |       0.947761 |   0.947761 |          0 |             0 |      1072 |      268 |           16 |           3 | False               |
| churn                                                      | binclass   |       0.952    |   0.952    |          0 |             0 |      4000 |     1000 |           20 |           2 | False               |
| contraceptive_method_choice                                | multiclass |       0.627119 |   0.627119 |          0 |             0 |      1178 |      295 |            9 |           3 | False               |
| drug_consumption                                           | multiclass |       0.403183 |   0.403183 |          0 |             0 |      1507 |      377 |           12 |           7 | False               |
| ibm-employee-performance                                   | binclass   |       1        |   1        |          0 |             0 |      1176 |      294 |           30 |           2 | False               |
| internet_firewall                                          | multiclass |       0.931029 |   0.931029 |          0 |             0 |     52425 |    13107 |            7 |           4 | False               |
| kc1                                                        | binclass   |       0.881517 |   0.881517 |          0 |             0 |      1687 |      422 |           21 |           2 | False               |
| mammography                                                | binclass   |       0.988824 |   0.988824 |          0 |             0 |      8946 |     2237 |            6 |           2 | False               |
| mfeat-factors                                              | multiclass |       0.975    |   0.975    |          0 |             0 |      1600 |      400 |          216 |          10 | False               |
| mice_protein_expression                                    | multiclass |       1        |   1        |          0 |             0 |       864 |      216 |           75 |           8 | False               |
| national-longitudinal-survey-binary                        | binclass   |       1        |   1        |          0 |             0 |      3926 |      982 |           16 |           2 | False               |
| pc3                                                        | binclass   |       0.900958 |   0.900958 |          0 |             0 |      1250 |      313 |           37 |           2 | False               |
| pc4                                                        | binclass   |       0.914384 |   0.914384 |          0 |             0 |      1166 |      292 |           37 |           2 | False               |
| pendigits                                                  | multiclass |       0.997271 |   0.997271 |          0 |             0 |      8793 |     2199 |           16 |          10 | False               |
| qsar                                                       | binclass   |       0.905213 |   0.905213 |          0 |             0 |       844 |      211 |           40 |           2 | False               |
| seismic+bumps                                              | binclass   |       0.934236 |   0.934236 |          0 |             0 |      2067 |      517 |           18 |           2 | False               |
| semeion                                                    | multiclass |       0.959248 |   0.959248 |          0 |             0 |      1274 |      319 |          256 |          10 | False               |
| thyroid                                                    | multiclass |       0.99375  |   0.99375  |          0 |             0 |      5760 |     1440 |           21 |           3 | False               |
| wine-quality-white                                         | multiclass |       0.682653 |   0.682653 |          0 |             0 |      3918 |      980 |           11 |           7 | False               |
