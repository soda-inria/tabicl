# ensmble_ttt_step8_ckpt_test vs v1.1_baseline

Note: 18 failed datasets were filled with v1.1_baseline accuracy.

- baseline ok datasets: 178
- test ok datasets after fallback: 178
- baseline fallback datasets: 18
- baseline mean acc: 0.837755
- test mean acc: 0.841143
- mean delta: +0.3388 pp
- median delta: +0.0000 pp
- improved / regressed / tie: 88 / 47 / 43
- Pearson corr(baseline_acc, delta_pp): -0.1336
- Spearman corr(baseline_acc, delta_pp): -0.1365

## Baseline Accuracy Quartiles

| base_q     |   n |   base_min |   base_max |   base_mean |   test_mean |   delta_mean_pp |   delta_median_pp |   improved |   regressed |   tie |   fallback |
|:-----------|----:|-----------:|-----------:|------------:|------------:|----------------:|------------------:|-----------:|------------:|------:|-----------:|
| Q1 lowest  |  45 |   0.356    |   0.74481  |    0.638872 |    0.643719 |          0.4846 |            0      |         21 |          13 |    11 |          7 |
| Q2         |  44 |   0.745    |   0.872    |    0.816737 |    0.823378 |          0.6641 |            0.1889 |         29 |           6 |     9 |          5 |
| Q3         |  44 |   0.875    |   0.956113 |    0.914141 |    0.915811 |          0.167  |            0.0121 |         22 |          13 |     9 |          2 |
| Q4 highest |  45 |   0.956607 |   1        |    0.9825   |    0.982929 |          0.0429 |            0      |         16 |          15 |    14 |          4 |

## Improved Datasets

| dataset                                                        | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:---------------------------------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| jungle_chess_2pcs_raw_endgame_complete                         | multiclass |       0.86245  |   0.917448 |     5.4998 |         6.377 |     35855 |     8964 |            6 |           3 | False               |
| mfeat-zernike                                                  | multiclass |       0.8525   |   0.9025   |     5      |         5.865 |      1600 |      400 |           47 |          10 | False               |
| compass                                                        | binclass   |       0.826675 |   0.875939 |     4.9264 |         5.959 |     13315 |     3329 |           17 |           2 | False               |
| eye_movements_bin                                              | binclass   |       0.625493 |   0.670828 |     4.5335 |         7.248 |      6086 |     1522 |           20 |           2 | False               |
| eye_movements                                                  | multiclass |       0.714808 |   0.752742 |     3.7934 |         5.307 |      8748 |     2188 |           27 |           3 | False               |
| hill-valley                                                    | binclass   |       0.930041 |   0.967078 |     3.7037 |         3.982 |       969 |      243 |          100 |           2 | False               |
| autoUniv-au4-2500                                              | multiclass |       0.522    |   0.558    |     3.6    |         6.897 |      2000 |      500 |          100 |           3 | False               |
| Pima_Indians_Diabetes_Database                                 | binclass   |       0.746753 |   0.779221 |     3.2468 |         4.348 |       614 |      154 |            8 |           2 | False               |
| artificial-characters                                          | multiclass |       0.863992 |   0.895303 |     3.1311 |         3.624 |      8174 |     2044 |            7 |          10 | False               |
| vehicle                                                        | multiclass |       0.858824 |   0.882353 |     2.3529 |         2.74  |       676 |      170 |           18 |           4 | False               |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass   |       0.6025   |   0.625    |     2.25   |         3.734 |      1600 |      400 |            7 |           2 | False               |
| FOREX_audusd-hour-High                                         | binclass   |       0.681689 |   0.703137 |     2.1449 |         3.146 |     35060 |     8765 |           10 |           2 | False               |
| FOREX_audsgd-hour-High                                         | binclass   |       0.683856 |   0.704392 |     2.0536 |         3.003 |     35060 |     8765 |           10 |           2 | False               |
| contraceptive_method_choice                                    | multiclass |       0.627119 |   0.647458 |     2.0339 |         3.243 |      1178 |      295 |            9 |           3 | False               |
| GesturePhaseSegmentationProcessed                              | multiclass |       0.786329 |   0.804557 |     1.8228 |         2.318 |      7898 |     1975 |           32 |           5 | False               |
| FOREX_audjpy-hour-High                                         | binclass   |       0.708614 |   0.724701 |     1.6087 |         2.27  |     35060 |     8765 |           10 |           2 | False               |
| website_phishing                                               | multiclass |       0.911439 |   0.926199 |     1.476  |         1.619 |      1082 |      271 |            9 |           3 | False               |
| car-evaluation                                                 | multiclass |       0.973988 |   0.988439 |     1.4451 |         1.484 |      1382 |      346 |           21 |           4 | False               |
| FOREX_cadjpy-hour-High                                         | binclass   |       0.700742 |   0.714889 |     1.4147 |         2.019 |     35060 |     8765 |           10 |           2 | False               |
| autoUniv-au7-1100                                              | multiclass |       0.404545 |   0.418182 |     1.3636 |         3.371 |       880 |      220 |           12 |           5 | False               |
| FOREX_audcad-day-High                                          | binclass   |       0.743869 |   0.757493 |     1.3624 |         1.832 |      1467 |      367 |           10 |           2 | False               |
| Fitness_Club_c                                                 | binclass   |       0.786667 |   0.8      |     1.3333 |         1.695 |      1200 |      300 |            6 |           2 | False               |
| mfeat-fourier                                                  | multiclass |       0.885    |   0.8975   |     1.25   |         1.412 |      1600 |      400 |           76 |          10 | False               |
| waveform-5000                                                  | multiclass |       0.866    |   0.876    |     1      |         1.155 |      4000 |     1000 |           40 |           3 | False               |
| mfeat-morphological                                            | multiclass |       0.76     |   0.77     |     1      |         1.316 |      1600 |      400 |            6 |          10 | False               |
| GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1                    | binclass   |       0.66875  |   0.678125 |     0.9375 |         1.402 |      1280 |      320 |           20 |           2 | False               |
| predict_students_dropout_and_academic_success                  | multiclass |       0.762712 |   0.770621 |     0.791  |         1.037 |      3539 |      885 |           34 |           3 | False               |
| ada_prior                                                      | binclass   |       0.841183 |   0.84885  |     0.7667 |         0.911 |      3649 |      913 |           14 |           2 | False               |
| Mobile_Price_Classification                                    | multiclass |       0.94     |   0.9475   |     0.75   |         0.798 |      1600 |      400 |           20 |           4 | False               |
| banknote_authentication                                        | binclass   |       0.549091 |   0.556364 |     0.7273 |         1.325 |      1097 |      275 |            4 |           2 | False               |
| satimage                                                       | multiclass |       0.92846  |   0.935459 |     0.6998 |         0.754 |      5144 |     1286 |           36 |           6 | False               |
| madeline                                                       | binclass   |       0.753185 |   0.759554 |     0.6369 |         0.846 |      2512 |      628 |          259 |           2 | False               |
| led7                                                           | multiclass |       0.7375   |   0.74375  |     0.625  |         0.847 |      2560 |      640 |            7 |          10 | False               |
| thyroid-ann                                                    | multiclass |       0.988079 |   0.993377 |     0.5298 |         0.536 |      3017 |      755 |           21 |           3 | False               |
| electricity                                                    | binclass   |       0.903785 |   0.908529 |     0.4745 |         0.525 |     36249 |     9063 |            8 |           2 | False               |
| pc1                                                            | binclass   |       0.941441 |   0.945946 |     0.4505 |         0.478 |       887 |      222 |           21 |           2 | False               |
| Marketing_Campaign                                             | binclass   |       0.890625 |   0.895089 |     0.4464 |         0.501 |      1792 |      448 |           27 |           2 | False               |
| segment                                                        | multiclass |       0.935065 |   0.939394 |     0.4329 |         0.463 |      1848 |      462 |           17 |           7 | False               |
| gina_agnostic                                                  | binclass   |       0.927954 |   0.932277 |     0.4323 |         0.466 |      2774 |      694 |          970 |           2 | False               |
| telco-customer-churn                                           | binclass   |       0.804826 |   0.809084 |     0.4258 |         0.529 |      5634 |     1409 |           18 |           2 | False               |
| allbp                                                          | multiclass |       0.978808 |   0.982781 |     0.3974 |         0.406 |      3017 |      755 |           29 |           3 | False               |
| baseball                                                       | multiclass |       0.947761 |   0.951493 |     0.3731 |         0.394 |      1072 |      268 |           16 |           3 | False               |
| pc4                                                            | binclass   |       0.914384 |   0.917808 |     0.3425 |         0.375 |      1166 |      292 |           37 |           2 | False               |
| MagicTelescope                                                 | binclass   |       0.884858 |   0.888275 |     0.3417 |         0.386 |     15216 |     3804 |            9 |           2 | False               |
| IBM_HR_Analytics_Employee_Attrition_and_Performance            | binclass   |       0.863946 |   0.867347 |     0.3401 |         0.394 |      1176 |      294 |           31 |           2 | False               |
| yeast                                                          | multiclass |       0.62963  |   0.632997 |     0.3367 |         0.535 |      1187 |      297 |            8 |          10 | False               |
| semeion                                                        | multiclass |       0.959248 |   0.962382 |     0.3135 |         0.327 |      1274 |      319 |          256 |          10 | False               |
| FOREX_audcad-hour-High                                         | binclass   |       0.712037 |   0.715117 |     0.308  |         0.433 |     35060 |     8765 |           10 |           2 | False               |
| waveform_database_generator_version_1                          | multiclass |       0.866    |   0.869    |     0.3    |         0.346 |      4000 |     1000 |           21 |           3 | False               |
| default_of_credit_card_clients                                 | binclass   |       0.826167 |   0.829    |     0.2833 |         0.343 |     24000 |     6000 |           23 |           2 | False               |
| Indian_pines                                                   | multiclass |       0.962274 |   0.965008 |     0.2734 |         0.284 |      7315 |     1829 |          220 |           8 | False               |
| steel_plates_faults                                            | multiclass |       0.838046 |   0.840617 |     0.2571 |         0.307 |      1552 |      389 |           27 |           7 | False               |
| heloc                                                          | binclass   |       0.7255   |   0.728    |     0.25   |         0.345 |      8000 |     2000 |           22 |           2 | False               |
| kc1                                                            | binclass   |       0.881517 |   0.883886 |     0.237  |         0.269 |      1687 |      422 |           21 |           2 | False               |
| mobile_c36_oversampling                                        | binclass   |       0.990533 |   0.992755 |     0.2222 |         0.224 |     41408 |    10352 |            6 |           2 | False               |
| National_Health_and_Nutrition_Health_Survey                    | binclass   |       0.839912 |   0.842105 |     0.2193 |         0.261 |      1822 |      456 |            7 |           2 | False               |
| Employee                                                       | binclass   |       0.852846 |   0.854995 |     0.2148 |         0.252 |      3722 |      931 |            8 |           2 | False               |
| credit                                                         | binclass   |       0.78283  |   0.784924 |     0.2094 |         0.267 |     13371 |     3343 |           10 |           2 | False               |
| Pumpkin_Seeds                                                  | binclass   |       0.872    |   0.874    |     0.2    |         0.229 |      2000 |      500 |           12 |           2 | False               |
| thyroid-dis                                                    | multiclass |       0.691071 |   0.692857 |     0.1786 |         0.258 |      2240 |      560 |           26 |           5 | False               |
| BNG(tic-tac-toe)                                               | binclass   |       0.815088 |   0.816866 |     0.1778 |         0.218 |     31492 |     7874 |            9 |           2 | False               |
| htru                                                           | binclass   |       0.97933  |   0.981006 |     0.1676 |         0.171 |     14318 |     3580 |            8 |           2 | False               |
| online_shoppers                                                | binclass   |       0.904298 |   0.905921 |     0.1622 |         0.179 |      9864 |     2466 |           14 |           2 | False               |
| naticusdroid+android+permissions+dataset                       | binclass   |       0.971024 |   0.972558 |     0.1534 |         0.158 |     23465 |     5867 |           86 |           2 | False               |
| BLE_RSSI_dataset_for_Indoor_localization                       | multiclass |       0.740611 |   0.742113 |     0.1502 |         0.203 |      7987 |     1997 |            3 |           3 | False               |
| dry_bean_dataset                                               | multiclass |       0.927653 |   0.929122 |     0.1469 |         0.158 |     10888 |     2723 |           16 |           7 | False               |
| taiwanese_bankruptcy_prediction                                | binclass   |       0.970674 |   0.972141 |     0.1466 |         0.151 |      5455 |     1364 |           95 |           2 | False               |
| eeg-eye-state                                                  | binclass   |       0.991322 |   0.992657 |     0.1335 |         0.135 |     11984 |     2996 |           14 |           2 | False               |
| dis                                                            | binclass   |       0.984106 |   0.98543  |     0.1325 |         0.135 |      3017 |      755 |           29 |           2 | False               |
| Gender_Gap_in_Spanish_WP                                       | multiclass |       0.605263 |   0.606316 |     0.1053 |         0.174 |      3796 |      950 |           13 |           3 | False               |
| HR_Analytics_Job_Change_of_Data_Scientists                     | binclass   |       0.799322 |   0.800365 |     0.1044 |         0.131 |     15326 |     3832 |           13 |           2 | False               |
| wine-quality-white                                             | multiclass |       0.682653 |   0.683673 |     0.102  |         0.149 |      3918 |      980 |           11 |           7 | False               |
| rl                                                             | binclass   |       0.854125 |   0.855131 |     0.1006 |         0.118 |      3976 |      994 |           12 |           2 | False               |
| okcupid_stem                                                   | multiclass |       0.747564 |   0.748501 |     0.0937 |         0.125 |     21341 |     5336 |           13 |           3 | False               |
| Firm-Teacher_Clave-Direction_Classification                    | multiclass |       0.878241 |   0.879167 |     0.0926 |         0.105 |      8640 |     2160 |           16 |           4 | False               |
| phoneme                                                        | binclass   |       0.903793 |   0.904718 |     0.0925 |         0.102 |      4323 |     1081 |            5 |           2 | False               |
| jm1                                                            | binclass   |       0.821314 |   0.822232 |     0.0919 |         0.112 |      8708 |     2177 |           21 |           2 | False               |
| Amazon_employee_access                                         | binclass   |       0.944309 |   0.945224 |     0.0915 |         0.097 |     26215 |     6554 |            7 |           2 | False               |
| optdigits                                                      | multiclass |       0.995552 |   0.996441 |     0.089  |         0.089 |      4496 |     1124 |           64 |          10 | False               |
| gas-drift                                                      | multiclass |       0.996405 |   0.997124 |     0.0719 |         0.072 |     11128 |     2782 |          128 |           6 | False               |
| water_quality                                                  | binclass   |       0.908125 |   0.90875  |     0.0625 |         0.069 |      6396 |     1600 |           20 |           2 | False               |
| PhishingWebsites                                               | binclass   |       0.979195 |   0.979647 |     0.0452 |         0.046 |      8844 |     2211 |           30 |           2 | False               |
| mammography                                                    | binclass   |       0.988824 |   0.989271 |     0.0447 |         0.045 |      8946 |     2237 |            6 |           2 | False               |
| in_vehicle_coupon_recommendation                               | binclass   |       0.790698 |   0.791092 |     0.0394 |         0.05  |     10147 |     2537 |           21 |           2 | False               |
| INNHotelsGroup                                                 | binclass   |       0.902136 |   0.902412 |     0.0276 |         0.031 |     29020 |     7255 |           17 |           2 | False               |
| BNG(breast-w)                                                  | binclass   |       0.988316 |   0.98857  |     0.0254 |         0.026 |     31492 |     7874 |            9 |           2 | False               |
| Click_prediction_small                                         | binclass   |       0.83204  |   0.83229  |     0.025  |         0.03  |     31958 |     7990 |            3 |           2 | False               |
| California-Housing-Classification                              | binclass   |       0.912791 |   0.913033 |     0.0242 |         0.027 |     16512 |     4128 |            8 |           2 | False               |

## Regressed Datasets

| dataset                                  | task       |   baseline_acc |   test_acc |   delta_pp |   rel_delta_% |   n_train |   n_test |   n_features |   n_classes | baseline_fallback   |
|:-----------------------------------------|:-----------|---------------:|-----------:|-----------:|--------------:|----------:|---------:|-------------:|------------:|:--------------------|
| Basketball_c                             | binclass   |       0.712687 |   0.69403  |    -1.8657 |        -2.618 |      1072 |      268 |           11 |           2 | False               |
| FOREX_audchf-day-High                    | binclass   |       0.754768 |   0.73842  |    -1.6349 |        -2.166 |      1466 |      367 |           10 |           2 | False               |
| wine-quality-red                         | multiclass |       0.66875  |   0.65625  |    -1.25   |        -1.869 |      1279 |      320 |            4 |           6 | False               |
| Water_Quality_and_Potability             | binclass   |       0.641768 |   0.629573 |    -1.2195 |        -1.9   |      2620 |      656 |            8 |           2 | False               |
| MIC                                      | binclass   |       0.909091 |   0.89697  |    -1.2121 |        -1.333 |      1319 |      330 |          104 |           2 | False               |
| statlog                                  | binclass   |       0.745    |   0.735    |    -1      |        -1.342 |       800 |      200 |           20 |           2 | False               |
| sports_articles_for_objectivity_analysis | binclass   |       0.855    |   0.845    |    -1      |        -1.17  |       800 |      200 |           59 |           2 | False               |
| QSAR_biodegradation                      | binclass   |       0.886256 |   0.876777 |    -0.9479 |        -1.07  |       843 |      211 |           41 |           2 | False               |
| FOREX_audjpy-day-High                    | binclass   |       0.765668 |   0.757493 |    -0.8174 |        -1.068 |      1465 |      367 |           10 |           2 | False               |
| first-order-theorem-proving              | multiclass |       0.637255 |   0.629902 |    -0.7353 |        -1.154 |      4894 |     1224 |           51 |           6 | False               |
| ada                                      | binclass   |       0.848193 |   0.840964 |    -0.7229 |        -0.852 |      3317 |      830 |           48 |           2 | False               |
| abalone                                  | multiclass |       0.651914 |   0.644737 |    -0.7177 |        -1.101 |      3341 |      836 |            8 |           3 | False               |
| turiye_student_evaluation                | multiclass |       0.522337 |   0.515464 |    -0.6873 |        -1.316 |      4656 |     1164 |           32 |           5 | False               |
| rice_cammeo_and_osmancik                 | binclass   |       0.929134 |   0.922572 |    -0.6562 |        -0.706 |      3048 |      762 |            7 |           2 | False               |
| led24                                    | multiclass |       0.734375 |   0.729688 |    -0.4687 |        -0.638 |      2560 |      640 |           24 |          10 | False               |
| golf_play_dataset_extended               | binclass   |       0.936073 |   0.931507 |    -0.4566 |        -0.488 |       876 |      219 |            9 |           2 | False               |
| microaggregation2                        | multiclass |       0.6405   |   0.637    |    -0.35   |        -0.546 |     16000 |     4000 |           20 |           5 | False               |
| cmc                                      | multiclass |       0.589831 |   0.586441 |    -0.339  |        -0.575 |      1178 |      295 |            9 |           3 | False               |
| ringnorm                                 | binclass   |       0.97973  |   0.976351 |    -0.3378 |        -0.345 |      5920 |     1480 |           20 |           2 | False               |
| dna                                      | multiclass |       0.965517 |   0.962382 |    -0.3135 |        -0.325 |      2548 |      638 |          180 |           3 | False               |
| splice                                   | multiclass |       0.956113 |   0.952978 |    -0.3135 |        -0.328 |      2552 |      638 |           60 |           3 | False               |
| mfeat-karhunen                           | multiclass |       0.975    |   0.9725   |    -0.25   |        -0.256 |      1600 |      400 |           64 |          10 | False               |
| pol                                      | binclass   |       0.983639 |   0.98116  |    -0.2479 |        -0.252 |      8065 |     2017 |           26 |           2 | False               |
| Customer_Personality_Analysis            | binclass   |       0.883929 |   0.881696 |    -0.2232 |        -0.253 |      1792 |      448 |           24 |           2 | False               |
| churn                                    | binclass   |       0.952    |   0.95     |    -0.2    |        -0.21  |      4000 |     1000 |           20 |           2 | False               |
| ozone-level-8hr                          | binclass   |       0.956607 |   0.954635 |    -0.1972 |        -0.206 |      2027 |      507 |           72 |           2 | False               |
| wine                                     | binclass   |       0.774951 |   0.772994 |    -0.1957 |        -0.253 |      2043 |      511 |            4 |           2 | False               |
| house_16H                                | binclass   |       0.89066  |   0.888807 |    -0.1853 |        -0.208 |     10790 |     2698 |           16 |           2 | False               |
| E-CommereShippingData                    | binclass   |       0.678182 |   0.676364 |    -0.1818 |        -0.268 |      8799 |     2200 |           10 |           2 | False               |
| Bank_Customer_Churn_Dataset              | binclass   |       0.877    |   0.8755   |    -0.15   |        -0.171 |      8000 |     2000 |           10 |           2 | False               |
| Telecom_Churn_Dataset                    | binclass   |       0.955022 |   0.953523 |    -0.1499 |        -0.157 |      2666 |      667 |           17 |           2 | False               |
| delta_ailerons                           | binclass   |       0.950912 |   0.949509 |    -0.1403 |        -0.147 |      5703 |     1426 |            5 |           2 | False               |
| allrep                                   | multiclass |       0.98543  |   0.984106 |    -0.1325 |        -0.134 |      3017 |      755 |           29 |           4 | False               |
| Cardiovascular-Disease-dataset           | binclass   |       0.734143 |   0.732857 |    -0.1286 |        -0.175 |     56000 |    14000 |           11 |           2 | False               |
| spambase                                 | binclass   |       0.959826 |   0.95874  |    -0.1086 |        -0.113 |      3680 |      921 |           57 |           2 | False               |
| Wilt                                     | binclass   |       0.992746 |   0.99171  |    -0.1036 |        -0.104 |      3856 |      965 |            5 |           2 | False               |
| national-longitudinal-survey-binary      | binclass   |       1        |   0.998982 |    -0.1018 |        -0.102 |      3926 |      982 |           16 |           2 | False               |
| waveform_database_generator              | multiclass |       0.356    |   0.355    |    -0.1    |        -0.281 |      3999 |     1000 |           21 |           3 | False               |
| Satellite                                | binclass   |       0.993137 |   0.992157 |    -0.098  |        -0.099 |      4080 |     1020 |           36 |           2 | False               |
| kdd_ipums_la_97-small                    | binclass   |       0.888247 |   0.887283 |    -0.0963 |        -0.108 |      4150 |     1038 |           20 |           2 | False               |
| wall-robot-navigation                    | multiclass |       0.986264 |   0.985348 |    -0.0916 |        -0.093 |      4364 |     1092 |           24 |           4 | False               |
| page-blocks                              | multiclass |       0.978995 |   0.978082 |    -0.0913 |        -0.093 |      4378 |     1095 |           10 |           5 | False               |
| company_bankruptcy_prediction            | binclass   |       0.970674 |   0.969941 |    -0.0733 |        -0.076 |      5455 |     1364 |           95 |           2 | False               |
| twonorm                                  | binclass   |       0.979054 |   0.978378 |    -0.0676 |        -0.069 |      5920 |     1480 |           20 |           2 | False               |
| pendigits                                | multiclass |       0.997271 |   0.996817 |    -0.0455 |        -0.046 |      8793 |     2199 |           16 |          10 | False               |
| bank                                     | binclass   |       0.90899  |   0.908659 |    -0.0332 |        -0.036 |     36168 |     9043 |           16 |           2 | False               |
| BNG(cmc)                                 | multiclass |       0.586257 |   0.585986 |    -0.0271 |        -0.046 |     44236 |    11060 |            9 |           3 | False               |

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
| FICO-HELOC-cleaned                                         | binclass   |       0.74481  |   0.74481  |          0 |             0 |      7896 |     1975 |           23 |           2 | False               |
| FOREX_cadjpy-day-High                                      | binclass   |       0.716621 |   0.716621 |          0 |             0 |      1467 |      367 |           10 |           2 | False               |
| GAMETES_Heterogeneity_20atts_1600_Het_0.4_0.2_50_EDM-2_001 | binclass   |       0.68125  |   0.68125  |          0 |             0 |      1280 |      320 |           20 |           2 | False               |
| JapaneseVowels                                             | multiclass |       0.998996 |   0.998996 |          0 |             0 |      7968 |     1993 |           14 |           9 | False               |
| KDDCup09_upselling                                         | binclass   |       0.811891 |   0.811891 |          0 |             0 |      4102 |     1026 |           49 |           2 | False               |
| PieChart3                                                  | binclass   |       0.875    |   0.875    |          0 |             0 |       861 |      216 |           37 |           2 | False               |
| PizzaCutter3                                               | binclass   |       0.880383 |   0.880383 |          0 |             0 |       834 |      209 |           37 |           2 | False               |
| ada_agnostic                                               | binclass   |       0.840088 |   0.840088 |          0 |             0 |      3649 |      913 |           48 |           2 | False               |
| analcatdata_authorship                                     | multiclass |       0.988166 |   0.988166 |          0 |             0 |       672 |      169 |           69 |           4 | False               |
| drug_consumption                                           | multiclass |       0.403183 |   0.403183 |          0 |             0 |      1507 |      377 |           12 |           7 | False               |
| estimation_of_obesity_levels                               | multiclass |       0.985816 |   0.985816 |          0 |             0 |      1688 |      423 |           16 |           7 | False               |
| ibm-employee-performance                                   | binclass   |       1        |   1        |          0 |             0 |      1176 |      294 |           30 |           2 | False               |
| internet_firewall                                          | multiclass |       0.931029 |   0.931029 |          0 |             0 |     52425 |    13107 |            7 |           4 | False               |
| maternal_health_risk                                       | multiclass |       0.871921 |   0.871921 |          0 |             0 |       811 |      203 |            6 |           3 | False               |
| mfeat-factors                                              | multiclass |       0.975    |   0.975    |          0 |             0 |      1600 |      400 |          216 |          10 | False               |
| mfeat-pixel                                                | multiclass |       0.975    |   0.975    |          0 |             0 |      1600 |      400 |          240 |          10 | False               |
| mice_protein_expression                                    | multiclass |       1        |   1        |          0 |             0 |       864 |      216 |           75 |           8 | False               |
| mozilla4                                                   | binclass   |       0.940174 |   0.940174 |          0 |             0 |     12436 |     3109 |            4 |           2 | False               |
| pc3                                                        | binclass   |       0.900958 |   0.900958 |          0 |             0 |      1250 |      313 |           37 |           2 | False               |
| qsar                                                       | binclass   |       0.905213 |   0.905213 |          0 |             0 |       844 |      211 |           40 |           2 | False               |
| seismic+bumps                                              | binclass   |       0.934236 |   0.934236 |          0 |             0 |      2067 |      517 |           18 |           2 | False               |
| shuttle                                                    | multiclass |       0.999483 |   0.999483 |          0 |             0 |     46400 |    11600 |            9 |           7 | False               |
| sylvine                                                    | binclass   |       0.970732 |   0.970732 |          0 |             0 |      4099 |     1025 |           20 |           2 | False               |
| thyroid                                                    | multiclass |       0.99375  |   0.99375  |          0 |             0 |      5760 |     1440 |           21 |           3 | False               |
