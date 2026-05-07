# step8_ckpt_test 与 step9_lr5e-6 对比总结

Note: 当前比较使用 fallback 后结果；step8 和 step9 中原失败数据集已按 baseline accuracy 填充。

- 对比数据集：178
- step8 平均 accuracy：0.841143
- step9 平均 accuracy：0.841106
- step8-step9 平均 delta：+0.0037 pp
- step8 更好 / step9 更好 / 持平：38 / 55 / 85

## step8 优势最大

| dataset                        | task       |   step8_acc |   step9_acc |   delta_pp |
|:-------------------------------|:-----------|------------:|------------:|-----------:|
| Pima_Indians_Diabetes_Database | binclass   |    0.779221 |    0.766234 |     1.2987 |
| Water_Quality_and_Potability   | binclass   |    0.629573 |    0.621951 |     0.7622 |
| mfeat-morphological            | multiclass |    0.77     |    0.7625   |     0.75   |
| contraceptive_method_choice    | multiclass |    0.647458 |    0.640678 |     0.678  |
| Fitness_Club_c                 | binclass   |    0.8      |    0.793333 |     0.6667 |
| FOREX_audcad-day-High          | binclass   |    0.757493 |    0.752044 |     0.545  |
| waveform_database_generator    | multiclass |    0.355    |    0.35     |     0.5    |
| led7                           | multiclass |    0.74375  |    0.739062 |     0.4688 |
| hill-valley                    | binclass   |    0.967078 |    0.962963 |     0.4115 |
| baseball                       | multiclass |    0.951493 |    0.947761 |     0.3731 |

## step9 优势最大

| dataset                                | task       |   step8_acc |   step9_acc |   delta_pp |
|:---------------------------------------|:-----------|------------:|------------:|-----------:|
| autoUniv-au7-1100                      | multiclass |    0.418182 |    0.427273 |    -0.9091 |
| jungle_chess_2pcs_raw_endgame_complete | multiclass |    0.917448 |    0.924253 |    -0.6805 |
| vehicle                                | multiclass |    0.882353 |    0.888235 |    -0.5882 |
| steel_plates_faults                    | multiclass |    0.840617 |    0.845758 |    -0.5141 |
| golf_play_dataset_extended             | binclass   |    0.931507 |    0.936073 |    -0.4566 |
| Diabetic_Retinopathy_Debrecen          | binclass   |    0.748918 |    0.753247 |    -0.4329 |
| ada                                    | binclass   |    0.840964 |    0.844578 |    -0.3614 |
| FOREX_audsgd-hour-High                 | binclass   |    0.704392 |    0.707587 |    -0.3195 |
| wine-quality-red                       | multiclass |    0.65625  |    0.659375 |    -0.3125 |
| MIC                                    | binclass   |    0.89697  |    0.9      |    -0.303  |
