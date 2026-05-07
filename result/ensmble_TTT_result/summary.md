# ensmble_TTT_result summary

## 输入目录
- ensemble: `result/ensmble_TTT_result/ensmble_ttt_step10_lr5e-6`
- ICL training: `result/ensmble_TTT_result/iclv1.1_ttt_step10`
- 明细文件: `result/ensmble_TTT_result/detail.csv`

## 主要结论
- 原始全量 `avg_accuracy_ok` 上，ensemble 更高：`0.841789` vs `0.838531`，差值 `+0.3258 pp`。
- 但这不能直接当作公平对比，因为 ensemble 失败了 `22` 个数据集，而 ICL training 只失败了 `6` 个。
- 在两边都成功的 `156` 个共同数据集上，ensemble 平均 acc 为 `0.841789`，ICL training 为 `0.839353`，ensemble 平均领先 `+0.2436 pp`。
- 逐 dataset 计数上，ensemble / ICL training / 持平 为 `72` / `47` / `37`，说明 ensemble 在交集上总体占优，但并非全面压制。
- 只看这 `156` 个交集数据集的总耗时，ensemble 为 `12025.616s`，ICL training 为 `2137.947s`，ensemble 是 ICL training 的 `5.62x`。
- ensemble 的额外失败集共有 `16` 个，主要是大表和高负载任务；共享失败集有 `6` 个，都是两边都没跑通的数据集。

## 原始运行概况
| method | ok_count | failed_count | avg_accuracy_ok | wall_seconds |
| --- | --- | --- | --- | --- |
| ensemble_ttt_step10_lr5e-6 | 156 | 22 | 0.841789 | 12078.614 |
| iclv1.1_ttt_step10 | 172 | 6 | 0.838531 | 1763.958 |

## 共同成功交集对比
| common_ok | ensemble_avg_acc | icl_avg_acc | mean_delta | median_delta | ensemble_win | icl_win | tie | \|delta\|>=0.1pp | \|delta\|>=0.5pp | ensemble_total_seconds | icl_total_seconds | time_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 156 | 0.841789 | 0.839353 | +0.2436 pp | +0.0000 pp | 72 | 47 | 37 | 101 | 41 | 12025.616 | 2137.947 | 5.62x |

## 交集时间拆解
| metric | ensemble | ICL training |
| --- | --- | --- |
| fit_seconds_sum | 10957.801 | 1090.010 |
| predict_seconds_sum | 1067.815 | 1047.937 |
| total_seconds_sum | 12025.616 | 2137.947 |

## Ensemble 优势最大的 Dataset
| dataset | task | ensemble_acc | icl_acc | delta |
| --- | --- | --- | --- | --- |
| eye_movements_bin | binclass | 0.674770 | 0.628121 | +4.6649 pp |
| compass | binclass | 0.878642 | 0.834785 | +4.3857 pp |
| mfeat-zernike | multiclass | 0.905000 | 0.862500 | +4.2500 pp |
| hill-valley | binclass | 0.967078 | 0.930041 | +3.7037 pp |
| eye_movements | multiclass | 0.756399 | 0.719378 | +3.7020 pp |
| autoUniv-au4-2500 | multiclass | 0.558000 | 0.522000 | +3.6000 pp |
| vehicle | multiclass | 0.888235 | 0.858824 | +2.9412 pp |
| artificial-characters | multiclass | 0.898239 | 0.870841 | +2.7397 pp |
| Pima_Indians_Diabetes_Database | binclass | 0.766234 | 0.746753 | +1.9481 pp |
| autoUniv-au7-1100 | multiclass | 0.422727 | 0.404545 | +1.8182 pp |

## ICL training 优势最大的 Dataset
| dataset | task | ensemble_acc | icl_acc | delta |
| --- | --- | --- | --- | --- |
| Water_Quality_and_Potability | binclass | 0.626524 | 0.641768 | -1.5244 pp |
| Basketball_c | binclass | 0.694030 | 0.708955 | -1.4925 pp |
| QSAR_biodegradation | binclass | 0.876777 | 0.890995 | -1.4218 pp |
| FOREX_audjpy-day-High | binclass | 0.757493 | 0.771117 | -1.3624 pp |
| wine-quality-red | multiclass | 0.656250 | 0.668750 | -1.2500 pp |
| turiye_student_evaluation | multiclass | 0.513746 | 0.525773 | -1.2027 pp |
| MIC | binclass | 0.900000 | 0.909091 | -0.9091 pp |
| waveform_database_generator | multiclass | 0.345000 | 0.353000 | -0.8000 pp |
| splice | multiclass | 0.951411 | 0.959248 | -0.7837 pp |
| in_vehicle_coupon_recommendation | binclass | 0.789121 | 0.795822 | -0.6701 pp |

## 失败集说明
- ensemble 额外失败集 `16` 个: `BNG(cmc), Cardiovascular-Disease-dataset, FOREX_audcad-hour-High, FOREX_audjpy-hour-High, FOREX_audsgd-hour-High, FOREX_audusd-hour-High, FOREX_cadjpy-hour-High, Indian_pines, bank, electricity, gina_agnostic, internet_firewall, jungle_chess_2pcs_raw_endgame_complete, mobile_c36_oversampling, naticusdroid+android+permissions+dataset, shuttle`。
- 共享失败集 `6` 个: `Credit_c, Rain_in_Australia, SDSS17, accelerometer, customer_satisfaction_in_airline, dabetes_130-us_hospitals`。
