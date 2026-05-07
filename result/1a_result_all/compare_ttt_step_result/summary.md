# 1a_result_all vs ICL training

## 输入
- full training root: `result/1a_result_all`
- ICL training root: `result/ttt_step_result`
- 使用固定 reference set 比较: `step10/20/30/40/50` 共同成功交集，共 `150` 个数据集。
- step1 也强制使用这组 reference set，以避免因样本集合不同带来偏差。
- 产物目录: `result/1a_result_all/compare_ttt_step_result`

## 主要结论
- 在公平的共同成功数据集上，两者平均 accuracy 非常接近。`1a_result_all` 最有利的 step 是 `30`，均值领先 `+0.0092 pp`；`ICL training` 最有利的 step 是 `40`，均值领先 `-0.0574 pp`。
- 真正更稳定的是 `ICL training`：它每个 step 都有 `150` 或 `150` 个共同成功数据集，而 `1a_result_all` 持续多出 `20-22` 个额外失败数据集。
- 高 step 时胜负天平转向 `ICL training`。在 step40/50，共同数据集上 `ICL training` 分别赢 `45` / `45` 个数据集，而 `1a_result_all` 只赢 `31` / `34` 个。
- step1 原始共同成功集有 `152` 个，但为与后续 step 保持同一基准，额外排除了 `2` 个 step1-only 数据集: `BNG(breast-w), BNG(tic-tac-toe)`。
- `1a_result_all` 的额外失败集高度稳定，几乎固定在同一批大表/高负载数据集上: `BNG(cmc), Cardiovascular-Disease-dataset, FOREX_audcad-hour-High, FOREX_audjpy-hour-High, FOREX_audsgd-hour-High, FOREX_audusd-hour-High, FOREX_cadjpy-hour-High, INNHotelsGroup, Indian_pines, bank, default_of_credit_card_clients, electricity, gas-drift, gina_agnostic, internet_firewall, jungle_chess_2pcs_raw_endgame_complete, madeline, mobile_c36_oversampling, naticusdroid+android+permissions+dataset, shuttle`。

## Step 汇总
| step | common_ok | full_avg | step_avg | mean_gap | median_gap | full_win | step_win | tie | extra_full_fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 150 | 0.837503 | 0.837476 | +0.0027 pp | +0.0000 pp | 14 | 9 | 127 | 20 |
| 10 | 150 | 0.837730 | 0.837705 | +0.0025 pp | +0.0000 pp | 23 | 26 | 101 | 22 |
| 20 | 150 | 0.837929 | 0.837937 | -0.0008 pp | +0.0000 pp | 27 | 34 | 89 | 22 |
| 30 | 150 | 0.837760 | 0.837668 | +0.0092 pp | +0.0000 pp | 36 | 39 | 75 | 22 |
| 40 | 150 | 0.837316 | 0.837889 | -0.0574 pp | +0.0000 pp | 31 | 45 | 74 | 22 |
| 50 | 150 | 0.837240 | 0.837680 | -0.0440 pp | +0.0000 pp | 34 | 45 | 71 | 22 |

## 每个 Step 中 full training 优势最大的样本
| step | dataset | task | full_acc | step_acc | gap |
| --- | --- | --- | --- | --- | --- |
| 1 | yeast | multiclass | 0.622896 | 0.619529 | +0.3367 pp |
| 1 | autoUniv-au4-2500 | multiclass | 0.522000 | 0.520000 | +0.2000 pp |
| 1 | thyroid-dis | multiclass | 0.692857 | 0.691071 | +0.1786 pp |
| 10 | contraceptive_method_choice | multiclass | 0.630508 | 0.623729 | +0.6780 pp |
| 10 | FOREX_cadjpy-day-High | binclass | 0.713896 | 0.708447 | +0.5450 pp |
| 10 | FOREX_audchf-day-High | binclass | 0.743869 | 0.738420 | +0.5450 pp |
| 20 | FOREX_cadjpy-day-High | binclass | 0.727520 | 0.719346 | +0.8174 pp |
| 20 | GesturePhaseSegmentationProcessed | multiclass | 0.805570 | 0.798481 | +0.7089 pp |
| 20 | contraceptive_method_choice | multiclass | 0.630508 | 0.623729 | +0.6780 pp |
| 30 | GesturePhaseSegmentationProcessed | multiclass | 0.806076 | 0.796962 | +0.9114 pp |
| 30 | yeast | multiclass | 0.632997 | 0.626263 | +0.6734 pp |
| 30 | waveform_database_generator | multiclass | 0.362000 | 0.356000 | +0.6000 pp |
| 40 | airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass | 0.615000 | 0.607500 | +0.7500 pp |
| 40 | contraceptive_method_choice | multiclass | 0.633898 | 0.627119 | +0.6780 pp |
| 40 | GesturePhaseSegmentationProcessed | multiclass | 0.802025 | 0.796456 | +0.5570 pp |
| 50 | FOREX_audchf-day-High | binclass | 0.741144 | 0.730245 | +1.0899 pp |
| 50 | eye_movements | multiclass | 0.732176 | 0.723949 | +0.8227 pp |
| 50 | contraceptive_method_choice | multiclass | 0.630508 | 0.623729 | +0.6780 pp |

## 每个 Step 中 ICL training 优势最大的样本
| step | dataset | task | full_acc | step_acc | gap |
| --- | --- | --- | --- | --- | --- |
| 1 | mfeat-zernike | multiclass | 0.852500 | 0.855000 | -0.2500 pp |
| 1 | waveform_database_generator | multiclass | 0.358000 | 0.360000 | -0.2000 pp |
| 1 | dna | multiclass | 0.963950 | 0.965517 | -0.1567 pp |
| 10 | QSAR_biodegradation | binclass | 0.876777 | 0.890995 | -1.4218 pp |
| 10 | hill-valley | binclass | 0.925926 | 0.930041 | -0.4115 pp |
| 10 | abalone | multiclass | 0.649522 | 0.653110 | -0.3589 pp |
| 20 | website_phishing | multiclass | 0.915129 | 0.922509 | -0.7380 pp |
| 20 | thyroid-dis | multiclass | 0.691071 | 0.696429 | -0.5357 pp |
| 20 | sports_articles_for_objectivity_analysis | binclass | 0.850000 | 0.855000 | -0.5000 pp |
| 30 | FOREX_audchf-day-High | binclass | 0.732970 | 0.743869 | -1.0899 pp |
| 30 | abalone | multiclass | 0.636364 | 0.642344 | -0.5981 pp |
| 30 | thyroid-dis | multiclass | 0.691071 | 0.696429 | -0.5357 pp |
| 40 | banknote_authentication | binclass | 0.516364 | 0.538182 | -2.1818 pp |
| 40 | FOREX_cadjpy-day-High | binclass | 0.700272 | 0.719346 | -1.9074 pp |
| 40 | abalone | multiclass | 0.625598 | 0.638756 | -1.3158 pp |
| 50 | waveform_database_generator | multiclass | 0.342000 | 0.359000 | -1.7000 pp |
| 50 | yeast | multiclass | 0.616162 | 0.629630 | -1.3468 pp |
| 50 | abalone | multiclass | 0.617225 | 0.629187 | -1.1962 pp |

## 图像说明
- `shared_mean_accuracy.png`: 共同成功数据集上的均值 accuracy 曲线，标注的是 `full - step` 的均值差。
- `win_tie_loss.png`: 每个 step 上 full training / ICL training / tie 的数据集计数。
- `coverage_gap.png`: 共同成功数据集数量，以及 full training 相比 ICL training 的额外失败数量。
- `accuracy_gap_distribution.png`: 逐 dataset 的 accuracy gap 分布；正值代表 full training 更好，负值代表 ICL training 更好。
