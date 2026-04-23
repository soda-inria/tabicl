# dataset_up_summary

## 总体判断
- 提升数据集数量随 step 增大严格上升，从 step1 的 25 个增加到 step50 的 72 个。
- 共有 `97` 个 dataset 至少在一个 step 出现该方向变化。
- 按任务类型看: `binclass`=60, `multiclass`=37。

## 按 Step 统计
| step | 提升数量 | 下降数量 | 持平数量 |
| --- | --- | --- | --- |
| 1 | 25 | 18 | 129 |
| 2 | 27 | 26 | 119 |
| 3 | 33 | 31 | 108 |
| 4 | 34 | 33 | 105 |
| 5 | 40 | 35 | 97 |
| 10 | 53 | 36 | 83 |
| 20 | 58 | 37 | 77 |
| 30 | 61 | 48 | 63 |
| 40 | 66 | 44 | 62 |
| 50 | 72 | 47 | 53 |

## best_step 分布
| best_step | dataset_count |
| --- | --- |
| 1 | 2 |
| 2 | 3 |
| 3 | 5 |
| 4 | 1 |
| 5 | 6 |
| 10 | 8 |
| 20 | 19 |
| 30 | 16 |
| 40 | 13 |
| 50 | 24 |

## Pattern 分布
| pattern | dataset_count |
| --- | --- |
| nonnegative_gain | 32 |
| mixed_recovered_to_gain | 28 |
| mixed_ended_in_decline | 16 |
| all_steps_up | 14 |
| mixed_ended_flat | 7 |

## Task 分布
| task_type | dataset_count |
| --- | --- |
| binclass | 60 |
| multiclass | 37 |

## Top Dataset
| dataset | task | best_step | baseline_acc | delta | up_steps | down_steps | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| artificial-characters | multiclass | 50 | 0.863992 | +1.8591 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| autoUniv-au7-1100 | multiclass | 50 | 0.404545 | +1.8182 pp | 5,40,50 | 30 | mixed_recovered_to_gain |
| mfeat-zernike | multiclass | 50 | 0.852500 | +1.7500 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| jungle_chess_2pcs_raw_endgame_complete | multiclass | 50 | 0.862450 | +1.7180 pp | 4,5,10,20,30,40,50 | 1,2 | mixed_recovered_to_gain |
| autoUniv-au4-2500 | multiclass | 50 | 0.522000 | +1.6000 pp | 20,30,40,50 | 1,2 | mixed_recovered_to_gain |
| FOREX_audsgd-hour-High | binclass | 40 | 0.683856 | +1.5174 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| satimage | multiclass | 30 | 0.928460 | +1.3997 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| FOREX_audusd-hour-High | binclass | 30 | 0.681689 | +1.3919 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| turiye_student_evaluation | multiclass | 50 | 0.522337 | +1.2887 pp | 10,20,30,40,50 | 1,3,4,5 | mixed_recovered_to_gain |
| compass | binclass | 50 | 0.826675 | +1.2616 pp | 2,3,4,5,10,20,30,40,50 |  | nonnegative_gain |
| mfeat-fourier | multiclass | 50 | 0.885000 | +1.2500 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| eye_movements_bin | binclass | 50 | 0.625493 | +1.2484 pp | 3,4,5,10,20,30,40,50 | 1 | mixed_recovered_to_gain |
| GesturePhaseSegmentationProcessed | multiclass | 20 | 0.786329 | +1.2152 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |
| website_phishing | multiclass | 20 | 0.911439 | +1.1070 pp | 20,30,40,50 |  | nonnegative_gain |
| eye_movements | multiclass | 40 | 0.714808 | +1.0055 pp | 1,2,3,4,5,10,20,30,40,50 |  | all_steps_up |

## 字段说明
- `up_steps`: 该 dataset 相对 baseline accuracy 提升的 step 列表。
- `down_steps`: 该 dataset 相对 baseline accuracy 下降的 step 列表。
- `pattern`: 基于所有 step 的正/负/持平轨迹归类。
- `delta_pp_stepX`: step X 相对 baseline 的 accuracy 差值，单位是百分点 pp。
