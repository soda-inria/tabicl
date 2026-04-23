# dataset_down_summary

## 总体判断
- 下降数据集数量整体随 step 增大上升，从 step1 的 18 个增加到 step50 的 47 个；但不是单调变化，step30 为 48 个，step40 回落到 44 个。
- 共有 `86` 个 dataset 至少在一个 step 出现该方向变化。
- 按任务类型看: `binclass`=60, `multiclass`=26。

## 按 Step 统计
| step | 下降数量 | 提升数量 | 持平数量 |
| --- | --- | --- | --- |
| 1 | 18 | 25 | 129 |
| 2 | 26 | 27 | 119 |
| 3 | 31 | 33 | 108 |
| 4 | 33 | 34 | 105 |
| 5 | 35 | 40 | 97 |
| 10 | 36 | 53 | 83 |
| 20 | 37 | 58 | 77 |
| 30 | 48 | 61 | 63 |
| 40 | 44 | 66 | 62 |
| 50 | 47 | 72 | 53 |

## worst_step 分布
| worst_step | dataset_count |
| --- | --- |
| 1 | 6 |
| 2 | 3 |
| 3 | 9 |
| 4 | 2 |
| 5 | 8 |
| 10 | 8 |
| 20 | 10 |
| 30 | 13 |
| 40 | 9 |
| 50 | 18 |

## Pattern 分布
| pattern | dataset_count |
| --- | --- |
| nonpositive_decline | 33 |
| mixed_recovered_to_gain | 28 |
| mixed_ended_in_decline | 16 |
| mixed_ended_flat | 7 |
| all_steps_down | 2 |

## Task 分布
| task_type | dataset_count |
| --- | --- |
| binclass | 60 |
| multiclass | 26 |

## Top Dataset
| dataset | task | worst_step | baseline_acc | delta | up_steps | down_steps | pattern |
| --- | --- | --- | --- | --- | --- | --- | --- |
| banknote_authentication | binclass | 50 | 0.549091 | -3.2727 pp | 2,3,4,5,20,30 | 40,50 | mixed_ended_in_decline |
| FOREX_audchf-day-High | binclass | 50 | 0.754768 | -2.4523 pp | 1 | 2,3,4,5,10,20,30,40,50 | mixed_ended_in_decline |
| abalone | multiclass | 50 | 0.651914 | -2.2727 pp | 2,3,10 | 4,5,20,30,40,50 | mixed_ended_in_decline |
| FOREX_cadjpy-day-High | binclass | 50 | 0.716621 | -2.1798 pp | 20,40 | 2,3,4,5,10,50 | mixed_ended_in_decline |
| sports_articles_for_objectivity_analysis | binclass | 50 | 0.855000 | -2.0000 pp |  | 30,40,50 | nonpositive_decline |
| autoUniv-au7-1100 | multiclass | 30 | 0.404545 | -1.3636 pp | 5,40,50 | 30 | mixed_recovered_to_gain |
| yeast | multiclass | 5 | 0.629630 | -1.3468 pp |  | 1,2,3,4,5,10,20,30 | nonpositive_decline |
| statlog | binclass | 10 | 0.745000 | -1.0000 pp |  | 10,20,30,40 | nonpositive_decline |
| first-order-theorem-proving | multiclass | 30 | 0.637255 | -0.9804 pp | 3 | 1,5,10,20,30,40,50 | mixed_ended_in_decline |
| semeion | multiclass | 30 | 0.959248 | -0.9404 pp |  | 20,30,40,50 | nonpositive_decline |
| golf_play_dataset_extended | binclass | 30 | 0.936073 | -0.9132 pp |  | 30,40,50 | nonpositive_decline |
| spambase | binclass | 50 | 0.959826 | -0.8686 pp |  | 3,4,5,10,30,40,50 | nonpositive_decline |
| FOREX_audcad-day-High | binclass | 20 | 0.743869 | -0.8174 pp | 2,40,50 | 5,10,20,30 | mixed_recovered_to_gain |
| cmc | multiclass | 40 | 0.589831 | -0.6780 pp |  | 4,5,10,20,30,40,50 | nonpositive_decline |
| phoneme | binclass | 50 | 0.903793 | -0.6475 pp |  | 1,2,3,4,5,10,20,30,40,50 | all_steps_down |

## 字段说明
- `up_steps`: 该 dataset 相对 baseline accuracy 提升的 step 列表。
- `down_steps`: 该 dataset 相对 baseline accuracy 下降的 step 列表。
- `pattern`: 基于所有 step 的正/负/持平轨迹归类。
- `delta_pp_stepX`: step X 相对 baseline 的 accuracy 差值，单位是百分点 pp。
