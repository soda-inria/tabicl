# dataset_down_trend

## 主要趋势
- 下降 dataset 数量整体随 step 增大而上升: step1 为 `18` 个，step50 为 `47` 个。
- 下降数量峰值出现在 step30: `48` 个，占共同成功数据集的 `27.91%`。
- 最大单点下降出现在 step50: `3.2727 pp`。
- 至少一次下降的数据集共有 `86` 个；其中持续非正收益模式 `35` 个，后期 step30/40/50 仍出现下降的 `65` 个。
- 结论上，高 step 同时带来更多提升 dataset 和更明显的下降尾部；下降风险不是简单单调扩大，step30 的下降数量最高，step40 略有回落，step50 再次增加。

## 按 Step 下降趋势
| step | 下降数量 | 下降比例 | 平均下降幅度 | 中位下降幅度 | 最大下降幅度 | >=0.1pp | >=0.5pp | >=1.0pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 18 | 10.47% | -0.1461 pp | -0.0479 pp | -1.0101 pp | 4 | 1 | 1 |
| 2 | 26 | 15.12% | -0.1586 pp | -0.0631 pp | -1.0101 pp | 10 | 1 | 1 |
| 3 | 31 | 18.02% | -0.1432 pp | -0.0919 pp | -1.0101 pp | 14 | 1 | 1 |
| 4 | 33 | 19.19% | -0.1682 pp | -0.1196 pp | -1.0101 pp | 22 | 2 | 1 |
| 5 | 35 | 20.35% | -0.2090 pp | -0.1205 pp | -1.3468 pp | 22 | 4 | 2 |
| 10 | 36 | 20.93% | -0.2616 pp | -0.1532 pp | -1.6349 pp | 28 | 5 | 2 |
| 20 | 37 | 21.51% | -0.3345 pp | -0.2500 pp | -1.6349 pp | 30 | 6 | 2 |
| 30 | 48 | 27.91% | -0.3649 pp | -0.2500 pp | -1.5000 pp | 40 | 9 | 4 |
| 40 | 44 | 25.58% | -0.4141 pp | -0.2951 pp | -2.1798 pp | 36 | 11 | 4 |
| 50 | 47 | 27.33% | -0.5404 pp | -0.2999 pp | -3.2727 pp | 40 | 13 | 5 |

## 下降最严重的 Dataset
| dataset | task | worst_step | worst_delta | down_steps | up_steps | pattern |
| --- | --- | --- | --- | --- | --- | --- |
| banknote_authentication | binclass | 50 | -3.2727 pp | 40,50 | 2,3,4,5,20,30 | mixed_ended_in_decline |
| FOREX_audchf-day-High | binclass | 50 | -2.4523 pp | 2,3,4,5,10,20,30,40,50 | 1 | mixed_ended_in_decline |
| abalone | multiclass | 50 | -2.2727 pp | 4,5,20,30,40,50 | 2,3,10 | mixed_ended_in_decline |
| FOREX_cadjpy-day-High | binclass | 50 | -2.1798 pp | 2,3,4,5,10,50 | 20,40 | mixed_ended_in_decline |
| sports_articles_for_objectivity_analysis | binclass | 50 | -2.0000 pp | 30,40,50 |  | nonpositive_decline |
| autoUniv-au7-1100 | multiclass | 30 | -1.3636 pp | 30 | 5,40,50 | mixed_recovered_to_gain |
| yeast | multiclass | 5 | -1.3468 pp | 1,2,3,4,5,10,20,30 |  | nonpositive_decline |
| statlog | binclass | 10 | -1.0000 pp | 10,20,30,40 |  | nonpositive_decline |
| first-order-theorem-proving | multiclass | 30 | -0.9804 pp | 1,5,10,20,30,40,50 | 3 | mixed_ended_in_decline |
| semeion | multiclass | 30 | -0.9404 pp | 20,30,40,50 |  | nonpositive_decline |
| golf_play_dataset_extended | binclass | 30 | -0.9132 pp | 30,40,50 |  | nonpositive_decline |
| spambase | binclass | 50 | -0.8686 pp | 3,4,5,10,30,40,50 |  | nonpositive_decline |

## 图像
- `ttt_step_down_trend.png`: 上半部分是下降 dataset 数量及 `>=0.5pp` 明显下降数量；下半部分是每个 step 下下降幅度分布。
