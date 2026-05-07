# Baseline Accuracy Comparison

## Inputs
- benchmark baseline: `baseline/tabiclv1.1_baseline/all_classification_results.csv`
- casual ensemble32 baseline: `baseline/iclv1.1_casual_ensmble32/all_classification_results.csv`
- detail CSV: `baseline/baseline_acc_comparison/per_dataset_acc_comparison.csv`

## Visualizations
- scatter: `baseline/baseline_acc_comparison/accuracy_scatter.svg`
- all-dataset sorted delta: `baseline/baseline_acc_comparison/delta_by_dataset.svg`
- top gains/losses: `baseline/baseline_acc_comparison/top_changes.svg`
- delta distribution: `baseline/baseline_acc_comparison/delta_distribution.svg`
- causal feature ratio vs delta: `baseline/baseline_acc_comparison/feature_ratio_vs_delta.svg`

## Overall
| shared_ok | benchmark_avg_acc | casual_avg_acc | mean_delta | median_delta | std_delta | min_delta | max_delta | casual_wins | benchmark_wins | ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 178 | 0.837755 | 0.835388 | -0.2367 pp | -0.0068 pp | +1.1639 pp | -10.9873 pp | +3.0000 pp | 50 | 90 | 38 |

## Delta Magnitude
| threshold | datasets | share |
| --- | --- | --- |
| \|delta\| >= 0.01 pp | 137 | 77.0% |
| \|delta\| >= 0.05 pp | 120 | 67.4% |
| \|delta\| >= 0.10 pp | 101 | 56.7% |
| \|delta\| >= 0.25 pp | 74 | 41.6% |
| \|delta\| >= 0.50 pp | 46 | 25.8% |
| \|delta\| >= 1.00 pp | 21 | 11.8% |

## Task Type Summary
| task_type | datasets | benchmark_avg_acc | casual_avg_acc | mean_delta | median_delta | casual_wins | benchmark_wins | ties |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| binclass | 99 | 0.854052 | 0.851818 | -0.2234 pp | +0.0000 pp | 29 | 47 | 23 |
| multiclass | 79 | 0.817332 | 0.814798 | -0.2534 pp | -0.0367 pp | 21 | 43 | 15 |

## Runtime On Shared OK Datasets
| metric | benchmark | casual ensemble32 | ratio casual/benchmark |
| --- | --- | --- | --- |
| fit + predict seconds | 2823.216 | 5427.957 | 1.92x |
| mean seconds / dataset | 15.861 | 30.494 | 1.92x |

## Largest Gains For Casual Ensemble32
| dataset | task | benchmark_acc | casual_acc | delta | selected_features |
| --- | --- | --- | --- | --- | --- |
| mfeat-zernike | multiclass | 0.852500 | 0.882500 | +3.0000 pp | 6/47 |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass | 0.602500 | 0.625000 | +2.2500 pp | 3/7 |
| volkert | multiclass | 0.733751 | 0.750129 | +1.6378 pp | 22/180 |
| ASP-POTASSCO-classification | multiclass | 0.440154 | 0.455598 | +1.5444 pp | 6/141 |
| vehicle | multiclass | 0.858824 | 0.870588 | +1.1765 pp | 5/18 |
| cmc | multiclass | 0.589831 | 0.600000 | +1.0169 pp | 4/9 |
| Marketing_Campaign | binclass | 0.890625 | 0.899554 | +0.8929 pp | 8/27 |
| car-evaluation | multiclass | 0.973988 | 0.982659 | +0.8671 pp | 9/21 |
| FOREX_audcad-day-High | binclass | 0.743869 | 0.752044 | +0.8174 pp | 3/10 |
| baseball | multiclass | 0.947761 | 0.955224 | +0.7463 pp | 3/16 |
| IBM_HR_Analytics_Employee_Attrition_and_Performance | binclass | 0.863946 | 0.870748 | +0.6803 pp | 11/31 |
| Water_Quality_and_Potability | binclass | 0.641768 | 0.647866 | +0.6098 pp | 3/8 |
| MIC | binclass | 0.909091 | 0.915152 | +0.6061 pp | 21/104 |
| compass | binclass | 0.826675 | 0.832682 | +0.6008 pp | 8/17 |
| wine-quality-white | multiclass | 0.682653 | 0.687755 | +0.5102 pp | 6/11 |

## Largest Losses For Casual Ensemble32
| dataset | task | benchmark_acc | casual_acc | delta | selected_features |
| --- | --- | --- | --- | --- | --- |
| madeline | binclass | 0.753185 | 0.643312 | -10.9873 pp | 10/259 |
| semeion | multiclass | 0.959248 | 0.912226 | -4.7022 pp | 25/256 |
| Indian_pines | multiclass | 0.962274 | 0.923455 | -3.8819 pp | 3/220 |
| autoUniv-au7-1100 | multiclass | 0.404545 | 0.368182 | -3.6364 pp | 6/12 |
| gina_agnostic | binclass | 0.927954 | 0.899135 | -2.8818 pp | 60/970 |
| dna | multiclass | 0.965517 | 0.937304 | -2.8213 pp | 22/180 |
| one-hundred-plants-margin | multiclass | 0.909375 | 0.884375 | -2.5000 pp | 7/64 |
| hill-valley | binclass | 0.930041 | 0.909465 | -2.0576 pp | 3/100 |
| UJI_Pen_Characters | multiclass | 0.542125 | 0.523810 | -1.8315 pp | 3/80 |
| qsar | binclass | 0.905213 | 0.890995 | -1.4218 pp | 7/40 |
| rl | binclass | 0.854125 | 0.841046 | -1.3078 pp | 4/12 |
| Mobile_Price_Classification | multiclass | 0.940000 | 0.927500 | -1.2500 pp | 4/20 |
| internet_usage | multiclass | 0.529674 | 0.518299 | -1.1375 pp | 14/70 |
| mfeat-pixel | multiclass | 0.975000 | 0.965000 | -1.0000 pp | 11/240 |
| sports_articles_for_objectivity_analysis | binclass | 0.855000 | 0.845000 | -1.0000 pp | 8/59 |

## Interpretation
- Casual ensemble32 的平均 acc 比 benchmark baseline 低 `-0.2367 pp`，属于轻微整体下降。
- 逐数据集看，casual ensemble32 提升 `50` 个，下降 `90` 个，持平 `38` 个。
- 变化主要集中在小幅区间：`|delta| < 0.10 pp` 的数据集有 `77` 个。
- Casual ensemble32 的总 fit+predict 时间是 benchmark baseline 的 `1.92x`。
