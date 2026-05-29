# TTT step8 data178 分组总结

- 成功交集 `170` 个；TTT step8 更高 `64` 个，ICLv2 更高 `59` 个，持平 `47` 个。
- 平均收益 `+0.0554 pp`，加权净收益约 `+602.0` 个预测；这说明 step8 有小幅正收益，但不是稳定强提升。
- TTT step8 更适合：TTT 实际执行、类别数不超过 10、ICLv2 有中等 headroom、数据中存在可适配边界的数据集。
- TTT step8 不稳定：低基线/高难度、多类别边界复杂、部分时间/生成式结构或小样本高维数据集，固定 step8 可能过适配。
- 持平 `47` 个中既包含未执行 TTT 的数据集，也包含执行后预测未变化的数据集，不能把 tie 简单理解为 TTT 无效。
- 实验建议：把 step8 放进 protected selector，以 ICLv2 作为 step0 候选；不要无条件用固定 step8 替换 ICLv2。

## 输出文件
- 交集明细：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/detail.csv`
- 富化明细：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/data178_group_feature_detail.csv`
- 详细分析：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/data178_group_feature_analysis.md`
- 成功交集汇总：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/summary.md`

## 三组中位数画像
| group | n | mean_delta | median_train | median_test | median_features | median_classes | ttt_applied_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTT step8更高 | 64 | +0.5766 pp | 3919.5 | 1225.0 | 14.0 | 2.0 | 64 |
| ICLv2更高 | 59 | -0.4659 pp | 3036.0 | 950.0 | 20.0 | 2.0 | 59 |
| 持平 | 47 | +0.0000 pp | 1350.0 | 423.0 | 29.0 | 4.0 | 35 |
