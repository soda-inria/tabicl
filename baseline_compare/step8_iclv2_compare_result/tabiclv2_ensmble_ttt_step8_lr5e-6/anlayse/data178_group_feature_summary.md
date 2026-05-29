# data178 分组特征总结

## 一句话结论
TTT step8 相对 ICLv2 ensemble32 的平均收益很小但为正：成功交集 `170` 个数据集上，`64` 胜、`59` 负、`47` 平，平均 `+0.0554 pp`，净约 `+602.0` 个预测；它不是全面更强，而是少数强收益数据集抵消了一批小幅下降。

## 三类数据集的主要特征

### 1. TTT step8 更高的 64 个
- 平均 delta `+0.5766 pp`，但中位数只有 `+0.2184 pp`，说明大部分是小幅提升，少数强提升拉高均值。
- 更常见于 TTT 实际执行、类别数 `<=10`、ICLv2 acc 处于 `0.8-0.95` 的有结构但仍有 headroom 区间。
- 代表性强收益：`jungle_chess_2pcs_raw_endgame_complete`、`compass`、`eye_movements`，这些不是随机噪声，按 test 样本数贡献分别达到明显正值。

### 2. ICLv2 ensemble32 更高的 59 个
- 平均 delta `-0.4659 pp`，其中强下降 `3` 个，中/弱下降占多数。
- 风险集中在 step8 实际执行后：所有 59 个下降样本都执行了 TTT，说明问题不是 skip，而是更新方向在这些数据集上破坏了原有边界。
- 小 test 或小 delta_correct 的下降不应过度解读；但 `waveform_database_generator`、`first-order-theorem-proving`、`Gender_Gap_in_Spanish_WP`、`FOREX_audjpy-hour-High` 等应进入风险集。

### 3. 持平的 47 个
- 持平组分两类：`12` 个 TTT 未执行，`35` 个 TTT 执行但预测未变。
- 未执行部分多为 `n_classes > 10` 或主动 OOM skip，例如 `one-hundred-plants-*`、`letter`、`internet_usage`、`walking-activity`；这些不应作为 TTT 稳定性的正证据。
- 执行但持平的部分多是高 acc ceiling 或边界很稳定的数据，例如 `Satellite`、`PhishingWebsites`、`thyroid`、`wall-robot-navigation`。

## 实验选择规则
- 默认不要无条件使用 step8；应做 step selector 或 protected fallback。
- 优先保留：TTT 实际执行、类别数 `<=10`、ICLv2 acc `0.8-0.95`、`delta_correct_est` 明显为正的数据集。
- 谨慎使用：ICLv2 acc 已 `>0.95` 的 ceiling 数据、test 很小且只差 1-2 个样本的数据、FOREX/time-like 数据。
- 默认跳过或回退：类别数 `>10`、已知 OOM skip、大幅下降 `<= -1pp` 且 `delta_correct_est` 明显为负的数据集。
- 下一步最有价值的实验：在这些分组上做 `step0/4/8/12` 保护性选择，直接验证强收益是否保留、强下降是否被挡住。

## 文件
- 详细分析：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/data178_group_feature_analysis.md`
- 富集明细：`1b_result_v2/v2_ensmble_ttt_step8_lr5e-6/data178_group_feature_detail.csv`