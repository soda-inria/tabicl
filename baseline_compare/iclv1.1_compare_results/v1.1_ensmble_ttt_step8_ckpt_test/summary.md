# ensmble_ttt_step8_ckpt_test 与 baseline 对比总结

- 失败的 18 个数据集已按 baseline fallback 处理，accuracy 替换为 `v1.1_baseline` 对应值，delta 记为 0。
- 全部 178 个数据集：baseline 0.837755 -> test/fallback 0.841143，平均 delta +0.3388 pp。
- 提升 / 下降 / 持平：88 / 47 / 43。
- baseline fallback 数据集：ASP-POTASSCO-classification, Credit_c, Rain_in_Australia, SDSS17, UJI_Pen_Characters, accelerometer, customer_satisfaction_in_airline, dabetes_130-us_hospitals, internet_usage, kr-vs-k, kropt, letter, one-hundred-plants-margin, one-hundred-plants-shape, one-hundred-plants-texture, texture, volkert, walking-activity。

详见 `compare_with_v1.1_baseline.md`。
