# baseline 两个结果目录 accuracy 对比

对比对象：

- 旧目录：`baseline/iclv2_ensmble32`
- 新目录：`baseline/iclv2_ensemble32`

## 结论

两个目录都包含 178 个数据集，且 178 个数据集全部 `status=ok`。按 `dataset_name` 对齐后，只有 1 个数据集的 `accuracy` 不一致：

| dataset_name | old_accuracy | new_accuracy | delta | n_test | n_features | n_classes |
|---|---:|---:|---:|---:|---:|---:|
| maternal_health_risk | 0.876847 | 0.871921 | -0.004926 | 203 | 6 | 3 |

整体平均：

| 目录 | ok_count | avg_accuracy_ok |
|---|---:|---:|
| `baseline/iclv2_ensmble32` | 178 | 0.847192 |
| `baseline/iclv2_ensemble32` | 178 | 0.847164 |

平均 accuracy 差值为 `-0.0000277`，主要由 `maternal_health_risk` 少预测对 1 个测试样本造成，因为 `1 / 203 = 0.004926`。

## 原因判断

`benchmark.py` 新增指标后，推理标签生成路径从旧逻辑的：

```python
y_pred = classifier.predict(X_test)
```

变成了：

```python
y_pred, y_proba, proba_classes = predict_from_proba_or_model(classifier, X_test)
```

也就是先调用 `predict_proba()`，再由脚本自己做 `argmax` 和类别映射。当前 `src/tabicl/_sklearn/classifier.py` 中 `TabICLClassifier.predict()` 本身也是 `predict_proba()` 后 `argmax` 再 `inverse_transform`，所以按当前源码看，这个改动不应该系统性改变 accuracy。

这次实际只出现 `maternal_health_risk` 的 1 个样本差异，更像是边界样本在两次独立 GPU 推理运行中的数值微小差异，而不是数据集加载、成功交集、失败行混入或平均口径的问题。两个目录的 worker 分配也一致，差异只出现在 worker 0 的这个数据集。

## 额外注意

`f1` 在 158 个数据集上发生变化，但这不等同于 accuracy 变化。新目录同时写出了 `balanced_accuracy`、`roc_auc`、`log_loss`，且 `f1` 口径应以当前脚本中的 `sklearn.metrics.f1_score(..., average="weighted")` 为准。旧目录的 `f1` 值与新目录大面积不同，说明旧目录的 `f1` 不是同一版指标实现下的可比结果。

完整明细见：`baseline/baseline_accuracy_diff_detail.csv`。
