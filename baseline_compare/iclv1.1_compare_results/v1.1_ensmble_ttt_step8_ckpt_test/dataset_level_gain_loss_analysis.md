# TTT Step8 Dataset-Level Gain/Loss Analysis

口径：`ensmble_ttt_step8_ckpt_test` 对比 `v1.1_baseline`，排除 baseline fallback 数据集；稳定性使用 early steps `{8,9,12,13,14,15}`。

## 总览

- 提升数据集：88
- 下降数据集：47
- 提升平均 delta：+0.9179 pp
- 下降平均 delta：-0.4355 pp

## 下降数据集逐项分析

### Basketball_c

- 指标：baseline 0.712687 -> step8 0.694030，delta -1.8657 pp，约 -5.0 个 test 样本。
- 任务/规模：binclass，train=1072，test=268，features=11，classes=2。
- JSON 属性：num=11，cat=0，modality=numeric_only，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-1.928 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；低/中维纯数值特征，TTT 更新较容易形成平滑修正；test 很小，几个样本即可造成 pp 级波动；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### FOREX_audchf-day-High

- 指标：baseline 0.754768 -> step8 0.738420，delta -1.6349 pp，约 -6.0 个 test 样本。
- 任务/规模：binclass，train=1466，test=367，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.028，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-2.225 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### wine-quality-red

- 指标：baseline 0.668750 -> step8 0.656250，delta -1.2500 pp，约 -4.0 个 test 样本。
- 任务/规模：multiclass，train=1279，test=320，features=4，classes=6。
- JSON 属性：num=4，cat=0，modality=numeric_only，imbalance=68.100，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.937 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别极不平衡，accuracy 变化可能主要来自多数类边界；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Water_Quality_and_Potability

- 指标：baseline 0.641768 -> step8 0.629573，delta -1.2195 pp，约 -8.0 个 test 样本。
- 任务/规模：binclass，train=2620，test=656，features=8，classes=2。
- JSON 属性：num=8，cat=0，modality=numeric_only，imbalance=NA，source=kaggle，domain=scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-2.109 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低 baseline 二分类仍可能通过校正决策边界获益；低/中维纯数值特征，TTT 更新较容易形成平滑修正；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### MIC

- 指标：baseline 0.909091 -> step8 0.896970，delta -1.2121 pp，约 -4.0 个 test 样本。
- 任务/规模：binclass，train=1319，test=330，features=104，classes=2。
- JSON 属性：num=9，cat=95，modality=mixed，imbalance=5.416，source=openml，domain=scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-0.960 pp。
- 具体判断：科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；类别特征极多，离散稀疏模式容易被 TTT 过拟合；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### sports_articles_for_objectivity_analysis

- 指标：baseline 0.855000 -> step8 0.845000，delta -1.0000 pp，约 -2.0 个 test 样本。
- 任务/规模：binclass，train=800，test=200，features=59，classes=2。
- JSON 属性：num=57，cat=2，modality=mixed，imbalance=NA，source=uci，domain=text/sequence/symbolic。
- 稳定性：early 6/6 均下降，early mean delta=-0.750 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### statlog

- 指标：baseline 0.745000 -> step8 0.735000，delta -1.0000 pp，约 -2.0 个 test 样本。
- 任务/规模：binclass，train=800，test=200，features=20，classes=2。
- JSON 属性：num=7，cat=13，modality=mixed，imbalance=NA，source=uci，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-1.417 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动；step8 明显下降，应优先加入排除或早停保护。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### QSAR_biodegradation

- 指标：baseline 0.886256 -> step8 0.876777，delta -0.9479 pp，约 -2.0 个 test 样本。
- 任务/规模：binclass，train=843，test=211，features=41，classes=2。
- JSON 属性：num=41，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=scientific/health/environment。
- 稳定性：early 不稳定，提升/下降 2/3/6，early mean delta=-0.079 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；test 很小，几个样本即可造成 pp 级波动。
- 建议：需要保护性早停；用 validation-free proxy 或 baseline fallback。

### FOREX_audjpy-day-High

- 指标：baseline 0.765668 -> step8 0.757493，delta -0.8174 pp，约 -3.0 个 test 样本。
- 任务/规模：binclass，train=1465，test=367，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.056，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-1.998 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### first-order-theorem-proving

- 指标：baseline 0.637255 -> step8 0.629902，delta -0.7353 pp，约 -9.0 个 test 样本。
- 任务/规模：multiclass，train=4894，test=1224，features=51，classes=6。
- JSON 属性：num=51，cat=0，modality=numeric_only，imbalance=5.255，source=openml，domain=text/sequence/symbolic。
- 稳定性：early 6/6 均下降，early mean delta=-0.422 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低 baseline 多分类通常更像结构性难题，TTT 风险高。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### ada

- 指标：baseline 0.848193 -> step8 0.840964，delta -0.7229 pp，约 -6.0 个 test 样本。
- 任务/规模：binclass，train=3317，test=830，features=48，classes=2。
- JSON 属性：num=48，cat=0，modality=numeric_only，imbalance=3.030，source=openml，domain=scientific/health/environment; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.321 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### abalone

- 指标：baseline 0.651914 -> step8 0.644737，delta -0.7177 pp，约 -6.0 个 test 样本。
- 任务/规模：multiclass，train=3341，test=836，features=8，classes=3。
- JSON 属性：num=7，cat=1，modality=mixed，imbalance=1.094，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.877 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### turiye_student_evaluation

- 指标：baseline 0.522337 -> step8 0.515464，delta -0.6873 pp，约 -8.0 个 test 样本。
- 任务/规模：multiclass，train=4656，test=1164，features=32，classes=5。
- JSON 属性：num=30，cat=2，modality=mixed，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.873 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### rice_cammeo_and_osmancik

- 指标：baseline 0.929134 -> step8 0.922572，delta -0.6562 pp，约 -5.0 个 test 样本。
- 任务/规模：binclass，train=3048，test=762，features=7，classes=2。
- JSON 属性：num=7，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-0.547 pp。
- 具体判断：科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### led24

- 指标：baseline 0.734375 -> step8 0.729688，delta -0.4687 pp，约 -3.0 个 test 样本。
- 任务/规模：multiclass，train=2560，test=640，features=24，classes=10。
- JSON 属性：num=0，cat=24，modality=categorical_only，imbalance=1.139，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.469 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；纯类别特征，TTT 主要在离散组合上调整；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### golf_play_dataset_extended

- 指标：baseline 0.936073 -> step8 0.931507，delta -0.4566 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=876，test=219，features=9，classes=2。
- JSON 属性：num=3，cat=6，modality=mixed，imbalance=NA，source=kaggle，domain=other。
- 稳定性：early 多数下降 5/6，early mean delta=-0.837 pp。
- 具体判断：数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### microaggregation2

- 指标：baseline 0.640500 -> step8 0.637000，delta -0.3500 pp，约 -14.0 个 test 样本。
- 任务/规模：multiclass，train=16000，test=4000，features=20，classes=5。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=15.023，source=openml，domain=other。
- 稳定性：early 多数下降 5/6，early mean delta=-0.129 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：需要保护性早停；用 validation-free proxy 或 baseline fallback。

### cmc

- 指标：baseline 0.589831 -> step8 0.586441，delta -0.3390 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=1178，test=295，features=9，classes=3。
- JSON 属性：num=2，cat=7，modality=mixed，imbalance=1.889，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均下降，early mean delta=-1.130 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### ringnorm

- 指标：baseline 0.979730 -> step8 0.976351，delta -0.3378 pp，约 -5.0 个 test 样本。
- 任务/规模：binclass，train=5920，test=1480，features=20，classes=2。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=1.020，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 多数下降 5/6，early mean delta=-0.191 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：需要保护性早停；用 validation-free proxy 或 baseline fallback。

### dna

- 指标：baseline 0.965517 -> step8 0.962382，delta -0.3135 pp，约 -2.0 个 test 样本。
- 任务/规模：multiclass，train=2548，test=638，features=180，classes=3。
- JSON 属性：num=0，cat=180，modality=categorical_only，imbalance=2.162，source=openml，domain=business/customer/finance; scientific/health/environment; text/sequence/symbolic。
- 稳定性：early 6/6 均下降，early mean delta=-0.575 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；业务/客户类字段分布可能更依赖离散类别和采样口径；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；类别特征极多，离散稀疏模式容易被 TTT 过拟合。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### splice

- 指标：baseline 0.956113 -> step8 0.952978，delta -0.3135 pp，约 -2.0 个 test 样本。
- 任务/规模：multiclass，train=2552，test=638，features=60，classes=3。
- JSON 属性：num=0，cat=60，modality=categorical_only，imbalance=2.158，source=openml，domain=business/customer/finance; scientific/health/environment; text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.444 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### mfeat-karhunen

- 指标：baseline 0.975000 -> step8 0.972500，delta -0.2500 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=1600，test=400，features=64，classes=10。
- JSON 属性：num=64，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均下降，early mean delta=-0.250 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### pol

- 指标：baseline 0.983639 -> step8 0.981160，delta -0.2479 pp，约 -5.0 个 test 样本。
- 任务/规模：binclass，train=8065，test=2017，features=26，classes=2。
- JSON 属性：num=26，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.240 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；类别较均衡，delta 更可能反映真实边界变化。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Customer_Personality_Analysis

- 指标：baseline 0.883929 -> step8 0.881696，delta -0.2232 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=1792，test=448，features=24，classes=2。
- JSON 属性：num=22，cat=2，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-0.298 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### churn

- 指标：baseline 0.952000 -> step8 0.950000，delta -0.2000 pp，约 -2.0 个 test 样本。
- 任务/规模：binclass，train=4000，test=1000，features=20，classes=2。
- JSON 属性：num=16，cat=4，modality=mixed，imbalance=6.072，source=openml，domain=business/customer/finance; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 0/3/6，early mean delta=-0.083 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：需要保护性早停；用 validation-free proxy 或 baseline fallback。

### ozone-level-8hr

- 指标：baseline 0.956607 -> step8 0.954635，delta -0.1972 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=2027，test=507，features=72，classes=2。
- JSON 属性：num=72，cat=0，modality=numeric_only，imbalance=14.838，source=openml，domain=scientific/health/environment; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.493 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；类别极不平衡，accuracy 变化可能主要来自多数类边界；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### wine

- 指标：baseline 0.774951 -> step8 0.772994，delta -0.1957 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=2043，test=511，features=4，classes=2。
- JSON 属性：num=4，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=business/customer/finance。
- 稳定性：early 不稳定，提升/下降 3/3/6，early mean delta=0.000 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### house_16H

- 指标：baseline 0.890660 -> step8 0.888807，delta -0.1853 pp，约 -5.0 个 test 样本。
- 任务/规模：binclass，train=10790，test=2698，features=16，classes=2。
- JSON 属性：num=16，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.321 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### E-CommereShippingData

- 指标：baseline 0.678182 -> step8 0.676364，delta -0.1818 pp，约 -4.0 个 test 样本。
- 任务/规模：binclass，train=8799，test=2200，features=10，classes=2。
- JSON 属性：num=6，cat=4，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-0.697 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；业务/客户类字段分布可能更依赖离散类别和采样口径；低 baseline 二分类仍可能通过校正决策边界获益；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Bank_Customer_Churn_Dataset

- 指标：baseline 0.877000 -> step8 0.875500，delta -0.1500 pp，约 -3.0 个 test 样本。
- 任务/规模：binclass，train=8000，test=2000，features=10，classes=2。
- JSON 属性：num=6，cat=4，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-0.317 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Telecom_Churn_Dataset

- 指标：baseline 0.955022 -> step8 0.953523，delta -0.1499 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=2666，test=667，features=17，classes=2。
- JSON 属性：num=14，cat=3，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 不稳定，提升/下降 4/2/6，early mean delta=0.050 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### delta_ailerons

- 指标：baseline 0.950912 -> step8 0.949509，delta -0.1403 pp，约 -2.0 个 test 样本。
- 任务/规模：binclass，train=5703，test=1426，features=5，classes=2。
- JSON 属性：num=5，cat=0，modality=numeric_only，imbalance=1.131，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.152 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### allrep

- 指标：baseline 0.985430 -> step8 0.984106，delta -0.1325 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=3017，test=755，features=29，classes=4。
- JSON 属性：num=6，cat=23，modality=mixed，imbalance=107.294，source=openml，domain=other。
- 稳定性：early 多数下降 5/6，early mean delta=-0.110 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；数值+类别混合特征，收益更依赖类别编码是否稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### Cardiovascular-Disease-dataset

- 指标：baseline 0.734143 -> step8 0.732857，delta -0.1286 pp，约 -18.0 个 test 样本。
- 任务/规模：binclass，train=56000，test=14000，features=11，classes=2。
- JSON 属性：num=5，cat=6，modality=mixed，imbalance=1.001，source=openml，domain=scientific/health/environment。
- 稳定性：early 6/6 均下降，early mean delta=-0.082 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；数值+类别混合特征，收益更依赖类别编码是否稳定；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### spambase

- 指标：baseline 0.959826 -> step8 0.958740，delta -0.1086 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=3680，test=921，features=57，classes=2。
- JSON 属性：num=57，cat=0，modality=numeric_only，imbalance=1.538，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.199 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Wilt

- 指标：baseline 0.992746 -> step8 0.991710，delta -0.1036 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=3856，test=965，features=5，classes=2。
- JSON 属性：num=5，cat=0，modality=numeric_only，imbalance=NA，source=openml，domain=other。
- 稳定性：early 6/6 均下降，early mean delta=-0.104 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### national-longitudinal-survey-binary

- 指标：baseline 1.000000 -> step8 0.998982，delta -0.1018 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=3926，test=982，features=16，classes=2。
- JSON 属性：num=9，cat=7，modality=mixed，imbalance=1.649，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 0/2/6，early mean delta=-0.034 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### waveform_database_generator

- 指标：baseline 0.356000 -> step8 0.355000，delta -0.1000 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=3999，test=1000，features=21，classes=3。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-1.567 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低 baseline 多分类通常更像结构性难题，TTT 风险高；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### Satellite

- 指标：baseline 0.993137 -> step8 0.992157，delta -0.0980 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=4080，test=1020，features=36，classes=2。
- JSON 属性：num=36，cat=0，modality=numeric_only，imbalance=67.000，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.131 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### kdd_ipums_la_97-small

- 指标：baseline 0.888247 -> step8 0.887283，delta -0.0963 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=4150，test=1038，features=20，classes=2。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 3/1/6，early mean delta=0.032 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### wall-robot-navigation

- 指标：baseline 0.986264 -> step8 0.985348，delta -0.0916 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=4364，test=1092，features=24，classes=4。
- JSON 属性：num=24，cat=0，modality=numeric_only，imbalance=6.723，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.153 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### page-blocks

- 指标：baseline 0.978995 -> step8 0.978082，delta -0.0913 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=4378，test=1095，features=10，classes=5。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=175.464，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均下降，early mean delta=-0.091 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### company_bankruptcy_prediction

- 指标：baseline 0.970674 -> step8 0.969941，delta -0.0733 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=5455，test=1364，features=95，classes=2。
- JSON 属性：num=93，cat=2，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-0.073 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### twonorm

- 指标：baseline 0.979054 -> step8 0.978378，delta -0.0676 pp，约 -1.0 个 test 样本。
- 任务/规模：binclass，train=5920，test=1480，features=20，classes=2。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=1.002，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 3/1/6，early mean delta=0.034 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；下降幅度很小，更像统计/离散样本波动。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### pendigits

- 指标：baseline 0.997271 -> step8 0.996817，delta -0.0455 pp，约 -1.0 个 test 样本。
- 任务/规模：multiclass，train=8793，test=2199，features=16，classes=10。
- JSON 属性：num=16，cat=0，modality=numeric_only，imbalance=1.084，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 0/2/6，early mean delta=-0.015 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：下降样本数很少；先视为噪声波动，后续多 seed 验证。

### bank

- 指标：baseline 0.908990 -> step8 0.908659，delta -0.0332 pp，约 -3.0 个 test 样本。
- 任务/规模：binclass，train=36168，test=9043，features=16，classes=2。
- JSON 属性：num=7，cat=9，modality=mixed，imbalance=NA，source=uci，domain=business/customer/finance。
- 稳定性：early 6/6 均下降，early mean delta=-0.046 pp。
- 具体判断：业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：列入 TTT 风险集；默认 baseline 或缩短/跳过 TTT。

### BNG(cmc)

- 指标：baseline 0.586257 -> step8 0.585986，delta -0.0271 pp，约 -3.0 个 test 样本。
- 任务/规模：multiclass，train=44236，test=11060，features=9，classes=3。
- JSON 属性：num=2，cat=7，modality=mixed，imbalance=1.893，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 2/4/6，early mean delta=-0.023 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定；下降幅度很小，更像统计/离散样本波动。
- 建议：需要保护性早停；用 validation-free proxy 或 baseline fallback。

## 提升数据集逐项分析

### jungle_chess_2pcs_raw_endgame_complete

- 指标：baseline 0.862450 -> step8 0.917448，delta +5.4998 pp，约 +493.0 个 test 样本。
- 任务/规模：multiclass，train=35855，test=8964，features=6，classes=3。
- JSON 属性：num=6，cat=0，modality=numeric_only，imbalance=5.320，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=7.651 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；低/中维纯数值特征，TTT 更新较容易形成平滑修正；delta_correct=493.0，不是单纯小样本抖动。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### mfeat-zernike

- 指标：baseline 0.852500 -> step8 0.902500，delta +5.0000 pp，约 +20.0 个 test 样本。
- 任务/规模：multiclass，train=1600，test=400，features=47，classes=10。
- JSON 属性：num=47，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=5.167 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；类别较均衡，delta 更可能反映真实边界变化；step8 提升幅度大，且需要结合 early-step 稳定性判断是否可复用。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### compass

- 指标：baseline 0.826675 -> step8 0.875939，delta +4.9264 pp，约 +164.0 个 test 样本。
- 任务/规模：binclass，train=13315，test=3329，features=17，classes=2。
- JSON 属性：num=8，cat=9，modality=mixed，imbalance=1.000，source=openml，domain=business/customer/finance。
- 稳定性：early 6/6 均提升，early mean delta=5.492 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；类别较均衡，delta 更可能反映真实边界变化；delta_correct=164.0，不是单纯小样本抖动。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### eye_movements_bin

- 指标：baseline 0.625493 -> step8 0.670828，delta +4.5335 pp，约 +69.0 个 test 样本。
- 任务/规模：binclass，train=6086，test=1522，features=20，classes=2。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=5.497 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低 baseline 二分类仍可能通过校正决策边界获益；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### eye_movements

- 指标：baseline 0.714808 -> step8 0.752742，delta +3.7934 pp，约 +83.0 个 test 样本。
- 任务/规模：multiclass，train=8748，test=2188，features=27，classes=3。
- JSON 属性：num=24，cat=3，modality=mixed，imbalance=1.485，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=4.639 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；中等 baseline 多分类是本轮最强受益组合；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### hill-valley

- 指标：baseline 0.930041 -> step8 0.967078，delta +3.7037 pp，约 +9.0 个 test 样本。
- 任务/规模：binclass，train=969，test=243，features=100，classes=2。
- JSON 属性：num=100，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=3.635 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别较均衡，delta 更可能反映真实边界变化；test 很小，几个样本即可造成 pp 级波动；step8 提升幅度大，且需要结合 early-step 稳定性判断是否可复用。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### autoUniv-au4-2500

- 指标：baseline 0.522000 -> step8 0.558000，delta +3.6000 pp，约 +18.0 个 test 样本。
- 任务/规模：multiclass，train=2000，test=500，features=100，classes=3。
- JSON 属性：num=58，cat=42，modality=mixed，imbalance=5.985，source=openml，domain=text/sequence/symbolic; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=3.800 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### Pima_Indians_Diabetes_Database

- 指标：baseline 0.746753 -> step8 0.779221，delta +3.2468 pp，约 +5.0 个 test 样本。
- 任务/规模：binclass，train=614，test=154，features=8，classes=2。
- JSON 属性：num=8，cat=0，modality=numeric_only，imbalance=NA，source=kaggle，domain=scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=1.840 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；test 很小，几个样本即可造成 pp 级波动；step8 提升幅度大，且需要结合 early-step 稳定性判断是否可复用。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### artificial-characters

- 指标：baseline 0.863992 -> step8 0.895303，delta +3.1311 pp，约 +64.0 个 test 样本。
- 任务/规模：multiclass，train=8174，test=2044，features=7，classes=10。
- JSON 属性：num=7，cat=0，modality=numeric_only，imbalance=2.360，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=3.612 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；中等 baseline 多分类是本轮最强受益组合；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### vehicle

- 指标：baseline 0.858824 -> step8 0.882353，delta +2.3529 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=676，test=170，features=18，classes=4。
- JSON 属性：num=18，cat=0，modality=numeric_only，imbalance=1.095，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=2.549 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True

- 指标：baseline 0.602500 -> step8 0.625000，delta +2.2500 pp，约 +9.0 个 test 样本。
- 任务/规模：binclass，train=1600，test=400，features=7，classes=2。
- JSON 属性：num=3，cat=4，modality=mixed，imbalance=1.245，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=1.083 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 二分类仍可能通过校正决策边界获益；数值+类别混合特征，收益更依赖类别编码是否稳定；step8 提升幅度大，且需要结合 early-step 稳定性判断是否可复用。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### FOREX_audusd-hour-High

- 指标：baseline 0.681689 -> step8 0.703137，delta +2.1449 pp，约 +188.0 个 test 样本。
- 任务/规模：binclass，train=35060，test=8765，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.126，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=2.297 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低 baseline 二分类仍可能通过校正决策边界获益；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### FOREX_audsgd-hour-High

- 指标：baseline 0.683856 -> step8 0.704392，delta +2.0536 pp，约 +180.0 个 test 样本。
- 任务/规模：binclass，train=35060，test=8765，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.061，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=2.354 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低 baseline 二分类仍可能通过校正决策边界获益；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### contraceptive_method_choice

- 指标：baseline 0.627119 -> step8 0.647458，delta +2.0339 pp，约 +6.0 个 test 样本。
- 任务/规模：multiclass，train=1178，test=295，features=9，classes=3。
- JSON 属性：num=5，cat=4，modality=mixed，imbalance=NA，source=uci，domain=other。
- 稳定性：early 多数提升 5/6，early mean delta=0.847 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动；step8 提升幅度大，且需要结合 early-step 稳定性判断是否可复用。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### GesturePhaseSegmentationProcessed

- 指标：baseline 0.786329 -> step8 0.804557，delta +1.8228 pp，约 +36.0 个 test 样本。
- 任务/规模：multiclass，train=7898，test=1975，features=32，classes=5。
- JSON 属性：num=32，cat=0，modality=numeric_only，imbalance=2.956，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=2.084 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### FOREX_audjpy-hour-High

- 指标：baseline 0.708614 -> step8 0.724701，delta +1.6087 pp，约 +141.0 个 test 样本。
- 任务/规模：binclass，train=35060，test=8765，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.062，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=1.588 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### website_phishing

- 指标：baseline 0.911439 -> step8 0.926199，delta +1.4760 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=1082，test=271，features=9，classes=3。
- JSON 属性：num=0，cat=9，modality=categorical_only，imbalance=NA，source=uci，domain=business/customer/finance; text/sequence/symbolic。
- 稳定性：early 6/6 均提升，early mean delta=1.353 pp。
- 具体判断：业务/客户类字段分布可能更依赖离散类别和采样口径；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；纯类别特征，TTT 主要在离散组合上调整；test 很小，几个样本即可造成 pp 级波动。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### car-evaluation

- 指标：baseline 0.973988 -> step8 0.988439，delta +1.4451 pp，约 +5.0 个 test 样本。
- 任务/规模：multiclass，train=1382，test=346，features=21，classes=4。
- JSON 属性：num=0，cat=21，modality=categorical_only，imbalance=18.615，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=1.445 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；纯类别特征，TTT 主要在离散组合上调整；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### FOREX_cadjpy-hour-High

- 指标：baseline 0.700742 -> step8 0.714889，delta +1.4147 pp，约 +124.0 个 test 样本。
- 任务/规模：binclass，train=35060，test=8765，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.074，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=1.308 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### autoUniv-au7-1100

- 指标：baseline 0.404545 -> step8 0.418182，delta +1.3636 pp，约 +3.0 个 test 样本。
- 任务/规模：multiclass，train=880，test=220，features=12，classes=5。
- JSON 属性：num=8，cat=4，modality=mixed，imbalance=1.993，source=openml，domain=text/sequence/symbolic; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 2/2/6，early mean delta=0.455 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### FOREX_audcad-day-High

- 指标：baseline 0.743869 -> step8 0.757493，delta +1.3624 pp，约 +5.0 个 test 样本。
- 任务/规模：binclass，train=1467，test=367，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.031，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 不稳定，提升/下降 2/4/6，early mean delta=-0.409 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### Fitness_Club_c

- 指标：baseline 0.786667 -> step8 0.800000，delta +1.3333 pp，约 +4.0 个 test 样本。
- 任务/规模：binclass，train=1200，test=300，features=6，classes=2。
- JSON 属性：num=3，cat=3，modality=mixed，imbalance=NA，source=kaggle，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=1.222 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### mfeat-fourier

- 指标：baseline 0.885000 -> step8 0.897500，delta +1.2500 pp，约 +5.0 个 test 样本。
- 任务/规模：multiclass，train=1600，test=400，features=76，classes=10。
- JSON 属性：num=76，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=1.292 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### mfeat-morphological

- 指标：baseline 0.760000 -> step8 0.770000，delta +1.0000 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=1600，test=400，features=6，classes=10。
- JSON 属性：num=6，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=2.333 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### waveform-5000

- 指标：baseline 0.866000 -> step8 0.876000，delta +1.0000 pp，约 +10.0 个 test 样本。
- 任务/规模：multiclass，train=4000，test=1000，features=40，classes=3。
- JSON 属性：num=40，cat=0，modality=numeric_only，imbalance=1.024，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=1.083 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### GAMETES_Epistasis_2-Way_20atts_0.1H_EDM-1_1

- 指标：baseline 0.668750 -> step8 0.678125，delta +0.9375 pp，约 +3.0 个 test 样本。
- 任务/规模：binclass，train=1280，test=320，features=20，classes=2。
- JSON 属性：num=0，cat=20，modality=categorical_only，imbalance=1.000，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.833 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低 baseline 二分类仍可能通过校正决策边界获益；纯类别特征，TTT 主要在离散组合上调整；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### predict_students_dropout_and_academic_success

- 指标：baseline 0.762712 -> step8 0.770621，delta +0.7910 pp，约 +7.0 个 test 样本。
- 任务/规模：multiclass，train=3539，test=885，features=34，classes=3。
- JSON 属性：num=5，cat=29，modality=mixed，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.603 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；中等 baseline 多分类是本轮最强受益组合；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### ada_prior

- 指标：baseline 0.841183 -> step8 0.848850，delta +0.7667 pp，约 +7.0 个 test 样本。
- 任务/规模：binclass，train=3649，test=913，features=14，classes=2。
- JSON 属性：num=6，cat=8，modality=mixed，imbalance=3.030，source=openml，domain=business/customer/finance; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.639 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### Mobile_Price_Classification

- 指标：baseline 0.940000 -> step8 0.947500，delta +0.7500 pp，约 +3.0 个 test 样本。
- 任务/规模：multiclass，train=1600，test=400，features=20，classes=4。
- JSON 属性：num=14，cat=6，modality=mixed，imbalance=NA，source=kaggle，domain=scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=0.750 pp。
- 具体判断：科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### banknote_authentication

- 指标：baseline 0.549091 -> step8 0.556364，delta +0.7273 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=1097，test=275，features=4，classes=2。
- JSON 属性：num=4，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=business/customer/finance。
- 稳定性：early 多数提升 5/6，early mean delta=0.424 pp。
- 具体判断：baseline 很低，任务可能不仅是边界未调好；业务/客户类字段分布可能更依赖离散类别和采样口径；低 baseline 二分类仍可能通过校正决策边界获益；低/中维纯数值特征，TTT 更新较容易形成平滑修正；test 很小，几个样本即可造成 pp 级波动。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### satimage

- 指标：baseline 0.928460 -> step8 0.935459，delta +0.6998 pp，约 +9.0 个 test 样本。
- 任务/规模：multiclass，train=5144，test=1286，features=36，classes=6。
- JSON 属性：num=36，cat=0，modality=numeric_only，imbalance=2.450，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=0.570 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### madeline

- 指标：baseline 0.753185 -> step8 0.759554，delta +0.6369 pp，约 +4.0 个 test 样本。
- 任务/规模：binclass，train=2512，test=628，features=259，classes=2。
- JSON 属性：num=259，cat=0，modality=numeric_only，imbalance=1.012，source=openml，domain=scientific/health/environment; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=1.141 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；类别较均衡，delta 更可能反映真实边界变化。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### led7

- 指标：baseline 0.737500 -> step8 0.743750，delta +0.6250 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=2560，test=640，features=7，classes=10。
- JSON 属性：num=0，cat=7，modality=categorical_only，imbalance=1.263，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.234 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合；纯类别特征，TTT 主要在离散组合上调整。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### thyroid-ann

- 指标：baseline 0.988079 -> step8 0.993377，delta +0.5298 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=3017，test=755，features=21，classes=3。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=37.505，source=openml，domain=scientific/health/environment; text/sequence/symbolic; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.486 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：保留为 TTT 正样本；优先在 step8/12/15 做早停选择。

### electricity

- 指标：baseline 0.903785 -> step8 0.908529，delta +0.4745 pp，约 +43.0 个 test 样本。
- 任务/规模：binclass，train=36249，test=9063，features=8，classes=2。
- JSON 属性：num=7，cat=1，modality=mixed，imbalance=1.355，source=openml，domain=scientific/health/environment; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.717 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### pc1

- 指标：baseline 0.941441 -> step8 0.945946，delta +0.4505 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=887，test=222，features=21，classes=2。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=13.403，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=0.450 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界；test 很小，几个样本即可造成 pp 级波动。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Marketing_Campaign

- 指标：baseline 0.890625 -> step8 0.895089，delta +0.4464 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=1792，test=448，features=27，classes=2。
- JSON 属性：num=19，cat=8，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 不稳定，提升/下降 2/3/6，early mean delta=0.037 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### segment

- 指标：baseline 0.935065 -> step8 0.939394，delta +0.4329 pp，约 +2.0 个 test 样本。
- 任务/规模：multiclass，train=1848，test=462，features=17，classes=7。
- JSON 属性：num=17，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 多数提升 5/6，early mean delta=0.325 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### gina_agnostic

- 指标：baseline 0.927954 -> step8 0.932277，delta +0.4323 pp，约 +3.0 个 test 样本。
- 任务/规模：binclass，train=2774，test=694，features=970，classes=2。
- JSON 属性：num=970，cat=0，modality=numeric_only，imbalance=NA，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.961 pp。
- 具体判断：。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### telco-customer-churn

- 指标：baseline 0.804826 -> step8 0.809084，delta +0.4258 pp，约 +6.0 个 test 样本。
- 任务/规模：binclass，train=5634，test=1409，features=18，classes=2。
- JSON 属性：num=3，cat=15，modality=mixed，imbalance=2.768，source=openml，domain=business/customer/finance; text/sequence/symbolic; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 4/1/6，early mean delta=0.118 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### allbp

- 指标：baseline 0.978808 -> step8 0.982781，delta +0.3974 pp，约 +3.0 个 test 样本。
- 任务/规模：multiclass，train=3017，test=755，features=29，classes=3。
- JSON 属性：num=6，cat=23，modality=mixed，imbalance=257.786，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.243 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；数值+类别混合特征，收益更依赖类别编码是否稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：有 ceiling，收益有限；不要为这类数据过度调参。

### baseball

- 指标：baseline 0.947761 -> step8 0.951493，delta +0.3731 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=1072，test=268，features=16，classes=3。
- JSON 属性：num=15，cat=1，modality=mixed，imbalance=21.316，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 1/0/6，early mean delta=0.062 pp。
- 具体判断：数值+类别混合特征，收益更依赖类别编码是否稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界；test 很小，几个样本即可造成 pp 级波动。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### pc4

- 指标：baseline 0.914384 -> step8 0.917808，delta +0.3425 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=1166，test=292，features=37，classes=2。
- JSON 属性：num=37，cat=0，modality=numeric_only，imbalance=7.191，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 不稳定，提升/下降 2/2/6，early mean delta=0.000 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；test 很小，几个样本即可造成 pp 级波动。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### MagicTelescope

- 指标：baseline 0.884858 -> step8 0.888275，delta +0.3417 pp，约 +13.0 个 test 样本。
- 任务/规模：binclass，train=15216，test=3804，features=9，classes=2。
- JSON 属性：num=9，cat=0，modality=numeric_only，imbalance=1.844，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.443 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### IBM_HR_Analytics_Employee_Attrition_and_Performance

- 指标：baseline 0.863946 -> step8 0.867347，delta +0.3401 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=1176，test=294，features=31，classes=2。
- JSON 属性：num=25，cat=6，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 6/6 均提升，early mean delta=0.340 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定；test 很小，几个样本即可造成 pp 级波动。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### yeast

- 指标：baseline 0.629630 -> step8 0.632997，delta +0.3367 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=1187，test=297，features=8，classes=10。
- JSON 属性：num=8，cat=0，modality=numeric_only，imbalance=92.600，source=openml，domain=other。
- 稳定性：early 多数提升 5/6，early mean delta=0.281 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别极不平衡，accuracy 变化可能主要来自多数类边界；test 很小，几个样本即可造成 pp 级波动。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### semeion

- 指标：baseline 0.959248 -> step8 0.962382，delta +0.3135 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=1274，test=319，features=256，classes=10。
- JSON 属性：num=256，cat=0，modality=numeric_only，imbalance=1.045，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 2/0/6，early mean delta=0.104 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别较均衡，delta 更可能反映真实边界变化。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### FOREX_audcad-hour-High

- 指标：baseline 0.712037 -> step8 0.715117，delta +0.3080 pp，约 +27.0 个 test 样本。
- 任务/规模：binclass，train=35060，test=8765，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.061，source=openml，domain=financial/FOREX; scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=0.228 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；FOREX 数据存在时间粒度差异，hour 与 day 的 TTT 方向不同；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### waveform_database_generator_version_1

- 指标：baseline 0.866000 -> step8 0.869000，delta +0.3000 pp，约 +3.0 个 test 样本。
- 任务/规模：multiclass，train=4000，test=1000，features=21，classes=3。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=synthetic/rule-generated。
- 稳定性：early 多数提升 5/6，early mean delta=0.283 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；中等 baseline 多分类是本轮最强受益组合。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### default_of_credit_card_clients

- 指标：baseline 0.826167 -> step8 0.829000，delta +0.2833 pp，约 +17.0 个 test 样本。
- 任务/规模：binclass，train=24000，test=6000，features=23，classes=2。
- JSON 属性：num=14，cat=9，modality=mixed，imbalance=NA，source=uci，domain=business/customer/finance。
- 稳定性：early 6/6 均提升，early mean delta=0.222 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### Indian_pines

- 指标：baseline 0.962274 -> step8 0.965008，delta +0.2734 pp，约 +5.0 个 test 样本。
- 任务/规模：multiclass，train=7315，test=1829，features=220，classes=8。
- JSON 属性：num=220，cat=0，modality=numeric_only，imbalance=202.500，source=openml，domain=scientific/health/environment; vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=0.428 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；类别极不平衡，accuracy 变化可能主要来自多数类边界。
- 建议：有 ceiling，收益有限；不要为这类数据过度调参。

### steel_plates_faults

- 指标：baseline 0.838046 -> step8 0.840617，delta +0.2571 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=1552，test=389，features=27，classes=7。
- JSON 属性：num=25，cat=2，modality=mixed，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.728 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；中等 baseline 多分类是本轮最强受益组合；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### heloc

- 指标：baseline 0.725500 -> step8 0.728000，delta +0.2500 pp，约 +5.0 个 test 样本。
- 任务/规模：binclass，train=8000，test=2000，features=22，classes=2。
- JSON 属性：num=22，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=business/customer/finance。
- 稳定性：early 不稳定，提升/下降 2/3/6，early mean delta=-0.042 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；类别较均衡，delta 更可能反映真实边界变化。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### kc1

- 指标：baseline 0.881517 -> step8 0.883886，delta +0.2370 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=1687，test=422，features=21，classes=2。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=5.469，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 不稳定，提升/下降 4/0/6，early mean delta=0.158 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### mobile_c36_oversampling

- 指标：baseline 0.990533 -> step8 0.992755，delta +0.2222 pp，约 +23.0 个 test 样本。
- 任务/规模：binclass，train=41408，test=10352，features=6，classes=2。
- JSON 属性：num=6，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.248 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：有 ceiling，收益有限；不要为这类数据过度调参。

### National_Health_and_Nutrition_Health_Survey

- 指标：baseline 0.839912 -> step8 0.842105，delta +0.2193 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=1822，test=456，features=7，classes=2。
- JSON 属性：num=7，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=scientific/health/environment。
- 稳定性：early 不稳定，提升/下降 1/4/6，early mean delta=-0.146 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Employee

- 指标：baseline 0.852846 -> step8 0.854995，delta +0.2148 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=3722，test=931，features=8，classes=2。
- JSON 属性：num=3，cat=5，modality=mixed，imbalance=NA，source=kaggle，domain=business/customer/finance。
- 稳定性：early 不稳定，提升/下降 2/3/6，early mean delta=-0.143 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### credit

- 指标：baseline 0.782830 -> step8 0.784924，delta +0.2094 pp，约 +7.0 个 test 样本。
- 任务/规模：binclass，train=13371，test=3343，features=10，classes=2。
- JSON 属性：num=10，cat=0，modality=numeric_only，imbalance=1.000，source=openml，domain=business/customer/finance。
- 稳定性：early 6/6 均提升，early mean delta=0.199 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；业务/客户类字段分布可能更依赖离散类别和采样口径；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化。
- 建议：可保留，但建议用 early-step 稳定性筛选。

### Pumpkin_Seeds

- 指标：baseline 0.872000 -> step8 0.874000，delta +0.2000 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=2000，test=500，features=12，classes=2。
- JSON 属性：num=12，cat=0，modality=numeric_only，imbalance=NA，source=kaggle，domain=other。
- 稳定性：early 不稳定，提升/下降 2/3/6，early mean delta=-0.167 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### thyroid-dis

- 指标：baseline 0.691071 -> step8 0.692857，delta +0.1786 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=2240，test=560，features=26，classes=5。
- JSON 属性：num=6，cat=20，modality=mixed，imbalance=52.645，source=openml，domain=scientific/health/environment; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 4/0/6，early mean delta=0.119 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低 baseline 多分类通常更像结构性难题，TTT 风险高；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### BNG(tic-tac-toe)

- 指标：baseline 0.815088 -> step8 0.816866，delta +0.1778 pp，约 +14.0 个 test 样本。
- 任务/规模：binclass，train=31492，test=7874，features=9，classes=2。
- JSON 属性：num=0，cat=9，modality=categorical_only，imbalance=1.881，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.169 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；纯类别特征，TTT 主要在离散组合上调整；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### htru

- 指标：baseline 0.979330 -> step8 0.981006，delta +0.1676 pp，约 +6.0 个 test 样本。
- 任务/规模：binclass，train=14318，test=3580，features=8，classes=2。
- JSON 属性：num=8，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.154 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### online_shoppers

- 指标：baseline 0.904298 -> step8 0.905921，delta +0.1622 pp，约 +4.0 个 test 样本。
- 任务/规模：binclass，train=9864，test=2466，features=14，classes=2。
- JSON 属性：num=5，cat=9，modality=mixed，imbalance=5.462，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.128 pp。
- 具体判断：数值+类别混合特征，收益更依赖类别编码是否稳定；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### naticusdroid+android+permissions+dataset

- 指标：baseline 0.971024 -> step8 0.972558，delta +0.1534 pp，约 +9.0 个 test 样本。
- 任务/规模：binclass，train=23465，test=5867，features=86，classes=2。
- JSON 属性：num=0，cat=86，modality=categorical_only，imbalance=NA，source=uci，domain=text/sequence/symbolic。
- 稳定性：early 6/6 均提升，early mean delta=0.111 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；类别特征极多，离散稀疏模式容易被 TTT 过拟合；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### BLE_RSSI_dataset_for_Indoor_localization

- 指标：baseline 0.740611 -> step8 0.742113，delta +0.1502 pp，约 +3.0 个 test 样本。
- 任务/规模：multiclass，train=7987，test=1997，features=3，classes=3。
- JSON 属性：num=3，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.342 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；中等 baseline 多分类是本轮最强受益组合；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### dry_bean_dataset

- 指标：baseline 0.927653 -> step8 0.929122，delta +0.1469 pp，约 +4.0 个 test 样本。
- 任务/规模：multiclass，train=10888，test=2723，features=16，classes=7。
- JSON 属性：num=16，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=0.208 pp。
- 具体判断：科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### taiwanese_bankruptcy_prediction

- 指标：baseline 0.970674 -> step8 0.972141，delta +0.1466 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=5455，test=1364，features=95，classes=2。
- JSON 属性：num=95，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=business/customer/finance。
- 稳定性：early 6/6 均提升，early mean delta=0.147 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；业务/客户类字段分布可能更依赖离散类别和采样口径；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### eeg-eye-state

- 指标：baseline 0.991322 -> step8 0.992657，delta +0.1335 pp，约 +4.0 个 test 样本。
- 任务/规模：binclass，train=11984，test=2996，features=14，classes=2。
- JSON 属性：num=14，cat=0，modality=numeric_only，imbalance=1.228，source=openml，domain=synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.211 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### dis

- 指标：baseline 0.984106 -> step8 0.985430，delta +0.1325 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=3017，test=755，features=29，classes=2。
- JSON 属性：num=6，cat=23，modality=mixed，imbalance=64.034，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.221 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；数值+类别混合特征，收益更依赖类别编码是否稳定；类别极不平衡，accuracy 变化可能主要来自多数类边界；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Gender_Gap_in_Spanish_WP

- 指标：baseline 0.605263 -> step8 0.606316，delta +0.1053 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=3796，test=950，features=13，classes=3。
- JSON 属性：num=13，cat=0，modality=numeric_only，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.316 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；低 baseline 多分类通常更像结构性难题，TTT 风险高；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### HR_Analytics_Job_Change_of_Data_Scientists

- 指标：baseline 0.799322 -> step8 0.800365，delta +0.1044 pp，约 +4.0 个 test 样本。
- 任务/规模：binclass，train=15326，test=3832，features=13，classes=2。
- JSON 属性：num=3，cat=10，modality=mixed，imbalance=NA，source=kaggle，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.157 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；数值+类别混合特征，收益更依赖类别编码是否稳定；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### wine-quality-white

- 指标：baseline 0.682653 -> step8 0.683673，delta +0.1020 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=3918，test=980，features=11，classes=7。
- JSON 属性：num=11，cat=0，modality=numeric_only，imbalance=439.600，source=openml，domain=scientific/health/environment; vision/sensor/pattern。
- 稳定性：early 不稳定，提升/下降 3/0/6，early mean delta=0.136 pp。
- 具体判断：baseline 偏低但仍有一定结构信号；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低 baseline 多分类通常更像结构性难题，TTT 风险高；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### rl

- 指标：baseline 0.854125 -> step8 0.855131，delta +0.1006 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=3976，test=994，features=12，classes=2。
- JSON 属性：num=5，cat=7，modality=mixed，imbalance=1.000，source=openml，domain=scientific/health/environment; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.419 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；数值+类别混合特征，收益更依赖类别编码是否稳定；类别较均衡，delta 更可能反映真实边界变化。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### okcupid_stem

- 指标：baseline 0.747564 -> step8 0.748501，delta +0.0937 pp，约 +5.0 个 test 样本。
- 任务/规模：multiclass，train=21341，test=5336，features=13，classes=3。
- JSON 属性：num=2，cat=11，modality=mixed，imbalance=6.826，source=openml，domain=text/sequence/symbolic; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.131 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；中等 baseline 多分类是本轮最强受益组合；数值+类别混合特征，收益更依赖类别编码是否稳定。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Firm-Teacher_Clave-Direction_Classification

- 指标：baseline 0.878241 -> step8 0.879167，delta +0.0926 pp，约 +2.0 个 test 样本。
- 任务/规模：multiclass，train=8640，test=2160，features=16，classes=4。
- JSON 属性：num=0，cat=16，modality=categorical_only，imbalance=NA，source=uci，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.193 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；中等 baseline 多分类是本轮最强受益组合；纯类别特征，TTT 主要在离散组合上调整；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### phoneme

- 指标：baseline 0.903793 -> step8 0.904718，delta +0.0925 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=4323，test=1081，features=5，classes=2。
- JSON 属性：num=5，cat=0，modality=numeric_only，imbalance=2.407，source=openml，domain=business/customer/finance; scientific/health/environment; text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 不稳定，提升/下降 4/2/6，early mean delta=0.031 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；低/中维纯数值特征，TTT 更新较容易形成平滑修正。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### jm1

- 指标：baseline 0.821314 -> step8 0.822232，delta +0.0919 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=8708，test=2177，features=21，classes=2。
- JSON 属性：num=21，cat=0，modality=numeric_only，imbalance=4.169，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.268 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Amazon_employee_access

- 指标：baseline 0.944309 -> step8 0.945224，delta +0.0915 pp，约 +6.0 个 test 样本。
- 任务/规模：binclass，train=26215，test=6554，features=7，classes=2。
- JSON 属性：num=0，cat=7，modality=categorical_only，imbalance=16.274，source=openml，domain=business/customer/finance; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.145 pp。
- 具体判断：JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；纯类别特征，TTT 主要在离散组合上调整；类别极不平衡，accuracy 变化可能主要来自多数类边界；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### optdigits

- 指标：baseline 0.995552 -> step8 0.996441，delta +0.0890 pp，约 +1.0 个 test 样本。
- 任务/规模：multiclass，train=4496，test=1124，features=64，classes=10。
- JSON 属性：num=64，cat=0，modality=numeric_only，imbalance=1.032，source=openml，domain=vision/sensor/pattern。
- 稳定性：early 6/6 均提升，early mean delta=0.089 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；类别较均衡，delta 更可能反映真实边界变化；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### gas-drift

- 指标：baseline 0.996405 -> step8 0.997124，delta +0.0719 pp，约 +2.0 个 test 样本。
- 任务/规模：multiclass，train=11128，test=2782，features=128，classes=6。
- JSON 属性：num=128，cat=0，modality=numeric_only，imbalance=1.834，source=openml，domain=vision/sensor/pattern; synthetic/rule-generated。
- 稳定性：early 6/6 均提升，early mean delta=0.054 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### water_quality

- 指标：baseline 0.908125 -> step8 0.908750，delta +0.0625 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=6396，test=1600，features=20，classes=2。
- JSON 属性：num=20，cat=0，modality=numeric_only，imbalance=NA，source=kaggle，domain=scientific/health/environment。
- 稳定性：early 6/6 均提升，early mean delta=0.167 pp。
- 具体判断：科学/医疗/环境类常见小样本或噪声测量，TTT 容易受局部 batch 影响；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### PhishingWebsites

- 指标：baseline 0.979195 -> step8 0.979647，delta +0.0452 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=8844，test=2211，features=30，classes=2。
- JSON 属性：num=0，cat=30，modality=categorical_only，imbalance=1.257，source=openml，domain=text/sequence/symbolic; vision/sensor/pattern。
- 稳定性：early 不稳定，提升/下降 2/2/6，early mean delta=-0.015 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；序列/符号表格化特征依赖位置或 token 结构，TTT 可能放大或破坏局部模式；纯类别特征，TTT 主要在离散组合上调整；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### mammography

- 指标：baseline 0.988824 -> step8 0.989271，delta +0.0447 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=8946，test=2237，features=6，classes=2。
- JSON 属性：num=6，cat=0，modality=numeric_only，imbalance=42.012，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 2/1/6，early mean delta=-0.007 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别极不平衡，accuracy 变化可能主要来自多数类边界；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### in_vehicle_coupon_recommendation

- 指标：baseline 0.790698 -> step8 0.791092，delta +0.0394 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=10147，test=2537，features=21，classes=2。
- JSON 属性：num=0，cat=21，modality=categorical_only，imbalance=NA，source=uci，domain=business/customer/finance; vision/sensor/pattern。
- 稳定性：early 不稳定，提升/下降 4/2/6，early mean delta=0.033 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；JSON 描述显示为规则/传感器/模式识别类，适配信号通常较稳定；业务/客户类字段分布可能更依赖离散类别和采样口径；纯类别特征，TTT 主要在离散组合上调整；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### INNHotelsGroup

- 指标：baseline 0.902136 -> step8 0.902412，delta +0.0276 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=29020，test=7255，features=17，classes=2。
- JSON 属性：num=11，cat=6，modality=mixed，imbalance=NA，source=kaggle，domain=other。
- 稳定性：early 不稳定，提升/下降 4/0/6，early mean delta=0.021 pp。
- 具体判断：数值+类别混合特征，收益更依赖类别编码是否稳定；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### BNG(breast-w)

- 指标：baseline 0.988316 -> step8 0.988570，delta +0.0254 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=31492，test=7874，features=9，classes=2。
- JSON 属性：num=9，cat=0，modality=numeric_only，imbalance=1.906，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.028 pp。
- 具体判断：baseline 接近 ceiling，剩余 headroom 很小；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### Click_prediction_small

- 指标：baseline 0.832040 -> step8 0.832290，delta +0.0250 pp，约 +2.0 个 test 样本。
- 任务/规模：binclass，train=31958，test=7990，features=3，classes=2。
- JSON 属性：num=3，cat=0，modality=numeric_only，imbalance=NA，source=openml，domain=other。
- 稳定性：early 6/6 均提升，early mean delta=0.025 pp。
- 具体判断：baseline 位于 0.7-0.9 的有效适配区间；低/中维纯数值特征，TTT 更新较容易形成平滑修正；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

### California-Housing-Classification

- 指标：baseline 0.912791 -> step8 0.913033，delta +0.0242 pp，约 +1.0 个 test 样本。
- 任务/规模：binclass，train=16512，test=4128，features=8，classes=2。
- JSON 属性：num=8，cat=0，modality=numeric_only，imbalance=1.001，source=openml，domain=other。
- 稳定性：early 不稳定，提升/下降 2/4/6，early mean delta=-0.057 pp。
- 具体判断：低/中维纯数值特征，TTT 更新较容易形成平滑修正；类别较均衡，delta 更可能反映真实边界变化；提升幅度很小，应视为弱信号。
- 建议：只作弱提升样本；需要多 seed 或更大评测确认。

## 索引表

### Top 25 下降

| dataset_name                             | task_type   |   baseline_acc |   delta_pp |   delta_correct | stability                     | feature_modality   | domain_tag                                                                                                                        |
|:-----------------------------------------|:------------|---------------:|-----------:|----------------:|:------------------------------|:-------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| Basketball_c                             | binclass    |         0.7127 |    -1.8657 |         -5.0000 | early 6/6 均下降              | numeric_only       | business/customer/finance                                                                                                         |
| FOREX_audchf-day-High                    | binclass    |         0.7548 |    -1.6349 |         -6.0000 | early 6/6 均下降              | numeric_only       | financial/FOREX; scientific/health/environment                                                                                    |
| wine-quality-red                         | multiclass  |         0.6687 |    -1.2500 |         -4.0000 | early 6/6 均下降              | numeric_only       | other                                                                                                                             |
| Water_Quality_and_Potability             | binclass    |         0.6418 |    -1.2195 |         -8.0000 | early 6/6 均下降              | numeric_only       | scientific/health/environment                                                                                                     |
| MIC                                      | binclass    |         0.9091 |    -1.2121 |         -4.0000 | early 6/6 均下降              | mixed              | scientific/health/environment                                                                                                     |
| sports_articles_for_objectivity_analysis | binclass    |         0.8550 |    -1.0000 |         -2.0000 | early 6/6 均下降              | mixed              | text/sequence/symbolic                                                                                                            |
| statlog                                  | binclass    |         0.7450 |    -1.0000 |         -2.0000 | early 6/6 均下降              | mixed              | business/customer/finance                                                                                                         |
| QSAR_biodegradation                      | binclass    |         0.8863 |    -0.9479 |         -2.0000 | early 不稳定，提升/下降 2/3/6 | numeric_only       | scientific/health/environment                                                                                                     |
| FOREX_audjpy-day-High                    | binclass    |         0.7657 |    -0.8174 |         -3.0000 | early 6/6 均下降              | numeric_only       | financial/FOREX; scientific/health/environment                                                                                    |
| first-order-theorem-proving              | multiclass  |         0.6373 |    -0.7353 |         -9.0000 | early 6/6 均下降              | numeric_only       | text/sequence/symbolic                                                                                                            |
| ada                                      | binclass    |         0.8482 |    -0.7229 |         -6.0000 | early 6/6 均下降              | numeric_only       | scientific/health/environment; vision/sensor/pattern; synthetic/rule-generated                                                    |
| abalone                                  | multiclass  |         0.6519 |    -0.7177 |         -6.0000 | early 6/6 均下降              | mixed              | other                                                                                                                             |
| turiye_student_evaluation                | multiclass  |         0.5223 |    -0.6873 |         -8.0000 | early 6/6 均下降              | mixed              | other                                                                                                                             |
| rice_cammeo_and_osmancik                 | binclass    |         0.9291 |    -0.6562 |         -5.0000 | early 6/6 均下降              | numeric_only       | scientific/health/environment                                                                                                     |
| led24                                    | multiclass  |         0.7344 |    -0.4687 |         -3.0000 | early 6/6 均下降              | categorical_only   | synthetic/rule-generated                                                                                                          |
| golf_play_dataset_extended               | binclass    |         0.9361 |    -0.4566 |         -1.0000 | early 多数下降 5/6            | mixed              | other                                                                                                                             |
| microaggregation2                        | multiclass  |         0.6405 |    -0.3500 |        -14.0000 | early 多数下降 5/6            | numeric_only       | other                                                                                                                             |
| cmc                                      | multiclass  |         0.5898 |    -0.3390 |         -1.0000 | early 6/6 均下降              | mixed              | vision/sensor/pattern                                                                                                             |
| ringnorm                                 | binclass    |         0.9797 |    -0.3378 |         -5.0000 | early 多数下降 5/6            | numeric_only       | synthetic/rule-generated                                                                                                          |
| dna                                      | multiclass  |         0.9655 |    -0.3135 |         -2.0000 | early 6/6 均下降              | categorical_only   | business/customer/finance; scientific/health/environment; text/sequence/symbolic                                                  |
| splice                                   | multiclass  |         0.9561 |    -0.3135 |         -2.0000 | early 6/6 均下降              | categorical_only   | business/customer/finance; scientific/health/environment; text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated |
| mfeat-karhunen                           | multiclass  |         0.9750 |    -0.2500 |         -1.0000 | early 6/6 均下降              | numeric_only       | vision/sensor/pattern                                                                                                             |
| pol                                      | binclass    |         0.9836 |    -0.2479 |         -5.0000 | early 6/6 均下降              | numeric_only       | other                                                                                                                             |
| Customer_Personality_Analysis            | binclass    |         0.8839 |    -0.2232 |         -1.0000 | early 6/6 均下降              | mixed              | business/customer/finance                                                                                                         |
| churn                                    | binclass    |         0.9520 |    -0.2000 |         -2.0000 | early 不稳定，提升/下降 0/3/6 | mixed              | business/customer/finance; vision/sensor/pattern; synthetic/rule-generated                                                        |

### Top 25 提升

| dataset_name                                                   | task_type   |   baseline_acc |   delta_pp |   delta_correct | stability                     | feature_modality   | domain_tag                                                              |
|:---------------------------------------------------------------|:------------|---------------:|-----------:|----------------:|:------------------------------|:-------------------|:------------------------------------------------------------------------|
| jungle_chess_2pcs_raw_endgame_complete                         | multiclass  |         0.8624 |     5.4998 |        493.0000 | early 6/6 均提升              | numeric_only       | synthetic/rule-generated                                                |
| mfeat-zernike                                                  | multiclass  |         0.8525 |     5.0000 |         20.0000 | early 6/6 均提升              | numeric_only       | vision/sensor/pattern                                                   |
| compass                                                        | binclass    |         0.8267 |     4.9264 |        164.0000 | early 6/6 均提升              | mixed              | business/customer/finance                                               |
| eye_movements_bin                                              | binclass    |         0.6255 |     4.5335 |         69.0000 | early 6/6 均提升              | numeric_only       | text/sequence/symbolic; vision/sensor/pattern                           |
| eye_movements                                                  | multiclass  |         0.7148 |     3.7934 |         83.0000 | early 6/6 均提升              | mixed              | text/sequence/symbolic; vision/sensor/pattern                           |
| hill-valley                                                    | binclass    |         0.9300 |     3.7037 |          9.0000 | early 6/6 均提升              | numeric_only       | synthetic/rule-generated                                                |
| autoUniv-au4-2500                                              | multiclass  |         0.5220 |     3.6000 |         18.0000 | early 6/6 均提升              | mixed              | text/sequence/symbolic; synthetic/rule-generated                        |
| Pima_Indians_Diabetes_Database                                 | binclass    |         0.7468 |     3.2468 |          5.0000 | early 6/6 均提升              | numeric_only       | scientific/health/environment                                           |
| artificial-characters                                          | multiclass  |         0.8640 |     3.1311 |         64.0000 | early 6/6 均提升              | numeric_only       | text/sequence/symbolic; vision/sensor/pattern; synthetic/rule-generated |
| vehicle                                                        | multiclass  |         0.8588 |     2.3529 |          4.0000 | early 6/6 均提升              | numeric_only       | vision/sensor/pattern; synthetic/rule-generated                         |
| airlines_seed_0_nrows_2000_nclasses_10_ncols_100_stratify_True | binclass    |         0.6025 |     2.2500 |          9.0000 | early 6/6 均提升              | mixed              | other                                                                   |
| FOREX_audusd-hour-High                                         | binclass    |         0.6817 |     2.1449 |        188.0000 | early 6/6 均提升              | numeric_only       | financial/FOREX; scientific/health/environment                          |
| FOREX_audsgd-hour-High                                         | binclass    |         0.6839 |     2.0536 |        180.0000 | early 6/6 均提升              | numeric_only       | financial/FOREX; scientific/health/environment                          |
| contraceptive_method_choice                                    | multiclass  |         0.6271 |     2.0339 |          6.0000 | early 多数提升 5/6            | mixed              | other                                                                   |
| GesturePhaseSegmentationProcessed                              | multiclass  |         0.7863 |     1.8228 |         36.0000 | early 6/6 均提升              | numeric_only       | vision/sensor/pattern; synthetic/rule-generated                         |
| FOREX_audjpy-hour-High                                         | binclass    |         0.7086 |     1.6087 |        141.0000 | early 6/6 均提升              | numeric_only       | financial/FOREX; scientific/health/environment                          |
| website_phishing                                               | multiclass  |         0.9114 |     1.4760 |          4.0000 | early 6/6 均提升              | categorical_only   | business/customer/finance; text/sequence/symbolic                       |
| car-evaluation                                                 | multiclass  |         0.9740 |     1.4451 |          5.0000 | early 6/6 均提升              | categorical_only   | other                                                                   |
| FOREX_cadjpy-hour-High                                         | binclass    |         0.7007 |     1.4147 |        124.0000 | early 6/6 均提升              | numeric_only       | financial/FOREX; scientific/health/environment                          |
| autoUniv-au7-1100                                              | multiclass  |         0.4045 |     1.3636 |          3.0000 | early 不稳定，提升/下降 2/2/6 | mixed              | text/sequence/symbolic; synthetic/rule-generated                        |
| FOREX_audcad-day-High                                          | binclass    |         0.7439 |     1.3624 |          5.0000 | early 不稳定，提升/下降 2/4/6 | numeric_only       | financial/FOREX; scientific/health/environment                          |
| Fitness_Club_c                                                 | binclass    |         0.7867 |     1.3333 |          4.0000 | early 6/6 均提升              | mixed              | other                                                                   |
| mfeat-fourier                                                  | multiclass  |         0.8850 |     1.2500 |          5.0000 | early 6/6 均提升              | numeric_only       | vision/sensor/pattern                                                   |
| mfeat-morphological                                            | multiclass  |         0.7600 |     1.0000 |          4.0000 | early 6/6 均提升              | numeric_only       | vision/sensor/pattern                                                   |
| waveform-5000                                                  | multiclass  |         0.8660 |     1.0000 |         10.0000 | early 6/6 均提升              | numeric_only       | synthetic/rule-generated                                                |
