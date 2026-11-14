# 2026-11-14 实验与结果归档

本轮聚焦 **MI、SSVEP、信号可靠性**，没有新增情绪训练。所有数字均为公开代理数据上的开发验证结果，不是比赛官方成绩；固定测试池未评分。不同数据集与标签任务分表，不能直接混算平均分。

完整模型、保存预测、采样索引和曲线见 [当日 Release](https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/artifacts-20261114)。附件逐文件与分卷清单为 [release_artifacts_manifest_20261114.json](../../storage/release_artifacts_manifest_20261114.json)。Git保存全部小型报告、配置、结果表、日志及核验记录；原始EEG与原始第三方预训练权重不在附件中。

## 1. 本轮计数

| 实验 | 新增正式神经拟合 | 短流程拟合 | 其他评分/校准 |
|---|---:|---:|---|
| PhysioNet MI专用基线 | 6 | 2 | 3次固定融合评分 |
| EEGConformer与CBraMod初始化对照 | 9 | 3 | 无 |
| CBraMod归一化对照 | 6 | 0 | 复用6次旧控制，不重复计拟合 |
| Kalunga2016经典SSVEP | 0 | 0 | 12次源阈值校准、24条验证记录 |
| 合计 | **21** | **5** | 各类评分另列 |

另有一次CPU流程检查在数据加载阶段取消，没有生成模型或成绩，不计拟合。信号可靠性只新增数据接口审计，不重复队友的训练。队友已发布的九组结构对照见 [experiments/tuar_multitask_20261113](../../experiments/tuar_multitask_20261113/)，不计入这里的21次。

## 2. MI：专用模型与基础模型

PhysioNet MI：109人、9837个四类想象trial，64通道、200Hz、4秒。三个本地开发重划分各87人训练/11人验证，固定11人测试池不评分。验证集合可能重叠；标准差混合划分与训练种子变化，不代表纯训练随机性。统一逐步采样与1500更新预算，架构之间的归一化和优化配方差异另列。

| 条件 | 平均验证逐被试BA |
|---|---:|
| EEGNet | 59.13% |
| Shallow | 61.29% |
| 固定0.75 EEGNet＋0.25 Shallow融合 | **62.26%** |
| HUST-BCIML EEGConformer实现 | 58.01% |
| 预训练CBraMod，uV/100 | 61.01% |
| 随机CBraMod，uV/100 | 55.28% |
| 预训练CBraMod，逐trial时间zscore | 61.56% |
| 随机CBraMod，逐trial时间zscore | 54.62% |

同架构同预算下，预训练相对随机在uV/100和zscore条件分别提高5.73pp、6.94pp，三个开发划分方向均为正。此结果不是随机模型充分收敛的上限比较。

zscore相对uV/100：预训练组逐划分+1.64/+1.23/−1.22pp，均值+0.55pp；随机组均值−0.66pp。未达到事前三划分方向一致要求，不把zscore宣布为稳定改进。既有轻量融合仍比zscore预训练高约0.70pp。

- [专用模型报告](../../outputs/physionet_mi_20261114/report/结果报告.md)
- [基础模型对照报告](../../outputs/physionet_models_20261114/report/结果报告.md)
- [归一化报告](../../outputs/physionet_normalization_20261114/report/结果报告.md)
- [归一化回传完整性与独立复算回执](../../outputs/physionet_normalization_20261114/completion_verification.json)

## 3. SSVEP：独立代理库

Kalunga2016开发侧10人、23个官方run、736个事件，8个真实EEG导联，统一200Hz。被试11/12未下载或评分；三次开发验证覆盖1–6，被试7–10仅在训练侧。固定CCA/FBCCA配方，无验证侧参数搜索。

| 方法 | 1秒三类激活 | 2秒三类激活 | 1秒含rest四类 | 2秒含rest四类 |
|---|---:|---:|---:|---:|
| CCA | 67.36% | 74.31% | 51.04% | 64.06% |
| FBCCA | **76.74%** | **87.15%** | **67.19%** | **74.74%** |

三类激活为免训练频率识别；四类包含仅在训练被试选择的rest拒识阈值，不称免校准四类模型。事件映射遵循MOABB的17/21Hz修正，从原始FIF重建，不使用其他项目缓存。全run离线滤波使用窗外上下文；输入1/2秒不是严格在线延迟。

[SSVEP报告](../../outputs/kalunga_classical_20261114/结果报告.md)与[24份预测独立核验](../../outputs/kalunga_classical_20261114/independent_audit.json)。两个窗口来自同一数据集，不作为比赛两个独立数据集。

## 4. 信号可靠性与文献资源

已读取TUAR v3.0.1的310个EDF头与TUEV v2.0.1的518个EDF头。TUAR包含多采样率、复合伪迹标签，4个文件缺同名主CSV；TUEV REC通道号对应TCP双极montage，不能按原始EDF信号序号直接取通道。没有自行改变队友标签定义，也没有新增该方向模型成绩。

- [TUAR/TUEV输入核对](../../plans/tuh_input_audit_20261114.md)
- [IEEE/期刊、基础模型及实验室资源调研](../../plans/model_research_20261114.md)
- [队友截图转录与判读](../../plans/teammate_results_20261114.md)仅作沟通记录，不是独立复算成绩。

## 5. 复现与恢复

```powershell
python scripts/restore_release_artifacts.py --manifest storage/release_artifacts_manifest_20261114.json --destination D:/OmniEEG-20261114-artifacts
```

目标目录必须为空；恢复脚本逐分卷及逐文件核对SHA256。档案保留训练时原始配置和源文件字节，绝对路径是来源记录，不代表新机器可直接使用该目录。新机器运行时通过CLI指定数据、权重与模型代码位置，不改写旧记录冒充原实验。

外部依赖见 [external_dependencies_20261114.json](../../storage/external_dependencies_20261114.json)：CBraMod按现有submodule commit固定；HUST-BCIML固定到本轮核验的commit；原始公开CBraMod权重按来源URL获取并核对SHA256。未将其他人的SSVEP私有预训练检查点纳入发布。

主要入口：`run_physionet_mi.py`、`run_physionet_models.py`、`run_physionet_normalization.py`、`run_kalunga_classical.py`及对应`report_*`和`verify_normalization_artifacts.py`。后两个MI入口依赖前序保存的控制与采样索引；本次Release完整包含这些依赖。

## 6. 结论边界

当前证据支持固定适配预算下的预训练收益，以及经典SSVEP基线在独立代理库的可运行性；不支持归一化具有稳定普遍收益，不证明复杂专家或学习路由有效，也不证明比赛隐藏数据集上的通用性能。没有新增独立测试成绩，没有重新使用锁定测试选择模型。
