# OmniEEG 实验工作区

公开代码与实验记录仓库：https://github.com/ChenshuoXing0406/omnieeg-experiments 。
2026-11-14实验入口与完整结果索引：[当日归档](reports/20261114/README.md)；模型及预测附件：[当日Release](https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/artifacts-20261114)。
队友协作见 [CONTRIBUTING.md](CONTRIBUTING.md)，数据与第三方代码获取见 [DATASETS.md](DATASETS.md)。
实验模型、验证预测及日志已提供为 [Release 分卷附件](https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/artifacts-20261113)，清单见 `storage/release_artifacts_manifest.json`。
除单列Release中的BNCI原始T会话文件外，其他原始EEG、DE/EEG特征缓存、原始第三方预训练权重和SEED压缩包未公开，需按授权获取或从个人服务器私有备份恢复。
恢复实验附件：`python scripts/restore_release_artifacts.py --destination D:/OmniEEG-artifacts`（目标目录需为空，需预留分卷和解包空间）。
主分支可用 `git clone --recurse-submodules https://github.com/ChenshuoXing0406/omnieeg-experiments.git` 获取五个锁定版本外部仓库；EmT单文件及其CBCR非商业许可证随仓库保留。

2026-11-13完成私有服务器备份与大文件清理：三个SEED压缩包、16个工作区分卷及服务器解包后的7048个文件均通过SHA256校验。本机移除5470个数据、模型及临时文件，共54.38GB，其中原有项目文件37.89GB、迁移临时归档16.49GB。代码、结果表和恢复清单保留；少量本地迁移审计资料暂留。
私有服务器恢复位置记录在本机 `storage/SERVER_BACKUP_MANIFEST_PRIVATE.json`，清理回执在 `storage/LOCAL_CLEANUP_RECEIPT_PRIVATE.json`，两者不提交公开仓库。虚拟环境与大文件已清理，重跑前需要重新建立环境并恢复所需数据；服务器已保存实际解包副本。

本工作区从组内研究方案出发，先运行公开数据基线，再检验多任务训练和 MoE。
已经接入官方 CBraMod，并完成 MI、SSVEP 及历史情绪方向的真实数据搜索。
2026-11-14 当前参赛范围确定为 MI、SSVEP、信号可靠性，不再参加情绪赛道；历史情绪结果保留作研究记录。
队友负责联合架构与专家路由对照；本工作区另开64通道MI数据集验证，实验安排见 `plans/physionet_mi_20261114.md`，入口 `scripts/run_physionet_mi.py`。
本地公开数据实验不是赛事官方数据或正式成绩，也不证明预训练或 MoE 有效。

2026-11-14 MI归一化2×2配对已完成：复用6次uV/100控制，新增6次zscore训练；协议与结论见`plans/physionet_normalization_20261114.md`。

## 本轮完成结果

### 2026-11-14 MI归一化配对（6次新增正式拟合完成）

预训练CBraMod的uV/100与zscore平均验证逐被试BA为61.01%／61.56%；随机初始化为55.28%／54.62%。
预训练组归一化改动逐划分+1.64/+1.23/−1.22pp，平均+0.55pp，未达到事前三划分方向一致要求，不宣布稳定改进。
两种归一化下预训练均高于随机：平均+5.73pp／+6.94pp，结论限1500步预算。
既有轻量融合62.26%仍高于zscore预训练约0.70pp。十二份预测复算、六组配置/初值/采样配对和六个新增检查点重载核验通过；未评估测试池。
详见`outputs/physionet_normalization_20261114/report/结果报告.md`和同目录`fold_results.csv`。

### 2026-11-14 独立SSVEP经典基线（12次源阈值校准完成）

Kalunga2016开发侧10人、23个官方run、736个事件，8个EEG通道，统一200Hz。
锁定11/12被试未下载或评分；三次开发验证覆盖1–6，7–10仅在训练侧。
FBCCA三类激活识别1秒/2秒BA为76.74%／87.15%；源标签校准rest阈值后的四类BA为67.19%／74.74%。
CCA对应三类67.36%／74.31%，四类51.04%／64.06%。两种任务分表，不将三类结果冒充完整四类。
24条验证记录的试次身份、阈值应用与逐类召回BA独立核验通过；没有新增神经模型拟合。
详见`outputs/kalunga_classical_20261114/结果报告.md`和`plans/kalunga_classical_20261114.md`。

### 2026-11-14 实验室模型与64通道MI预训练对照（9次正式拟合完成）

复用PhysioNet MI缓存、三个开发重划分和既有逐步采样索引；固定测试池不评分。
实验室HUST-BCIML EEGConformer、预训练CBraMod、随机CBraMod平均验证逐被试BA为58.01%／61.01%／55.28%。
两个CBraMod条件架构、头初值、输入uV/100和优化预算相同；预训练逐划分+8.11/+4.84/+4.24pp，均值+5.73pp。
该结果限于1500更新预算，不代表随机模型充分训练上限；预训练CBraMod仍低于既有轻量融合62.26%。
9次检查点重载、预测复算、被试身份、采样索引和初始化对照均核验通过；3次短流程另列。
详见 `outputs/physionet_models_20261114/report/结果报告.md`、`plans/physionet_models_20261114.md`。
文献与资源增量见 `plans/model_research_20261114.md`；TUAR/TUEV逐文件接口盘点见 `plans/tuh_input_audit_20261114.md`。

### 2026-11-14 独立64通道MI验证（6次正式拟合完成）

从服务器已有公共数据选取PhysioNet MI原始EDF，109人、9837个有效四类想象trial；真实64通道、4秒、200Hz。
固定11人测试池不评分，三个本地开发重划分各87人训练/11人验证；不等同官方三个种子协议。
EEGNet、Shallow、固定0.75/0.25融合的平均验证逐被试BA为59.13%／61.29%／62.26%。
融合相对Shallow增加0.97pp，三个划分方向一致，但不能据此宣称普遍或显著收益。
6个检查点重载和9份验证预测复算通过；2次真实数据短流程拟合另列，不计入正式6次。
详见 `outputs/physionet_mi_20261114/report/结果报告.md` 和 `plans/physionet_mi_20261114.md`。

### 完整实验汇总（截至2026-11-13）

最新全量汇总：`outputs/experiment_summary_complete_20261113/OmniEEG总实验汇总.xlsx`，
文字解读：同目录 `总实验汇总.md`。
共376次独立拟合、412条拟合任务结果、114条复用评分、254个参数与评分配置；
含2次无新增准确率的部署重拟合，历史测试/集成10条另列，12份打包/解包副本不重复计数。
完整参数、划分、原始配置与来源哈希均保留；4182条训练曲线另存CSV。
旧218次拟合的源记录哈希核验通过，新纳入288份保存预测重新计算BA一致。
Excel六个工作表的计数与均值公式检查、版面检查及导出错误扫描均通过，未新增训练或测试。

### 2026-11-13 面向比赛的共享边界补充对照（9次训练完成）

新增P2/P4各三折及冻结前缀P4三折，复用同协议S/J共12次训练。
统一预训练初始化、flatten分类头、采样索引、每任务1500次更新；当前均为公开开发验证。
结果和资源见 `outputs/partial_sharing_v1/结果报告.md`、`complete_results.csv`、`参数与资源.csv`。

P4相对完全共享J：MI −0.71pp、SSVEP +0.90pp、情绪 +1.43pp，三任务均值仅 +0.54pp；
未达到事前1pp保留门槛，未追加条件性容量对照，不能宣称部分共享有效。
冻结前8层、训练任务末4层后，按联合检查点选模为33.64%／43.11%／17.40%；
在同一次训练中独立选任务分支后为33.93%／44.54%／19.20%（MI／SSVEP两秒／FACED八类BA）。
冻结共享权重的哈希保持不变，最佳任务分支可精确组合为同一个 `model_taskwise.pt`，无需额外训练。
独立选模增加验证选优自由度，不能把该变化视作独立测试收益，也不支持三个任务统一冻结。

36份新增任务预测复算及完整验证集重载通过，最大误差0；六种官方输入形状的12项合成接口检查通过。
未使用官方EEG数据或新增独立测试；决赛从零预训练尚未验证。所有训练与临时唤醒进程已退出。

### 2026-11-13 三任务专用候选搜索（105次对照拟合完成）

最新结果和全部逐折参数记录见 `outputs/next_round_20261113/结果报告.md`、`逐折结果.csv`、`配置汇总.csv`。
候选权重路径、SHA256和条件见 `outputs/best_candidates_20261113/candidate_manifest.json`。

同一三折开发验证：SSVEP 1秒FBCCA与源模板融合82.01%，2秒95.67%；MI均衡EEGNet与Shallow概率融合52.53%。
SEED三类：仅训练侧归一化的TCN55.64%；历史/当前窗口归一化后TCN/EmT组合78.12%；整会话离线归一化组合78.38%。
后两者使用无标签目标数据，且上游DE-LDS仍离线，不能写成严格零校准或端到端在线成绩。
FACED原八类离线对齐TCN27.34%；作者123人九类特征基准39.57%，两种数据和任务分表，不横向比较百分比。

105次对照拟合之外，另保存两套SSVEP全30开发被试重拟合候选，没有新增测试分数。
SSVEP30组、SEED54组、MI27组分数组合评分分别记录，未膨胀拟合数。
所有训练队列结束、129份主记录预测复算通过。没有重新使用旧测试集选参数，没有独立新测试成绩。
MIRepNet、EA和对比学习的负结果完整保留；当前候选只代表已测开发条件下的选择，不证明全局最优。

### 2026-11-13 异构残差适配（27次训练已完成）

按队友图示增加空间、时间、整窗频谱残差分支，比较固定均匀融合、样本级学习门控和去谱。
保持同一预训练骨干、flatten与任务头初值、每数据集1500步、采样顺序和开发折；均为独立任务适配。
全部有效分支密集计算，不是原先的稀疏FFN模型。

| 条件 | MI验证BA | SSVEP两秒验证BA | 情绪八类验证BA |
| --- | ---: | ---: | ---: |
| 原单任务基线 | 43.89% | 67.63% | 16.42% |
| 均匀融合 | 43.79% | 68.07% | 17.22% |
| 学习门控 | 43.73% | 67.75% | 16.88% |
| 去谱学习门控 | 43.60% | 68.50% | 16.87% |

均为三折开发验证均值，不能当作独立测试成绩。新增27项保存预测重算、27个检查点重载均通过，
重载子集预测最大绝对误差约7.0e-6。
MI无收益；SSVEP和情绪的最高均值差都不到1个百分点，未达到事前保留门槛。
当前不将复杂门控或整窗频谱旁路作为已验证贡献，不据此排除其他设置下可能有效。

结果：`outputs/residual_adapters_v1/结果解读.md`、`comparison.png`、`逐折结果.csv`。
总表已追加到 `outputs/experiment_summary_20261113/`，累计218次独立拟合、236条任务记录、158个配置组合。

### 2026-11-13 三折研究对照与初始化控制（已完成）

新入口为 `scripts/run_preliminary.py`，输出独立保存至 `outputs/preliminary_v1/`。
队列依次执行三个开发折：18 次核心训练（单任务稠密、联合稠密、固定任务路由、内容路由），
以及12次补充拟合（各折 FBCNet 式网络、EEGNet、一秒频带序列 TCN、同特征时间平均 Logistic）。
30次作业已经完成，保存预测重算48项、核心检查点重载36项均通过。
11月13日检查到旧进程已退出，保留完成结果并恢复剩余作业，状态现为finished。

新增 `scripts/run_initialization_controls.py` 完成18次配对训练，复用原9个S组为全量微调基线。
保持相同flatten、分类头初值、样本顺序、1500次更新和优化设置，比较冻结与全量、预训练与随机。

| 任务 | 预训练全量微调 | 预训练冻结主干 | 随机初始化全量训练 |
| --- | ---: | ---: | ---: |
| MI | 43.89% | 31.39% | 41.67% |
| SSVEP两秒 | 67.63% | 23.83% | 29.24% |
| 情绪八类 | 16.42% | 16.35% | 18.08% |

均为三折开发验证BA；预训练收益限于本轮下游预算，不能推断随机模型充分训练的性能上限。
冻结组保留训练态dropout，只冻结梯度；验证统一eval。27项保存预测核验、18个新增检查点初值重建与重载核验通过。
结果和图表见 `outputs/research_update_20261113/RESULTS.md` 与同目录PNG。

新划分只使用旧 train+val 被试，旧 test 被试排除；三折与训练种子共同变化，
不是官方三个划分种子，也不代表纯训练种子标准差。原有结果保持不变。
Top-1 学习路由采用前向单位权重、选中概率的直通梯度估计；只计算选中专家。
四专家固定路由只分配三个专家，完整对照仍需后续容量/激活计算量匹配稠密模型。

```powershell
# 读取已完成队列的状态和日志。
Get-Content outputs/preliminary_v1/queue_status.json
Get-Content outputs/preliminary_v1/queue.stdout.log -Tail 8

# 导出当前已完成模型的验证结果，并可独立重载检查核心模型。
& .venv/Scripts/python.exe scripts/run_preliminary.py export
& .venv/Scripts/python.exe scripts/report_preliminary.py --checkpoints
```

`smoke.json` 验证复制专家初始输出与稠密输出接近、分类损失可传至路由器；
真实情绪输入 B16 前后向约0.0495秒/批、峰值已分配显存约995 MiB，
不含优化器与验证，不作为完整训练资源估算。训练完成后逐项重算保存预测。

2026-11-11 完成118组不同配置、140次训练或拟合，含细化与训练种子复核。
所有140次保存的验证预测与最终测试预测均重新计算核验。

| 条件 | 验证集选定的配置 | 验证BA | 对应测试BA |
| --- | --- | --- | --- |
| MI | EEGNet式CNN，F1=16，lr=0.003，dropout=0.5，weight decay=0.01 | 57.29% ± 1.59% | 55.56% |
| SSVEP 1秒 | FBCCA，后部9通道，5谐波，5频带，权重指数1.0 | 77.81% | 53.75% |
| SSVEP 2秒 | FBCCA，同上，权重指数0.5 | 98.33% | 84.92% |
| 情绪8类 | 单试次相对频带功率，训练集标准化，Logistic Regression，C=0.1 | 22.97% | 21.28% |

MI 验证列为三个训练种子均值和样本标准差，测试列为预先固定的三模型概率均值集成；
该集成的验证BA为58.33%。其余方法确定性拟合一次，没有用重复拟合制造种子方差。
这里的最佳指已测试配置中按验证规则选出的模型，不是测试集排名或全局最优。
SSVEP 1秒存在明显验证到测试回落；情绪仍受12名被试、8类与处理版本限制。
MI 测试被试曾在前一轮查看过，不能当作全新盲测。

汇总和交付：

- `outputs/search_v1/champions_summary.csv`：所选配置、验证/测试指标。
- `outputs/search_v1/all_runs.csv`：全部140次拟合的参数与验证结果。
- `outputs/search_v1/validation_search.png`：验证集搜索曲线。
- `outputs/search_v1/audit.json`：逐次预测重算和划分核验。
- `outputs/search_v1/packages/best_models_9d3d0f3f9d5c.zip`：模型、配置、依赖版本和本地预测入口。
- `outputs/search_v1/inference_check/report.json`：从独立解包目录运行，通道重排后复现预测；缺失通道拒绝。

模型包59个内容文件已核对SHA256，不包含EEG数组。四个条件的打包模型均通过预测一致性检查。

## 三方向模型搜索

本轮入口为 `scripts/search_models.py`，输出在 `outputs/search_v1/`。
它先筛模型和超参数，再细化验证集领先的两个模型族，最后复核前三个配置。
随机训练模型用 42/43/44 三个开发种子；确定性方法拟合一次，不用重复运行伪造三种子方差。
搜索只看逐被试平衡准确率的等权平均，不运行测试指标。

| 条件 | 数据范围 | 输入 | 训练 验证 测试被试数 |
| --- | --- | --- | --- |
| MI | BNCI2014-001，9 名被试 T 会话 | 22×800，四分类 | 5 / 2 / 2 |
| SSVEP 1秒 | Benchmark，全部35名被试 | 64×200，40分类 | 26 / 4 / 5 |
| SSVEP 2秒 | 同一 Benchmark 的独立窗口条件 | 64×400，40分类 | 26 / 4 / 5 |
| 情绪10秒 | FACED处理导出，固定前12名被试，1875个窗口 | 26×2000，八分类 | 8 / 2 / 2 |

SSVEP 的1秒和2秒来自同一公开数据集，不能当成比赛的两个独立数据集。
SSVEP 的各划分均包含有经验和无经验被试。情绪数据使用 Braindecode 的固定版本处理导出，
不是原方案中的30通道原始复现；它不代表整个认知与情绪赛道。
SSVEP 使用整试次离线滤波后裁窗，滤波可使用窗外信号上下文；这里的1秒/2秒指模型输入长度，
不构成严格只观察该时段的在线时延实验。所有候选在同一离线预处理下比较。

| 候选 | 作用与搜索范围 |
| --- | --- |
| CBraMod 全量微调 | 作者权重严格加载；骨干学习率、正则化和读出方式 |
| CBraMod 冻结表征加 Ridge | 线性迁移对照；训练集内标准化与正则强度 |
| EEGNet式 CNN | 本地 PyTorch 实现；滤波器数、学习率、dropout、权重衰减 |
| MI 滤波组 CSP 加 LDA | 本地实现的任务基线；频带组、空间滤波器数、协方差收缩 |
| SSVEP CCA 和 FBCCA | 已知刺激频率的任务基线；后部通道子集、谐波数、子频带和权重 |
| SSVEP 频谱加 Ridge | 复频谱与对数幅值对照；正则强度 |
| 情绪频带功率加 LDA SVM Logistic | 传统任务基线；C、gamma 和分类器 |

这些是已测试候选，不包含组内全部16个基础模型，也不是全球最优搜索。
任务专用模型可能领先，它们用于建立强基线，不能拼接后宣称得到了共享基础模型。

初筛最多20轮，细化和种子复核最多40轮；验证指标连续10轮不改善则早停。
同预算下比较配置，记录实际训练轮数和更新数。学习率采用余弦衰减。
验证表现驱动的通道读出补充实验单独保留计划、旧冠军和新结果，避免覆盖历史。
固定被试划分下的训练种子波动不等于跨划分或跨数据集不确定性。

```powershell
# 数据准备：SSVEP 全35人；情绪先使用已验证可下载的公开处理版本。
python scripts/download_tracks.py ssvep --subjects 35
& .venv/Scripts/python.exe scripts/prepare_tracks.py ssvep
& .venv/Scripts/python.exe scripts/prepare_faced_hub.py

# 验证集搜索，可根据配置指纹恢复已经完成的实验。
& .venv/Scripts/python.exe scripts/search_models.py search mi
& .venv/Scripts/python.exe scripts/supplement_readout.py
# 情绪初筛的绝对幅度特征偏弱后，补充仅依赖单试次的相对功率和空间居中对照。
& .venv/Scripts/python.exe scripts/supplement_emotion_features.py
& .venv/Scripts/python.exe scripts/run_search_queue.py

# 各方向的 champion.json 确定后才运行，已有最终结果会直接读回。
& .venv/Scripts/python.exe scripts/search_models.py finalize mi
& .venv/Scripts/python.exe scripts/search_models.py finalize ssvep_1s
& .venv/Scripts/python.exe scripts/search_models.py finalize ssvep_2s
& .venv/Scripts/python.exe scripts/search_models.py finalize emotion
& .venv/Scripts/python.exe scripts/export_search_results.py
& .venv/Scripts/python.exe scripts/package_best_models.py
```

`plan.json` 固定筛选规则，`refinement_plan.json` 记录验证集驱动的下一轮配置，
`confirmation.json` 记录复核结果，`champion.json` 指向保存的权重或模型。
`final_test.json` 和 `final_test_predictions.npz` 只包含最后选定配置的测试评估；
随机模型同时记录各种子指标和预先固定的概率均值集成。
模型身份与预处理改变时，必须建立新配置/缓存版本；不得拿已查看的测试分数反复调参。
MI 测试被试在上一轮已评估过，因此本轮标为开发测试结果，不称为全新盲测。

情绪处理版本来源：
https://huggingface.co/datasets/braindecode/faced/tree/18c73355fc71c055e92828384da85bb49bc6e3c0
。导出元数据明确包含 V×1e6；本地不重复转换单位。
模型输入只包括 EEG，不包括主观评分、视频索引、被试编号或事件时刻。
上游的26通道、200 Hz、0.1 Hz高通与非重叠10秒窗口均完整保存；
本地另加 CAR 与逐窗口0.5–45 Hz滤波。
原始 BDF 下载因存储端连续连接失败而未完成，保留断点文件，不参与训练。
标签审计发现本次固定版本导出中没有 Neutral，虽然上游数据卡和 mapping 列出9类。
当前将实际存在的8种情绪映射为连续模型标签，并保留原始标签映射与缺失类记录。
本轮情绪结果不代表包含中性类的完整FACED九分类结果。
完整 FACED 官方下载接口经核验对匿名用户返回403；用户选择先完成公开8类实验。
依赖的观测版本保存在 `requirements-experiments.txt`。最佳模型包只含所选模型、代码、
参数和本地预测入口，不含 EEG 数据。输入需按同一预处理准备，单位、通道身份、采样率和长度都会检查。

## 环境

- 本机：RTX 5060 Laptop GPU，8 GB 显存。
- 运行入口：`.venv/Scripts/python.exe`。
- 虚拟环境通过 `--system-site-packages` 复用已有 `hustbciml-gpu` 的 PyTorch 2.11.0+cu128。
  没有改动已有环境；因此它并非完全独立的依赖快照。
- 默认 `python` 是 CPU 版，仅用于下载。训练务必指定上述虚拟环境入口。
- 每次实验在 `config.json` 中记录解释器、依赖版本、GPU、上游 commit 与数据/权重 SHA256。
- 当前训练使用 FP32、无 AMP、无学习率搜索。固定开发种子 42 不是官方种子。

## 数据入口与赛事状态

2026-11-11 核查：

| 入口 | 核查结果 |
| --- | --- |
| [主办方研究页](https://omni-intel.cn/research) | 链接到 OmniEEG-Bench；展示其他公开数据，不等于比赛训练集 |
| [腾讯官方赛事列表](https://science.tencent.com/dayanAI4S/activities/) | 链接到微信赛事公告；本次公告访问要求验证 |
| [OmniEEG-Bench](https://omni-eegbench.omni-intel.cn/bench) | 本次直接请求返回 502，暂无法核验数据入口 |
| [官方研究基准代码](https://github.com/ncclab-sustech/omni-eegbench) | 已取得代码；配置使用作者本地 Zarr 路径，未找到赛事数据下载链接 |
| [清华 SSVEP 原始数据](https://bci.med.tsinghua.edu.cn/download.html) | 包含 Benchmark 和 BETA 下载入口；尚未批量下载 |
| [BNCI 原始 MI 数据](https://bnci-horizon-2020.eu/database/data-sets) | 已下载 9 名被试 T 会话，22 EEG + 3 EOG，四分类 |
| [FACED 原始数据](https://www.synapse.org/Synapse:syn50614194) | 已找到 THU-EMOTION-BCI 发布页；下载权限尚未完成核验，尚未下载 |

工作区原有 `tmp/pdfs/handbook.txt` 的手册写明：初赛不开放训练数据；
计划于 11 月 23 日复赛开放训练数据与共享算力，以参赛群最新通知为准。
当前不能据公开网站访问情况断言赛事数据绝未发布。
上述公开数据均未确认是比赛实际采用的数据集，不根据输入尺寸推定身份。

## 复现已启动的实验

首次取得上游代码：

```powershell
git clone --depth 1 https://github.com/wjq-learning/CBraMod.git external/CBraMod
git clone --depth 1 https://github.com/ncclab-sustech/omni-eegbench.git external/omni-eegbench
```

下载原始 MI 数据及作者链接的公开权重：

```powershell
python scripts/download_pilot.py
```

原始数据和权重旁保存 `.source.json`，包含来源 URL、文件大小、下载时间和 SHA256。
BNCI 数据按来源页面的 CC BY-ND 4.0 使用，署名为 Graz University of Technology 的
Institute for Knowledge Discovery；原始和处理数据均不加入 Git。
相关论文：Tangermann et al. (2012), Review of the BCI Competition IV,
https://doi.org/10.3389/fnins.2012.00055。

数据准备与六种输入检查：

```powershell
& .venv/Scripts/python.exe scripts/prepare_bnci.py
& .venv/Scripts/python.exe scripts/run_cbramod.py smoke
```

首轮真实数据短训：

```powershell
& .venv/Scripts/python.exe scripts/run_cbramod.py pilot --epochs 5 --batch-size 16 --tag initial_5ep
```

该命令分别运行公开权重线性探测、公开权重全量微调、同架构随机初始化训练。
同一 tag 的运行目录存在时会拒绝覆盖；复跑时换 tag，避免丢失历史实验。
所有模式使用相同数据、划分、种子、batch、轮数、头结构与对应学习率。
训练预算只有 5 轮，不能代表模型收敛后的能力，尤其不能根据随机初始化短训结果认定预训练必然更好。

## 预处理与划分

- 从 MATLAB 试次索引减 1，正确转换为 Python 索引。
- 按空标签识别眼电校准块，不硬跳过前 3 块，保留 A04T 的全部 MI 试次。
- 只取 22 EEG，排除 3 EOG；保留原始电极顺序并记录名称。
- 每个连续 run 内 CAR，然后 0.3–40 Hz 四阶 Butterworth 双向滤波。
- 截取相对试次开始的 `[2, 6)` 秒，即提示出现后的 4 秒；每试次只产生一个样本。
- 多相抗混叠重采样 250 → 200 Hz，最终 `22×800`；原始单位 µV，训练时 `/100`。
- 保留源数据的伪迹试次和标记，无按类别或测试分数筛选样本。
- 训练被试 1–5，验证 6–7，测试 8–9，只有 T 会话。共 1440/576/576 试次。
- 这是作者式 5/2/2 被试划分的 T-only 版本；不是比赛 8:1:1，也不是原论文完整复现。
- 本地滤波与重采样实现和作者预处理不同，不能直接比较论文分数。
- 只按验证集平衡准确率保存最优权重，平分取较早轮；选模完成后评估一次测试集。
- 测试集不进入梯度、归一化拟合和选模。今后不能用本轮测试分数反复调参。

本轮分类头为 flatten → LayerNorm → Linear，保留固定电极与时间块位置。
它是单数据集基线头，不是支持任意输入布局的联合分类器。
CBraMod 本身已有时域与频域分支；要检验新增频域模块，必须先与原分支做受控比较。

## 输出与下一阶段

2026-11-11 首轮已完成，所有结果由保存的测试预测重新计算核验：

| 设置 | 最优验证 BA | 对应测试 BA | 选中轮数 |
| --- | --- | --- | --- |
| 公开权重冻结编码器 | 43.58% | 45.66% | 3 |
| 公开权重全量微调 | 43.92% | 39.24% | 4 |
| 同架构随机初始化 | 42.01% | 40.63% | 4 |

三组均为 5 轮、450 次更新，采用验证 BA 选模。当前只能说明流程能实际完成，
不足以判断全量微调、随机初始化或线性探测的收敛表现与普遍优劣。
冻结组保存的全部骨干张量与公开权重一致；其余两组有更新。
试次与被试划分无交叉、测试概率归一化和验证集选模均已核验。
原始伪迹标记共 264 个，均被保留。

- `outputs/smoke/summary.json`：六种模拟输入的更新、有限梯度和峰值已分配显存。
  每种仅 3 步、batch=2、无独立热身；耗时包含首次开销，不能用于速度排名或正式资源预算。
  另检查遮挡块扰动不会影响 patch embedding，验证原始波形不会从频域分支泄露。
- `outputs/pilot/initial_5ep_summary.csv`：首轮三种训练设置的汇总。
- 各运行目录保存 `config.json`、`history.json`、`best.pt`、`result.json`、`test_predictions.npz`。
- 数据缓存旁保存预处理元数据、被试划分、原始文件与处理文件的哈希。

下一阶段先复核预处理与短训稳定性，补齐 SSVEP 和情绪公开数据，建立各任务稠密基线。
随后固定输入表示，比较同架构单任务稠密与联合稠密；观察真实负迁移后再接 MoE。
MoE 必须增加固定路由与计算量/参数量对照，不能把更大容量或额外更新数算作路由收益。
公开数据预实验用于方法开发；赛事数据、官方划分、指定种子和官方算力接口取得后另建正式配置。

## SSVEP / MI / TUAR 三任务对照（2026-11-13）

本轮九组对照已单独归档于 [experiments/tuar_multitask_20261113](experiments/tuar_multitask_20261113/README.md)：包含冻结的 CBraMod 主干后残差专家代码、TUAR 患者隔离预处理、固定协议、可迁移启动入口，以及聚合验证结果。任务为 22 通道 MI、2 秒 SSVEP 与 TUAR 伪迹检测代理任务，不包含情绪识别。六种结构为普通主干、单适配器、同构/异构专家各自的固定平均与学习路由；全部专家执行，不宣称稀疏加速。本目录不含原始数据、患者明细或权重，且不覆盖原有实验。
