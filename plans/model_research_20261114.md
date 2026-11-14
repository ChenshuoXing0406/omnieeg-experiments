# 模型与范式调研增量：MI / SSVEP / 信号可靠性

日期：2026-11-14。范围为论文主站/作者稿、官方代码与本地实现核对；不是系统综述，不声称已精读全部全文。补充已打开EEGConformer作者完整PDF及CBraMod正式PDF，针对方法、数据与评测相关段落核对。IEEE是出版平台，期刊论文与ICLR/NeurIPS会议论文分别标明。

## 直接影响本轮实验的证据

| 工作 | 发表与原始来源 | 本轮采用或暂缓理由 |
|---|---|---|
| EEG Conformer | IEEE TNSRE 31:710–719，2023；DOI 10.1109/TNSRE.2022.3230250；[作者PDF](https://bingchuanliu.github.io/assets/pdf/song_tnsre.pdf)、[官方代码](https://github.com/eeyhsong/EEG-Conformer) | 卷积提取局部特征再进行时间token注意力，适合作为小CNN与基础模型之间的专用对照；使用实验室已核对实现。IEEE DOI直开本次失败，交叉核对作者PDF入口、公开书目和代码引用；未伪称已从IEEE下载全文。 |
| EEG-FM-Compass | National Science Review，2026-08-03，nwag466；[期刊原文](https://doi.org/10.1093/nsr/nwag466)、[作者代码](https://github.com/Dingkun0817/EEG-FM-Benchmark) | 公开摘要和作者说明指出，统一预处理和适配协议下，专用模型仍有竞争力、冻结线性探测常不足。由此推导本轮保留小模型参照，并先做同架构预训练/随机对照，而非只增大模型。不是将该论文结论当作我们实验的已证实结果。 |
| CBraMod | ICLR2025会议论文；[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/bbbd6d915cb90be21c1254a82d45cedd-Paper-Conference.pdf)、[官方代码](https://github.com/wjq-learning/CBraMod) | 交叉空间/时间注意力、掩码EEG重建，作者报告TUEG预训练；现有权重来源可核对，先测64通道MI预训练收益。ICLR不是IEEE/SCI期刊。 |
| EEGPT（BINE022） | NeurIPS2024会议；[正式摘要](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)、[官方预训练数据说明](https://github.com/BINE022/EEGPT/blob/main/datasets/pretrain/readme.md) | 官方预训练说明包含PhysioNetMI与TSUBenchmark；因此不能把使用其公开权重的这两库结果视为确认未见数据泛化。先记录数据重叠风险，不混进本轮严格配对的无标签预训练参照。注意与另一篇同名EEGPT自回归论文区分。 |
| LaBraM | ICLR2024会议；[正式论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf)、[官方代码](https://github.com/935963004/LaBraM) | 跨数据集通道patch、谱tokenizer与掩码token学习是后续候选；本轮尚未确认服务器中可用权重与依赖，不为增加模型数量阻塞已有可执行实验。 |

## 实验室资源核对

[HUST-BCIML](https://github.com/sylyoung/HUST-BCIML) GitHub README已读取，本地D盘克隆和服务器克隆均存在。本地含EEGNet、EEGConformer、CTNet、EEGDeformer、FBMSNet等实现。代码实现、预训练权重、已训练任务检查点必须分别登记；不能因为有模型类就写“已取得预训练模型”。

本地克隆commit c806d2cf8796b366742fd85b9b32f6c916211f43；本次未更新或覆盖该克隆。EEGConformer本地/服务器源文件哈希一致。GitHub当前README的版本比本地新，在线条目不得当作本地版本已具备的功能。

针对原文的进一步核对：EEGConformer论文主要实验采用被试内协议，并使用同类trial分段重组增强及Adam lr=.0002、beta1=.5；本轮是跨被试、无增强、统一短预算AdamW配方，因此即使表现不佳也不能写“原论文方法失效”。CBraMod原文PhysioNet划分是1–70/71–89/90–109，本轮是固定测试池加三次开发重划分，两者百分比不可直接比较。两者都使用四秒四类任务，但输入处理、头和预算并未完整复刻原论文。

## 三个赛道的研究边界

- MI：当前先完成新64通道数据上的专用模型与基础模型对照。BNCI的舌头类与PhysioNet的双拳类不是同一标签，不直接拼成共同四分类标签空间。
- SSVEP：保留已跑FBCCA和源模板强基线。EEGPT公开权重预训练含TSUBenchmark，使用它时应单列数据暴露条件；1秒/2秒窗口仍来自同一公开库，不能冒充比赛两个数据集。
- 信号可靠性：公共盘TUAR v3.0.1和TUEV v2.0.1已读README并盘点。TUAR是按通道区间的伪迹标注、存在复合标签；TUEV是六类事件标签，不自动等价二分类质量分数。队友已有训练，本轮不自行改标签任务重复训练。官方手册仍未披露数据名称及类别数，不把公开代理库认定为隐藏赛题。

下一轮新增模型应先审计输入采样率/电极与单位、原权重预训练数据、任务标签和被试隔离，再固定配方运行。无需为下载一个新模型而重复已做的EA、MIRepNet或专家队列。

## SSVEP公共盘新增资源线索（已读元信息，未纳入本轮训练）

- `/mnt/ssd2/2022_BCI_competition/SSVEP/2022SSVEPPublic`：README标明10个公开被试、原始1000Hz；不能把示例程序下采样后的250Hz当原始频率。
- `/mnt/ssd2/2025_BCI_competition/SSVEP/open_async_datasets`：有YSU_async_SSVEP_v3、SSVEP2025_robot、Kalunga2016_SSVEP_exoskeleton目录。
- `/mnt/ssd2/2025_BCI_competition/SSVEP/ssvep_pretrain_mvp`：已有SSVEP预训练项目、外部基线（CCA-Net、SSVEP-DAN、TBMSCCN等）及训练检查点。没有执行或修改其他人的训练脚本。
- 该项目`data/prepared/manifest.json`实际列出Nakanishi2015与Wang2016，8个真实后部电极、250Hz、4秒、逐被试NPZ；预处理单位与来源仍需从prepare_data.py及原始文件核对，不能直接将全部数值当uV。
- `outputs/pretrain_crossarch_cbramod_official_seed42_v1/best.pt`存在，run_config记录在官方CBraMod初始化上，用Nakanishi/Wang做频率条件预训练；它不是未经过目标库暴露的原始公开CBraMod权重。其发布状态、许可与数据隔离未经独立核验，不加载到本轮比赛候选。
- 该运行的完成元信息属于已有其他研究记录，没有独立复算，不并入本项目新增拟合数或模型比较成绩。

优先的后续SSVEP问题是：在真正独立的第二个数据集上，以固定FBCCA等经典配方为参照检验迁移，而不是用已包含Wang/Nakanishi的预训练权重在同库上宣称未见泛化。本轮先完成正在运行的MI九次配对，避免同时改变输入处理、数据集和模型。
