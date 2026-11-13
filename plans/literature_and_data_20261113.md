# MI、情绪识别与 HUST-BCIML 文献和数据行动方案

核查日期：2026-11-13。研究记录，未新增训练成绩。公开资料不代表实验室内部指导意见。

## 结论与已有实验的关系

下一轮优先解决跨被试分布差异、情绪训练被试不足和评测条件差异。已有残差专家实验没有显示稳定收益，因此暂不继续扩大 MoE 搜索。短期采用任务专用基线与严格对照；统一主干保留为研究问题。

当前三折开发实验 EEGNet-MI 的平均 BA 为 47.74%，情绪序列 TCN 为 18.21%。旧固定划分测试 MI 集成为 55.56%、情绪相对功率 Logistic 为 21.28%。这些来自不同协议，不能横向排名。SSVEP 2 秒 FBCCA 的 98.33% 是验证准确率，已看过的测试为 84.92%。

## 一、实验室公开资源

- [HUST-BCIML 开源入口](https://github.com/sylyoung/HUST-BCIML)、[论文代码检索页](https://sylyoung.github.io/HUST-BCIML/)、[伍冬睿老师完整论文列表](https://sites.google.com/site/drwuhust/publications/completepubs)。此处指公开可核实的 BCIML 团队；并不推断所有 HUST BCI 研究都属于同一团队。
- 该基准包含对齐、增强、模型、训练策略和集成模块。其公开三个 MI 表均为二分类，BNCI2014001 仅左手/右手；网络表还使用无标签目标被试 EA。不能将其准确率与我们的四分类、严格未见被试结果直接比较。其 README 区分历史重实现和纠正后的嵌套 LOSO，移植应固定版本并检查算法卡。
- [EEG-FM-Compass / EEG-FM-Benchmark](https://github.com/Dingkun0817/EEG-FM-Benchmark)，National Science Review 2026，[DOI](https://doi.org/10.1093/nsr/nwag466)。可借鉴专用模型与基础模型统一预处理、初始化和读出对照，而非只追逐新模型名单。不要与另一作者的 EEG-FM-Bench 混淆。

| 方法 / 发表信息 | 可利用内容 | 对我们实验的限制与建议 |
|---|---|---|
| [MIRepNet](https://github.com/staraink/MIRepNet)，Knowledge-Based Systems 343:115966, 2026，[论文](https://doi.org/10.1016/j.knosys.2026.115966) | MI 专用预训练，生理通道模板、分布对齐、掩码重建与分类联合预训练；官方权重已在本地仓库 | 本地审阅的默认 train_subject 在单个被试内切训练/验证，不等于零校准跨被试。45 通道模板，公开读取器 250 Hz、4 秒；我们的 22 通道、200 Hz 不能直接 reshape。应按电极坐标映射/插值、正确重采样、重写被试划分，并检查预训练数据重叠 |
| [MVCNet](https://github.com/wzwvv/MVCNet)，KBS 328:114205, 2025 | 时间/频率/空间多视图增强与对比学习，有单独 LOSO 脚本 | README 高分含 CO 被试内设置，不能拿 83% 当我们跨被试预期。先引入单项训练增强，再测试对比损失 |
| [EA 原始工作](https://doi.org/10.1109/TBME.2019.2913914)，IEEE TBME 67(2):399–410, 2020；[正确使用综述](https://doi.org/10.1088/1741-2552/addd49)，JNE 22:031005, 2025 | 按被试平均协方差白化，降低被试/会话差异 | 目标被试的整批无标签 EA 属传导设置；逐试次累计 EA 属在线设置；都应与不使用目标集合统计的严格泛化单独报告。仅对源被试做 EA 不等价于标准完整 EA |
| [T-TIME / DeepTransferEEG](https://github.com/sylyoung/DeepTransferEEG)，IEEE TBME 2024，[DOI](https://doi.org/10.1109/TBME.2023.3303289) | 在线增量对齐、信息最大化、集成；同仓库 BFT 为 JBHI 2026 已接收工作 | 使用无标签目标流；须检查比赛推理是否允许状态更新、试次顺序和目标统计。不能混入严格泛化主表 |
| [CKD](https://github.com/wzwvv/CKD)，IEEE TBME 2026 in press，[DOI](https://doi.org/10.1109/TBME.2026.3701548) | 多数据集教师向轻量学生进行对比蒸馏，适合扩充 MI 数据后的第二阶段 | 学生使用目标数据集的其他训练被试标签，默认还有累计在线 EA；不是完全零样本新数据集。决赛外部教师限制需另外遵守 |
| [SDDA](https://github.com/Dingkun0817/SDDA)，IEEE TBME 73(7):2400–2411, 2026，[DOI](https://doi.org/10.1109/TBME.2025.3631604) | 针对跨设备/通道数差异的空间蒸馏与多层对齐 | 公开 README 仍说明完整代码后续开源；涉及目标无标签数据，不能承诺直接复现 |
| [CMCRD](https://github.com/kssyyy/CMCRD)，IEEE TAFFC 2026 in press（作者论文列表状态）；[作者稿](https://arxiv.org/abs/2504.09221) | 训练时利用眼动等跨模态信息，部署只用 EEG | 当前公开 FACED EEG 导出无配对眼动，因此不作为第一轮；需要已获授权的多模态数据，不应虚构当前可运行 |

适合阅读的指导材料：[Transfer Learning for Motor Imagery Based BCIs: A Tutorial](https://doi.org/10.1016/j.neunet.2022.06.008)，Neural Networks 153:235–253, 2022；[Affective BCIs: A Tutorial](https://doi.org/10.1109/JPROC.2023.3277471)，Proceedings of the IEEE 111(10):1314–1332, 2023。引导我们区分校准/泛化、刺激类别/个体评分、离散类别/效价唤醒度，而非改变标签后直接比较准确率。

## 二、其他 IEEE / SCI 论文筛选

1. **MI：SCLDGN**，[论文](https://doi.org/10.1109/TBME.2024.3432934)，IEEE TBME 72(1):401–412, 2025；[官方代码](https://github.com/hongyizhi/SCLDGN)。使用源被试之间的分布约束、监督对比学习和混合增强解决域泛化，不依赖无标签目标被试适应。最适合先拆出 EEGNet + 源域 CORAL 的小对照，再做完整算法。官方滤波器组采样率默认 250 Hz，移植必须改成真实采样率；不能把原论文所有 MI/ME 任务都称为 MI。
2. **情绪：CLISA**，[论文](https://doi.org/10.1109/TAFFC.2022.3164516)，IEEE TAFFC，2022 在线 / 2023 卷期；[作者稿](https://arxiv.org/abs/2109.09559)，[FACED 作者实现](https://github.com/Rmin-cmd/FACED_Code)。用不同训练被试同一刺激时间段建立正样本，再用对齐后的表征提取 DE 特征。适合当前已知视频索引的 FACED。必须先划分被试，再构造正样本，所有预训练、归一化和模型选择不得使用保留被试；视频 ID 只用于训练配对，不输入推理分类器。另做留出视频诊断，防止把同刺激跨人泛化解释为跨刺激泛化。
3. **情绪：EmT**，[IEEE TNNLS 36(6):10381–10393, 2025](https://doi.org/10.1109/TNNLS.2025.3552603)，[官方代码](https://github.com/yi-ding-cs/EmT)。以通道×频带特征构造时间图，图网络提取空间关系，Transformer 学习较长时间上下文。与我们已有 1 秒频带序列 TCN 更直接衔接，是补充结构对照的优先候选；先比较相同特征上的平均分类、TCN、EmT，避免同时改变特征和网络。不是 HUST 团队工作。
4. **情绪：UACL-Net**，[IEEE TCYB 56(4):1845–1858, 2026](https://doi.org/10.1109/TCYB.2026.3665442)，[作者 PDF](https://www.fst.um.edu.mo/personal/wp-content/uploads/2026/03/EEGemotion.pdf)。不确定性对比学习与频域注意力可作为标签模糊问题的思路。论文同时列五折和额外 LOSO / 跨数据集实验，FACED 高分表含效价/唤醒度，不能视为我们八分类跨被试上限；未找到已核实官方代码链接，预训练 1000 epoch，不列为短期第一优先。
5. **评测证据**：[Subject Identity Confounds qEEG Emotion Recognition on DEAP and DREAMER](https://pmc.ncbi.nlm.nih.gov/articles/PMC13567904/)，Sensors 26(17):5327, 2026。作者在相同特征/模型下观察到随机窗口与独立被试评估差异，说明协议核对必要；其 qEEG 二分类结果不能推广为所有深度情绪模型无效。
6. 已调研未跑的 **DMMR**（[官方代码](https://github.com/CodeBreathing/DMMR)，AAAI 2024）仍可作为源域泛化重建方案，但属于会议，不标为 IEEE / SCI 期刊。**EMOD**（[官方代码](https://github.com/cyn4396/EMOD)，AAAI 2026）是多数据集情绪预训练方向，不是 HUST 成果；当前权重适配问题未解决，排在数据扩充和小基线之后。

## 三、更多数据：实际访问核验

| 数据 | 范式和规模 | 2026-11-13 本机核查 | 用途 / 注意事项 |
|---|---|---|---|
| [现有 FACED 固定版本公开导出](https://huggingface.co/datasets/braindecode/faced/tree/18c73355fc71c055e92828384da85bb49bc6e3c0) | 123 个 recording，26 EEG，200 Hz，10 秒窗口 | **已校验 123 份 metadata.tsv 的 Git blob SHA-1**，19,217 窗口、8 类，全部缺 Neutral；第 123 份组元数据也可访问 | 最直接扩充现有情绪实验；尚未下载新增 111 人信号。固定被试映射、单位/通道审计和新划分后才可训练；不覆盖旧 12 人版本。公开镜像许可证仍需保留来源并核实，不再分发数据 |
| [FACED / GNN4EEG 作者特征](https://github.com/Miracle-2001/GNN4EEG) | 123 人，九类任务特征，DE-LDS 数组 | **已实际下载** 123,984,200 bytes MATLAB 文件，变量 de_lds 形状 **123×840×150** | 150=作者说明的 30 通道×5 频带；不是原始 EEG，不能直接喂给 CBraMod。文件不含标签变量，必须从作者协议校验视频类别顺序、被试顺序、LDS 时序信息范围和标签，再构建独立九类特征基准；目前没有九类成绩 |
| [PhysioNet EEGMMIDB](https://physionet.org/content/eegmmidb/1.0.0/) | 109 人，64 EEG，160 Hz，运动执行与想象 | **已下载一个官方 MI EDF** 2,596,896 bytes；头部核对 65 信号含注释，EEG 160 Hz | MI 只选 R04/08/12 左右拳、R06/10/14 双拳/双脚；T1/T2 含义随 run 变化。可跨数据集预训练/独立验证，不把其标签直接拼成含舌头的 BNCI 四分类 |
| [SHU multi-session](https://figshare.com/articles/dataset/SHU_Multi-session_Dataset/19228725)，[Scientific Data 2022 论文](https://doi.org/10.1038/s41597-022-01647-1) | 25 人，32 EEG，250 Hz，5 日，左右手 MI | Figshare API 可访问，CC BY 4.0；MAT ZIP 1,433,568,381 bytes，范围请求返回有效 ZIP 头；README 下载本次 403 | 适合分离跨日与跨被试困难。**未全量下载、未验证解压**；ZIP 头显示加密条目标记，不能宣称已无障碍读取数据。全部文件约 2.23 GB 包括重复格式，不应当作纯 EEG 大小 |
| [OpenBMI / Lee2019](https://doi.org/10.1093/gigascience/giz002)，[GigaDB](https://gigadb.org/dataset/100542)，[代码](https://github.com/PatternRecognition/OpenBMI) | 54 人，62 EEG，两会话，MI 左右手 | 本机 HTTPS FTP 入口 SSL 失败；尚无本轮下载 | 有价值的大样本补充，但短期先用确定可访问数据；只选 MI，不混入同库 ERP/SSVEP |
| [SEED 系列官方申请页](https://bcmi.sjtu.edu.cn/home/seed/downloads.html) | SEED 三类、SEED-IV 四类、SEED-V 五类 | 官方明确申请与授权后提供访问 | 若实验室已有授权，可提供本地路径接入；未申请、未发送邮件。多模态方案还需核对哪些人有眼动 |
| [DEAP 官方 EULA](https://eecs.qmul.ac.uk/mmv/datasets/deap/doc/eula.pdf) / [DREAMER 官方记录](https://zenodo.org/records/546113) | 32 人 / 23 人；效价、唤醒度等评分 | DEAP 需签署申请；DREAMER 本次 Zenodo API access_right=restricted、files=[] | 不标注匿名可下载；二分类结果与 FACED 八类/九类不可直接比较 |

FACED 全体八类窗口数：Sadness 2214、Fear 2583、Anger 2583、Disgust 2706、Inspiration 3223、Amusement 2218、Tenderness 2214、Joy 1476。按窗口数而非被试数计；不得把窗口当独立受试者做显著性检验。

## 四、下一轮受控实验顺序

### A. 快速诊断：现有数据即可做

1. MI 同一批被试上比较被试内按完整 trial 分组验证与未见被试验证；情绪按完整视频/trial 分组，禁止相邻窗口跨集合。只在开发被试做诊断。两种协议分别报告，不用被试内高分替代正式结果。
2. MI：当前 EEGNet ERM → 加源被试分组采样 → 源域 CORAL。只从训练被试构造损失，验证被试不参与反向传播或归一化拟合。CORAL 权重候选 0、0.01、0.1，学习率候选 0.001、0.003，先固定种子筛选再补其他种子，预算和选择次数对齐；保留按被试 BA。
3. MI：补收缩协方差→切空间→正则 Logistic / SVM，所有参考点和缩放仅训练拟合，构建不依赖复杂主干的空间统计基线。
4. 若比赛推理约束允许，另建 offline-target-EA 和 causal-online-EA 两张独立表，比较同一 EEGNet；不使用目标标签、不用未来试次做在线参考，不把目标集合 EA 与严格 DG 混算。

### B. 情绪：先增加训练人，再比表征

1. 新建 faced_hub_123_v2 数据版本。按被试固定 train/val/test，并将旧已看过测试被试 3、11 留在开发侧；新增测试被试预先固定，任何标签使用只限加载/最终评分，特征拟合和调参隔离。先保存 manifest、被试列表与 SHA，再训练。新队列不与旧队列成绩声称配对提升。
2. 相同新划分比较 relative power + Logistic、1 秒 DE/logvar 序列 + TCN、同序列 + EmT；低维特征标准化仅训练拟合。对论文/代码的改动完整注明，不称完整作者复现。
3. CLISA 只用训练被试配对预训练；先固定低算力配置，再与无对比预训练做同头同预算对照。增加视频留出作为独立诊断。
4. GNN4EEG 文件另建特征实验，先复现 DE + 正则 SVM，再用轻量图网络。LDS 非因果可能使用后续窗口信息，应标为离线特征条件；不要跨被试一起拟合平滑或标准化。

### C. 扩数据后的模型迁移

MIRepNet 先完成官方权重加载与前向一致性核验，再做统一四分类跨被试试验；如做少量目标标签校准，单列 K-shot 曲线，不与零校准混算。PhysioNet / SHU 的二分类标签可用于专用预训练或独立二分类评估，不能直接混合所有任务标签。CKD 排在读取器/数据和协议审计完成之后。

保留当前 SSVEP FBCCA 和全部负结果。外部预训练权重和教师只用于手册允许阶段；决赛随机初始化、官方数据限定需独立实现，不用本调研建议覆盖比赛规则。

## 五、本轮已完成与证据

- 官方代码已保存：external/MIRepNet，commit b35d113cd7b1629b1e0f35a5f2495ee15c69c291；external/SCLDGN，commit 405d0c6d087747f54b0eae5f7d1fab3276567d0c。仅读取，未执行第三方训练脚本，未安装它们的旧依赖。
- access_audit.json：tmp/research_20261113/，记录 PhysioNet / SHU / FACED / OpenBMI 的真实访问状态。
- faced_expansion_audit.json：全部 123 份标签元数据哈希核验和类别计数；原始 TSV 留在 tmp/research_20261113/faced_metadata。
- faced_feature_file_audit.json：特征 MAT 结构、大小、SHA-256；文件在 data/raw/FACED_GNN4EEG/FACED_dataset_9_labels.mat。
- FACED 特征 SHA-256：7f27e02cdb316a97e01c080fd298975ade941eb11d3132b6908dc0e28ef261bb。
- PhysioNet 样本 SHA-256：3d161f88e1c00632585287d2ce584c2bc0f08862438eb255ea8723e00fac693d。
- 本轮没有新的模型训练结果，也没有新增 111 人 EEG 信号已下载的声明。数据入口可访问不等于完成预处理或复现论文成绩。
