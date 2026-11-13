# SEED 三分类跨被试特征实验 v1

数据 D:/seed/SEED/SEED.zip，仅使用作者 ExtractedFeatures/de_LDS 与包内 label.mat。45份MAT、15人每人3会话。读取完整ZIP成员时CRC校验，并保留逐成员SHA256；原始压缩包不改写。

标签原始 -1/0/+1 → 模型0/1/2（negative/neutral/positive）。按每条视频的10个连续1秒特征帧组成非重叠10秒窗口，不跨trial拼接。丢弃每个视频末尾不足10帧部分并计数；输入[N,10,62,5]。作者LDS平滑可能含同视频前后文，本轮明确为离线特征基准，不声称实时10秒推理或原始EEG实验。

三折验证被试预先固定1–5、6–10、11–15，其他10人为训练；同一人所有会话始终在一侧。全部15人用于开发交叉验证，没有保留独立测试。不能把配置选择后的最佳均值称为盲测或正式比赛成绩；折与种子共同变化不作统计显著性声明。

18个拟合：每折时间平均DE+StandardScaler+balanced Logistic(C=.1/1)、时间平均DE+StandardScaler+balanced LinearSVC(C=.01/.1)、序列TCN、官方EmT分类器的小配置各一。所有标准化仅训练被试拟合；SVM softmax margin未校准，用作分类和视频聚合，不解释为校准概率。

神经网络用相同训练样本序列：每批8个不同训练被试，各类各抽1个窗口，共24。seed42/43/44，1500更新，AdamW lr.001 wd.01，cosine，dropout.5，梯度范数裁剪1，每100步验证并按被试平均窗口BA选择。TCN沿用现有local SequenceTCN；EmT来自yi-ding-cs/EmT官方EmT_Cls.py，原文件SHA纳入配置；不称完全复现原论文训练。

EmT配置：graph layers[1,2], transformer1, adjacency2, hidden16, Cheby K3, heads4, dim_head16, alpha.25, 62channels,5features,3classes。两种神经网络的模型容量不同，作为方法对照，不作为单一结构因果消融。

保存窗口级预测，另外按视频trial平均窗口分数报告视频级BA。主指标为每个验证被试窗口BA的平均值。全部配置和逐折结果都报告，最终从保存checkpoint重载核验。SEED三类不与FACED八类直接比较高低；SEED-IV、SEED-VIG本轮不使用。
