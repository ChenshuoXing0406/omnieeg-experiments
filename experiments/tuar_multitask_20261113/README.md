# SSVEP / MI / TUAR 三任务九组对照（2026-11-13）

本目录归档这一轮 CBraMod 主干后的残差专家实验，包含实际运行的模型、训练和数据处理代码，以及可迁移的启动入口。它与仓库原有“替换末两层 FFN”的 MoE 分开保存，不覆盖旧实验。

**范围**：MI 为 BNCI2014-001 的 22 通道 T 会话；SSVEP 为清华 Benchmark 64 通道、2 秒；信号可靠性使用 TUAR v3.0.1 的真实伪迹二分类代理任务。本轮不含情绪识别、EEGdenoiseNet、64 通道 MI 或 SSVEP 1 秒实验。开发结果不是官方赛事成绩。

## 文件与来源

- `model.py`、`train.py`、`prepare_aligned.py`、`prepare_tuar.py`、`check_setup.py`：保留实验时的原始字节。
- `source_manifest.json`：原始文件 SHA256、参考仓库提交及公开权重地址/哈希。模型与训练源码哈希已与九组运行记录逐一核对。
- `vendor/models/`：本轮使用的 CBraMod 源码，来源 [wjq-learning/CBraMod](https://github.com/wjq-learning/CBraMod)，MIT 授权原文位于 `vendor/LICENSE_CBraMod`。
- `launch.py`：新增的可迁移启动器，保留九组名称、训练参数与队列分组，使用显式 GPU 参数替代历史机器的 UUID 和前置队列依赖。
- `prepare_public.py`：新增的本地 MAT 数据入口，沿用原始 MI 处理函数，按相同滤波/裁窗/重采样规则重新生成 SSVEP 缓存。新生成缓存记录该入口哈希。
- `results_summary.json`：2026-11-14 完成快照的聚合验证指标与配置，不含逐患者指标或逐窗口预测。
- `model_parameter_counts.json`：原始模型检查记录；`verify_publication.py` 校验源码及队列对应关系，不启动训练。

不上传原始数据、患者名单、逐记录/逐窗口标签、缓存、预训练或微调权重、机器登录配置。授权用户可在本地使用 `prepare_tuar.py` 重建确定性患者划分。新增便携入口未经完整九组重训，已有分数对应冻结原始核心代码及历史缓存。

## 环境与准备

建议 Linux + Python 3.11，安装与驱动匹配的 PyTorch CUDA 版本，再安装本目录 `requirements.txt`。依赖文件给出所需库，不代表所有版本组合都能逐位复现；运行时保存 torch 版本、GPU 和代码哈希。原训练为 FP32，并将当前进程的 CUDA 缓存分配上限设为可见卡显存的 30%，不等同于 GPU 算力配额。

以下命令均在本目录执行：

```bash
python -m pip install -r requirements.txt
python verify_publication.py
```

### 1. 数据

从原始发布方获得数据并遵循其许可：

- MI：[BNCI2014-001](https://lampx.tugraz.at/~bci/database/001-2014/)，仅 `A01T.mat` 至 `A09T.mat`。
- SSVEP：[清华 Benchmark](https://bci.med.tsinghua.edu.cn/download.html)，`S1.mat` 至 `S35.mat` 和 `64-channels.loc`。
- TUAR：[Temple EEG resources](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)，需要相应官方访问授权。本代码不提供受限数据下载凭据或镜像。

MI/SSVEP 原始文件目录结构：

```text
/path/to/raw/
  mi/A01T.mat ... A09T.mat
  ssvep/S1.mat ... S35.mat
  ssvep/64-channels.loc
```

```bash
python prepare_public.py --task both --raw-root /path/to/raw
python prepare_tuar.py --source /authorized/path/tuh_eeg_artifact/v3.0.1/edf
```

缓存生成于本目录 `processed/`，不进入 Git。TUAR 会把完整患者划分和窗口信息保存到本地 `split_plan.json`。同一预处理器和同一有效源文件集合确定同一患者划分；源数据版本或排除记录变化时，须使用新实验目录，不能沿用旧分数。

若迁移已有本轮缓存，可直接将 `mi/`、`ssvep_2s/`、`tuar/` 目录置于 `processed/`，同时保留各自 manifest 与 split plan。不要混入旧版未做 CAR 的 MI 缓存。

历史 `prepare_aligned.py` 的主入口依赖相邻 `training_20261113` 缓存，供审计保留；新机器应使用上面的便携入口。

### 2. 公开权重

将 [CBraMod 公开权重](https://huggingface.co/weighting666/CBraMod/resolve/main/pretrained_weights.pth) 保存到 `weights/pretrained_weights.pth`。SHA256 应为：

```text
0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178
```

本轮严格加载权重后移除重建输出层。决赛随机初始化预训练属于后续方案，不由本轮微调代码或结果替代。

### 3. 检查与训练

```bash
python check_setup.py
python launch.py --queue all --device 0 --dry-run
```

`check_setup.py` 会用公开权重执行 CPU 形状、反向和容量检查，并在已有缓存时检查有限值与被试隔离。它不产生性能分数。

在有空余资源且已获使用授权的机器上，以下三条命令分别运行对应队列；`3/4/6` 是历史队列标签，`--device` 才是当前机器实际设备。每个队列内部顺序运行，启动器使用 Linux 文件锁防止重复运行相同实验。

```bash
python launch.py --queue 3 --device 3
python launch.py --queue 4 --device 4
python launch.py --queue 6 --device 6
# 或在一张卡上顺序跑全部九组：
python launch.py --queue all --device 0
```

独立运行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --name joint_homo_gate_seed42 \
  --mode homo_gate --tasks mi ssvep_2s tuar \
  --steps 1500 --batch 16 --eval-every 100 --seed 42
```

输出在 `results/<name>/` 和 `logs/`，均忽略。训练器会检查恢复配置是否一致；若改变代码或预处理，请使用新目录和实验名称。不要同时直接运行同名 `train.py` 与队列启动器。

## 固定数据协议

| 任务 | 输入 / 类别 | 训练 / 验证 / 测试 | 样本数 |
|---|---|---|---|
| MI | 22 × 4 × 200 / 4 类 | 被试 1–5 / 6–7 / 8–9，仅 T 会话 | 1440 / 576 / 576 |
| SSVEP | 64 × 2 × 200 / 40 类 | 1–5、9–29 / 6、30–32 / 7、8、33–35 | 6240 / 960 / 1200 |
| TUAR | 20 × 2 × 200 / 2 类 | 168 / 21 / 21 名患者 | 28156 / 13124 / 14194 |

MI：22 个 EEG 导联，连续 run 的 CAR 和 0.3–40 Hz 四阶零相位滤波，trial 起点取 [2,6) 秒，250→200 Hz，保留原始伪迹标志试次。SSVEP：整试次 6–90 Hz 四阶零相位滤波，250 Hz 下取 160:660，即 0.5 秒基线加 0.14 秒潜伏期之后的 2 秒，重采样至 200 Hz。两者单位为微伏 / 100。

TUAR 使用 20 个共同 TCP 双极导联，连续记录 0.5–80 Hz 滤波并重采样至 200 Hz，2 秒不重叠窗，不进行每窗幅度标准化。标签 0 为所选导联无伪迹标注；标签 1 为伪迹时间并集覆盖 ≥0.5 秒。覆盖介于 0 与 0.5 秒、低置信度和发作上下文窗口排除。先按患者划分，再切窗；训练每记录每类最多 64 窗口，验证/测试保留合格窗。

原始 310 条记录中保留 304 条（4 条缺标注、2 条标注时间超界）。患者按照 `SHA256('TUAR_patient_split_v1|42|' + patient_id)` 排序，前 80%、接着 10%、余下患者分别用于训练、验证、测试。完整 ID 与窗口计划只在获授权环境本地保存。

离线零相位滤波会使用窗口外上下文，不代表严格在线延迟。历史团队实验曾接触 MI/SSVEP 的测试被试，因此不称团队盲测。本轮不评估测试集。公开 CBraMod 的 TUEG 预训练与 TUAR 患者重叠尚未排除；这里只确认本轮监督划分的患者隔离。

## 九组结构与共同训练设置

| 组别 | 训练任务 | 新增模块 |
|---|---|---|
| `independent_mi_seed42` | MI | 无 |
| `independent_ssvep_2s_seed42` | SSVEP | 无 |
| `independent_tuar_seed42` | TUAR | 无 |
| `joint_plain_seed42` | 三任务 | 无 |
| `joint_adapter_seed42` | 三任务 | 单 MLP 适配器 |
| `joint_homo_fixed_seed42` | 三任务 | 3 个同构 MLP，等权 |
| `joint_homo_gate_seed42` | 三任务 | 3 个同构 MLP，学习路由 |
| `joint_hetero_fixed_seed42` | 三任务 | 频谱/空间/时间专家，等权 |
| `joint_hetero_gate_seed42` | 三任务 | 频谱/空间/时间专家，学习路由 |

专家共同输入为 `H(200) + 整窗幅度谱(201) + 原始块(200) + 统计量(4)`，合计 605 维。频谱在真实时间信号上计算，插值不提高真实分辨率。输出 `H' = H + 0.1 * sum(g_e * E_e)`。路由用 H 的全局均值与标准差生成样本级权重。**所有专家都执行，本轮不是 Top-k 稀疏计算。**

主干约 4.884M 参数；单适配器 180744，同构池 179532，异构池 180728，差异 <1%；学习路由另有 12931 参数。所有任务头为 Flatten→LayerNorm→Dropout(0.5)→Linear，相同任务在不同组中初始任务头一致。

每任务 1500 次更新、batch 16、seed 42。AdamW 主干学习率 1e-4，其余 1e-3，weight decay 0.01，余弦衰减，梯度范数上限 1。联合组每轮逐任务独立更新，共 4500 次优化器更新。类别权重仅由训练标签计算。参考训练配置来源为仓库提交 `ca188b799dd3281b3db8e783a361d9518353fe8b`。

## 已完成结果与边界

下表是**验证集被试平均 BA（%）**，不是测试或官方成绩。联合组按三任务等权平均选择同一个检查点；单任务按自身指标选模，不能将差异全部归因于梯度冲突。TUAR 有 2 名验证患者只有一种类别，其个体 BA 按出现类别的召回率计算。

| 模型 | MI | SSVEP 2 秒 | TUAR | 联合均值 |
|---|---:|---:|---:|---:|
| MI 单任务 | 50.69 | — | — | — |
| SSVEP 单任务 | — | 87.81 | — | — |
| TUAR 单任务 | — | — | 81.62 | — |
| 普通联合 | 49.65 | 87.92 | 81.13 | 72.90 |
| 单适配器 | 50.00 | 86.98 | 81.53 | 72.84 |
| 同构固定 | 49.31 | 88.02 | 81.29 | 72.87 |
| 同构学习 | 49.48 | 88.23 | 81.21 | 72.97 |
| 异构固定 | 49.48 | 86.77 | 81.62 | 72.62 |
| 异构学习 | 49.83 | 86.88 | 81.50 | 72.73 |

九组训练完成，最佳检查点重载后的验证概率与保存结果一致。当前支持联合训练流程可运行；同构学习路由只比普通联合平均高约 0.07 个百分点，不能证明专家机制有稳定收益。尚缺 MI+SSVEP 双任务对照，不能单独归因 TUAR 的净作用；需进一步重复种子及控制实验。

## 引用

- Wang J, et al. **CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding.** ICLR 2025. <https://arxiv.org/abs/2412.07236>
- Hamid A, et al. **The Temple University Artifact Corpus: An Annotated Corpus of EEG Artifacts.** IEEE SPMB 2020. <https://isip.piconepress.com/conferences/ieee_spmb/2020/papers/p01_09.pdf>
- Wang Y, et al. **A Benchmark Dataset for SSVEP-Based Brain-Computer Interfaces.** IEEE TNSRE 2017. <https://doi.org/10.1109/TNSRE.2016.2627556>
- BNCI2014-001 / BCI Competition IV dataset 2a. <https://www.bbci.de/competition/iv/>
