# 团队协作

仓库为公开研究代码与公开开发实验记录，不包含原始受试者波形或受限数据集。

1. Fork 仓库并克隆自己的 Fork，创建分支，例如 `experiment/mi-alignment`。
2. 使用 Python 3.11 建立虚拟环境，按 `requirements-experiments.txt` 安装依赖。CUDA版本需匹配本机驱动；纯CPU环境应另装对应torch。
3. 按 `DATASETS.md` 获取所需数据及第三方代码。训练入口在 `scripts/`，协议在 `plans/`。
4. 为新实验使用新的输出目录。不要覆盖已有结果、源代码快照或依据旧测试分数选参数。
5. 提交代码、配置与开发验证结果，通过 Pull Request 说明数据划分、参数、预算、是否使用目标侧统计，以及验证BA。不要提交数据、权重、密码、令牌或虚拟环境。

`outputs/`默认忽略，提交新结果时显式选择小文件，例如 `git add -f outputs/新实验/config.json outputs/新实验/result.json`；不要对整个outputs使用强制添加。源文件、配置和快照保留原始字节以维持哈希审计。

公开仓库允许任何人克隆和提交PR；合并仍由仓库维护者执行。不知道队友用户名不影响Fork协作。公开不代表已授予第三方代码、数据或权重的再分发许可。

## 实验入口

- `scripts/run_preliminary.py`：单任务、联合稠密与路由控制。
- `scripts/run_initialization_controls.py`：预训练/随机初始化、冻结/全量微调。
- `scripts/run_partial_sharing.py`、`run_partial_frozen.py`：共享层边界与冻结控制。
- `scripts/run_mi_domain.py`、`run_mi_specialists.py`：MI均衡采样、对齐与专用网络。
- `scripts/run_seed_features.py`、`run_seed_adapted_neural.py`：SEED特征基线及归一化控制。
- `scripts/run_faced_normalization_search.py`、`run_faced123_features.py`：FACED八类与九类作者特征，不能混表。
- `scripts/collect_complete_summary.py`：汇总既有结果，不启动训练；生成Excel还需原工作环境的artifact工具，CSV/Markdown不依赖该工具。

已有结果的config含源文件哈希，修改训练源码后应使用新协议/新目录，不要把失败的哈希校验绕过后当作原实验续跑。部分数据准备入口仍保留原机器的D盘路径，迁移前请检查入口常量并在新协议中记录调整。
