# 公共盘TUAR/TUEV输入与标注核对

2026-11-14通过自己的10022账号读取。只读EDF头、CSV/REC标注；没有训练、输出原始波形或对eval作模型评分。

入口：`scripts/audit_tuh_inputs.py`；完整逐文件清单：`outputs/physionet_models_20261114/tuh_input_audit.json`。

## TUAR v3.0.1

目录：`/mnt/data2/LaBraM_data/TUEG/tuh_eeg_artifact/v3.0.1`。

- 310个EDF均可读取头；由文件名得到213个被试。
- 采样率为250/256/400/512/1000Hz，各201/93/9/6/1个文件；不得把整个数据集固定当作250Hz。
- EDF原始信号数为27–128，包含不同采集配置，不能将其总信号数直接认作统一EEG电极数，也不能直接等同比赛公布的58或18通道。
- 存在按TCP双极通道标注的CSV，以及eyem_musc、musc_elec等复合标签。必须根据具体任务定义处理，不能默默丢弃或猜成单标签；seiz CSV与伪迹CSV不可混淆。
- 发现4个EDF没有同名主CSV，也没有同stem其他文件。未核验其是否应被视为全背景，因此本轮只标注缺失，后续读取器不得静默生成背景标签：
  - `edf/01_tcp_ar/aaaaadmi_s013_t013.edf`
  - `edf/01_tcp_ar/aaaaajoo_s002_t001.edf`
  - `edf/01_tcp_ar/aaaaalaf_s001_t000.edf`
  - `edf/01_tcp_ar/aaaaapnl_s008_t004.edf`

## TUEV v2.0.1

目录：`/mnt/data2/LaBraM_data/TUEV/v2.0.1`。

- 518个EDF与518个同名REC可读，未发现缺失REC；全部EDF头采样率250Hz。
- EDF原始信号数24–33，REC的channel索引指向README定义的TCP双极montage，不能按原EDF信号序号直接取通道。
- REC记录条数：spsw(1)1212、gped(2)15931、pled(3)8182、eyem(4)1399、artf(5)13257、bckg(6)73372。这是标注行计数，不是独立trial或被试数，不能拿它直接当训练样本比例。
- README说明eval目录的数字为evaluation index；本轮未核验它与原始患者身份的映射，不自动把eval/032之类目录当作新被试ID重划分。
- 这些六类事件标签并不自动构成“干净/不干净”二分类。任务映射需和队友及最终比赛标签协议对齐，本轮不新增标签定义。

## 对后续模型适配的影响

先处理真实电极标签、AR/LE参考与TCP差分关系，再做采样率统一和2秒裁窗；不能仅靠padding/复制电极凑比赛输入。缺失标注、复合标签、窗口与标签区间重叠规则、被试身份应写进独立数据版本元信息。

这份盘点服务队友已有信号可靠性实验，不构成新的模型成绩，也不认定两库就是官方隐藏赛题。
