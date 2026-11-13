# 数据、依赖与存储

Git主分支提供代码和实验记录；实验训练检查点、保存预测和日志另附于 [Release](https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/artifacts-20261113)。共13个分卷，内部2784个文件具有SHA256清单；所有卷与GitHub服务端摘要一致，本地完整归档读取核验通过。GitHub附件未做全量回程下载；后续私有服务器备份已另行完成实际解包和逐文件SHA256校验，本机大文件据此清理。

原始数据、DE/EEG处理缓存、原始第三方预训练权重没有上传；公开仓库不等于所有数据的云备份。SEED官方协议禁止再分发数据或其中的部分，见 https://bcmi.sjtu.edu.cn/home/seed/downloads_cn.html 。FACED公开导出的本地数据卡标注license unknown，不将“可下载”视为已获再分发许可。

## 数据获取

- MI：BNCI2014-001，官方入口 https://bnci-horizon-2020.eu/database/data-sets 。官方明确标注CC BY-ND 4.0。9个未修改T会话原始MATLAB文件已提供于 https://github.com/ChenshuoXing0406/omnieeg-experiments/releases/tag/raw-bnci-20261113 ，附署名、许可与哈希。只做tar.gz打包，无E会话和处理衍生物。准备脚本见 `scripts/prepare_bnci.py`。
- SSVEP：Tsinghua Benchmark，官方入口 https://bci.med.tsinghua.edu.cn/download.html 。遵守原站条款；缓存位于 `data/processed/ssvep_1s.npz` 和 `ssvep_2s.npz`。
- FACED：完整原始数据需从作者/Synapse入口申请。八类公开导出和123人九类作者特征是两套不同实验，获取来源及SHA见各运行config和数据审计记录。不得据此宣称获得完整原始九分类数据。

NEMAR的nm000112 v1.1.4明确采用CC-BY-4.0（https://nemar.org/dataset/nm000112?v=v1.1.4），但本机从该版本仅取得元数据和下载清单，没有据此取得完整原始EEG。已有八类Hugging Face导出的数据卡标注license unknown，不能把NEMAR版本的许可直接套在来源映射尚未独立核验的导出文件上。SSVEP已查到官方下载入口，但尚未找到明确再分发条款；这两部分保留私有备份。
- SEED：https://bcmi.sjtu.edu.cn/home/seed/ ，按作者要求申请并遵守使用协议。原机器的 `D:/seed/SEED/SEED.zip` 不上传；作者DE-LDS特征被用于本项目特征实验。SEED、SEED-IV、SEED-VIG不能互换，后两者未据此宣称完成实验。

数据拥有者的许可适用于原始文件及其衍生数据；未经核实不向公开仓库再分发。清单中的哈希和文件大小用于迁移核验，不代替下载权限。

## 第三方代码和权重

按来源获取到 `external/`，保留各自许可证，不将第三方项目重新授权为本项目代码：

- CBraMod：https://github.com/wjq-learning/CBraMod ，已用版本见 `storage/external_repositories.json`；公开预训练权重路径 `checkpoints/cbramod.pth`，SHA见实验config。
- MIRepNet：https://github.com/staraink/MIRepNet 。
- EmT：https://github.com/yi-ding-cs/EmT 。
- 其他依赖的来源与版本见外部代码清单，和 `plans/literature_and_data_20261113.md`。

## 释放本地空间

2026-11-13已将工作区快照与D盘三个SEED压缩包迁移至个人服务器的私有目录；服务器逐文件解包验证通过后，本机大体积原件、虚拟环境与本项目缓存已经清理。代码、结果表及私有恢复清单保留，少量迁移审计资料暂留。

私有恢复清单在本机 `storage/SERVER_BACKUP_MANIFEST_PRIVATE.json`；公开清理摘要为 `storage/migration_summary.json`。旧 `storage/local_inventory.json` 是迁移前库存，不代表当前仍占用本地空间，也不是数据副本。Git普通文件限制及LFS计费见 https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github 和 https://docs.github.com/en/billing/concepts/product-billing/git-lfs 。
