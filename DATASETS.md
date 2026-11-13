# 数据、依赖与存储

本仓库仅上传代码和实验记录。原始数据、处理缓存、模型权重没有随Git提交；克隆仓库不会获得这些二进制文件。公开仓库不等于全量云备份。

## 数据获取

- MI：BNCI2014-001，官方入口 https://bnci-horizon-2020.eu/database/data-sets 。仅T会话的本地协议见README及plans；原始数据获取后放在 `data/raw/`，准备脚本见 `scripts/prepare_bnci.py`。
- SSVEP：Tsinghua Benchmark，官方入口 https://bci.med.tsinghua.edu.cn/download.html 。遵守原站条款；缓存位于 `data/processed/ssvep_1s.npz` 和 `ssvep_2s.npz`。
- FACED：完整原始数据需从作者/Synapse入口申请。八类公开导出和123人九类作者特征是两套不同实验，获取来源及SHA见各运行config和数据审计记录。不得据此宣称获得完整原始九分类数据。
- SEED：https://bcmi.sjtu.edu.cn/home/seed/ ，按作者要求申请并遵守使用协议。原机器的 `D:/seed/SEED/SEED.zip` 不上传；作者DE-LDS特征被用于本项目特征实验。SEED、SEED-IV、SEED-VIG不能互换，后两者未据此宣称完成实验。

数据拥有者的许可适用于原始文件及其衍生数据；未经核实不向公开仓库再分发。清单中的哈希和文件大小用于迁移核验，不代替下载权限。

## 第三方代码和权重

按来源获取到 `external/`，保留各自许可证，不将第三方项目重新授权为本项目代码：

- CBraMod：https://github.com/wjq-learning/CBraMod ，已用版本见 `storage/external_repositories.json`；公开预训练权重路径 `checkpoints/cbramod.pth`，SHA见实验config。
- MIRepNet：https://github.com/staraink/MIRepNet 。
- EmT：https://github.com/yi-ding-cs/EmT 。
- 其他依赖的来源与版本见外部代码清单，和 `plans/literature_and_data_20261113.md`。

## 释放本地空间

原始数据与权重仍在本地。本次Git上传不能释放约12GB工作数据、约5.6GB二进制输出及D盘约20.9GB压缩包。不要仅凭 `git push` 成功删除它们。

后续需将有权备份的数据迁移至团队网盘、实验室服务器或对象存储，下载恢复并校验SHA256后再清理本地。仓库中的 `storage/local_inventory.json` 只记录相对位置、大小及迁移状态；它不是数据副本。Git普通文件限制及LFS计费见 https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github 和 https://docs.github.com/en/billing/concepts/product-billing/git-lfs 。
