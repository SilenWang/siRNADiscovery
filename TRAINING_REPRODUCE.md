# siRNADiscovery 训练复现说明

本仓库包含 siRNADiscovery 的完整训练复现流程。

## 环境配置

使用 pixi 管理环境，配置文件为 `pixi.toml`。

```bash
# 安装环境（conda-forge + pypi）
pixi install
```

- 修正 Agent 给的结果，直接使用12版本，避免处理依赖问题

## siRNA-split 实验结果（10折交叉验证, GPU）

| 指标 | 复现值 | 论文值 (Briefings in Bioinformatics, 2024) |
|------|--------|-------------------------------------------|
| PCC  | 0.7678 | 0.754 ± 0.041 |
| SPCC | 0.7682 | 0.756 ± 0.042 |
| MSE  | 0.0202 | 0.021 ± 0.002 |
| AUC  | 0.8741 | 0.838 ± 0.034 |

## mRNA-split 实验结果（10折交叉验证, GPU）

| 指标 | 复现值 | 论文值 |
|------|--------|--------|
| PCC  | 0.6631 | 0.681 ± 0.050 |
| SPCC | 0.6647 | 0.693 ± 0.046 |
| MSE  | 0.0302 | 0.026 ± 0.005 |
| AUC  | 0.8440 | 0.821 ± 0.034 |

## 训练脚本位置

- siRNA 层面分割: `siRNA_split/siRNADiscovery.py` (超参数: `siRNA_split/siRNA_param.json`)
- mRNA 层面分割: `mRNA_split/siRNADiscovery.py` (超参数: `mRNA_split/mRNA_param.json`)

## 数据预处理说明

数据预处理中间产物已包含在仓库中：
- `siRNA_split/siRNA_split_preprocess/` - 共折叠 / 自折叠特征 (从 RNAcofold/RNAfold 计算并经 SVD 降维)
- `siRNA_split/RNA_AGO2/` - AGO2 相互作用概率 (从 RPISeq 计算)
- `siRNA_split/siRNA_split_datasets/` - 10 折交叉验证的 train/dev/test 分割

如需重新运行预处理，参见 `Data_preprocess/` 目录下的脚本（需要安装 ViennaRNA 和 RPISeq）。

## 参考文献

Long R, Guo Z, Han D, et al. siRNADiscovery: a graph neural network for siRNA efficacy prediction via deep RNA sequence analysis. *Briefings in Bioinformatics*, 2024. doi: 10.1093/bib/bbae563
