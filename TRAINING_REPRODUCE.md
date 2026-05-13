# siRNADiscovery 训练复现说明

本仓库包含 siRNADiscovery 的完整训练复现流程。

## 环境配置

使用 pixi 管理环境，配置文件为 `pixi.toml`。

```bash
# 安装环境（conda-forge + pypi）
pixi install
```

### 关键依赖
- Python 3.10, TensorFlow 2.14.0, stellargraph 1.2.1
- CUDA 11.8 (cudatoolkit), cuDNN 8.9.7

## GPU 配置

本机显卡: NVIDIA GeForce RTX 3060 (12GB)

### ptxas
系统缺少 `ptxas`，需要从完整 CUDA Toolkit 安装或拷贝：

```bash
# ptxas 放入 pixi 环境 bin 目录
cp /path/to/ptxas .pixi/envs/default/bin/
chmod +x .pixi/envs/default/bin/ptxas

# libdevice.10.bc 软链接（供 XLA 编译使用）
mkdir -p .pixi/envs/default/nvvm/libdevice
ln -sf .pixi/envs/default/lib/libdevice.10.bc \
       .pixi/envs/default/nvvm/libdevice/libdevice.10.bc
```

### 设置环境变量并训练

```bash
CONDA_ENV_LIB="$(pwd)/.pixi/envs/default/lib"
CONDA_PREFIX="$(pwd)/.pixi/envs/default"
LD_LIBRARY_PATH="$CONDA_ENV_LIB:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64" \
XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX" \
PATH="$CONDA_PREFIX/bin:$PATH" \
pixi run python siRNADiscovery.py
```

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
