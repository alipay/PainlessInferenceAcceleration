# Painless Inference Acceleration (PIA)


<p align="center">
  
<p align="center">
   A toolkit for accelerating LLM inference without painness. Currently it only contains LOOKAHEAD, which accelerates LLM without loss of accuracy.
</p>

</p>

## News （最近更新）



## Introduction （介绍）

该代码库简称为PIA，用于LLM推理加速。

PIA的目录如下所示：
- lookahead						# lookahead框架
- lookahead/common              # lookahead公共代码
- lookahead/models			    # lookahead支持的模型
- lookahead/tests			    # 本地测试脚本及数据
- LEGAL.md 					# 合法性声明
- README.md					# 使用指南
- requirements.txt	# 依赖包


## Lincense （使用协议）

协议为CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

使用本项目前，请先阅读LICENSE.txt。如果您不同意该使用协议中列出的条款、法律免责声明和许可，您将不得使用本项目中的这些内容。

## Installation （安装指南）

- PIA的安装步骤如下所示：
```
# 新建环境（Python版本3.8）
conda create -n pia python=3.8
source activate pia

# clone项目地址
git clone xxx

# 安装包依赖
cd pia
pip install -r requirements.txt
```


## Quick Start （快速启动）

PIA提供了本地测试脚本，可以快速进行安装正确性验证：
```
# 终端运行
cd lookahead/examples
python usage.python
```

## FAQ （问答）

## Citations （引用）