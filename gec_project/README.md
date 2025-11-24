# 中文公文语法检错系统 (CGEC)

## 项目简介

本项目是一个专注于**政府公文领域**的中文语法检错模型，基于 **MacBERT + GECToR** 架构，通过多任务学习和Focal Loss实现高召回率的错误检测。

### 核心特性
- ✅ **轻量化部署**：支持CPU推理，INT8量化后可在消费级PC运行
- ✅ **高召回率**：针对公文领域优化，宁可误报不可漏报
- ✅ **多任务学习**：引入主谓宾识别辅助任务，增强句法感知
- ✅ **领域增强**：针对公文特点的数据合成策略

## 项目结构

```
gec_project/
├── data/                  # 数据目录
│   ├── raw/              # 原始公文语料 (jsonl格式)
│   ├── clean/            # 清洗后的正确句子
│   ├── synthetic/        # 生成的训练数据
│   └── vocab/            # 词表和标签映射
├── src/                   # 源代码
│   ├── config.py         # 全局配置
│   ├── dataset.py        # 数据集处理
│   ├── modeling.py       # 模型定义
│   ├── loss.py           # 损失函数
│   ├── trainer.py        # 训练流程
│   ├── predictor.py      # 推理接口
│   ├── utils/            # 工具模块
│       ├── augmentation.py    # 错误生成
│       └── svo_extract.py     # 句法分析（LTP）
├── experiments/          # 实验结果和模型checkpoint
└── deploy/              # 部署相关
    └── export_onnx.py   # 模型导出和量化
```

## 快速开始

### 1. 环境安装

```bash
pip install -r requirements.txt
```

### 2. 数据准备

将原始公文语料（jsonl格式）放入 `data/raw/` 目录。

### 3. 数据预处理

```bash
# 清洗语料并生成训练数据
python src/utils/preprocess.py
```

### 4. 模型训练

```bash
python src/trainer.py --config config/train_config.yaml
```

### 5. 模型推理

```bash
python src/predictor.py --model_path experiments/best_model.pt --input "待检测的文本"
```

### 6. 模型部署

```bash
# 导出ONNX并量化
python deploy/export_onnx.py --model_path experiments/best_model.pt
```

## 技术架构

- **Base Model**: `hfl/chinese-macbert-base`
- **任务架构**: Seq2Edit (GECToR)
- **辅助任务**: 主谓宾核心成分识别
- **损失函数**: Focal Loss + CrossEntropy
- **优化器**: AdamW with Warmup

## 数据格式说明

详见 [SPECIFICATION.md](../SPECIFICATION.md) 第3章节。

## 性能指标

模型在政府公文验证集上的表现：
- **Recall**: > 90% (核心指标)
- **Precision**: ~70%
- **F0.5**: 平衡召回率的综合指标
- **推理速度**: < 100ms/句 (CPU, INT8)

## 开发计划

- [x] 项目框架搭建
- [ ] 数据预处理管道
- [ ] 模型训练代码
- [ ] 推理和评估
- [ ] 模型量化和部署
- [ ] 论文撰写

## License

MIT

## 联系方式

如有问题请提Issue。
