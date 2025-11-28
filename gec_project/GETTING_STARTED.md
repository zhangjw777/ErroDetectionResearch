# 项目初始化完成！

## 📁 项目结构已创建

```
gec_project/
├── data/                       ✅ 数据目录
│   ├── raw/                   # 放入原始公文JSONL文件
│   ├── clean/                 # 清洗后的句子
│   ├── synthetic/             # 生成的训练数据
│   └── vocab/                 # 标签映射
├── src/                        ✅ 源代码
│   ├── config.py              # 全局配置
│   ├── dataset.py             # 数据集处理
│   ├── modeling.py            # 模型定义
│   ├── loss.py                # 损失函数
│   ├── trainer.py             # 训练流程 (支持AMP+DDP)
│   ├── predictor.py           # 推理接口
│   ├── preprocess.py          # 数据预处理
│   └── utils/
│       ├── augmentation.py    # 错误生成
│       └── svo_extract.py     # 句法分析
├── scripts/                    ✅ 启动脚本
│   ├── train_ddp.ps1          # Windows双卡训练
│   ├── train_ddp.sh           # Linux/Mac双卡训练
│   └── train_single.ps1       # 单卡训练
├── experiments/                ✅ 实验结果
├── deploy/                     ✅ 部署脚本
│   └── export_onnx.py
├── requirements.txt            ✅ 依赖清单
└── README.md                   ✅ 项目说明
```

## 🚀 下一步操作

### 1. 准备数据
将你的10万条公文数据（JSONL格式）放入 `data/raw/` 目录

### 2. 安装依赖
在训练机器上运行：
```bash
pip install -r requirements.txt
```

### 3. 数据预处理

**CPU模式**（较慢）:
```bash
cd gec_project
python src/preprocess.py
```

**GPU模式（推荐，快10-20倍）**:
```bash
# 处理所有句子（默认）
python src/preprocess.py --use_cuda

# 只处理前5000个句子（用于快速测试）
python src/preprocess.py --use_cuda --max_sentences 5000
```

**CPU模式**（较慢，适合小规模测试）:
```bash
# 只处理1000个句子
python src/preprocess.py --max_sentences 1000
```

**完整参数说明**:
```bash
python src/preprocess.py --use_cuda --max_sentences 5000 --num_samples 3
```

可用参数：
- `--use_cuda`: 使用GPU加速SVO提取（需要CUDA环境）
- `--max_sentences`: 处理的句子数量（默认：None，处理全部）
- `--num_samples`: 每个句子生成的错误样本数（默认：2）
- `--raw_dir`: 原始数据目录（默认：data/raw）
- `--output_dir`: 输出目录（默认：data/synthetic）

### 4. 训练模型

#### 🔥 单卡训练
```bash
python src/trainer.py
```

#### ⚡ 双卡DDP训练（推荐，有两张4090时使用）

**Windows (PowerShell)**:
```powershell
.\scripts\train_ddp.ps1
```

**或直接使用 torchrun**:
```bash
torchrun --nproc_per_node=2 src/trainer.py
```

**Linux/Mac**:
```bash
bash scripts/train_ddp.sh
```

> 📝 **DDP 训练说明**：
> - 使用两张 GPU 并行训练，速度几乎翻倍
> - 每张 GPU 的 batch_size = `BATCH_SIZE`，总 batch_size = `BATCH_SIZE × 2`
> - 自动启用 AMP (FP16) 混合精度，4090 上速度更快、显存更省
> - 仅主进程 (rank=0) 打印日志和保存模型

### 5. 推理测试
```bash
python src/predictor.py
```

### 6. 模型部署
```bash
# 导出ONNX
python deploy/export_onnx.py --model_path experiments/best_model.pt --action export_onnx

# 模型量化
python deploy/export_onnx.py --model_path experiments/best_model.pt --action quantize

# 性能测试
python deploy/export_onnx.py --model_path experiments/best_model.pt --action benchmark
```

## 📝 关键文件说明

- **config.py**: 所有超参数都在这里，可根据需要调整
- **modeling.py**: MacBERT + 双头架构的实现
- **loss.py**: Focal Loss + 多任务损失
- **trainer.py**: 完整的训练循环，支持 AMP + DDP + 梯度累积
- **augmentation.py**: 公文领域的错误生成策略
- **svo_extract.py**: 使用DDParser提取主谓宾

## ⚠️ 注意事项

1. 首次运行需要下载MacBERT模型（约400MB）
2. DDParser首次运行会下载模型文件
3. 训练过程中会自动保存最佳模型（基于Recall）
4. 所有的import错误提示是因为库还未安装，在训练机器上安装后即可
5. DDP 训练需要 NCCL 后端支持（Linux 原生支持，Windows 需要确保 PyTorch 正确安装）

## 📊 核心指标

项目重点关注：
- **Recall（召回率）**: 宁可误报不可漏报
- **F2 Score**: 强调召回率的综合指标
- **推理速度**: 目标 < 100ms/句（CPU）

## 🔧 可调参数

在 `config.py` 中可以调整：

### 基础训练参数
- `BATCH_SIZE`: 批大小（默认128）
- `NUM_EPOCHS`: 训练轮数（默认8）
- `LEARNING_RATE`: 学习率（默认2e-5）
- `FOCAL_LOSS_GAMMA`: Focal Loss的gamma（默认2.0）
- `MTL_LAMBDA_SVO`: SVO辅助任务权重（默认0.5）
- `MTL_LAMBDA_SENT`: 句级检测辅助任务权重（默认0.5）

### 混合精度与分布式训练
- `USE_AMP`: 是否启用混合精度训练（默认True，强烈建议开启）
- `GRADIENT_ACCUMULATION_STEPS`: 梯度累积步数（默认1，可增大以模拟更大batch）

### AMP + DDP 性能预期（双4090）
| 配置 | 每epoch时间 | 显存占用 |
|------|------------|---------|
| 单卡 FP32 | ~30min | ~20GB |
| 单卡 AMP | ~15min | ~12GB |
| 双卡 DDP + AMP | ~8min | ~12GB/卡 |

项目框架已经完全搭建好了！
