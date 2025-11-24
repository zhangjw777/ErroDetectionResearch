# 实验记录

本目录存放训练实验的checkpoint和日志。

## 目录结构

```
experiments/
├── exp_20241224_120000/    # 实验时间戳
│   ├── best_model.pt       # 最佳模型
│   ├── checkpoint_epoch_5.pt
│   ├── config.json         # 训练配置
│   └── tensorboard/        # TensorBoard日志
├── exp_20241224_150000/
└── ...
```

## Checkpoint内容

每个checkpoint包含：
- `model_state_dict`: 模型参数
- `optimizer_state_dict`: 优化器状态
- `scheduler_state_dict`: 学习率调度器状态
- `epoch`: 当前epoch
- `best_recall`: 最佳召回率
- `metrics`: 验证集指标

## 加载Checkpoint

```python
import torch
from modeling import GECModelWithMTL

checkpoint = torch.load('experiments/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Best Recall: {checkpoint['best_recall']}")
```

## TensorBoard可视化

```bash
tensorboard --logdir experiments/exp_20241224_120000/tensorboard
```

然后访问 http://localhost:6006
