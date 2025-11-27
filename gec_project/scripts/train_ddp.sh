#!/bin/bash
# DDP分布式训练启动脚本 (Linux/Mac)
# 使用方式: bash scripts/train_ddp.sh

# 设置环境变量 (可选，调试时使用)
export CUDA_VISIBLE_DEVICES="0,1"  # 使用GPU 0和1

# 切换到项目目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Starting DDP Training with 2 GPUs"
echo "========================================"

# 使用torchrun启动DDP训练
# --nproc_per_node: 每个节点的进程数（GPU数量）
# --master_port: 主节点端口（避免冲突可修改）
torchrun --nproc_per_node=2 --master_port=12355 src/trainer.py
