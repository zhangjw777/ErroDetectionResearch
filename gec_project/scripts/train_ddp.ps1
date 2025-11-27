# DDP分布式训练启动脚本 (PowerShell)
# 使用方式: .\scripts\train_ddp.ps1

# 设置环境变量 (可选，调试时使用)
$env:CUDA_VISIBLE_DEVICES = "0,1"  # 使用GPU 0和1

# 切换到项目目录
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting DDP Training with 2 GPUs" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 使用torchrun启动DDP训练
# --nproc_per_node: 每个节点的进程数（GPU数量）
# --master_port: 主节点端口（避免冲突可修改）
torchrun --nproc_per_node=2 --master_port=12355 src/trainer.py

Pop-Location
