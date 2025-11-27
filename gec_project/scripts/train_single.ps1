# 单卡训练启动脚本 (PowerShell)
# 使用方式: .\scripts\train_single.ps1

# 设置环境变量 (可选)
$env:CUDA_VISIBLE_DEVICES = "0"  # 使用GPU 0

# 切换到项目目录
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Green
Write-Host "Starting Single GPU Training" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

# 直接运行Python脚本（单卡模式）
python src/trainer.py

Pop-Location
