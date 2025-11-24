"""
模型部署脚本
将训练好的PyTorch模型导出为ONNX格式并进行INT8量化
"""
import torch
import onnx
import onnxruntime
from pathlib import Path
import logging
import argparse

from config import default_config as cfg
from modeling import GECModelWithMTL
from predictor import GECPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_onnx(
    model_path: str,
    output_path: str,
    dummy_input_size: int = 128
):
    """
    将PyTorch模型导出为ONNX格式
    
    Args:
        model_path: PyTorch checkpoint路径
        output_path: ONNX模型输出路径
        dummy_input_size: 示例输入的序列长度
    """
    logger.info(f"Loading model from {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 假设有label_map
    from dataset import build_label_maps
    gec_label_map, svo_label_map = build_label_maps(str(cfg.VOCAB_DIR))
    
    from modeling import create_model
    model = create_model(
        bert_model_name=cfg.BERT_MODEL,
        num_gec_labels=len(gec_label_map),
        num_svo_labels=len(svo_label_map),
        device='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    
    # 创建示例输入
    batch_size = 1
    dummy_input_ids = torch.randint(
        0, 21128, (batch_size, dummy_input_size), dtype=torch.long
    )
    dummy_attention_mask = torch.ones(
        batch_size, dummy_input_size, dtype=torch.long
    )
    
    # 导出ONNX
    logger.info(f"Exporting to ONNX format: {output_path}")
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['gec_logits', 'svo_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'gec_logits': {0: 'batch_size', 1: 'sequence_length'},
            'svo_logits': {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=cfg.ONNX_OPSET_VERSION,
        do_constant_folding=True,
    )
    
    logger.info("ONNX export completed")
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")


def quantize_model(
    model_path: str,
    quantized_path: str
):
    """
    对PyTorch模型进行INT8量化
    
    Args:
        model_path: 原始模型路径
        quantized_path: 量化后模型输出路径
    """
    logger.info(f"Quantizing model from {model_path}")
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    from dataset import build_label_maps
    gec_label_map, svo_label_map = build_label_maps(str(cfg.VOCAB_DIR))
    
    from modeling import create_model
    model = create_model(
        bert_model_name=cfg.BERT_MODEL,
        num_gec_labels=len(gec_label_map),
        num_svo_labels=len(svo_label_map),
        device='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 动态量化（适用于推理）
    logger.info("Applying dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 只量化Linear层
        dtype=torch.qint8
    )
    
    # 保存量化模型
    checkpoint['model_state_dict'] = quantized_model.state_dict()
    torch.save(checkpoint, quantized_path)
    
    logger.info(f"Quantized model saved to {quantized_path}")
    
    # 比较模型大小
    import os
    original_size = os.path.getsize(model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
    
    logger.info(f"Original model size: {original_size:.2f} MB")
    logger.info(f"Quantized model size: {quantized_size:.2f} MB")
    logger.info(f"Compression ratio: {original_size / quantized_size:.2f}x")


def benchmark_inference(
    model_path: str,
    num_samples: int = 100
):
    """
    测试推理性能
    
    Args:
        model_path: 模型路径
        num_samples: 测试样本数
    """
    import time
    
    logger.info(f"Benchmarking inference speed...")
    
    # 创建预测器
    predictor = GECPredictor(
        model_path=model_path,
        vocab_dir=str(cfg.VOCAB_DIR),
        device='cpu'
    )
    
    # 测试句子
    test_text = "我们要认真学习贯彻会议精神，切实提高工作质量。"
    
    # 预热
    for _ in range(10):
        predictor.predict(test_text)
    
    # 测试
    start_time = time.time()
    for _ in range(num_samples):
        predictor.predict(test_text)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_samples * 1000  # 转为毫秒
    logger.info(f"Average inference time: {avg_time:.2f} ms/sample")
    logger.info(f"Throughput: {1000 / avg_time:.2f} samples/second")


def main():
    parser = argparse.ArgumentParser(description="Model Deployment")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--action', type=str, required=True, 
                       choices=['export_onnx', 'quantize', 'benchmark'],
                       help="Action to perform")
    parser.add_argument('--output_path', type=str, help="Output path for export/quantization")
    
    args = parser.parse_args()
    
    if args.action == 'export_onnx':
        if not args.output_path:
            args.output_path = str(cfg.DEPLOY_DIR / "model.onnx")
        export_to_onnx(args.model_path, args.output_path)
    
    elif args.action == 'quantize':
        if not args.output_path:
            args.output_path = str(cfg.DEPLOY_DIR / "model_quantized.pt")
        quantize_model(args.model_path, args.output_path)
    
    elif args.action == 'benchmark':
        benchmark_inference(args.model_path)


if __name__ == "__main__":
    # 示例用法
    print("Model Deployment Script")
    print("\nUsage examples:")
    print("1. Export to ONNX:")
    print("   python export_onnx.py --model_path experiments/best_model.pt --action export_onnx")
    print("\n2. Quantize model:")
    print("   python export_onnx.py --model_path experiments/best_model.pt --action quantize")
    print("\n3. Benchmark:")
    print("   python export_onnx.py --model_path experiments/best_model.pt --action benchmark")
