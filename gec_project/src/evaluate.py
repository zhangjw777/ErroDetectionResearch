"""
评估模块
负责：
1. 字符级错误检测评估
2. 句级错误检测评估
3. 输出评估报告
"""
import torch
import json
import logging
from typing import List, Dict
from pathlib import Path

from config import default_config as cfg
from predictor import GEDPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEDEvaluator:
    """
    GED评估器
    
    支持多种评估模式：
    1. 字符级评估：比较 source 和 target 的逐字符差异，计算错误检测指标
    2. 句级评估：判断句子是否有错（只要 source != target 就是有错）
    
    支持的数据格式：
    - 字典格式: {'source': [...], 'target': [...], 'type': [...]}
    - 列表格式: [{'source': str, 'target': str}, ...]
    """
    
    def __init__(self, predictor: GEDPredictor):
        """
        Args:
            predictor: GEDPredictor 实例
        """
        self.predictor = predictor
    
    @staticmethod
    def _compute_char_diff(source: str, target: str) -> List[Dict]:
        """
        计算 source 和 target 之间的字符级差异
        
        返回每个 source 字符的 ground truth 标签：
        - 如果该字符在 target 中对应位置相同 -> KEEP
        - 如果该字符在 target 中被删除或替换 -> ERROR
        
        简化策略：逐字符对齐，使用编辑距离思想
        """
        # 使用简单的逐字符比较来标记错误位置
        # 注意：这是一个简化的实现，真实场景可能需要更复杂的对齐算法
        errors = []
        
        min_len = min(len(source), len(target))
        
        for i in range(min_len):
            if source[i] != target[i]:
                errors.append({
                    'position': i,
                    'source_char': source[i],
                    'target_char': target[i],
                    'type': 'REPLACE'
                })
        
        # 处理长度差异
        if len(source) > len(target):
            # source 比 target 长，说明有多余字符需要删除
            for i in range(min_len, len(source)):
                errors.append({
                    'position': i,
                    'source_char': source[i],
                    'target_char': '',
                    'type': 'DELETE'
                })
        elif len(target) > len(source):
            # target 比 source 长，说明 source 缺少字符，需要插入
            # 这里我们将缺失标记在最后一个有效位置
            for i in range(min_len, len(target)):
                # 插入错误通常在前一个位置标记
                errors.append({
                    'position': min_len - 1 if min_len > 0 else 0,
                    'source_char': '',
                    'target_char': target[i],
                    'type': 'INSERT'
                })
        
        return errors
    
    @staticmethod
    def _get_error_positions_from_labels(labels: List[str]) -> set:
        """
        从预测标签中提取错误位置
        
        只要不是 $KEEP 就认为是检测到了错误
        """
        error_positions = set()
        for i, label in enumerate(labels):
            if label != cfg.GEC_KEEP_LABEL:
                error_positions.add(i)
        return error_positions
    
    def _normalize_data(self, data) -> List[Dict]:
        """
        将输入数据归一化为统一格式
        
        支持格式：
        1. {'source': [...], 'target': [...], 'type': [...]}
        2. [{'source': str, 'target': str}, ...]
        
        Returns:
            [{'source': str, 'target': str, 'type': str}, ...]
        """
        if isinstance(data, dict):
            # 字典格式：{'source': [...], 'target': [...], 'type': [...]}
            sources = data.get('source', [])
            targets = data.get('target', [])
            types = data.get('type', ['unknown'] * len(sources))
            
            return [
                {'source': s, 'target': t, 'type': tp}
                for s, t, tp in zip(sources, targets, types)
            ]
        elif isinstance(data, list):
            # 列表格式
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    normalized.append({
                        'source': item.get('source', ''),
                        'target': item.get('target', ''),
                        'type': item.get('type', 'unknown')
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    # 元组格式 (source, target)
                    normalized.append({
                        'source': item[0],
                        'target': item[1],
                        'type': item[2] if len(item) > 2 else 'unknown'
                    })
            return normalized
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    def evaluate_char_level(
        self, 
        data,
        verbose: bool = False
    ) -> Dict:
        """
        字符级评估
        
        计算模型在字符级别的错误检测能力：
        - TP: 模型正确检测到的错误字符数
        - FP: 模型错误标记为错误的字符数（实际无错）
        - FN: 模型漏检的错误字符数（实际有错但未检出）
        
        Args:
            data: 评估数据
            verbose: 是否打印详细信息
        
        Returns:
            {
                'precision': 精确率,
                'recall': 召回率,
                'f1': F1 分数,
                'f2': F2 分数 (侧重召回率),
                'f0.5': F0.5 分数 (侧重精确率),
                'tp': 真正例数,
                'fp': 假正例数,
                'fn': 假负例数,
                'total_errors': 总错误字符数,
                'total_chars': 总字符数,
                'samples_evaluated': 评估样本数
            }
        """
        samples = self._normalize_data(data)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_errors = 0
        total_chars = 0
        
        for i, sample in enumerate(samples):
            source = sample['source']
            target = sample['target']
            
            # 获取真实错误位置
            gt_errors = self._compute_char_diff(source, target)
            gt_error_positions = {e['position'] for e in gt_errors}
            total_errors += len(gt_error_positions)
            total_chars += len(source)
            
            # 模型预测
            result = self.predictor.predict(source)
            pred_error_positions = self._get_error_positions_from_labels(result['labels'])
            
            # 计算 TP, FP, FN
            tp = len(gt_error_positions & pred_error_positions)
            fp = len(pred_error_positions - gt_error_positions)
            fn = len(gt_error_positions - pred_error_positions)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            
            if verbose and (tp > 0 or fp > 0 or fn > 0):
                logger.info(f"\n样本 {i+1}:")
                logger.info(f"  Source: {source}")
                logger.info(f"  Target: {target}")
                logger.info(f"  真实错误位置: {sorted(gt_error_positions)}")
                logger.info(f"  预测错误位置: {sorted(pred_error_positions)}")
                logger.info(f"  TP={tp}, FP={fp}, FN={fn}")
        
        # 计算指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F2 (beta=2, 侧重召回率)
        beta = 2.0
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
        
        # F0.5 (beta=0.5, 侧重精确率)
        beta = 0.5
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'f0.5': f05,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'total_errors': total_errors,
            'total_chars': total_chars,
            'samples_evaluated': len(samples)
        }
    
    def evaluate_sentence_level(
        self, 
        data,
        error_threshold: float = 0.2,
        verbose: bool = False
    ) -> Dict:
        """
        句级评估
        
        计算模型在句子级别的错误检测能力：
        - 正样本（有错）：source != target
        - 负样本（无错）：source == target
        
        Args:
            data: 评估数据
            error_threshold: 句级错误判断阈值 (sent_error_prob > threshold 判定为有错)
            verbose: 是否打印详细信息
        
        Returns:
            {
                'precision': 精确率,
                'recall': 召回率,
                'f1': F1 分数,
                'f2': F2 分数,
                'accuracy': 准确率,
                'tp': 真正例数 (正确检测到有错的句子),
                'fp': 假正例数 (误报为有错的句子),
                'fn': 假负例数 (漏检的有错句子),
                'tn': 真负例数 (正确判断为无错的句子),
                'total_positive': 真实有错的句子数,
                'total_negative': 真实无错的句子数,
                'samples_evaluated': 评估样本数
            }
        """
        samples = self._normalize_data(data)
        
        tp = 0  # 有错，预测有错
        fp = 0  # 无错，预测有错
        fn = 0  # 有错，预测无错
        tn = 0  # 无错，预测无错
        
        for i, sample in enumerate(samples):
            source = sample['source']
            target = sample['target']
            
            # 真实标签：source != target 表示有错
            gt_has_error = source != target
            
            # 模型预测
            result = self.predictor.predict(source)
            pred_has_error = result['sent_error_prob'] > error_threshold
            
            # 统计
            if gt_has_error and pred_has_error:
                tp += 1
            elif not gt_has_error and pred_has_error:
                fp += 1
            elif gt_has_error and not pred_has_error:
                fn += 1
            else:
                tn += 1
            
            if verbose:
                status = "✓" if (gt_has_error == pred_has_error) else "✗"
                logger.info(f"{status} 样本 {i+1}: 真实={'有错' if gt_has_error else '无错'}, "
                          f"预测={'有错' if pred_has_error else '无错'} (prob={result['sent_error_prob']:.3f})")
        
        # 计算指标
        total_positive = tp + fn  # 真实有错的句子数
        total_negative = tn + fp  # 真实无错的句子数
        total = tp + fp + fn + tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # F2 (beta=2, 侧重召回率)
        beta = 2.0
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'total_positive': total_positive,
            'total_negative': total_negative,
            'samples_evaluated': len(samples)
        }
    
    def evaluate(
        self, 
        data,
        error_threshold: float = 0.2,
        verbose: bool = False
    ) -> Dict:
        """
        综合评估（同时计算字符级和句级指标）
        
        Args:
            data: 评估数据
            error_threshold: 句级错误判断阈值
            verbose: 是否打印详细信息
        
        Returns:
            {
                'char_level': 字符级评估结果,
                'sentence_level': 句级评估结果
            }
        """
        char_results = self.evaluate_char_level(data, verbose=verbose)
        sent_results = self.evaluate_sentence_level(data, error_threshold=error_threshold, verbose=verbose)
        
        return {
            'char_level': char_results,
            'sentence_level': sent_results
        }
    
    def print_report(self, results: Dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("GED 评估报告")
        print("=" * 60)
        
        if 'char_level' in results:
            char = results['char_level']
            print("\n【字符级评估】")
            print(f"  样本数: {char['samples_evaluated']}")
            print(f"  总字符数: {char['total_chars']}")
            print(f"  总错误数: {char['total_errors']}")
            print(f"  TP: {char['tp']}, FP: {char['fp']}, FN: {char['fn']}")
            print(f"  Precision: {char['precision']:.4f}")
            print(f"  Recall:    {char['recall']:.4f}")
            print(f"  F1:        {char['f1']:.4f}")
            print(f"  F2:        {char['f2']:.4f}")
            print(f"  F0.5:      {char['f0.5']:.4f}")
        
        if 'sentence_level' in results:
            sent = results['sentence_level']
            print("\n【句级评估】")
            print(f"  样本数: {sent['samples_evaluated']}")
            print(f"  有错句子数: {sent['total_positive']}")
            print(f"  无错句子数: {sent['total_negative']}")
            print(f"  TP: {sent['tp']}, FP: {sent['fp']}, FN: {sent['fn']}, TN: {sent['tn']}")
            print(f"  Accuracy:  {sent['accuracy']:.4f}")
            print(f"  Precision: {sent['precision']:.4f}")
            print(f"  Recall:    {sent['recall']:.4f}")
            print(f"  F1:        {sent['f1']:.4f}")
            print(f"  F2:        {sent['f2']:.4f}")
        
        print("\n" + "=" * 60)


def main():
    """
    评估主函数
    
    使用方法：
        python evaluate.py --model_path <模型路径> --data_path <评估数据路径>
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='GED模型评估')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型checkpoint路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称（用于自动查找模型路径）')
    parser.add_argument('--data_path', type=str, default=None,
                        help='评估数据路径（JSON格式）')
    parser.add_argument('--error_threshold', type=float, default=0.2,
                        help='句级错误判断阈值')
    parser.add_argument('--verbose', action='store_true',
                        help='是否打印详细信息')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.exp_name:
        model_path = cfg.EXPERIMENTS_DIR / args.exp_name / "best_model.pt"
    else:
        # 默认使用最新的实验
        logger.error("请指定 --model_path 或 --exp_name")
        return
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 确定评估数据路径
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # 默认使用验证集
        data_path = cfg.SYNTHETIC_DIR / "dev.json"
    
    if not data_path.exists():
        logger.error(f"评估数据文件不存在: {data_path}")
        return
    
    # 设置设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 加载评估数据
    logger.info(f"加载评估数据: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)
    
    # 创建预测器
    logger.info(f"加载模型: {model_path}")
    predictor = GEDPredictor(
        model_path=str(model_path),
        vocab_dir=str(cfg.VOCAB_DIR),
        device=device
    )
    
    # 创建评估器
    evaluator = GEDEvaluator(predictor)
    
    # 运行评估
    logger.info("开始评估...")
    results = evaluator.evaluate(
        eval_data,
        error_threshold=args.error_threshold,
        verbose=args.verbose
    )
    
    # 打印报告
    evaluator.print_report(results)
    
    # 保存结果
    output_path = model_path.parent / "eval_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
