"""
评估模块
负责：
1. 字符级错误检测评估
2. 句级错误检测评估
3. 输出评估报告到文件（CSV/TXT）
"""
import torch
import json
import logging
import csv
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from config import default_config as cfg
from predictor import GEDPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEDEvaluator:
    """
    GED评估器
    
    支持多种评估模式：
    1. 字符级评估：比较 source 和 target 的逐字符差异，计算错误检测指标
    2. 句级评估：判断句子是否有错
    
    支持的数据格式：
    - 字典格式: {'source': [...], 'target': [...], 'type': [...]}
        - source: 原句列表
        - target: 正确句列表
        - type: 'positive'=正样本（无错）, 'negative'=负样本（有错）
    - 列表格式: [{'source': str, 'target': str, 'type': str}, ...]
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
            for i in range(min_len, len(target)):
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
           - type: 'positive'=正样本（无错）, 'negative'=负样本（有错）
        2. [{'source': str, 'target': str, 'type': str}, ...]
        
        Returns:
            [{'source': str, 'target': str, 'type': str, 'gt_has_error': bool}, ...]
        """
        if isinstance(data, dict):
            # 字典格式：{'source': [...], 'target': [...], 'type': [...]}
            sources = data.get('source', [])
            targets = data.get('target', [])
            types = data.get('type', ['unknown'] * len(sources))
            
            return [
                {
                    'source': s, 
                    'target': t, 
                    'type': tp,
                    # positive=正样本（无错），negative=负样本（有错）
                    'gt_has_error': (tp == 'negative')
                }
                for s, t, tp in zip(sources, targets, types)
            ]
        elif isinstance(data, list):
            # 列表格式
            normalized = []
            for item in data:
                if isinstance(item, dict):
                    sample_type = item.get('type', 'unknown')
                    normalized.append({
                        'source': item.get('source', ''),
                        'target': item.get('target', ''),
                        'type': sample_type,
                        'gt_has_error': (sample_type == 'negative')
                    })
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    sample_type = item[2] if len(item) > 2 else 'unknown'
                    normalized.append({
                        'source': item[0],
                        'target': item[1],
                        'type': sample_type,
                        'gt_has_error': (sample_type == 'negative')
                    })
            return normalized
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")
    
    @staticmethod
    def _compute_f_score(precision: float, recall: float, beta: float) -> float:
        """
        计算 F-beta 分数
        
        Args:
            precision: 精确率
            recall: 召回率
            beta: beta值，beta>1侧重召回率，beta<1侧重精确率
        
        Returns:
            F-beta分数
        """
        if precision + recall == 0:
            return 0.0
        return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    
    def evaluate_char_level(
        self, 
        data,
        verbose: bool = False,
        detail_records: Optional[List[Dict]] = None
    ) -> Dict:
        """
        字符级评估
        
        计算模型在字符级别的错误检测能力：
        - TP: 模型正确检测到的错误字符数
        - FP: 模型错误标记为错误的字符数（实际无错）
        - FN: 模型漏检的错误字符数（实际有错但未检出）
        
        Args:
            data: 评估数据
            verbose: 是否在日志中记录详细信息（不在控制台打印）
            detail_records: 如果提供，将每个样本的详细记录追加到该列表
        
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
            
            # 记录详细信息
            if detail_records is not None:
                detail_records.append({
                    'index': i,
                    'level': 'char',
                    'source': source,
                    'target': target,
                    'sample_type': sample.get('type', ''),
                    'gt_error_positions': sorted(gt_error_positions),
                    'pred_error_positions': sorted(pred_error_positions),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn
                })
            
            if verbose and (tp > 0 or fp > 0 or fn > 0):
                logger.debug(f"样本 {i+1}: Source={source[:30]}... TP={tp}, FP={fp}, FN={fn}")
        
        # 计算指标
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        f1 = self._compute_f_score(precision, recall, 1.0)
        f2 = self._compute_f_score(precision, recall, 2.0)
        f05 = self._compute_f_score(precision, recall, 0.5)
        
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
        verbose: bool = False,
        detail_records: Optional[List[Dict]] = None
    ) -> Dict:
        """
        句级评估
        
        计算模型在句子级别的错误检测能力：
        - 正样本（无错）：type='positive'
        - 负样本（有错）：type='negative'
        
        Args:
            data: 评估数据，格式为 {'source': [...], 'target': [...], 'type': [...]}
                  type: 'positive'=正样本（无错）, 'negative'=负样本（有错）
            error_threshold: 句级错误判断阈值 (sent_error_prob > threshold 判定为有错)
            verbose: 是否在日志中记录详细信息
            detail_records: 如果提供，将每个样本的详细记录追加到该列表
        
        Returns:
            {
                'precision': 精确率,
                'recall': 召回率,
                'f1': F1 分数,
                'f2': F2 分数,
                'f0.5': F0.5 分数,
                'accuracy': 准确率,
                'tp': 真正例数 (正确检测到有错的句子),
                'fp': 假正例数 (误报为有错的句子),
                'fn': 假负例数 (漏检的有错句子),
                'tn': 真负例数 (正确判断为无错的句子),
                'total_positive': 真实有错的句子数 (负样本数),
                'total_negative': 真实无错的句子数 (正样本数),
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
            
            # 真实标签：从type字段获取，negative表示有错，positive表示无错
            gt_has_error = sample.get('gt_has_error', False)
            
            # 模型预测
            result = self.predictor.predict(source)
            pred_has_error = result['sent_has_error']  # 使用predictor中的判断逻辑
            
            # 统计
            if gt_has_error and pred_has_error:
                tp += 1
                status = 'TP'
            elif not gt_has_error and pred_has_error:
                fp += 1
                status = 'FP'
            elif gt_has_error and not pred_has_error:
                fn += 1
                status = 'FN'
            else:
                tn += 1
                status = 'TN'
            
            # 记录详细信息
            if detail_records is not None:
                detail_records.append({
                    'index': i,
                    'level': 'sent',
                    'source': source,
                    'target': target,
                    'sample_type': sample.get('type', ''),
                    'gt_has_error': gt_has_error,
                    'pred_has_error': pred_has_error,
                    'sent_error_prob': result['sent_error_prob'],
                    'num_edits': len(result.get('edits', [])),
                    'status': status
                })
            
            if verbose:
                logger.debug(f"样本 {i+1}: {status} | GT={gt_has_error} | Pred={pred_has_error} | prob={result['sent_error_prob']:.3f}")
        
        # 计算指标
        total_positive = tp + fn  # 真实有错的句子数（负样本）
        total_negative = tn + fp  # 真实无错的句子数（正样本）
        total = tp + fp + fn + tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        f1 = self._compute_f_score(precision, recall, 1.0)
        f2 = self._compute_f_score(precision, recall, 2.0)
        f05 = self._compute_f_score(precision, recall, 0.5)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2,
            'f0.5': f05,
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
        verbose: bool = False,
        collect_details: bool = False
    ) -> Dict:
        """
        综合评估（同时计算字符级和句级指标）
        
        Args:
            data: 评估数据
            error_threshold: 句级错误判断阈值
            verbose: 是否在日志中记录详细信息
            collect_details: 是否收集每个样本的详细记录
        
        Returns:
            {
                'char_level': 字符级评估结果,
                'sentence_level': 句级评估结果,
                'details': 详细记录列表（仅当collect_details=True时）
            }
        """
        detail_records = [] if collect_details else None
        
        char_results = self.evaluate_char_level(data, verbose=verbose, detail_records=detail_records)
        sent_results = self.evaluate_sentence_level(data, error_threshold=error_threshold, 
                                                     verbose=verbose, detail_records=detail_records)
        
        result = {
            'char_level': char_results,
            'sentence_level': sent_results
        }
        
        if collect_details:
            result['details'] = detail_records
        
        return result
    
    def save_results_to_csv(
        self, 
        results: Dict, 
        output_path: str,
        include_details: bool = True
    ) -> Dict[str, str]:
        """
        将评估结果保存到CSV文件
        
        Args:
            results: evaluate()的返回结果
            output_path: 输出文件路径（不含扩展名）
            include_details: 是否保存详细记录
        
        Returns:
            保存的文件路径字典
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # 保存汇总指标
        summary_path = output_path.with_suffix('.csv')
        with open(summary_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 写入句级评估结果
            writer.writerow(['=== 句级评估结果 ==='])
            writer.writerow(['指标', '值'])
            sent = results.get('sentence_level', {})
            writer.writerow(['样本数', sent.get('samples_evaluated', 0)])
            writer.writerow(['有错句子数(负样本)', sent.get('total_positive', 0)])
            writer.writerow(['无错句子数(正样本)', sent.get('total_negative', 0)])
            writer.writerow(['TP', sent.get('tp', 0)])
            writer.writerow(['FP', sent.get('fp', 0)])
            writer.writerow(['FN', sent.get('fn', 0)])
            writer.writerow(['TN', sent.get('tn', 0)])
            writer.writerow(['Accuracy', f"{sent.get('accuracy', 0):.4f}"])
            writer.writerow(['Precision', f"{sent.get('precision', 0):.4f}"])
            writer.writerow(['Recall', f"{sent.get('recall', 0):.4f}"])
            writer.writerow(['F0.5', f"{sent.get('f0.5', 0):.4f}"])
            writer.writerow(['F1', f"{sent.get('f1', 0):.4f}"])
            writer.writerow(['F2', f"{sent.get('f2', 0):.4f}"])
            
            writer.writerow([])
            
            # 写入字符级评估结果
            writer.writerow(['=== 字符级评估结果 ==='])
            writer.writerow(['指标', '值'])
            char = results.get('char_level', {})
            writer.writerow(['样本数', char.get('samples_evaluated', 0)])
            writer.writerow(['总字符数', char.get('total_chars', 0)])
            writer.writerow(['总错误数', char.get('total_errors', 0)])
            writer.writerow(['TP', char.get('tp', 0)])
            writer.writerow(['FP', char.get('fp', 0)])
            writer.writerow(['FN', char.get('fn', 0)])
            writer.writerow(['Precision', f"{char.get('precision', 0):.4f}"])
            writer.writerow(['Recall', f"{char.get('recall', 0):.4f}"])
            writer.writerow(['F0.5', f"{char.get('f0.5', 0):.4f}"])
            writer.writerow(['F1', f"{char.get('f1', 0):.4f}"])
            writer.writerow(['F2', f"{char.get('f2', 0):.4f}"])
        
        saved_files['summary'] = str(summary_path)
        logger.info(f"评估汇总结果已保存到: {summary_path}")
        
        # 保存详细记录
        if include_details and 'details' in results:
            details = results['details']
            
            # 分离句级和字符级详细记录
            sent_details = [d for d in details if d.get('level') == 'sent']
            
            if sent_details:
                details_path = output_path.parent / f"{output_path.stem}_sent_details.csv"
                with open(details_path, 'w', encoding='utf-8-sig', newline='') as f:
                    fieldnames = ['index', 'source', 'target', 'sample_type', 
                                  'gt_has_error', 'pred_has_error', 'sent_error_prob', 
                                  'num_edits', 'status']
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                    writer.writeheader()
                    for record in sent_details:
                        record['sent_error_prob'] = f"{record.get('sent_error_prob', 0):.4f}"
                        writer.writerow(record)
                
                saved_files['sent_details'] = str(details_path)
                logger.info(f"句级详细记录已保存到: {details_path}")
        
        return saved_files
    
    def print_report(self, results: Dict):
        """打印评估报告（简洁版，主要信息记录到日志）"""
        logger.info("=" * 60)
        logger.info("GED 评估报告")
        logger.info("=" * 60)
        
        if 'sentence_level' in results:
            sent = results['sentence_level']
            logger.info("【句级评估】")
            logger.info(f"  样本数: {sent['samples_evaluated']} (有错: {sent['total_positive']}, 无错: {sent['total_negative']})")
            logger.info(f"  TP={sent['tp']}, FP={sent['fp']}, FN={sent['fn']}, TN={sent['tn']}")
            logger.info(f"  Accuracy: {sent['accuracy']:.4f} | Precision: {sent['precision']:.4f} | Recall: {sent['recall']:.4f}")
            logger.info(f"  F0.5: {sent['f0.5']:.4f} | F1: {sent['f1']:.4f} | F2: {sent['f2']:.4f}")
        
        if 'char_level' in results:
            char = results['char_level']
            logger.info("【字符级评估】")
            logger.info(f"  样本数: {char['samples_evaluated']} | 总字符: {char['total_chars']} | 总错误: {char['total_errors']}")
            logger.info(f"  TP={char['tp']}, FP={char['fp']}, FN={char['fn']}")
            logger.info(f"  Precision: {char['precision']:.4f} | Recall: {char['recall']:.4f}")
            logger.info(f"  F0.5: {char['f0.5']:.4f} | F1: {char['f1']:.4f} | F2: {char['f2']:.4f}")
        
        logger.info("=" * 60)


def main():
    """
    评估主函数
    
    使用方法：
        python evaluate.py --model_path <模型路径> --data_path <评估数据路径>
        python evaluate.py --exp_name <实验名称> --data_path <评估数据路径>
    
    数据格式支持：
        {
            'source': ['原句1', '原句2', ...],
            'target': ['正确句1', '正确句2', ...],
            'type': ['positive', 'negative', ...]  # positive=无错, negative=有错
        }
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='GED模型评估')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型checkpoint路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称（用于自动查找模型路径）')
    parser.add_argument('--data_path', type=str, default=None,
                        help='评估数据路径（JSON格式）')
    parser.add_argument('--output_path', type=str, default=None,
                        help='输出文件路径（不含扩展名）')
    parser.add_argument('--error_threshold', type=float, default=0.2,
                        help='句级错误判断阈值')
    parser.add_argument('--verbose', action='store_true',
                        help='是否记录详细日志')
    parser.add_argument('--device', type=str, default='cuda',
                        help='推理设备')
    parser.add_argument('--no_details', action='store_true',
                        help='不保存详细记录')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.exp_name:
        model_path = cfg.EXPERIMENTS_DIR / args.exp_name / "best_f2_model" / "best_f2_model.pt"
    else:
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
        data_path = cfg.SYNTHETIC_DATA_DIR / "dev.json"
    
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
        verbose=args.verbose,
        collect_details=not args.no_details
    )
    
    # 打印简要报告到日志
    evaluator.print_report(results)
    
    # 确定输出路径
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = model_path.parent / f"eval_results_{timestamp}"
    
    # 保存结果到CSV
    saved_files = evaluator.save_results_to_csv(
        results, 
        output_path,
        include_details=not args.no_details
    )
    
    # 同时保存JSON格式（便于程序读取）
    json_path = output_path.with_suffix('.json')
    # 移除details字段（太大）
    results_for_json = {k: v for k, v in results.items() if k != 'details'}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, ensure_ascii=False, indent=2)
    logger.info(f"评估结果JSON已保存到: {json_path}")
    
    logger.info("评估完成！")


if __name__ == "__main__":
    main()
