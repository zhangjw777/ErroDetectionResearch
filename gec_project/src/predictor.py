"""
推理模块
负责：
1. 加载训练好的模型
2. 对输入文本进行预测
3. 将编辑操作转换为最终文本
"""
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import json

from config import default_config as cfg
from modeling import GEDModelWithMTL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GEDPredictor:
    """
    GED推理器（错误检测）
    
    流程：
    1. 对输入文本进行tokenization
    2. 模型预测编辑标签
    3. 应用编辑操作得到纠正后的文本
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_dir: str,
        device: str = 'cpu'
    ):
        """
        Args:
            model_path: 模型checkpoint路径
            vocab_dir: 词表目录
            device: 推理设备
        """
        self.device = device
        
        # 加载tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
        
        # 加载标签映射
        self.ged_label_map, self.svo_label_map = self._load_label_maps(vocab_dir)
        self.id2ged_label = {v: k for k, v in self.ged_label_map.items()}
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Predictor initialized on {device}")
    
    def _load_label_maps(self, vocab_dir: str) -> Tuple[Dict, Dict]:
        """加载标签映射"""
        # GED标签
        ged_label_path = Path(vocab_dir) / "label_map.txt"
        with open(ged_label_path, 'r', encoding='utf-8') as f:
            ged_labels = [line.strip() for line in f]
        ged_label_map = {label: idx for idx, label in enumerate(ged_labels)}
        
        # SVO标签
        svo_label_path = Path(vocab_dir) / "svo_labels.txt"
        with open(svo_label_path, 'r', encoding='utf-8') as f:
            svo_labels = [line.strip() for line in f]
        svo_label_map = {label: idx for idx, label in enumerate(svo_labels)}
        
        # 创建反向映射
        self.id2svo_label = {v: k for k, v in svo_label_map.items()}
        
        return ged_label_map, svo_label_map
    
    def _load_model(self, model_path: str) -> GEDModelWithMTL:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        num_ged_labels = len(self.ged_label_map)
        num_svo_labels = len(self.svo_label_map)
        
        from modeling import create_model
        model = create_model(
            bert_model_name=cfg.BERT_MODEL,
            num_ged_labels=num_ged_labels,
            num_svo_labels=num_svo_labels,
            device=self.device
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Checkpoint Info - Epoch: {checkpoint.get('epoch', 'N/A')}, "
                   f"Best Recall: {checkpoint.get('best_recall', 'N/A')}, "
                   f"Best F2: {checkpoint.get('best_f2', 'N/A')}")
        
        return model
    
    @torch.no_grad()
    def predict(self, text: str, return_svo: bool = False) -> Dict:
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
            return_svo: 是否返回SVO句法分析结果（BIO标签）
        
        Returns:
            {
                'original': 原始文本,
                'corrected': 纠正后的文本,
                'tokens': token列表,
                'labels': 编辑标签列表,
                'edits': 编辑操作详情,
                'sent_has_error': 句子是否有错 (bool),
                'sent_error_prob': 句子有错的概率 (float),
                'svo_labels': SVO标签列表 (仅当return_svo=True时),
                'svo_analysis': SVO分析结果 (仅当return_svo=True时)
            }
        """
        # Tokenize
        tokens = list(text)  # 简单按字符切分
        
        # BERT tokenization
        input_ids = [self.tokenizer.cls_token_id]
        token_to_ids = []  # 记录每个原始token对应的ID位置
        label_mask_list = [0]  # [CLS] 位置为 0
        
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                token_to_ids.append(len(input_ids))  # 记录第一个子词位置
                # 第一个子词标记为 1，后续子词标记为 0
                label_mask_list.append(1)  # 真实字符的第一个子词
                label_mask_list.extend([0] * (len(token_ids) - 1))  # 子词续接标记为 0
                input_ids.extend(token_ids)
        
        input_ids.append(self.tokenizer.sep_token_id)
        label_mask_list.append(0)  # [SEP] 位置为 0
        
        # 转为tensor
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids_tensor)
        # 构造 label_mask：标记哪些位置是真实字符的第一个子词
        # 这样 ErrorAwareSentenceHead 可以正确计算 valid_mask = attention_mask * label_mask
        label_mask = torch.tensor([label_mask_list], dtype=torch.long).to(self.device)
        
        # 模型预测 - 传入 label_mask 以确保句级头正确计算 valid_mask
        gec_logits, svo_logits, sent_logits = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            label_mask=label_mask  # 修复：传入 label_mask 用于 ErrorAwareSentenceHead
        )
        
        # 句级错误检测预测
        sent_probs = F.softmax(sent_logits, dim=-1).squeeze(0)  # [2]
        sent_error_prob = sent_probs[1].item()  # 预测"有错"的概率
        sent_has_error = sent_error_prob > 0.2  # 二分类阈值
        
        # 获取预测标签
        gec_preds = torch.argmax(gec_logits, dim=-1).squeeze(0).cpu().tolist()
        svo_preds = torch.argmax(svo_logits, dim=-1).squeeze(0).cpu().tolist()
        
        # 提取原始token的标签（只取第一个子词的预测）
        token_labels = []
        svo_labels = []
        edits = []
        
        for i, pos in enumerate(token_to_ids):
            # GEC标签
            label_id = gec_preds[pos]
            label_str = self.id2ged_label[label_id]
            token_labels.append(label_str)
            
            # SVO标签 (BIO格式)
            svo_id = svo_preds[pos]
            svo_str = self.id2svo_label[svo_id]
            svo_labels.append(svo_str)
            
            if label_str != cfg.GEC_KEEP_LABEL:
                edits.append({
                    'position': i,
                    'token': tokens[i],
                    'operation': label_str
                })
        
        # 应用编辑操作
        corrected_text = self._apply_edits(tokens, token_labels)
        
        result = {
            'original': text,
            'corrected': corrected_text,
            'tokens': tokens,
            'labels': token_labels,
            'edits': edits,
            'sent_has_error': sent_has_error,  # 句子是否有错（bool）
            'sent_error_prob': sent_error_prob  # 句子有错的概率（float）
        }
        
        # 可选：返回SVO分析结果
        if return_svo:
            result['svo_labels'] = svo_labels
            result['svo_analysis'] = self._extract_svo_spans(tokens, svo_labels)
        
        return result
    
    def _extract_svo_spans(self, tokens: List[str], svo_labels: List[str]) -> Dict:
        """
        从BIO标签中提取主谓宾成分的文本片段
        
        Args:
            tokens: 字符列表
            svo_labels: BIO格式的SVO标签列表
        
        Returns:
            {
                'subjects': ['主语1', '主语2', ...],
                'predicates': ['谓语1', '谓语2', ...],
                'objects': ['宾语1', '宾语2', ...]
            }
        """
        spans = {'subjects': [], 'predicates': [], 'objects': []}
        current_type = None
        current_span = []
        
        for token, label in zip(tokens, svo_labels):
            if label.startswith('B-'):
                # 保存前一个span
                if current_span and current_type:
                    span_text = ''.join(current_span)
                    if current_type == 'SUB':
                        spans['subjects'].append(span_text)
                    elif current_type == 'PRED':
                        spans['predicates'].append(span_text)
                    elif current_type == 'OBJ':
                        spans['objects'].append(span_text)
                
                # 开始新span
                current_type = label.split('-')[1]
                current_span = [token]
            elif label.startswith('I-') and current_type == label.split('-')[1]:
                # 继续当前span
                current_span.append(token)
            else:
                # O标签，结束当前span
                if current_span and current_type:
                    span_text = ''.join(current_span)
                    if current_type == 'SUB':
                        spans['subjects'].append(span_text)
                    elif current_type == 'PRED':
                        spans['predicates'].append(span_text)
                    elif current_type == 'OBJ':
                        spans['objects'].append(span_text)
                current_type = None
                current_span = []
        
        # 处理最后一个span
        if current_span and current_type:
            span_text = ''.join(current_span)
            if current_type == 'SUB':
                spans['subjects'].append(span_text)
            elif current_type == 'PRED':
                spans['predicates'].append(span_text)
            elif current_type == 'OBJ':
                spans['objects'].append(span_text)
        
        return spans
    
    def _apply_edits(self, tokens: List[str], labels: List[str]) -> str:
        """
        应用编辑操作到token序列
        
        GECToR编辑操作：
        - $KEEP: 保持不变
        - $DELETE: 删除该token
        - $APPEND_X: 在该token后添加字符X
        - $REPLACE_X: 将该token替换为字符X
        """
        result_tokens = []
        
        for token, label in zip(tokens, labels):
            if label == cfg.GEC_KEEP_LABEL:
                # 保持
                result_tokens.append(token)
            elif label == cfg.GEC_DELETE_LABEL:
                # 删除（不添加到结果）
                continue
            elif label.startswith(cfg.GEC_APPEND_PREFIX):
                # 添加
                result_tokens.append(token)
                append_char = label.replace(cfg.GEC_APPEND_PREFIX, '')
                if append_char:
                    result_tokens.append(append_char)
            elif label.startswith(cfg.GEC_REPLACE_PREFIX):
                # 替换
                replace_char = label.replace(cfg.GEC_REPLACE_PREFIX, '')
                if replace_char and replace_char != 'MASK':
                    result_tokens.append(replace_char)
                else:
                    result_tokens.append(token)  # 无法替换，保持原token
            else:
                # 未知操作，保持不变
                result_tokens.append(token)
        
        return ''.join(result_tokens)
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """批量预测"""
        return [self.predict(text) for text in texts]


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
    """测试推理"""
    # 示例
    model_path = cfg.EXPERIMENTS_DIR / "best_model.pt"
    vocab_dir = cfg.VOCAB_DIR
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # 创建预测器
    predictor = GEDPredictor(
        model_path=str(model_path),
        vocab_dir=str(vocab_dir),
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 测试样例
    test_texts = [
        "通过这次活动，使我们认识到了错误。",
        "在党的领导下，取得了伟大成就。",
        "我们要认真学习贯彻会议精神。"
    ]
    
    for text in test_texts:
        result = predictor.predict(text, return_svo=True)
        print(f"\n原文: {result['original']}")
        print(f"纠正: {result['corrected']}")
        print(f"句级判断: {'[有错]' if result['sent_has_error'] else '[无错]'} (概率: {result['sent_error_prob']:.3f})")
        if result['edits']:
            print(f"编辑: {result['edits']}")
        else:
            print("编辑: 无需修改")
        
        # 显示SVO分析结果
        if 'svo_analysis' in result:
            svo = result['svo_analysis']
            print(f"句法分析:")
            print(f"  - 主语: {svo['subjects'] if svo['subjects'] else '未识别到'}")
            print(f"  - 谓语: {svo['predicates'] if svo['predicates'] else '未识别到'}")
            print(f"  - 宾语: {svo['objects'] if svo['objects'] else '未识别到'}")


if __name__ == "__main__":
    main()
