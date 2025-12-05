"""
推理模块
负责：
1. 加载训练好的模型
2. 对输入文本进行预测
3. 将编辑操作转换为最终文本
4. 支持批量预测并输出到CSV文件
"""
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path
import json
import csv
from datetime import datetime

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
        
        # 兼容旧版checkpoint：将gec_classifier映射到ged_classifier
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # 将旧的gec_classifier映射到新的ged_classifier
            if k.startswith('gec_classifier'):
                new_key = k.replace('gec_classifier', 'ged_classifier')
                logger.info(f"Mapping checkpoint key: {k} -> {new_key}")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # 加载权重
        model.load_state_dict(new_state_dict)
        
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
            token_ids = self.tokenizer.encode(token, add_special_tokens=False) # 避免每个字都被加特殊符号
            if len(token_ids) > 0:#如果这个字/单词被编码为多个子词
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
        ged_logits, svo_logits, sent_logits = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask,
            label_mask=label_mask  # 传入 label_mask 用于 ErrorAwareSentenceHead
        )
        
        # 句级错误检测预测
        sent_probs = F.softmax(sent_logits, dim=-1).squeeze(0)  # [2]
        sent_error_prob = sent_probs[1].item()  # 预测"有错"的概率
        
        # 获取预测标签
        ged_preds = torch.argmax(ged_logits, dim=-1).squeeze(0).cpu().tolist()
        svo_preds = torch.argmax(svo_logits, dim=-1).squeeze(0).cpu().tolist()
        
        # 提取原始token的标签（只取第一个子词的预测）
        token_labels = []
        svo_labels = []
        edits = []
        
        for i, pos in enumerate(token_to_ids):
            # GED标签
            label_id = ged_preds[pos]
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
        
        sent_has_error = sent_error_prob > 0.2 or len(edits)>0 # 二分类阈值

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
        - $REPLACE_MASK / $APPEND_MASK: 模型检测到错误但无法确定具体内容，用[?]标记
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
                if append_char and append_char != 'MASK':
                    result_tokens.append(append_char)
                else:
                    # MASK情况：模型知道需要添加，但不确定添加什么
                    result_tokens.append('[?]')
            elif label.startswith(cfg.GEC_REPLACE_PREFIX):
                # 替换
                replace_char = label.replace(cfg.GEC_REPLACE_PREFIX, '')
                if replace_char and replace_char != 'MASK':
                    result_tokens.append(replace_char)
                else:
                    # MASK情况：模型知道需要替换，但不确定替换成什么
                    result_tokens.append('[?]')
            else:
                # 未知操作，保持不变
                result_tokens.append(token)
        
        return ''.join(result_tokens)
    
    def predict_batch(self, texts: List[str], return_svo: bool = False) -> List[Dict]:
        """
        批量预测
        
        Args:
            texts: 输入文本列表
            return_svo: 是否返回SVO句法分析结果
        
        Returns:
            预测结果列表
        """
        return [self.predict(text, return_svo=return_svo) for text in texts]
    
    def predict_from_data(
        self, 
        data: Dict[str, List], 
        return_svo: bool = False
    ) -> List[Dict]:
        """
        从数据字典格式进行批量预测
        
        Args:
            data: 数据字典，格式为 {'source': [...], 'target': [...], 'type': [...]}
                  - source: 原句列表
                  - target: 正确句列表（可选，用于评估）
                  - type: 样本类型列表，'positive'表示无错，'negative'表示有错
            return_svo: 是否返回SVO句法分析结果
        
        Returns:
            预测结果列表，每个结果包含原始数据信息和预测结果
        """
        sources = data.get('source', [])
        targets = data.get('target', sources)  # 如果没有target，用source
        types = data.get('type', ['unknown'] * len(sources))
        
        results = []
        for i, (source, target, sample_type) in enumerate(zip(sources, targets, types)):
            pred_result = self.predict(source, return_svo=return_svo)
            
            # 添加原始数据信息
            pred_result['target'] = target
            pred_result['sample_type'] = sample_type
            pred_result['gt_has_error'] = (sample_type == 'negative')  # negative表示有错
            pred_result['index'] = i
            
            results.append(pred_result)
        
        return results
    
    def save_predictions_to_csv(
        self, 
        predictions: List[Dict], 
        output_path: str,
        include_svo: bool = False
    ) -> str:
        """
        将预测结果保存到CSV文件
        
        Args:
            predictions: 预测结果列表
            output_path: 输出文件路径
            include_svo: 是否包含SVO信息
        
        Returns:
            实际保存的文件路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 定义CSV字段
        fieldnames = [
            'index', 'original', 'target', 'corrected', 
            'sample_type', 'gt_has_error', 'pred_has_error', 'sent_error_prob',
            'num_edits', 'edits_summary'
        ]
        
        if include_svo:
            fieldnames.extend(['subjects', 'predicates', 'objects'])
        
        with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for pred in predictions:
                row = {
                    'index': pred.get('index', ''),
                    'original': pred.get('original', ''),
                    'target': pred.get('target', ''),
                    'corrected': pred.get('corrected', ''),
                    'sample_type': pred.get('sample_type', ''),
                    'gt_has_error': pred.get('gt_has_error', ''),
                    'pred_has_error': pred.get('sent_has_error', ''),
                    'sent_error_prob': f"{pred.get('sent_error_prob', 0):.4f}",
                    'num_edits': len(pred.get('edits', [])),
                    'edits_summary': '; '.join([
                        f"{e['position']}:{e['token']}->{e['operation']}" 
                        for e in pred.get('edits', [])
                    ])
                }
                
                if include_svo and 'svo_analysis' in pred:
                    svo = pred['svo_analysis']
                    row['subjects'] = '|'.join(svo.get('subjects', []))
                    row['predicates'] = '|'.join(svo.get('predicates', []))
                    row['objects'] = '|'.join(svo.get('objects', []))
                
                writer.writerow(row)
        
        logger.info(f"预测结果已保存到: {output_path}")
        return str(output_path)

def main():
    """
    预测主函数
    
    使用方法：
        python predictor.py --model_path <模型路径>
        python predictor.py --exp_name <实验名称>
    
    参数说明：
        --model_path: 模型checkpoint路径
        --exp_name: 实验名称（用于自动查找模型路径）
        --return_svo: 是否输出SVO句法分析结果
    
    注意：数据输入使用datasets库在代码中获取，格式为字典 {'source': [...], 'target': [...], 'type': [...]}
    用户需要在代码中自行添加数据加载逻辑，调用 predict_from_data(data) 方法进行预测。
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='GED模型预测')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型checkpoint路径')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称（用于自动查找模型路径）')
    parser.add_argument('--return_svo', action='store_true',
                        help='是否输出SVO句法分析结果')
    
    args = parser.parse_args()
    
    # 确定模型路径
    if args.model_path:
        model_path = Path(args.model_path)
    elif args.exp_name:
        model_path = cfg.EXPERIMENTS_DIR / args.exp_name / "best_f2_model" / "best_f2_model.pt"
    else:
        logger.error("请指定 --model_path 或 --exp_name 参数")
        return
    
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        return
    
    # 设置设备（默认使用cuda）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")
    
    # 创建预测器
    logger.info(f"加载模型: {model_path}")
    predictor = GEDPredictor(
        model_path=str(model_path),
        vocab_dir=str(cfg.VOCAB_DIR),
        device=device
    )
    
    # ==================== 数据加载区域 ====================
    # 用户在此处添加数据加载代码，使用datasets库加载数据
    # 数据格式应为字典: {'source': [...], 'target': [...], 'type': [...]}
    # 示例:
    # from datasets import load_dataset
    # dataset = load_dataset("your_dataset")
    # data = {
    #     'source': dataset['test']['source'],
    #     'target': dataset['test']['target'],
    #     'type': dataset['test']['type']
    # }
    # ======================================================
    
    data = None  # 用户需要替换为实际数据
    
    if data is None:
        logger.warning("未加载数据，请在代码中添加数据加载逻辑")
        logger.info("数据格式示例: {'source': [...], 'target': [...], 'type': [...]}")
        return
    
    # 执行预测
    predictions = predictor.predict_from_data(data, return_svo=args.return_svo)
    
    # 确定输出路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = model_path.parent / f"predictions_{timestamp}.csv"
    
    # 保存结果
    predictor.save_predictions_to_csv(predictions, output_path, include_svo=args.return_svo)
    logger.info(f"批量预测完成，共处理 {len(predictions)} 条样本")


if __name__ == "__main__":
    main()
