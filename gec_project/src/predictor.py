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
from modeling import GECModelWithMTL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GECPredictor:
    """
    GEC推理器
    
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
        self.gec_label_map, self.svo_label_map = self._load_label_maps(vocab_dir)
        self.id2gec_label = {v: k for k, v in self.gec_label_map.items()}
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        logger.info(f"Predictor initialized on {device}")
    
    def _load_label_maps(self, vocab_dir: str) -> Tuple[Dict, Dict]:
        """加载标签映射"""
        # GEC标签
        gec_label_path = Path(vocab_dir) / "label_map.txt"
        with open(gec_label_path, 'r', encoding='utf-8') as f:
            gec_labels = [line.strip() for line in f]
        gec_label_map = {label: idx for idx, label in enumerate(gec_labels)}
        
        # SVO标签
        svo_label_path = Path(vocab_dir) / "svo_labels.txt"
        with open(svo_label_path, 'r', encoding='utf-8') as f:
            svo_labels = [line.strip() for line in f]
        svo_label_map = {label: idx for idx, label in enumerate(svo_labels)}
        
        return gec_label_map, svo_label_map
    
    def _load_model(self, model_path: str) -> GECModelWithMTL:
        """加载模型"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 获取模型配置
        num_gec_labels = len(self.gec_label_map)
        num_svo_labels = len(self.svo_label_map)
        
        from modeling import create_model
        model = create_model(
            bert_model_name=cfg.BERT_MODEL,
            num_gec_labels=num_gec_labels,
            num_svo_labels=num_svo_labels,
            device=self.device
        )
        
        # 加载权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Best Recall: {checkpoint.get('best_recall', 'N/A')}")
        
        return model
    
    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
        
        Returns:
            {
                'original': 原始文本,
                'corrected': 纠正后的文本,
                'tokens': token列表,
                'labels': 编辑标签列表,
                'edits': 编辑操作详情
            }
        """
        # Tokenize
        tokens = list(text)  # 简单按字符切分
        
        # BERT tokenization
        input_ids = [self.tokenizer.cls_token_id]
        token_to_ids = []  # 记录每个原始token对应的ID位置
        
        for token in tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) > 0:
                token_to_ids.append(len(input_ids))  # 记录第一个子词位置
                input_ids.extend(token_ids)
        
        input_ids.append(self.tokenizer.sep_token_id)
        
        # 转为tensor
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        attention_mask = torch.ones_like(input_ids_tensor)
        
        # 模型预测
        gec_logits, svo_logits = self.model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask
        )
        
        # 获取预测标签
        gec_preds = torch.argmax(gec_logits, dim=-1).squeeze(0).cpu().tolist()
        
        # 提取原始token的标签（只取第一个子词的预测）
        token_labels = []
        edits = []
        
        for i, pos in enumerate(token_to_ids):
            label_id = gec_preds[pos]
            label_str = self.id2gec_label[label_id]
            token_labels.append(label_str)
            
            if label_str != cfg.GEC_KEEP_LABEL:
                edits.append({
                    'position': i,
                    'token': tokens[i],
                    'operation': label_str
                })
        
        # 应用编辑操作
        corrected_text = self._apply_edits(tokens, token_labels)
        
        return {
            'original': text,
            'corrected': corrected_text,
            'tokens': tokens,
            'labels': token_labels,
            'edits': edits
        }
    
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
                if append_char and append_char != 'MASK':
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


def main():
    """测试推理"""
    # 示例
    model_path = cfg.EXPERIMENTS_DIR / "best_model.pt"
    vocab_dir = cfg.VOCAB_DIR
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
    
    # 创建预测器
    predictor = GECPredictor(
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
        result = predictor.predict(text)
        print(f"\n原文: {result['original']}")
        print(f"纠正: {result['corrected']}")
        if result['edits']:
            print(f"编辑: {result['edits']}")


if __name__ == "__main__":
    main()
