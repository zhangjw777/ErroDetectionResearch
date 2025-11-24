"""
数据集处理模块
负责：
1. 加载训练/验证数据
2. Token到Label的对齐
3. 构建PyTorch Dataset
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Tuple
import logging

from config import default_config as cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GECDataset(Dataset):
    """
    GEC数据集类
    
    输入格式 (JSON):
    {
        "uid": "doc_001_sent_05",
        "text": "通过这次活动，使我们认识到了错误。",
        "tokens": ["通", "过", ...],
        "gec_labels": ["$KEEP", "$KEEP", ..., "$DELETE", ...],
        "svo_labels": ["O", "O", ..., "B-SUB", ...]
        "sent_has_error": 1或0
    }
    
    输出格式 (Tensor):
    {
        "input_ids": [101, 6224, 6814, ..., 102],
        "attention_mask": [1, 1, 1, ..., 1, 0, 0],
        "gec_labels": [0, 0, 0, ..., 1, ...],
        "svo_labels": [0, 0, 0, ..., 1, 2, ...],
        "label_mask": [0, 1, 1, ..., 1, 0]  # 只对原始token计算loss
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        gec_label_map: Dict[str, int],
        svo_label_map: Dict[str, int],
        max_length: int = 128
    ):
        """
        Args:
            data_path: 训练/验证数据路径 (JSON格式)
            tokenizer: BERT tokenizer
            gec_label_map: GEC标签到ID的映射
            svo_label_map: SVO标签到ID的映射
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.gec_label_map = gec_label_map
        self.svo_label_map = svo_label_map
        self.max_length = max_length
        
        # 加载数据
        self.samples = self._load_data()
        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")
    
    def _load_data(self) -> List[Dict]:
        """加载JSON数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        核心逻辑：处理单个样本
        
        关键问题：如何对齐原始token的标签到BERT子词？
        策略：
        1. 对每个原始token进行tokenize
        2. 第一个子词继承标签，其余子词标记为ignore
        3. [CLS]和[SEP]的标签设为ignore
        """
        sample = self.samples[idx]
        
        # 原始数据
        text = sample['text']
        tokens = sample['tokens']  # 原姍token列表
        gec_labels_str = sample['gec_labels']  # 字符串标签
        svo_labels_str = sample['svo_labels']
        sent_has_error = sample.get('sent_has_error', 1)  # 句子级别错误标签，默认1（有错）
        
        # 转换标签为ID
        gec_label_ids = [self.gec_label_map.get(label, self.gec_label_map[cfg.GEC_KEEP_LABEL]) 
                         for label in gec_labels_str]
        svo_label_ids = [self.svo_label_map[label] for label in svo_labels_str]
        
        # Tokenize并对齐标签
        input_ids = [self.tokenizer.cls_token_id]
        gec_aligned_labels = [-100]  # [CLS]的标签ignore
        svo_aligned_labels = [-100]
        label_mask = [0]  # [CLS]不计算loss
        
        for i, token in enumerate(tokens):
            # 对单个token进行tokenize
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            
            if len(token_ids) == 0:  # 处理特殊情况
                continue
            
            # 添加token的第一个子词，继承标签
            input_ids.append(token_ids[0])
            gec_aligned_labels.append(gec_label_ids[i])
            svo_aligned_labels.append(svo_label_ids[i])
            label_mask.append(1)  # 原始token计算loss
            
            # 如果有多个子词，后续子词标签设为ignore
            for sub_token_id in token_ids[1:]:
                input_ids.append(sub_token_id)
                gec_aligned_labels.append(-100)  # ignore
                svo_aligned_labels.append(-100)
                label_mask.append(0)
        
        # 添加[SEP]
        input_ids.append(self.tokenizer.sep_token_id)
        gec_aligned_labels.append(-100)
        svo_aligned_labels.append(-100)
        label_mask.append(0)
        
        # Padding或截断
        seq_length = len(input_ids)
        if seq_length > self.max_length:
            # 截断
            input_ids = input_ids[:self.max_length]
            gec_aligned_labels = gec_aligned_labels[:self.max_length]
            svo_aligned_labels = svo_aligned_labels[:self.max_length]
            label_mask = label_mask[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            # Padding
            padding_length = self.max_length - seq_length
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            gec_aligned_labels += [-100] * padding_length
            svo_aligned_labels += [-100] * padding_length
            label_mask += [0] * padding_length
            attention_mask = [1] * seq_length + [0] * padding_length
        
        # 转为Tensor
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'gec_labels': torch.tensor(gec_aligned_labels, dtype=torch.long),
            'svo_labels': torch.tensor(svo_aligned_labels, dtype=torch.long),
            'sent_label': torch.tensor(sent_has_error, dtype=torch.long),
            'label_mask': torch.tensor(label_mask, dtype=torch.float),
        }


def build_label_maps(vocab_dir: str) -> Tuple[Dict, Dict]:
    """
    构建标签映射
    
    Args:
        vocab_dir: 词表目录
    
    Returns:
        gec_label_map: GEC标签 -> ID
        svo_label_map: SVO标签 -> ID
    """
    # 加载GEC标签
    gec_label_path = f"{vocab_dir}/label_map.txt"
    with open(gec_label_path, 'r', encoding='utf-8') as f:
        gec_labels = [line.strip() for line in f]
    gec_label_map = {label: idx for idx, label in enumerate(gec_labels)}
    
    # **关键验证**：确保 $KEEP 的 ID 是 0
    keep_id = gec_label_map.get(cfg.GEC_KEEP_LABEL)
    if keep_id != 0:
        raise ValueError(
            f"Critical Error: {cfg.GEC_KEEP_LABEL} must have ID 0, but got ID {keep_id}. "
            f"This will cause FocalLoss and metrics to fail. "
            f"Please regenerate label_map.txt using preprocess.py"
        )
    
    # 加载SVO标签
    svo_label_path = f"{vocab_dir}/svo_labels.txt"
    with open(svo_label_path, 'r', encoding='utf-8') as f:
        svo_labels = [line.strip() for line in f]
    svo_label_map = {label: idx for idx, label in enumerate(svo_labels)}
    
    logger.info(f"Loaded {len(gec_label_map)} GEC labels and {len(svo_label_map)} SVO labels")
    logger.info(f"Verified: {cfg.GEC_KEEP_LABEL} has ID {keep_id}")
    
    return gec_label_map, svo_label_map


def create_dataloaders(
    train_path: str,
    dev_path: str,
    tokenizer: BertTokenizer,
    gec_label_map: Dict,
    svo_label_map: Dict,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 128
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证的DataLoader
    
    Args:
        train_path: 训练集路径
        dev_path: 验证集路径
        tokenizer: BERT tokenizer
        gec_label_map: GEC标签映射
        svo_label_map: SVO标签映射
        batch_size: batch大小
        num_workers: worker数量
        max_length: 最大序列长度
    
    Returns:
        train_loader, dev_loader
    """
    train_dataset = GECDataset(
        train_path, tokenizer, gec_label_map, svo_label_map, max_length=max_length
    )
    dev_dataset = GECDataset(
        dev_path, tokenizer, gec_label_map, svo_label_map, max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader


if __name__ == "__main__":
    # 测试代码
    from transformers import BertTokenizer
    
    # TODO: 这里需要实际的数据文件才能测试
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_model)
    gec_label_map, svo_label_map = build_label_maps(cfg.vocab_dir)

    dataset = GECDataset(
        data_path=cfg.SYNTHETIC_DATA_DIR / "train.json",
        tokenizer=tokenizer,
        gec_label_map=gec_label_map,
        svo_label_map=svo_label_map
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    
    print("Dataset module loaded. Need actual data to test.")
