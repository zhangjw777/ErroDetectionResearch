"""
模型定义模块
实现 MacBERT + 双头架构 (GEC Head + SVO Head)
"""
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GECModelWithMTL(BertPreTrainedModel):
    """
    多任务学习的GEC模型
    
    架构：
    - Base: MacBERT Encoder
    - Head 1: GEC标签预测 (主任务)
    - Head 2: SVO成分识别 (辅助任务)
    
    输入：
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    
    输出：
        gec_logits: [batch_size, seq_len, num_gec_labels]
        svo_logits: [batch_size, seq_len, num_svo_labels]
    """
    
    def __init__(self, config, num_gec_labels: int, num_svo_labels: int):
        """
        Args:
            config: BertConfig
            num_gec_labels: GEC标签数量 (约5000)
            num_svo_labels: SVO标签数量 (7)
        """
        super().__init__(config)
        self.num_gec_labels = num_gec_labels
        self.num_svo_labels = num_svo_labels
        
        # BERT Encoder
        self.bert = BertModel(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # GEC Head (主任务)
        self.gec_classifier = nn.Linear(config.hidden_size, num_gec_labels)
        
        # SVO Head (辅助任务1)
        self.svo_classifier = nn.Linear(config.hidden_size, num_svo_labels)
        
        # Sentence-level Error Detection Head (辅助任务2)
        self.sent_error_classifier = nn.Linear(config.hidden_size, 2)  # 二分类: 0=正确, 1=有错
        
        # 初始化权重
        self.init_weights()
        
        logger.info(f"Model initialized with {num_gec_labels} GEC labels, {num_svo_labels} SVO labels, and sentence-level detection")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        gec_labels: Optional[torch.Tensor] = None,
        svo_labels: Optional[torch.Tensor] = None,
        sent_labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (可选)
            gec_labels: [batch_size, seq_len] (训练时提供)
            svo_labels: [batch_size, seq_len] (训练时提供)
            sent_labels: [batch_size] (句子级别标签: 0/1)
            label_mask: [batch_size, seq_len] (标记哪些位置需要计算loss)
        
        Returns:
            如果提供labels:
                (gec_logits, svo_logits, sent_logits)
            否则:
                (gec_logits, svo_logits, sent_logits)
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出 [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 获取[CLS]输出用于句子级分类 [batch_size, hidden_size]
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # GEC Head (token级)
        gec_logits = self.gec_classifier(sequence_output)  # [B, L, num_gec_labels]
        
        # SVO Head (token级)
        svo_logits = self.svo_classifier(sequence_output)  # [B, L, num_svo_labels]
        
        # Sentence Error Detection Head (句子级)
        sent_logits = self.sent_error_classifier(pooled_output)  # [B, 2]
        
        return gec_logits, svo_logits, sent_logits
    
    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取BERT编码器的输出（用于可视化或分析）
        
        Returns:
            sequence_output: [batch_size, seq_len, hidden_size]
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state


class GECModelConfig:
    """模型配置辅助类"""
    
    def __init__(
        self,
        bert_model_name: str = "hfl/chinese-macbert-base",
        num_gec_labels: int = 5000,
        num_svo_labels: int = 7,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        self.bert_model_name = bert_model_name
        self.num_gec_labels = num_gec_labels
        self.num_svo_labels = num_svo_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings


def create_model(
    bert_model_name: str,
    num_gec_labels: int,
    num_svo_labels: int,
    device: str = "cuda"
) -> GECModelWithMTL:
    """
    创建模型实例
    
    Args:
        bert_model_name: 预训练BERT模型名称
        num_gec_labels: GEC标签数量
        num_svo_labels: SVO标签数量
        device: 设备 (cuda/cpu)
    
    Returns:
        model: GECModelWithMTL实例
    """
    from transformers import BertConfig
    
    # 加载配置
    config = BertConfig.from_pretrained(bert_model_name)
    
    # 创建模型
    model = GECModelWithMTL.from_pretrained(
        bert_model_name,
        config=config,
        num_gec_labels=num_gec_labels,
        num_svo_labels=num_svo_labels
    )
    
    model.to(device)
    logger.info(f"Model loaded on {device}")
    
    return model


if __name__ == "__main__":
    # 测试模型创建
   pass
