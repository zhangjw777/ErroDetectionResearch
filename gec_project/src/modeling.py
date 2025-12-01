"""
模型定义模块
实现 MacBERT + 多任务架构

架构包含：
1. MacBERT Encoder - 共享表示 H_shared
2. SVO Head - 句法成分识别 (辅助任务1)
3. 句法-语义融合交互层 - 将 H_SVO 融合到 GED 表示中
4. GED Head - 编辑标签预测 (主任务，错误检测)
5. 错误感知多实例句级分类头 - 句级错误检测 (辅助任务2)
"""
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from typing import Optional, Tuple
import logging
from config import default_config as cfg
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 模块一：句法-语义融合交互层 ====================

class SyntaxSemanticInteractionLayer(nn.Module):
    """
    句法-语义融合交互层 (Syntactic-Semantic Interaction Layer)
    
    将SVO辅助任务的句法表示与BERT的语义表示进行门控融合，
    实现显式句法先验注入到GED表示中。
    
    公式：
    - G = σ(W_g · [H_shared ; H_SVO] + b_g)  # 门控向量
    - T = W_t · H_SVO + b_t                  # 语法变换
    - H_GED_input = H_shared + G ⊙ T         # 门控融合
    
    Args:
        hidden_size: 隐藏层维度 (与BERT一致，默认768)
        use_layer_norm: 是否在融合后使用LayerNorm
    """
    
    def __init__(self, hidden_size: int, use_layer_norm: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 门控投影: [H_shared; H_SVO] (2d) -> G (d)
        self.gate_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        # 语法变换: H_SVO (d) -> T (d)
        self.syntax_transform = nn.Linear(hidden_size, hidden_size)
        
        # 可选：融合后的LayerNorm
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, h_shared: torch.Tensor, h_svo: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h_shared: [B, L, d] - BERT编码器输出 (共享表示)
            h_svo: [B, L, d] - SVO中间表示
        
        Returns:
            h_ged_input: [B, L, d] - 融合后的表示，供GED Head使用
        """
        # 1. 拼接 H_shared 和 H_SVO: [B, L, 2d]
        concat = torch.cat([h_shared, h_svo], dim=-1)
        
        # 2. 计算门控向量: G = σ(W_g · concat + b_g), [B, L, d]
        gate = torch.sigmoid(self.gate_proj(concat))
        
        # 3. 语法变换: T = W_t · H_SVO + b_t, [B, L, d]
        syntax_transformed = self.syntax_transform(h_svo)
        
        # 4. 门控融合: H_GED_input = H_shared + G ⊙ T
        h_ged_input = h_shared + gate * syntax_transformed
        
        # 5. 可选的LayerNorm
        if self.use_layer_norm:
            h_ged_input = self.layer_norm(h_ged_input)
        
        return h_ged_input


# ==================== 模块三：错误感知多实例句级分类头 ====================

class ErrorAwareSentenceHead(nn.Module):
    """
    错误感知多实例句级分类头 (Error-Aware Multi-Instance Head)
    
    将句子视为token实例集合，使用GED预测的错误置信度驱动的
    注意力池化构造句级表示，显式对齐token-level与sentence-level。
    
    核心思想：
    - 从GED Head获取每个token是KEEP的概率 P(KEEP)
    - 错误置信度: e = 1 - P(KEEP)
    - 用错误置信度作为注意力权重做加权池化
    - 池化后的向量送入MLP做句级分类
    
    公式：
    - e_i = 1 - P(label_i = KEEP)     # 错误置信度
    - α_i = softmax(e_i)               # 错误驱动注意力
    - V_sent = Σ α_i · H_i             # 错误感知池化
    - y_sent = MLP(V_sent)             # 句级预测
    
    Args:
        hidden_size: 隐藏层维度
        num_classes: 句级分类数 (默认2，有错/无错)
        dropout_prob: Dropout概率
        detach_confidence: 是否detach错误置信度梯度（用于消融实验）
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_classes: int = 2,
        dropout_prob: float = 0.1,
        detach_confidence: bool = False
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.detach_confidence = detach_confidence
        
        # MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(
        self,
        h_tokens: torch.Tensor,        # [B, L, d]
        ged_logits: torch.Tensor,      # [B, L, C]
        valid_mask: torch.Tensor,      # [B, L] - 有效位置mask（真实字符位置，排除CLS/SEP/子词/padding）
        keep_label_idx: int = 0        # KEEP标签的索引
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            h_tokens: [B, L, d] - token级表示 (融合后的H_GED_input)
            ged_logits: [B, L, C] - GED Head的输出logits
            valid_mask: [B, L] - 有效位置mask (1=真实字符, 0=CLS/SEP/子词/padding)
                        这应该是 attention_mask * label_mask 的结果
            keep_label_idx: KEEP标签在标签表中的索引
        
        Returns:
            sent_logits: [B, num_classes] - 句级分类logits
            attention_weights: [B, L] - 错误驱动的注意力权重 (用于可视化)
        """
        batch_size, seq_len, _ = h_tokens.shape
        
        # 1. 计算错误置信度: e = 1 - P(KEEP)
        ged_probs = torch.softmax(ged_logits, dim=-1)  # [B, L, C]
        p_keep = ged_probs[:, :, keep_label_idx]       # [B, L]
        error_confidence = 1 - p_keep                  # [B, L]
        
        # 1.5 可选：detach错误置信度（用于消融实验）
        # 如果detach，则句级loss不会反向传播到GED表示
        if self.detach_confidence:
            error_confidence = error_confidence.detach()
        
        # 2. Mask无效位置 (CLS/SEP/子词/padding，设为极小值，softmax后趋近0)
        # valid_mask: 1=真实字符位置, 0=无效位置
        error_confidence = error_confidence.masked_fill(
            valid_mask == 0, 
            float('-inf')
        )
        
        # 3. 计算错误驱动的注意力权重: α = softmax(e)
        attention_weights = torch.softmax(error_confidence, dim=-1)  # [B, L]
        
        # 4. 错误感知池化: V_sent = Σ α_i · H_i
        # attention_weights: [B, L] -> [B, L, 1]
        alpha_expanded = attention_weights.unsqueeze(-1)
        v_sent = torch.sum(alpha_expanded * h_tokens, dim=1)  # [B, d]
        
        # 5. 句级分类
        sent_logits = self.classifier(v_sent)  # [B, num_classes]
        
        return sent_logits, attention_weights


class GEDModelWithMTL(BertPreTrainedModel):
    """
    多任务学习的GED模型（错误检测）
    
    架构（按数据流顺序）：
    1. MacBERT Encoder -> H_shared [B, L, d]
    2. SVO中间表示生成 -> H_SVO [B, L, d]
    3. SVO Head (辅助任务1): H_SVO -> svo_logits  # 关键：SVO头使用H_SVO，确保句法表示被直接监督
    4. 句法-语义融合层: (H_shared, H_SVO) -> H_GED_input
    5. GED Head (主任务): H_GED_input -> ged_logits
    6. 错误感知句级头 (辅助任务2): (H_GED_input, ged_logits, valid_mask) -> sent_logits
    
    输入：
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
    
    输出：
        ged_logits: [batch_size, seq_len, num_ged_labels]
        svo_logits: [batch_size, seq_len, num_svo_labels]
        sent_logits: [batch_size, 2]
        (可选) attention_weights: [batch_size, seq_len] - 错误驱动注意力
    """
    
    def __init__(
        self, 
        config, 
        num_ged_labels: int, 
        num_svo_labels: int,
        use_syntax_semantic_fusion: bool = True,
        use_error_aware_sent_head: bool = True,
        keep_label_idx: int = 0,
        detach_error_confidence: bool = False,
        syntax_fusion_use_layer_norm: bool = True
    ):
        """
        Args:
            config: BertConfig
            num_ged_labels: GED标签数量（编辑标签，用于错误检测）
            num_svo_labels: SVO标签数量 (7)
            use_syntax_semantic_fusion: 是否使用句法-语义融合层
            use_error_aware_sent_head: 是否使用错误感知句级头
            keep_label_idx: KEEP标签在标签表中的索引
            detach_error_confidence: 是否detach错误置信度梯度（用于消融实验）
            syntax_fusion_use_layer_norm: 句法-语义融合层是否使用LayerNorm
        """
        super().__init__(config)
        self.num_ged_labels = num_ged_labels
        self.num_svo_labels = num_svo_labels
        self.use_syntax_semantic_fusion = use_syntax_semantic_fusion
        self.use_error_aware_sent_head = use_error_aware_sent_head
        self.keep_label_idx = keep_label_idx
        self.detach_error_confidence = detach_error_confidence
        
        # ==================== BERT Encoder ====================
        # 注意：当使用 ErrorAwareSentenceHead 时，不需要 BERT 的 pooler 层
        # 如果不禁用，pooler 的参数不会参与梯度计算，导致 DDP 报错
        # RuntimeError: Expected to have finished reduction in the prior iteration...
        # Parameter indices which did not receive grad: 197 198 (bert.pooler.dense.weight/bias)
        self.bert = BertModel(config, add_pooling_layer=not use_error_aware_sent_head)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # ==================== SVO 中间表示生成层 ====================
        # 从 H_shared 生成 H_SVO，用于：
        # 1. SVO 分类（直接被 SVO 标签监督）
        # 2. 句法-语义融合（注入到 GEC 表示中）
        self.svo_hidden_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU()
        )
        
        # ==================== 模块一：句法-语义融合交互层 ====================
        if use_syntax_semantic_fusion:
            self.syntax_semantic_interaction = SyntaxSemanticInteractionLayer(
                hidden_size=config.hidden_size,
                use_layer_norm=syntax_fusion_use_layer_norm  # 从参数读取
            )
        else:
            self.syntax_semantic_interaction = None
        
        # ==================== GED Head (主任务) ====================
        # 使用融合后的 H_GED_input 进行预测
        self.ged_classifier = nn.Linear(config.hidden_size, num_ged_labels)
        
        # ==================== SVO Head (辅助任务1) ====================
        # **关键修改**：使用 H_SVO 进行预测，确保句法表示被直接监督
        self.svo_classifier = nn.Linear(config.hidden_size, num_svo_labels)
        
        # ==================== 模块三：错误感知句级分类头 (辅助任务2) ====================
        if use_error_aware_sent_head:
            self.sent_error_head = ErrorAwareSentenceHead(
                hidden_size=config.hidden_size,
                num_classes=2,
                dropout_prob=config.hidden_dropout_prob,
                detach_confidence=detach_error_confidence
            )
        else:
            # 降级为简单的 [CLS] + Linear
            self.sent_error_classifier = nn.Linear(config.hidden_size, 2)
            self.sent_error_head = None
        
        # 初始化权重
        self.init_weights()
        
        logger.info(
            f"Model initialized: "
            f"{num_ged_labels} GED labels, {num_svo_labels} SVO labels, "
            f"syntax_fusion={use_syntax_semantic_fusion}, "
            f"error_aware_head={use_error_aware_sent_head}, "
            f"detach_error_confidence={detach_error_confidence}"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        ged_labels: Optional[torch.Tensor] = None,
        svo_labels: Optional[torch.Tensor] = None,
        sent_labels: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len] (可选)
            ged_labels: [batch_size, seq_len] (训练时提供)
            svo_labels: [batch_size, seq_len] (训练时提供)
            sent_labels: [batch_size] (句子级别标签: 0/1)
            label_mask: [batch_size, seq_len] (标记哪些位置需要计算loss，1=真实字符)
            return_attention_weights: 是否返回错误驱动注意力权重
        
        Returns:
            ged_logits: [B, L, num_ged_labels]
            svo_logits: [B, L, num_svo_labels]
            sent_logits: [B, 2]
            (可选) attention_weights: [B, L] - 当 return_attention_weights=True
        """
        # ==================== Step 1: BERT 编码 ====================
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取序列输出: H_shared [B, L, d]
        h_shared = outputs.last_hidden_state
        h_shared = self.dropout(h_shared)
        
        # ==================== Step 2: 生成 SVO 中间表示 ====================
        # H_SVO = MLP(H_shared), [B, L, d]
        # 这个表示将被 SVO 任务直接监督，同时用于句法-语义融合
        h_svo = self.svo_hidden_proj(h_shared)
        
        # ==================== Step 3: SVO Head ====================
        # **关键修改**：使用 H_SVO（而非 H_shared）进行 SVO 分类
        # 这确保了 H_SVO 被 SVO 标签直接监督，真正学到句法信息
        svo_logits = self.svo_classifier(h_svo)  # [B, L, num_svo_labels]
        
        # ==================== Step 4: 句法-语义融合 ====================
        if self.use_syntax_semantic_fusion and self.syntax_semantic_interaction is not None:
            # 使用门控融合: H_GED_input = H_shared + G ⊙ T
            # H_SVO 现在是被 SVO 任务直接监督的句法表示
            h_ged_input = self.syntax_semantic_interaction(h_shared, h_svo)
        else:
            # 不使用融合，直接使用 H_shared
            h_ged_input = h_shared
        
        # ==================== Step 5: GED Head (使用融合后的表示) ====================
        ged_logits = self.ged_classifier(h_ged_input)  # [B, L, num_ged_labels]
        
        # ==================== Step 6: 错误感知句级分类头 ====================
        attention_weights = None
        
        if self.use_error_aware_sent_head and self.sent_error_head is not None:
            # **关键修改**：构造 valid_mask = attention_mask * label_mask
            # 确保只有真实字符位置参与“错误注意力”计算
            # 排除 [CLS], [SEP], 子词续接, padding
            if label_mask is not None:
                valid_mask = attention_mask * label_mask
            else:
                valid_mask = attention_mask
            
            # 使用错误感知多实例头
            sent_logits, attention_weights = self.sent_error_head(
                h_tokens=h_ged_input,
                ged_logits=ged_logits,
                valid_mask=valid_mask,  # 使用 valid_mask 而非 attention_mask
                keep_label_idx=self.keep_label_idx
            )
        else:
            # 降级使用 [CLS] + Linear
            # 注意：此时 add_pooling_layer=True，所以 pooler_output 可用
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                pooled_output = outputs.pooler_output
            else:
                # 备用方案：手动从 [CLS] token 获取表示
                pooled_output = h_shared[:, 0, :]  # [B, d]
            pooled_output = self.dropout(pooled_output)
            sent_logits = self.sent_error_classifier(pooled_output)  # [B, 2]
        
        # ==================== 返回结果 ====================
        if return_attention_weights and attention_weights is not None:
            return ged_logits, svo_logits, sent_logits, attention_weights
        else:
            return ged_logits, svo_logits, sent_logits
    
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


class GEDModelConfig:
    """模型配置辅助类"""
    
    def __init__(
        self,
        bert_model_name: str = "hfl/chinese-macbert-base",
        num_ged_labels: int = 5000,
        num_svo_labels: int = 7,
        hidden_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
    ):
        self.bert_model_name = bert_model_name
        self.num_ged_labels = num_ged_labels
        self.num_svo_labels = num_svo_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.max_position_embeddings = max_position_embeddings


def create_model(
    bert_model_name: str,
    num_ged_labels: int,
    num_svo_labels: int,
    device: str = "cuda",
    use_syntax_semantic_fusion: bool = True,
    use_error_aware_sent_head: bool = True,
    keep_label_idx: int = 0,
    detach_error_confidence: bool = False,
    syntax_fusion_use_layer_norm: bool = True
) -> GEDModelWithMTL:
    """
    创建模型实例
    
    Args:
        bert_model_name: 预训练BERT模型名称
        num_ged_labels: GED标签数量
        num_svo_labels: SVO标签数量
        device: 设备 (cuda/cpu)
        use_syntax_semantic_fusion: 是否使用句法-语义融合层
        use_error_aware_sent_head: 是否使用错误感知句级头
        keep_label_idx: KEEP标签索引
        detach_error_confidence: 是否detach错误置信度梯度
        syntax_fusion_use_layer_norm: 句法-语义融合层是否使用LayerNorm
    
    Returns:
        model: GEDModelWithMTL实例
    """
    from transformers import BertConfig
    
    # 加载配置
    config = BertConfig.from_pretrained(bert_model_name)
    
    # 创建模型
    model = GEDModelWithMTL.from_pretrained(
        bert_model_name,
        config=config,
        num_ged_labels=num_ged_labels,
        num_svo_labels=num_svo_labels,
        use_syntax_semantic_fusion=use_syntax_semantic_fusion,
        use_error_aware_sent_head=use_error_aware_sent_head,
        keep_label_idx=keep_label_idx,
        detach_error_confidence=detach_error_confidence,
        syntax_fusion_use_layer_norm=syntax_fusion_use_layer_norm
    )
    
    model.to(device)
    logger.info(f"Model loaded on {device}")
    
    return model


if __name__ == "__main__":
    # 测试模型创建
   pass
