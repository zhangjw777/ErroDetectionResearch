"""
损失函数模块
实现Focal Loss和多任务联合损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss实现
    
    公式: FL(p_t) = -α(1-p_t)^γ * log(p_t)
    
    用于解决GEC任务中极度不平衡的问题：
    - 90%以上的token是$KEEP
    - 错误token非常稀疏
    
    Args:
        alpha: 类别权重 (针对$KEEP类)
        gamma: 聚焦参数，越大越关注难分类样本
        reduction: 'mean' | 'sum' | 'none'
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, seq_len, num_labels] 模型输出
            targets: [batch_size, seq_len] 真实标签
            label_mask: [batch_size, seq_len] 标记有效位置
        
        Returns:
            loss: scalar
        """
        # Flatten
        logits = logits.view(-1, logits.size(-1))  # [B*L, num_labels]
        targets = targets.view(-1)  # [B*L]
        
        # 计算log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取目标类别的log概率
        log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # 计算p_t
        probs = torch.exp(log_probs)
        
        # Focal Loss权重: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Alpha平衡 (简化版：对所有非$KEEP类加权)
        # TODO: 可以为每个类别设置不同的alpha
        alpha_weight = torch.where(
            targets == 0,  # 假设$KEEP的ID是0
            torch.full_like(targets, self.alpha, dtype=torch.float),
            torch.full_like(targets, 1 - self.alpha, dtype=torch.float)
        )
        
        # 计算loss
        loss = -alpha_weight * focal_weight * log_probs
        
        # 处理ignore_index
        mask = (targets != self.ignore_index).float()
        loss = loss * mask
        
        # 如果提供了label_mask，进一步过滤
        if label_mask is not None:
            label_mask = label_mask.view(-1)
            loss = loss * label_mask
        
        # Reduction
        if self.reduction == 'mean':
            return loss.sum() / mask.sum()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    多任务联合损失
    
    L_total = L_GEC + λ1 * L_SVO + λ2 * L_SENT
    
    其中：
    - L_GEC使用Focal Loss (主任务)
    - L_SVO使用标准CrossEntropy (辅助任务1)
    - L_SENT使用标准CrossEntropy (辅助任务2: 句级错误检测)
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        mtl_lambda_svo: float = 0.5,
        mtl_lambda_sent: float = 0.3,
        ignore_index: int = -100
    ):
        """
        Args:
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            mtl_lambda_svo: SVO任务的权重
            mtl_lambda_sent: 句级错误检测任务的权重
            ignore_index: 忽略的标签ID
        """
        super(MultiTaskLoss, self).__init__()
        
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            ignore_index=ignore_index
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        self.sent_ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
        self.mtl_lambda_svo = mtl_lambda_svo
        self.mtl_lambda_sent = mtl_lambda_sent
        
        logger.info(f"MultiTaskLoss initialized: focal_alpha={focal_alpha}, "
                   f"focal_gamma={focal_gamma}, mtl_lambda_svo={mtl_lambda_svo}, "
                   f"mtl_lambda_sent={mtl_lambda_sent}")
    
    def forward(
        self,
        gec_logits: torch.Tensor,
        svo_logits: torch.Tensor,
        sent_logits: torch.Tensor,
        gec_labels: torch.Tensor,
        svo_labels: torch.Tensor,
        sent_labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        计算联合损失
        
        Args:
            gec_logits: [B, L, num_gec_labels]
            svo_logits: [B, L, num_svo_labels]
            sent_logits: [B, 2] (句级错误检测)
            gec_labels: [B, L]
            svo_labels: [B, L]
            sent_labels: [B] (0=正确, 1=有错)
            label_mask: [B, L]
        
        Returns:
            (total_loss, gec_loss, svo_loss, sent_loss)
        """
        # GEC损失 (Focal Loss)
        gec_loss = self.focal_loss(gec_logits, gec_labels, label_mask)
        
        # SVO损失 (CrossEntropy)
        svo_logits_flat = svo_logits.view(-1, svo_logits.size(-1))
        svo_labels_flat = svo_labels.view(-1)
        svo_loss = self.ce_loss(svo_logits_flat, svo_labels_flat)
        
        # 句级错误检测损失 (CrossEntropy)
        sent_loss = self.sent_ce_loss(sent_logits, sent_labels)
        
        # 总损失
        total_loss = gec_loss + self.mtl_lambda_svo * svo_loss + self.mtl_lambda_sent * sent_loss
        
        return total_loss, gec_loss, svo_loss, sent_loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    加权交叉熵（备用方案）
    
    如果Focal Loss效果不好，可以尝试简单的加权CE
    """
    
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights
        self.ignore_index = ignore_index
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [B, L, num_labels]
            targets: [B, L]
        
        Returns:
            loss: scalar
        """
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)
        
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss_fn(logits, targets)


if __name__ == "__main__":
    # 测试Focal Loss
    print("Testing Focal Loss...")
    
    batch_size, seq_len, num_labels = 4, 128, 5000
    
    # 模拟数据
    logits = torch.randn(batch_size, seq_len, num_labels)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long)  # 大部分是$KEEP (0)
    targets[0, 10] = 100  # 模拟一个错误
    targets[1, 20] = 200
    
    label_mask = torch.ones(batch_size, seq_len)
    label_mask[:, 0] = 0  # [CLS]不计算
    label_mask[:, -1] = 0  # [SEP]不计算
    
    # 计算loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(logits, targets, label_mask)
    print(f"Focal Loss: {loss.item():.4f}")
    
    # 测试多任务损失
    print("\nTesting MultiTask Loss...")
    svo_logits = torch.randn(batch_size, seq_len, 7)
    svo_labels = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    mtl_loss = MultiTaskLoss(focal_alpha=0.25, focal_gamma=2.0, mtl_lambda=0.5)
    total_loss, gec_loss, svo_loss = mtl_loss(
        logits, svo_logits, targets, svo_labels, label_mask
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"GEC Loss: {gec_loss.item():.4f}")
    print(f"SVO Loss: {svo_loss.item():.4f}")
