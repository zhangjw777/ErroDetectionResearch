"""
损失函数模块
实现Focal Loss、多任务联合损失和不确定性加权损失
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
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
        
        # **关键修复**：先处理 ignore_index，避免 gather 时访问非法索引
        # 创建 mask，标记有效位置
        mask = (targets != self.ignore_index).float()
        
        # 将 ignore_index 的位置临时映射到合法值（0，即 $KEEP）
        # 这些位置的 loss 会在后面通过 mask 被清零
        valid_targets = targets.clone()
        valid_targets[targets == self.ignore_index] = 0
        
        # 计算log_softmax
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 获取目标类别的log概率（现在所有 index 都是合法的）
        log_probs = log_probs.gather(dim=-1, index=valid_targets.unsqueeze(-1)).squeeze(-1)
        
        # 计算p_t
        probs = torch.exp(log_probs)
        
        # Focal Loss权重: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Alpha平衡 (简化版：对所有非$KEEP类加权)
        # 注意：这里仍然使用 valid_targets，因为 targets 中可能有 -100
        alpha_weight = torch.where(
            valid_targets == 0,  # $KEEP 的 ID 是 0（已验证）
            torch.full_like(valid_targets, self.alpha, dtype=torch.float),
            torch.full_like(valid_targets, 1 - self.alpha, dtype=torch.float)
        )
        
        # 计算loss
        loss = -alpha_weight * focal_weight * log_probs
        
        # 应用 mask，清零 ignore_index 位置的 loss
        loss = loss * mask
        
        # 如果提供了label_mask，进一步过滤
        if label_mask is not None:
            label_mask = label_mask.view(-1)
            loss = loss * label_mask
            # **修复归一化问题**：计算实际有效的token数量
            # 同时考虑 ignore_index 和 label_mask
            effective_mask = mask * label_mask
        else:
            effective_mask = mask
        
        # Reduction
        if self.reduction == 'mean':
            # **关键修复**：除以实际参与计算的token数量，而非所有非ignore的token数量
            effective_count = effective_mask.sum()
            if effective_count > 0:
                return loss.sum() / effective_count
            else:
                return loss.sum()  # 避免除零
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


# ==================== 模块二：基于不确定性的动态损失加权 ====================

class UncertaintyWeightedLoss(nn.Module):
    """
    基于同方差不确定性的动态多任务损失加权
    (Uncertainty-Weighted Multi-Task Loss)
    
    基于论文 "Multi-Task Learning Using Uncertainty to Weigh Losses" (CVPR 2018)
    
    核心思想：
    - 每个任务有一个可学习的不确定性参数 σ_i
    - 不确定性大的任务自动获得更小的权重
    - 使用对数方差 s_i = log(σ_i²) 形式提升数值稳定性
    
    公式：
    L_total = 1/2·exp(-s₁)·L_GEC + 1/2·exp(-s₂)·L_SVO + 1/2·exp(-s₃)·L_Sent
              + 1/2·(s₁ + s₂ + s₃)
    
    其中：
    - exp(-s_i) = 1/σ_i² 是任务权重（精度）
    - 1/2·(s₁+s₂+s₃) 是正则项，防止 σ 无限增大
    
    Args:
        init_log_var: 初始对数方差值 (默认0，对应σ=1)
    """
    
    def __init__(self, init_log_var: float = 0.0):
        super().__init__()
        
        # 可学习的对数方差参数: s_i = log(σ_i²)
        # 初始化为 init_log_var，对应 σ = exp(s/2)
        self.log_var_gec = nn.Parameter(torch.tensor([init_log_var]))
        self.log_var_svo = nn.Parameter(torch.tensor([init_log_var]))
        self.log_var_sent = nn.Parameter(torch.tensor([init_log_var]))
        
        logger.info(f"UncertaintyWeightedLoss initialized with log_var={init_log_var}")
    
    def forward(
        self,
        loss_gec: torch.Tensor,
        loss_svo: torch.Tensor,
        loss_sent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算不确定性加权的总损失
        
        Args:
            loss_gec: GEC主任务损失 (可以是Focal Loss)
            loss_svo: SVO辅助任务损失
            loss_sent: 句级错误检测损失
        
        Returns:
            total_loss: 加权后的总损失 (标量)
            loss_dict: 包含各项损失和不确定性参数的字典 (用于日志记录)
        """
        # 计算任务权重 (精度): precision = exp(-log_var) = 1/σ²
        precision_gec = torch.exp(-self.log_var_gec)
        precision_svo = torch.exp(-self.log_var_svo)
        precision_sent = torch.exp(-self.log_var_sent)
        
        # 加权损失: 0.5 * precision * loss
        weighted_gec = 0.5 * precision_gec * loss_gec
        weighted_svo = 0.5 * precision_svo * loss_svo
        weighted_sent = 0.5 * precision_sent * loss_sent
        
        # 正则项: 0.5 * (s₁ + s₂ + s₃)
        # 防止 σ 无限增大导致所有任务权重趋近于 0
        regularization = 0.5 * (self.log_var_gec + self.log_var_svo + self.log_var_sent)
        
        # 总损失
        total_loss = weighted_gec + weighted_svo + weighted_sent + regularization
        
        # 将 total_loss 从 [1] 变为标量
        total_loss = total_loss.squeeze()
        
        # 构建详细信息字典 (用于日志记录和TensorBoard)
        loss_dict = {
            # 原始损失
            'loss_total': total_loss.item(),
            'loss_gec': loss_gec.item(),
            'loss_svo': loss_svo.item(),
            'loss_sent': loss_sent.item(),
            # 加权后的损失
            'weighted_gec': weighted_gec.item(),
            'weighted_svo': weighted_svo.item(),
            'weighted_sent': weighted_sent.item(),
            # 对数方差 (可学习参数)
            'log_var_gec': self.log_var_gec.item(),
            'log_var_svo': self.log_var_svo.item(),
            'log_var_sent': self.log_var_sent.item(),
            # 标准差 σ = exp(s/2)
            'sigma_gec': torch.exp(0.5 * self.log_var_gec).item(),
            'sigma_svo': torch.exp(0.5 * self.log_var_svo).item(),
            'sigma_sent': torch.exp(0.5 * self.log_var_sent).item(),
            # 任务权重 (精度) = exp(-s)
            'weight_gec': precision_gec.item(),
            'weight_svo': precision_svo.item(),
            'weight_sent': precision_sent.item(),
        }
        
        return total_loss, loss_dict
    
    def get_task_weights(self) -> Dict[str, float]:
        """
        获取当前的任务权重 (用于分析)
        
        Returns:
            weights: 各任务的权重字典
        """
        with torch.no_grad():
            return {
                'gec': torch.exp(-self.log_var_gec).item(),
                'svo': torch.exp(-self.log_var_svo).item(),
                'sent': torch.exp(-self.log_var_sent).item(),
            }
    
    def get_uncertainties(self) -> Dict[str, float]:
        """
        获取当前的任务不确定性 σ (用于分析)
        
        Returns:
            uncertainties: 各任务的不确定性字典
        """
        with torch.no_grad():
            return {
                'gec': torch.exp(0.5 * self.log_var_gec).item(),
                'svo': torch.exp(0.5 * self.log_var_svo).item(),
                'sent': torch.exp(0.5 * self.log_var_sent).item(),
            }


class MultiTaskLossWithUncertainty(nn.Module):
    """
    集成不确定性加权的多任务损失
    
    整合了：
    - GEC任务的Focal Loss
    - SVO任务的CrossEntropy
    - 句级任务的CrossEntropy
    - 不确定性动态加权
    
    可以选择使用固定权重或不确定性加权
    """
    
    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        ignore_index: int = -100,
        use_uncertainty_weighting: bool = True,
        fixed_lambda_svo: float = 0.5,
        fixed_lambda_sent: float = 0.3,
        init_log_var: float = 0.0
    ):
        """
        Args:
            focal_alpha: Focal Loss的alpha参数
            focal_gamma: Focal Loss的gamma参数
            ignore_index: 忽略的标签ID
            use_uncertainty_weighting: 是否使用不确定性加权
            fixed_lambda_svo: 固定模式下SVO任务的权重
            fixed_lambda_sent: 固定模式下句级任务的权重
            init_log_var: 不确定性参数的初始值
        """
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.fixed_lambda_svo = fixed_lambda_svo
        self.fixed_lambda_sent = fixed_lambda_sent
        
        # GEC任务使用Focal Loss
        self.focal_loss = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            reduction='mean',
            ignore_index=ignore_index
        )
        
        # SVO任务使用CrossEntropy
        self.ce_loss = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            reduction='mean'
        )
        
        # 句级任务使用CrossEntropy
        self.sent_ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
        # 不确定性加权模块
        if use_uncertainty_weighting:
            self.uncertainty_weighter = UncertaintyWeightedLoss(init_log_var=init_log_var)
        else:
            self.uncertainty_weighter = None
        
        logger.info(
            f"MultiTaskLossWithUncertainty initialized: "
            f"uncertainty_weighting={use_uncertainty_weighting}, "
            f"focal_alpha={focal_alpha}, focal_gamma={focal_gamma}"
        )
    
    def forward(
        self,
        gec_logits: torch.Tensor,
        svo_logits: torch.Tensor,
        sent_logits: torch.Tensor,
        gec_labels: torch.Tensor,
        svo_labels: torch.Tensor,
        sent_labels: torch.Tensor,
        label_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多任务联合损失
        
        Args:
            gec_logits: [B, L, num_gec_labels]
            svo_logits: [B, L, num_svo_labels]
            sent_logits: [B, 2]
            gec_labels: [B, L]
            svo_labels: [B, L]
            sent_labels: [B]
            label_mask: [B, L]
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失详情字典
        """
        # 1. 计算GEC损失 (Focal Loss)
        gec_loss = self.focal_loss(gec_logits, gec_labels, label_mask)
        
        # 2. 计算SVO损失 (CrossEntropy)
        svo_logits_flat = svo_logits.view(-1, svo_logits.size(-1))
        svo_labels_flat = svo_labels.view(-1)
        svo_loss = self.ce_loss(svo_logits_flat, svo_labels_flat)
        
        # 3. 计算句级损失 (CrossEntropy)
        sent_loss = self.sent_ce_loss(sent_logits, sent_labels)
        
        # 4. 计算总损失
        if self.use_uncertainty_weighting and self.uncertainty_weighter is not None:
            # 使用不确定性加权
            total_loss, loss_dict = self.uncertainty_weighter(gec_loss, svo_loss, sent_loss)
        else:
            # 使用固定权重
            total_loss = gec_loss + self.fixed_lambda_svo * svo_loss + self.fixed_lambda_sent * sent_loss
            loss_dict = {
                'loss_total': total_loss.item(),
                'loss_gec': gec_loss.item(),
                'loss_svo': svo_loss.item(),
                'loss_sent': sent_loss.item(),
            }
        
        return total_loss, loss_dict
    
    def get_uncertainty_params(self) -> Optional[Dict[str, nn.Parameter]]:
        """
        获取不确定性参数（用于添加到优化器）
        
        Returns:
            params: 不确定性参数字典，如果不使用不确定性加权则返回None
        """
        if self.use_uncertainty_weighting and self.uncertainty_weighter is not None:
            return {
                'log_var_gec': self.uncertainty_weighter.log_var_gec,
                'log_var_svo': self.uncertainty_weighter.log_var_svo,
                'log_var_sent': self.uncertainty_weighter.log_var_sent,
            }
        return None


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
    sent_logits = torch.randn(batch_size, 2)
    sent_labels = torch.ones(batch_size, dtype=torch.long)  # 假设都有错
    
    mtl_loss = MultiTaskLoss(focal_alpha=0.25, focal_gamma=2.0, mtl_lambda_svo=0.5, mtl_lambda_sent=0.3)
    total_loss, gec_loss, svo_loss, sent_loss = mtl_loss(
        logits, svo_logits, sent_logits,
        targets, svo_labels, sent_labels,
        label_mask
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"GEC Loss: {gec_loss.item():.4f}")
    print(f"SVO Loss: {svo_loss.item():.4f}")
    print(f"Sent Loss: {sent_loss.item():.4f}")
    
    # ==================== 测试不确定性加权损失 ====================
    print("\n" + "=" * 50)
    print("Testing Uncertainty Weighted Loss...")
    
    # 模拟三个任务的损失
    loss_gec = torch.tensor(2.5)
    loss_svo = torch.tensor(1.2)
    loss_sent = torch.tensor(0.8)
    
    uncertainty_loss = UncertaintyWeightedLoss(init_log_var=0.0)
    total, loss_dict = uncertainty_loss(loss_gec, loss_svo, loss_sent)
    
    print(f"Total Loss: {total.item():.4f}")
    print(f"Task Weights: {uncertainty_loss.get_task_weights()}")
    print(f"Task Uncertainties: {uncertainty_loss.get_uncertainties()}")
    
    # 测试集成版本
    print("\n" + "=" * 50)
    print("Testing MultiTaskLossWithUncertainty...")
    
    mtl_uncertainty = MultiTaskLossWithUncertainty(
        focal_alpha=0.25,
        focal_gamma=2.0,
        use_uncertainty_weighting=True
    )
    
    total_loss, loss_dict = mtl_uncertainty(
        logits, svo_logits, sent_logits,
        targets, svo_labels, sent_labels,
        label_mask
    )
    
    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Loss Dict Keys: {list(loss_dict.keys())}")
    print(f"GEC Weight: {loss_dict.get('weight_gec', 'N/A')}")
    print(f"SVO Weight: {loss_dict.get('weight_svo', 'N/A')}")
    print(f"Sent Weight: {loss_dict.get('weight_sent', 'N/A')}")
