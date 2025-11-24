"""
训练流程模块
负责：
1. 模型训练循环
2. 验证和评估
3. Checkpoint保存
4. Metrics计算 (Recall, Precision, F0.5, F2)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import json
from typing import Dict, Tuple
from datetime import datetime

from config import default_config as cfg
from modeling import GECModelWithMTL
from loss import MultiTaskLoss
from dataset import create_dataloaders, build_label_maps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GECTrainer:
    """
    GEC模型训练器
    
    功能：
    - 训练和验证循环
    - 指标计算（重点是Recall）
    - Early Stopping
    - Checkpoint管理
    """
    
    def __init__(
        self,
        model: GECModelWithMTL,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: MultiTaskLoss,
        device: str,
        exp_dir: Path,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.exp_dir = exp_dir
        self.config = config
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_recall = 0.0
        self.best_f2 = 0.0
        self.patience_counter = 0
        
        # 创建实验目录
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard (可选)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")
        except ImportError:
            self.writer = None
            logger.warning("TensorBoard not available")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_gec_loss = 0.0
        total_svo_loss = 0.0
        total_sent_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gec_labels = batch['gec_labels'].to(self.device)
            svo_labels = batch['svo_labels'].to(self.device)
            sent_labels = batch['sent_label'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            
            # 前向传播
            gec_logits, svo_logits, sent_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                gec_labels=gec_labels,
                svo_labels=svo_labels,
                sent_labels=sent_labels,
                label_mask=label_mask
            )
            
            # 计算损失
            loss, gec_loss, svo_loss, sent_loss = self.criterion(
                gec_logits, svo_logits, sent_logits,
                gec_labels, svo_labels, sent_labels,
                label_mask
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=cfg.MAX_GRAD_NORM
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 累计损失
            total_loss += loss.item()
            total_gec_loss += gec_loss.item()
            total_svo_loss += svo_loss.item()
            total_sent_loss += sent_loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'gec': f"{gec_loss.item():.4f}",
                'svo': f"{svo_loss.item():.4f}",
                'sent': f"{sent_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            self.global_step += 1
            
            # TensorBoard记录
            if self.writer and batch_idx % cfg.LOG_INTERVAL == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/gec_loss', gec_loss.item(), self.global_step)
                self.writer.add_scalar('train/svo_loss', svo_loss.item(), self.global_step)
                self.writer.add_scalar('train/sent_loss', sent_loss.item(), self.global_step)
        
        # 计算平均损失
        avg_loss = total_loss / len(self.train_loader)
        avg_gec_loss = total_gec_loss / len(self.train_loader)
        avg_svo_loss = total_svo_loss / len(self.train_loader)
        avg_sent_loss = total_sent_loss / len(self.train_loader)
        
        return {
            'loss': avg_loss,
            'gec_loss': avg_gec_loss,
            'svo_loss': avg_svo_loss,
            'sent_loss': avg_sent_loss
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """验证集评估"""
        self.model.eval()
        
        total_loss = 0.0
        
        # 用于计算指标
        all_gec_preds = []
        all_gec_labels = []
        all_masks = []
        
        for batch in tqdm(self.dev_loader, desc="Evaluating"):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gec_labels = batch['gec_labels'].to(self.device)
            svo_labels = batch['svo_labels'].to(self.device)
            sent_labels = batch['sent_label'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            
            # 前向传播
            gec_logits, svo_logits, sent_logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # 计算损失
            loss, _, _, _ = self.criterion(
                gec_logits, svo_logits, sent_logits,
                gec_labels, svo_labels, sent_labels,
                label_mask
            )
            total_loss += loss.item()
            
            # 收集预测结果
            gec_preds = torch.argmax(gec_logits, dim=-1)  # [B, L]
            
            all_gec_preds.append(gec_preds.cpu().numpy())
            all_gec_labels.append(gec_labels.cpu().numpy())
            all_masks.append(label_mask.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / len(self.dev_loader)
        
        # 计算指标
        metrics = self._compute_metrics(
            np.concatenate(all_gec_preds),
            np.concatenate(all_gec_labels),
            np.concatenate(all_masks)
        )
        
        metrics['loss'] = avg_loss
        
        return metrics
    
    def _compute_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        核心指标：
        - Recall: TP / (TP + FN) - 最重要！
        - Precision: TP / (TP + FP)
        - F0.5: 强调Precision
        - F2: 强调Recall
        
        这里的"错误"指的是非$KEEP的标签
        """
        # Flatten并过滤无效位置
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()
        masks_flat = masks.flatten()
        
        # 只保留有效位置
        valid_idx = (labels_flat != -100) & (masks_flat > 0)
        preds_valid = preds_flat[valid_idx]
        labels_valid = labels_flat[valid_idx]
        
        # 将标签二值化：0是$KEEP，非0是错误
        # 假设$KEEP的ID是0
        pred_has_error = (preds_valid != 0).astype(int)
        label_has_error = (labels_valid != 0).astype(int)
        
        # 计算TP, FP, FN, TN
        tp = np.sum((pred_has_error == 1) & (label_has_error == 1))
        fp = np.sum((pred_has_error == 1) & (label_has_error == 0))
        fn = np.sum((pred_has_error == 0) & (label_has_error == 1))
        tn = np.sum((pred_has_error == 0) & (label_has_error == 0))
        
        # Recall
        recall = tp / (tp + fn + 1e-10)
        
        # Precision
        precision = tp / (tp + fp + 1e-10)
        
        # F0.5 (强调Precision)
        beta = cfg.EVAL_F_BETA
        f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
        
        # F2 (强调Recall)
        beta = cfg.EVAL_F2_BETA
        f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
        
        # Accuracy (不太重要，因为大部分是$KEEP)
        accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
        
        return {
            'recall': recall,
            'precision': precision,
            'f0.5': f05,
            'f2': f2,
            'accuracy': accuracy,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    def train(self, num_epochs: int):
        """完整训练流程"""
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch} Train - Loss: {train_metrics['loss']:.4f}")
            
            # 验证
            eval_metrics = self.evaluate()
            logger.info(
                f"Epoch {epoch} Eval - "
                f"Loss: {eval_metrics['loss']:.4f}, "
                f"Recall: {eval_metrics['recall']:.4f}, "
                f"Precision: {eval_metrics['precision']:.4f}, "
                f"F2: {eval_metrics['f2']:.4f}"
            )
            
            # TensorBoard记录
            if self.writer:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'eval/{key}', value, epoch)
            
            # 保存最佳模型 (基于Recall)
            if eval_metrics['recall'] > self.best_recall:
                self.best_recall = eval_metrics['recall']
                self.best_f2 = eval_metrics['f2']
                self.save_checkpoint('best_model.pt', eval_metrics)
                logger.info(f"✓ New best model saved! Recall: {self.best_recall:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Early Stopping
            if self.patience_counter >= cfg.PATIENCE:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存
            if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', eval_metrics)
        
        logger.info(f"Training completed! Best Recall: {self.best_recall:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Dict = None):
        """保存checkpoint"""
        checkpoint_path = self.exp_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_recall': self.best_recall,
            'config': self.config,
        }
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main():
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.USE_CUDA else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 加载tokenizer和标签映射
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
    gec_label_map, svo_label_map = build_label_maps(str(cfg.VOCAB_DIR))
    
    # 创建数据加载器
    train_loader, dev_loader = create_dataloaders(
        train_path=str(cfg.SYNTHETIC_DATA_DIR / "train.json"),
        dev_path=str(cfg.SYNTHETIC_DATA_DIR / "dev.json"),
        tokenizer=tokenizer,
        gec_label_map=gec_label_map,
        svo_label_map=svo_label_map,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        max_length=cfg.MAX_SEQ_LENGTH
    )
    
    # 创建模型
    from modeling import create_model
    model = create_model(
        bert_model_name=cfg.BERT_MODEL,
        num_gec_labels=len(gec_label_map),
        num_svo_labels=len(svo_label_map),
        device=device
    )
    
    # 优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 损失函数
    criterion = MultiTaskLoss(
        focal_alpha=cfg.FOCAL_LOSS_ALPHA,
        focal_gamma=cfg.FOCAL_LOSS_GAMMA,
        mtl_lambda_svo=cfg.MTL_LAMBDA_SVO,
        mtl_lambda_sent=cfg.MTL_LAMBDA_SENT
    )
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = cfg.EXPERIMENTS_DIR / f"exp_{timestamp}"
    
    # 创建训练器
    trainer = GECTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        exp_dir=exp_dir,
        config=cfg.to_dict()
    )
    
    # 开始训练
    trainer.train(num_epochs=cfg.NUM_EPOCHS)


if __name__ == "__main__":
    main()
