"""
训练流程模块
负责：
1. 模型训练循环
2. 验证和评估
3. Checkpoint保存
4. Metrics计算 (Recall, Precision, F0.5, F2)
5. 混合精度训练 (AMP/FP16) 支持
6. 分布式训练 (DDP) 支持
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import get_linear_schedule_with_warmup, BertTokenizer
import numpy as np
from tqdm import tqdm
from pathlib import Path
import logging
import json
from typing import Dict, Tuple, Optional
from datetime import datetime

from config import default_config as cfg
from modeling import GECModelWithMTL
from loss import MultiTaskLoss, MultiTaskLossWithUncertainty
from dataset import GECDataset, build_label_maps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== DDP工具函数 ====================

def setup_ddp(rank: int, world_size: int, backend: str = 'nccl'):
    """
    初始化分布式训练环境
    
    Args:
        rank: 当前进程的rank (GPU编号)
        world_size: 总进程数 (GPU数量)
        backend: 通信后端，GPU用'nccl'，CPU用'gloo'
    """
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    logger.info(f"DDP initialized: rank={rank}, world_size={world_size}")


def cleanup_ddp():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """判断是否是主进程（rank=0），用于日志打印和模型保存"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """获取当前进程的rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """获取总进程数"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_tensor(tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
    """
    在所有进程间聚合tensor
    
    Args:
        tensor: 要聚合的tensor
        op: 聚合操作，'mean'或'sum'
    """
    if not dist.is_initialized():
        return tensor
    
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    
    if op == 'mean':
        rt /= get_world_size()
    
    return rt


def create_ddp_dataloaders(
    train_path: str,
    dev_path: str,
    tokenizer,
    gec_label_map: Dict,
    svo_label_map: Dict,
    batch_size: int = 32,
    num_workers: int = 4,
    max_length: int = 128,
    use_ddp: bool = False
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    """
    创建支持DDP的DataLoader
    
    Args:
        use_ddp: 是否使用分布式训练
        
    Returns:
        train_loader, dev_loader, train_sampler (用于每个epoch设置)
    """
    train_dataset = GECDataset(
        train_path, tokenizer, gec_label_map, svo_label_map, max_length=max_length
    )
    dev_dataset = GECDataset(
        dev_path, tokenizer, gec_label_map, svo_label_map, max_length=max_length
    )
    
    train_sampler = None
    dev_sampler = None
    
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True
        )
        dev_sampler = DistributedSampler(
            dev_dataset,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=False
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # DDP时由sampler控制shuffle
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, dev_loader, train_sampler


class GECTrainer:
    """
    GEC模型训练器 (支持AMP + DDP + 不确定性加权)
    
    功能：
    - 混合精度训练 (AMP/FP16)
    - 分布式训练 (DDP)
    - 可选梯度累积
    - 不确定性加权多任务损失
    - 指标计算（重点是Recall）
    - Early Stopping
    - Checkpoint管理
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        dev_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        exp_dir: Path,
        config: dict,
        use_amp: bool = True,
        use_ddp: bool = False,
        train_sampler: Optional[DistributedSampler] = None,
        gradient_accumulation_steps: int = 1,
        use_uncertainty_weighting: bool = True
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
        
        # AMP设置
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        
        # DDP设置
        self.use_ddp = use_ddp
        self.train_sampler = train_sampler
        
        # 梯度累积
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # 不确定性加权
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_recall = 0.0
        self.best_f2 = 0.0
        self.patience_counter = 0
        
        # 用于记录不确定性参数的历史
        self.uncertainty_history = {
            'log_var_gec': [],
            'log_var_svo': [],
            'log_var_sent': [],
            'sigma_gec': [],
            'sigma_svo': [],
            'sigma_sent': [],
        }
        
        # 仅主进程创建实验目录和TensorBoard
        if is_main_process():
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            # 创建 F2 最优模型的单独保存目录
            self.f2_best_dir = self.exp_dir / "best_f2_model"
            self.f2_best_dir.mkdir(parents=True, exist_ok=True)
            
            # TensorBoard (可选)
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=self.exp_dir / "tensorboard")
            except ImportError:
                self.writer = None
                logger.warning("TensorBoard not available")
        else:
            self.f2_best_dir = None
            self.writer = None
        
        if is_main_process():
            logger.info(f"Trainer initialized: AMP={self.use_amp}, DDP={self.use_ddp}, "
                       f"gradient_accumulation_steps={self.gradient_accumulation_steps}, "
                       f"uncertainty_weighting={self.use_uncertainty_weighting}")
    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch（支持AMP、梯度累积和不确定性加权）"""
        self.model.train()
        
        # DDP: 设置sampler的epoch以确保每个epoch的shuffle不同
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)
        
        total_loss = 0.0
        total_gec_loss = 0.0
        total_svo_loss = 0.0
        total_sent_loss = 0.0
        
        # 累积不确定性参数 (如果使用)
        accumulated_log_vars = {'gec': 0.0, 'svo': 0.0, 'sent': 0.0}
        
        # 仅主进程显示进度条
        if is_main_process():
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            progress_bar = self.train_loader
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gec_labels = batch['gec_labels'].to(self.device)
            svo_labels = batch['svo_labels'].to(self.device)
            sent_labels = batch['sent_label'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            
            # 前向传播 (使用AMP)
            with autocast(device_type='cuda', enabled=self.use_amp):
                gec_logits, svo_logits, sent_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    gec_labels=gec_labels,
                    svo_labels=svo_labels,
                    sent_labels=sent_labels,
                    label_mask=label_mask
                )
                
                # 计算损失 - 支持两种损失函数接口
                if self.use_uncertainty_weighting:
                    # 新接口：返回 (total_loss, loss_dict)
                    loss, loss_dict = self.criterion(
                        gec_logits, svo_logits, sent_logits,
                        gec_labels, svo_labels, sent_labels,
                        label_mask
                    )
                    gec_loss = loss_dict['loss_gec']
                    svo_loss = loss_dict['loss_svo']
                    sent_loss = loss_dict['loss_sent']
                    
                    # 累积不确定性参数
                    if 'log_var_gec' in loss_dict:
                        accumulated_log_vars['gec'] += loss_dict['log_var_gec']
                        accumulated_log_vars['svo'] += loss_dict['log_var_svo']
                        accumulated_log_vars['sent'] += loss_dict['log_var_sent']
                else:
                    # 旧接口：返回 (total_loss, gec_loss, svo_loss, sent_loss)
                    loss, gec_loss, svo_loss, sent_loss = self.criterion(
                        gec_logits, svo_logits, sent_logits,
                        gec_labels, svo_labels, sent_labels,
                        label_mask
                    )
                    gec_loss = gec_loss.item()
                    svo_loss = svo_loss.item()
                    sent_loss = sent_loss.item()
                
                # 梯度累积：缩放损失
                loss = loss / self.gradient_accumulation_steps
            
            # 反向传播 (使用GradScaler)
            self.scaler.scale(loss).backward()
            
            # 梯度累积：达到累积步数后更新参数
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪 (需要先unscale)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=cfg.MAX_GRAD_NORM
                )
                
                # 更新参数
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # 累计损失 (还原缩放后的损失)
            actual_loss = loss.item() * self.gradient_accumulation_steps
            total_loss += actual_loss
            total_gec_loss += gec_loss if isinstance(gec_loss, float) else gec_loss
            total_svo_loss += svo_loss if isinstance(svo_loss, float) else svo_loss
            total_sent_loss += sent_loss if isinstance(sent_loss, float) else sent_loss
            
            # 更新进度条 (仅主进程)
            if is_main_process() and hasattr(progress_bar, 'set_postfix'):
                postfix = {
                    'loss': f"{actual_loss:.4f}",
                    'gec': f"{gec_loss:.4f}",
                    'svo': f"{svo_loss:.4f}",
                    'sent': f"{sent_loss:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                }
                # 如果使用不确定性加权，显示任务权重
                if self.use_uncertainty_weighting and 'weight_gec' in loss_dict:
                    postfix['w_gec'] = f"{loss_dict['weight_gec']:.2f}"
                progress_bar.set_postfix(postfix)
            
            # TensorBoard记录 (仅主进程)
            if self.writer and batch_idx % cfg.LOG_INTERVAL == 0:
                self.writer.add_scalar('train/loss', actual_loss, self.global_step)
                self.writer.add_scalar('train/gec_loss', gec_loss, self.global_step)
                self.writer.add_scalar('train/svo_loss', svo_loss, self.global_step)
                self.writer.add_scalar('train/sent_loss', sent_loss, self.global_step)
                
                # 记录不确定性参数 (详细记录)
                if self.use_uncertainty_weighting and 'log_var_gec' in loss_dict:
                    # 对数方差 s = log(σ²)
                    self.writer.add_scalar('uncertainty/log_var_gec', loss_dict['log_var_gec'], self.global_step)
                    self.writer.add_scalar('uncertainty/log_var_svo', loss_dict['log_var_svo'], self.global_step)
                    self.writer.add_scalar('uncertainty/log_var_sent', loss_dict['log_var_sent'], self.global_step)
                    
                    # 任务权重 = exp(-s) = 1/σ²
                    self.writer.add_scalar('uncertainty/weight_gec', loss_dict['weight_gec'], self.global_step)
                    self.writer.add_scalar('uncertainty/weight_svo', loss_dict['weight_svo'], self.global_step)
                    self.writer.add_scalar('uncertainty/weight_sent', loss_dict['weight_sent'], self.global_step)
                    
                    # 标准差 σ = exp(s/2)
                    self.writer.add_scalar('uncertainty/sigma_gec', loss_dict['sigma_gec'], self.global_step)
                    self.writer.add_scalar('uncertainty/sigma_svo', loss_dict['sigma_svo'], self.global_step)
                    self.writer.add_scalar('uncertainty/sigma_sent', loss_dict['sigma_sent'], self.global_step)
                    
                    # 加权后的损失
                    self.writer.add_scalar('uncertainty/weighted_gec', loss_dict['weighted_gec'], self.global_step)
                    self.writer.add_scalar('uncertainty/weighted_svo', loss_dict['weighted_svo'], self.global_step)
                    self.writer.add_scalar('uncertainty/weighted_sent', loss_dict['weighted_sent'], self.global_step)
        
        # 计算平均损失
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        avg_gec_loss = total_gec_loss / num_batches
        avg_svo_loss = total_svo_loss / num_batches
        avg_sent_loss = total_sent_loss / num_batches
        
        # 记录 epoch 级别的不确定性参数
        if self.use_uncertainty_weighting and num_batches > 0:
            avg_log_var_gec = accumulated_log_vars['gec'] / num_batches
            avg_log_var_svo = accumulated_log_vars['svo'] / num_batches
            avg_log_var_sent = accumulated_log_vars['sent'] / num_batches
            
            self.uncertainty_history['log_var_gec'].append(avg_log_var_gec)
            self.uncertainty_history['log_var_svo'].append(avg_log_var_svo)
            self.uncertainty_history['log_var_sent'].append(avg_log_var_sent)
            
            import math
            self.uncertainty_history['sigma_gec'].append(math.exp(0.5 * avg_log_var_gec))
            self.uncertainty_history['sigma_svo'].append(math.exp(0.5 * avg_log_var_svo))
            self.uncertainty_history['sigma_sent'].append(math.exp(0.5 * avg_log_var_sent))
        
        # DDP: 在所有进程间聚合损失
        if self.use_ddp:
            avg_loss = reduce_tensor(torch.tensor(avg_loss, device=self.device)).item()
            avg_gec_loss = reduce_tensor(torch.tensor(avg_gec_loss, device=self.device)).item()
            avg_svo_loss = reduce_tensor(torch.tensor(avg_svo_loss, device=self.device)).item()
            avg_sent_loss = reduce_tensor(torch.tensor(avg_sent_loss, device=self.device)).item()
        
        return {
            'loss': avg_loss,
            'gec_loss': avg_gec_loss,
            'svo_loss': avg_svo_loss,
            'sent_loss': avg_sent_loss
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """验证集评估（支持AMP、DDP和新的损失函数接口）"""
        self.model.eval()
        
        total_loss = 0.0
        
        # 用于计算指标
        all_gec_preds = []
        all_gec_labels = []
        all_masks = []
        
        # 仅主进程显示进度条
        if is_main_process():
            progress_bar = tqdm(self.dev_loader, desc="Evaluating")
        else:
            progress_bar = self.dev_loader
        
        for batch in progress_bar:
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            gec_labels = batch['gec_labels'].to(self.device)
            svo_labels = batch['svo_labels'].to(self.device)
            sent_labels = batch['sent_label'].to(self.device)
            label_mask = batch['label_mask'].to(self.device)
            
            # 前向传播 (使用AMP)
            with autocast(device_type='cuda', enabled=self.use_amp):
                gec_logits, svo_logits, sent_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # 计算损失 - 支持两种接口
                if self.use_uncertainty_weighting:
                    loss, _ = self.criterion(
                        gec_logits, svo_logits, sent_logits,
                        gec_labels, svo_labels, sent_labels,
                        label_mask
                    )
                else:
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
        
        # DDP: 在所有进程间聚合指标
        if self.use_ddp:
            # 聚合 TP, FP, FN, TN
            tp = reduce_tensor(torch.tensor(metrics['tp'], dtype=torch.float, device=self.device), op='sum').item()
            fp = reduce_tensor(torch.tensor(metrics['fp'], dtype=torch.float, device=self.device), op='sum').item()
            fn = reduce_tensor(torch.tensor(metrics['fn'], dtype=torch.float, device=self.device), op='sum').item()
            tn = reduce_tensor(torch.tensor(metrics['tn'], dtype=torch.float, device=self.device), op='sum').item()
            
            # 重新计算指标
            metrics['tp'] = int(tp)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tn'] = int(tn)
            
            recall = tp / (tp + fn + 1e-10)
            precision = tp / (tp + fp + 1e-10)
            
            beta = cfg.EVAL_F_BETA
            f05 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
            
            beta = cfg.EVAL_F2_BETA
            f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-10)
            
            accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-10)
            
            metrics['recall'] = recall
            metrics['precision'] = precision
            metrics['f0.5'] = f05
            metrics['f2'] = f2
            metrics['accuracy'] = accuracy
            metrics['loss'] = reduce_tensor(torch.tensor(avg_loss, device=self.device)).item()
        
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
        if is_main_process():
            logger.info(f"Starting training for {num_epochs} epochs")
            logger.info(f"  - AMP enabled: {self.use_amp}")
            logger.info(f"  - DDP enabled: {self.use_ddp}")
            logger.info(f"  - Gradient accumulation steps: {self.gradient_accumulation_steps}")
            logger.info(f"  - World size: {get_world_size()}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_epoch()
            if is_main_process():
                logger.info(f"Epoch {epoch} Train - Loss: {train_metrics['loss']:.4f}")
            
            # 验证
            eval_metrics = self.evaluate()
            if is_main_process():
                logger.info(
                    f"Epoch {epoch} Eval - "
                    f"Loss: {eval_metrics['loss']:.4f}, "
                    f"Recall: {eval_metrics['recall']:.4f}, "
                    f"Precision: {eval_metrics['precision']:.4f}, "
                    f"F2: {eval_metrics['f2']:.4f}"
                )
            
            # TensorBoard记录 (仅主进程)
            if self.writer:
                for key, value in eval_metrics.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f'eval/{key}', value, epoch)
            
            # 保存最佳模型 (基于Recall，仅主进程)
            if eval_metrics['recall'] > self.best_recall:
                self.best_recall = eval_metrics['recall']
                if is_main_process():
                    self.save_checkpoint('best_model.pt', eval_metrics)
                    logger.info(f"✓ New best Recall model saved! Recall: {self.best_recall:.4f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存最佳 F2 模型 (基于F2 Score，仅主进程)
            if eval_metrics['f2'] > self.best_f2:
                self.best_f2 = eval_metrics['f2']
                if is_main_process():
                    self.save_checkpoint('best_f2_model.pt', eval_metrics, save_dir=self.f2_best_dir)
                    logger.info(f"✓ New best F2 model saved! F2: {self.best_f2:.4f}")
            
            # Early Stopping
            if self.patience_counter >= cfg.PATIENCE:
                if is_main_process():
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # 定期保存 (仅主进程)
            if is_main_process() and (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt', eval_metrics)
            
            # DDP: 同步所有进程
            if self.use_ddp:
                dist.barrier()
        
        if is_main_process():
            logger.info(f"Training completed! Best Recall: {self.best_recall:.4f}, Best F2: {self.best_f2:.4f}")
    
    def save_checkpoint(self, filename: str, metrics: Dict = None, save_dir: Path = None):
        """
        保存checkpoint（自动处理DDP包装的模型和不确定性参数）
        
        Args:
            filename: checkpoint文件名
            metrics: 评估指标
            save_dir: 保存目录，默认为self.exp_dir
        """
        if save_dir is None:
            save_dir = self.exp_dir
        checkpoint_path = save_dir / filename
        
        # 获取原始模型（处理DDP包装）
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # 保存AMP状态
            'best_recall': self.best_recall,
            'best_f2': self.best_f2,  # 同时保存F2最优值
            'config': self.config,
        }
        
        # 保存损失函数状态（包含不确定性参数）
        if hasattr(self.criterion, 'state_dict'):
            checkpoint['criterion_state_dict'] = self.criterion.state_dict()
        
        # 保存不确定性参数历史
        if self.use_uncertainty_weighting and self.uncertainty_history:
            checkpoint['uncertainty_history'] = self.uncertainty_history
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def main(local_rank: int = -1):
    """
    主训练函数
    
    支持两种启动方式:
    1. 单卡训练: python trainer.py
    2. 多卡DDP训练: torchrun --nproc_per_node=2 trainer.py
    
    Args:
        local_rank: DDP时由torchrun自动设置，单卡时为-1
    """
    # ==================== 判断是否使用DDP ====================
    # torchrun会自动设置这些环境变量
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if use_ddp:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        setup_ddp(rank, world_size)
        device = torch.device(f'cuda:{local_rank}')
        
        # 仅在主进程打印信息
        if is_main_process():
            logger.info(f"DDP Mode: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() and cfg.USE_CUDA else 'cpu')
        logger.info(f"Single GPU Mode: device={device}")
    
    # ==================== 加载tokenizer和标签映射 ====================
    tokenizer = BertTokenizer.from_pretrained(cfg.BERT_MODEL)
    gec_label_map, svo_label_map = build_label_maps(str(cfg.VOCAB_DIR))
    
    # ==================== 创建数据加载器 ====================
    # DDP时每个GPU的实际batch_size = cfg.BATCH_SIZE
    # 总batch_size = cfg.BATCH_SIZE * world_size
    train_loader, dev_loader, train_sampler = create_ddp_dataloaders(
        train_path=str(cfg.SYNTHETIC_DATA_DIR / "train.json"),
        dev_path=str(cfg.SYNTHETIC_DATA_DIR / "dev.json"),
        tokenizer=tokenizer,
        gec_label_map=gec_label_map,
        svo_label_map=svo_label_map,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        max_length=cfg.MAX_SEQ_LENGTH,
        use_ddp=use_ddp
    )
    
    # ==================== 创建模型 ====================
    from modeling import create_model
    
    # 获取新模块配置（如果config中有定义）
    use_syntax_semantic_fusion = getattr(cfg, 'USE_SYNTAX_SEMANTIC_FUSION', True)
    use_error_aware_sent_head = getattr(cfg, 'USE_ERROR_AWARE_SENT_HEAD', True)
    keep_label_idx = getattr(cfg, 'KEEP_LABEL_IDX', 0)
    
    model = create_model(
        bert_model_name=cfg.BERT_MODEL,
        num_gec_labels=len(gec_label_map),
        num_svo_labels=len(svo_label_map),
        device=device,
        use_syntax_semantic_fusion=use_syntax_semantic_fusion,
        use_error_aware_sent_head=use_error_aware_sent_head,
        keep_label_idx=keep_label_idx
    )
    
    # DDP包装模型
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            logger.info("Model wrapped with DDP")
    
    # ==================== 损失函数 ====================
    use_uncertainty_weighting = getattr(cfg, 'USE_UNCERTAINTY_WEIGHTING', True)
    
    if use_uncertainty_weighting:
        # 使用不确定性加权的多任务损失
        criterion = MultiTaskLossWithUncertainty(
            focal_alpha=cfg.FOCAL_LOSS_ALPHA,
            focal_gamma=cfg.FOCAL_LOSS_GAMMA,
            use_uncertainty_weighting=True,
            fixed_lambda_svo=cfg.MTL_LAMBDA_SVO,
            fixed_lambda_sent=cfg.MTL_LAMBDA_SENT,
            uncertainty_params=cfg.UNCERTAINTY_INIT_LOG_VAR
        )
        if is_main_process():
            logger.info("Using UncertaintyWeighted MultiTask Loss")
    else:
        # 使用固定权重的多任务损失
        criterion = MultiTaskLoss(
            focal_alpha=cfg.FOCAL_LOSS_ALPHA,
            focal_gamma=cfg.FOCAL_LOSS_GAMMA,
            mtl_lambda_svo=cfg.MTL_LAMBDA_SVO,
            mtl_lambda_sent=cfg.MTL_LAMBDA_SENT
        )
        if is_main_process():
            logger.info("Using Fixed-weight MultiTask Loss")
    
    # ==================== 优化器和调度器 ====================
    # 构建参数组
    param_groups = [
        {'params': model.parameters(), 'lr': cfg.LEARNING_RATE}
    ]
    
    # 如果使用不确定性加权，将不确定性参数加入优化器
    # 通常不确定性参数使用较大的学习率
    if use_uncertainty_weighting and hasattr(criterion, 'uncertainty_weighter'):
        uncertainty_params = criterion.get_uncertainty_params()
        if uncertainty_params:
            param_groups.append({
                'params': list(uncertainty_params.values()),
                'lr': cfg.LEARNING_RATE * cfg.UNCERTAINTY_LR_MULTIPLIER  # 不确定性参数用更大的学习率
            })
            if is_main_process():
                logger.info("Added uncertainty parameters to optimizer with 10x learning rate")
    
    optimizer = AdamW(
        param_groups,
        weight_decay=cfg.WEIGHT_DECAY
    )
    
    # 计算总步数（考虑梯度累积）
    gradient_accumulation_steps = getattr(cfg, 'GRADIENT_ACCUMULATION_STEPS', 1)
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * cfg.NUM_EPOCHS
    warmup_steps = int(total_steps * cfg.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # ==================== 创建实验目录 ====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{timestamp}"
    if use_ddp:
        exp_name += f"_ddp{get_world_size()}gpu"
    if use_uncertainty_weighting:
        exp_name += "_uwl"  # Uncertainty Weighted Loss
    exp_dir = cfg.EXPERIMENTS_DIR / exp_name
    
    # ==================== 创建训练器 ====================
    use_amp = getattr(cfg, 'USE_AMP', True)  # 默认启用AMP
    
    trainer = GECTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        exp_dir=exp_dir,
        config=cfg.to_dict(),
        use_amp=use_amp,
        use_ddp=use_ddp,
        train_sampler=train_sampler,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_uncertainty_weighting=use_uncertainty_weighting
    )
    
    # ==================== 开始训练 ====================
    try:
        trainer.train(num_epochs=cfg.NUM_EPOCHS)
    finally:
        # 清理DDP
        if use_ddp:
            cleanup_ddp()


if __name__ == "__main__":
    main()
