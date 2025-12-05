"""
全局配置文件
定义所有超参数和路径配置
"""
import os
import json
from pathlib import Path

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
VOCAB_DIR = DATA_DIR / "vocab"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
DEPLOY_DIR = PROJECT_ROOT / "deploy"

# ==================== 模型配置 ====================
# 预训练模型
BERT_MODEL = "hfl/chinese-macbert-base"  # 使用MacBERT作为基础模型
BERT_HIDDEN_SIZE = 768

# 序列长度
MAX_SEQ_LENGTH = 192  

# ==================== 标签配置 ====================
# GECToR标签
GEC_KEEP_LABEL = "$KEEP"
GEC_DELETE_LABEL = "$DELETE"
GEC_APPEND_PREFIX = "$APPEND_"
GEC_REPLACE_PREFIX = "$REPLACE_"

# 句法标签 (BIO格式)
SVO_LABELS = [
    "O",        # 其他成分
    "B-SUB",    # 主语开始
    "I-SUB",    # 主语内部
    "B-PRED",   # 谓语开始
    "I-PRED",   # 谓语内部
    "B-OBJ",    # 宾语开始
    "I-OBJ",    # 宾语内部
]
NUM_SVO_LABELS = len(SVO_LABELS)

# 高频字词表大小（用于生成APPEND/REPLACE标签）
TOP_FREQ_CHARS = 3000  # 前3000个高频公文用字

# 高频虚词列表（用于 APPEND/REPLACE 标签轻量化）
# 只为这些字符生成专门的标签，其余使用 MASK
HIGH_FREQ_FUNCTION_WORDS = [
    # 结构助词
    "的", "地", "得", "了", "着", "过",
    # 判断词
    "是", "为",
    # 连词
    "和", "与", "及", "或", "而", "且",
    # 介词
    "在", "于", "对", "从", "把", "被", "由", "向", "到", "给", "让", "叫", "以",
    "通过", "经过", "根据", "依据", "按照", "关于", "为了",
    # 语气助词
    "吗", "呢", "吧", "啊", "呀", "哇",
    # 副词
    "不", "没", "都", "也", "还", "就", "才", "又", "再", "更", "很", "最", "非常",
    # 代词
    "我", "你", "他", "她", "它", "们", "这", "那", "哪", "什么", "怎么", "怎样",
    # 数词
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万",
    # 量词
    "个", "位", "名", "项", "件", "份", "次", "年", "月", "日",
]

# 标签压缩策略开关
ENABLE_LABEL_COMPRESSION = True  # 是否启用标签压缩（建议在标签词表超过5000时启用）

# ==================== 训练配置 ====================
# 基础训练参数
BATCH_SIZE = 128 #双4090配置
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 损失函数参数
FOCAL_LOSS_GAMMA = 2.5      # Focal Loss的聚焦参数
FOCAL_LOSS_ALPHA = 0.15     # 针对$KEEP标签的权重
MTL_LAMBDA_SVO = 0.5        # 多任务学习中SVO任务的权重
MTL_LAMBDA_SENT = 0.5       # 多任务学习中句级错误检测任务的权重

# Early Stopping
PATIENCE = 5  # 验证集连续3个epoch没提升就停止
MIN_DELTA = 0.001  # 最小改善幅度

# ==================== 数据增强配置 ====================
# 错误生成比例
AUG_PREPOSITION_ERROR_RATE = 0.15   # 介词滥用错误比例
AUG_WORD_ORDER_ERROR_RATE = 0.10    # 词序错误比例
AUG_CONFUSION_ERROR_RATE = 0.20     # 混淆词错误比例
AUG_DELETION_ERROR_RATE = 0.10      # 删除错误比例
AUG_INSERTION_ERROR_RATE = 0.10     # 插入错误比例
# ==================== 评估配置 ====================
# 评估指标的Beta值
EVAL_F_BETA = 0.5  # F0.5强调Precision，但我们主要关注Recall
EVAL_F2_BETA = 2.0  # F2强调Recall

# ==================== 部署配置 ====================
# 量化配置
QUANTIZATION_DTYPE = "qint8"  # INT8量化
ONNX_OPSET_VERSION = 14

# 推理配置
INFERENCE_BATCH_SIZE = 1  # 部署时单句推理
INFERENCE_DEVICE = "cpu"  # 默认CPU推理

# ==================== 日志配置 ====================
LOG_INTERVAL = 100  # 每100个batch打印一次日志
SAVE_INTERVAL = 1   # 每1个epoch保存一次模型
TENSORBOARD_DIR = EXPERIMENTS_DIR / "tensorboard"

# ==================== GPU配置 ====================
USE_CUDA = True  # 是否使用GPU（如果可用）
NUM_WORKERS = 4  # DataLoader的worker数量

# ==================== 混合精度与分布式训练配置 ====================
USE_AMP = True  # 是否启用混合精度训练 (FP16)，4090上强烈建议开启
GRADIENT_ACCUMULATION_STEPS = 1  # 梯度累积步数，增大等效batch_size

# ==================== 新模块配置 ====================
# 模块一：句法-语义融合交互层
USE_SYNTAX_SEMANTIC_FUSION = True  # 是否使用句法-语义融合层
SYNTAX_FUSION_USE_LAYER_NORM = True  # 融合后是否使用LayerNorm

# 模块二：不确定性加权损失
USE_UNCERTAINTY_WEIGHTING = True  # 是否使用不确定性动态加权
UNCERTAINTY_INIT_LOG_VAR = 0.0  # 不确定性参数初始值 (对应σ=1)
UNCERTAINTY_LR_MULTIPLIER = 10.0  # 不确定性参数的学习率倍数

# 模块三：错误感知多实例句级分类头
USE_ERROR_AWARE_SENT_HEAD = True  # 是否使用错误感知句级头
KEEP_LABEL_IDX = 0  # KEEP标签在标签表中的索引
DETACH_ERROR_CONFIDENCE = False  # 是否detach错误置信度梯度（用于消融实验）

# 公文特有词汇

# 介词集
PREPOSITIONS = ["通过", "经过", "在", "由于", "鉴于", "根据",
                "依据", "按照", "依照", "遵照", "据", "特别是",
                "为了", "为", "对于", "关于", "针对", "面向",
                "随着", "当", "值此", "之际","由", "被", "经",
                "结合", "伴随"]

# 混淆集（从JSON文件加载）
CONFUSIONS_FILE = VOCAB_DIR / "confusions.json"

def _load_confusions() -> dict:
    """从JSON文件加载混淆集"""
    if CONFUSIONS_FILE.exists():
        with open(CONFUSIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # 如果文件不存在，返回空字典并警告
        import logging
        logging.warning(f"混淆集文件不存在: {CONFUSIONS_FILE}")
        return {}

POLITICAL_CONFUSIONS = _load_confusions()




class Config:
    """配置类，方便后续扩展和管理"""
    
    def __init__(self):
        # 路径配置 (同时提供大写和小写访问)
        self.PROJECT_ROOT = self.project_root = PROJECT_ROOT
        self.DATA_DIR = self.data_dir = DATA_DIR
        self.RAW_DATA_DIR = self.raw_data_dir = RAW_DATA_DIR
        self.CLEAN_DATA_DIR = self.clean_data_dir = CLEAN_DATA_DIR
        self.SYNTHETIC_DATA_DIR = self.synthetic_data_dir = SYNTHETIC_DATA_DIR
        self.VOCAB_DIR = self.vocab_dir = VOCAB_DIR
        self.EXPERIMENTS_DIR = self.experiments_dir = EXPERIMENTS_DIR
        self.DEPLOY_DIR = self.deploy_dir = DEPLOY_DIR
        self.TENSORBOARD_DIR = self.tensorboard_dir = TENSORBOARD_DIR
        
        # 模型配置
        self.BERT_MODEL = self.bert_model = BERT_MODEL
        self.MAX_SEQ_LENGTH = self.max_seq_length = MAX_SEQ_LENGTH
        self.BERT_HIDDEN_SIZE = self.bert_hidden_size = BERT_HIDDEN_SIZE
        
        # GEC标签配置
        self.GEC_KEEP_LABEL = self.gec_keep_label = GEC_KEEP_LABEL
        self.GEC_DELETE_LABEL = self.gec_delete_label = GEC_DELETE_LABEL
        self.GEC_APPEND_PREFIX = self.gec_append_prefix = GEC_APPEND_PREFIX
        self.GEC_REPLACE_PREFIX = self.gec_replace_prefix = GEC_REPLACE_PREFIX
        
        # SVO标签配置
        self.SVO_LABELS = self.svo_labels = SVO_LABELS
        self.NUM_SVO_LABELS = self.num_svo_labels = NUM_SVO_LABELS
        self.TOP_FREQ_CHARS = self.top_freq_chars = TOP_FREQ_CHARS
        self.HIGH_FREQ_FUNCTION_WORDS = self.high_freq_function_words = HIGH_FREQ_FUNCTION_WORDS
        self.ENABLE_LABEL_COMPRESSION = self.enable_label_compression = ENABLE_LABEL_COMPRESSION
        
        # 训练配置
        self.BATCH_SIZE = self.batch_size = BATCH_SIZE
        self.NUM_EPOCHS = self.num_epochs = NUM_EPOCHS
        self.LEARNING_RATE = self.learning_rate = LEARNING_RATE
        self.WARMUP_RATIO = self.warmup_ratio = WARMUP_RATIO
        self.WEIGHT_DECAY = self.weight_decay = WEIGHT_DECAY
        self.MAX_GRAD_NORM = self.max_grad_norm = MAX_GRAD_NORM
        
        # 损失函数配置
        self.FOCAL_LOSS_GAMMA = self.focal_gamma = self.focal_loss_gamma = FOCAL_LOSS_GAMMA
        self.FOCAL_LOSS_ALPHA = self.focal_alpha = self.focal_loss_alpha = FOCAL_LOSS_ALPHA
        self.MTL_LAMBDA_SVO = self.mtl_lambda_svo = MTL_LAMBDA_SVO
        self.MTL_LAMBDA_SENT = self.mtl_lambda_sent = MTL_LAMBDA_SENT
        
        # Early Stopping
        self.PATIENCE = self.patience = PATIENCE
        self.MIN_DELTA = self.min_delta = MIN_DELTA
        
        # 数据增强配置
        self.AUG_PREPOSITION_ERROR_RATE = self.aug_preposition_error_rate = AUG_PREPOSITION_ERROR_RATE
        self.AUG_WORD_ORDER_ERROR_RATE = self.aug_word_order_error_rate = AUG_WORD_ORDER_ERROR_RATE
        self.AUG_CONFUSION_ERROR_RATE = self.aug_confusion_error_rate = AUG_CONFUSION_ERROR_RATE
        self.AUG_DELETION_ERROR_RATE = self.aug_deletion_error_rate = AUG_DELETION_ERROR_RATE
        self.AUG_INSERTION_ERROR_RATE = self.aug_insertion_error_rate = AUG_INSERTION_ERROR_RATE
        self.PREPOSITIONS = self.prepositions = PREPOSITIONS
        self.POLITICAL_CONFUSIONS = self.political_confusions = POLITICAL_CONFUSIONS
        
        # 评估配置
        self.EVAL_F_BETA = self.eval_f_beta = EVAL_F_BETA
        self.EVAL_F2_BETA = self.eval_f2_beta = EVAL_F2_BETA
        
        # 部署配置
        self.QUANTIZATION_DTYPE = self.quantization_dtype = QUANTIZATION_DTYPE
        self.ONNX_OPSET_VERSION = self.onnx_opset_version = ONNX_OPSET_VERSION
        self.INFERENCE_BATCH_SIZE = self.inference_batch_size = INFERENCE_BATCH_SIZE
        self.INFERENCE_DEVICE = self.inference_device = INFERENCE_DEVICE
        
        # 日志配置
        self.LOG_INTERVAL = self.log_interval = LOG_INTERVAL
        self.SAVE_INTERVAL = self.save_interval = SAVE_INTERVAL
        
        # GPU配置
        self.USE_CUDA = self.use_cuda = USE_CUDA
        self.NUM_WORKERS = self.num_workers = NUM_WORKERS
        
        # 混合精度与分布式训练配置
        self.USE_AMP = self.use_amp = USE_AMP
        self.GRADIENT_ACCUMULATION_STEPS = self.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
        
        # ==================== 新模块配置 ====================
        # 模块一：句法-语义融合交互层
        self.USE_SYNTAX_SEMANTIC_FUSION = self.use_syntax_semantic_fusion = USE_SYNTAX_SEMANTIC_FUSION
        self.SYNTAX_FUSION_USE_LAYER_NORM = self.syntax_fusion_use_layer_norm = SYNTAX_FUSION_USE_LAYER_NORM
        
        # 模块二：不确定性加权损失
        self.USE_UNCERTAINTY_WEIGHTING = self.use_uncertainty_weighting = USE_UNCERTAINTY_WEIGHTING
        self.UNCERTAINTY_INIT_LOG_VAR = self.uncertainty_init_log_var = UNCERTAINTY_INIT_LOG_VAR
        self.UNCERTAINTY_LR_MULTIPLIER = self.uncertainty_lr_multiplier = UNCERTAINTY_LR_MULTIPLIER
        
        # 模块三：错误感知多实例句级分类头
        self.USE_ERROR_AWARE_SENT_HEAD = self.use_error_aware_sent_head = USE_ERROR_AWARE_SENT_HEAD
        self.KEEP_LABEL_IDX = self.keep_label_idx = KEEP_LABEL_IDX
        self.DETACH_ERROR_CONFIDENCE = self.detach_error_confidence = DETACH_ERROR_CONFIDENCE
        
    def to_dict(self):
        """转换为字典，方便保存和序列化"""
        return {k: str(v) if isinstance(v, Path) else v 
                for k, v in self.__dict__.items()}
    
    def save(self, path):
        """保存配置到yaml文件"""
        import yaml
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True)
    
    @classmethod
    def load(cls, path):
        """从yaml文件加载配置"""
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        config = cls()
        for k, v in config_dict.items():
            setattr(config, k, v)
        return config
    
    def save_to_experiment(self, exp_dir: Path, filename: str = "config.yaml") -> Path:
        """
        将配置保存到实验目录（与tensorboard目录平级）
        
        Args:
            exp_dir: 实验目录路径，如 experiments/exp_xxx/
            filename: 配置文件名，默认为 config.yaml
        
        Returns:
            配置文件的完整路径
        """
        import yaml
        
        exp_dir = Path(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = exp_dir / filename
        
        # 过滤掉不需要序列化的属性（重复的小写别名）
        config_dict = {}
        for k, v in self.__dict__.items():
            # 只保留大写属性，避免重复
            if k.isupper() or k == 'project_root':
                if isinstance(v, Path):
                    config_dict[k] = str(v)
                elif isinstance(v, (dict, list)):
                    config_dict[k] = v
                else:
                    config_dict[k] = v
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        return config_path


# 创建默认配置实例
default_config = Config()


if __name__ == "__main__":
    # 测试配置
    print("配置测试:")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"BERT模型: {BERT_MODEL}")
    print(f"最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"SVO标签数: {NUM_SVO_LABELS}")
    print(f"Focal Loss Gamma: {FOCAL_LOSS_GAMMA}")
