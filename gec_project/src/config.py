"""
全局配置文件
定义所有超参数和路径配置
"""
import os
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
MAX_SEQ_LENGTH = 128  # 公文句子一般不会太长，128足够

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

# ==================== 训练配置 ====================
# 基础训练参数
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 损失函数参数
FOCAL_LOSS_GAMMA = 2.0      # Focal Loss的聚焦参数
FOCAL_LOSS_ALPHA = 0.25     # 针对$KEEP标签的权重
MTL_LAMBDA_SVO = 0.5        # 多任务学习中SVO任务的权重
MTL_LAMBDA_SENT = 0.3       # 多任务学习中句级错误检测任务的权重

# Early Stopping
PATIENCE = 5  # 验证集连续5个epoch没提升就停止
MIN_DELTA = 0.001  # 最小改善幅度

# ==================== 数据增强配置 ====================
# 错误生成比例
AUG_PREPOSITION_ERROR_RATE = 0.15   # 介词滥用错误比例
AUG_WORD_ORDER_ERROR_RATE = 0.10    # 词序错误比例
AUG_CONFUSION_ERROR_RATE = 0.20     # 混淆词错误比例
AUG_DELETION_ERROR_RATE = 0.10      # 删除错误比例
AUG_INSERTION_ERROR_RATE = 0.10     # 插入错误比例

# 公文特有词汇（用于构建混淆集和介词集）
PREPOSITIONS = ["通过", "经过", "在", "由于", "鉴于", "根据",
                "依据", "按照", "依照", "遵照", "据", "特别是",
                "为了", "为", "对于", "关于", "针对", "面向",
                "随着", "当", "值此", "之际","由", "被", "经",
                "结合", "伴随"]
POLITICAL_CONFUSIONS = {
    "权利": "权力",
    "权力": "权利",
    "制定": "制订",
    "制订": "制定",
    "启用": "起用",
    "起用": "启用",
    # 动作与结果/对象
    "反映": "反应",  # 公文常用来"反映情况"，而非"反应"
    "反应": "反映",
    "截至": "截止",  # 截至+具体时间点；截止+动词(报名已截止)
    "截止": "截至",
    "作出": "做出",  # 抽象名词多用"作"（作出决定）；具体动作多用"做"
    "做出": "作出",
    
    # 程度/逻辑关系
    "以至": "以致",  # 以至：程度加深；以致：导致（通常是不好）的结果
    "以致": "以至",
    "必须": "必需",  # 必须：一定要（副词）；必需：不可少（动词/形容词）
    "必需": "必须",
    
    # 审查/检查
    "考查": "考察",  # 考查：成绩/业务；考察：实地观察/干部任用
    "考察": "考查",
    "审定": "审订",  # 审定：决定/定稿；审订：审查修订
    "审订": "审定",
    
    # 资金/账目
    "账目": "帐目",  # 推荐使用"账"（贝字旁与钱有关）
    "帐目": "账目",
    "盈利": "营利",  # 盈利：赚了钱；营利：谋求利润（营利性机构）
    "营利": "盈利",
    
    # 范围/期间
    "期间": "其间",  # 期间：某个时期；其间：那中间
    "其间": "期间",
    "界限": "界线",  # 界限：抽象（思想界限）；界线：具体（国境界线）
    "界线": "界限",
    "部署": "布署",  # 常见错误写法
    "布署": "部署",
    "覆盖": "复盖",  # 常见简化错误
    "复盖": "覆盖",
    "宏大": "洪大",  # 规模宏大 vs 声音洪大
    "洪大": "宏大",
    "爆发": "暴发",  # 爆发战争/力量；暴发传染病/山洪
    "暴发": "爆发",
    "通信": "通讯",  # 通信产业/技术；通讯社/报道
    "通讯": "通信",
    "窜改": "篡改",  # 窜改：改动文字；篡改：伪造历史/理论
    "篡改": "窜改",
    "度过": "渡过",  # 度过时间；渡过难关（江河）
    "渡过": "度过",
    "合计": "核计",  # 合计：一共；核计：核算
    "核计": "合计",
    "启示": "启事",  # 获得启示；张贴启事
    "启事": "启示",
    "不但": "不单",
    "而且": "并且",
    "进而": "从而",  # 进而：进一步；从而：因此
    "从而": "进而",
    "颁布": "公布",  # 颁布法律/法令；公布消息/名单
    "公布": "颁布",
}

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
