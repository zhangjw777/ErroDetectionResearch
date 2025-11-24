"""
句法分析模块
使用LTP提取主谓宾成分，生成SVO标签
"""
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ltp import LTP
    LTP_AVAILABLE = True
except ImportError:
    LTP_AVAILABLE = False
    logger.warning("LTP not installed. SVO extraction will not work.")

# 尝试导入torch用于检测GPU
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SVOExtractor:
    """
    主谓宾提取器
    
    使用LTP进行依存句法分析，提取句子的核心成分：
    - Subject (主语): SBV
    - Predicate (谓语): HED (核心动词)
    - Object (宾语): VOB
    
    输出BIO格式的标签：
    - B-SUB, I-SUB (主语)
    - B-PRED, I-PRED (谓语)
    - B-OBJ, I-OBJ (宾语)
    - O (其他)
    """
    
    def __init__(self, use_cuda: bool = True):
        """
        Args:
            use_cuda: 是否使用GPU加速（默认True，会自动检测GPU是否可用）
        """
        if not LTP_AVAILABLE:
            raise ImportError("Please install ltp: pip install ltp")
        
        # LTP GPU支持
        # 只有在use_cuda=True且GPU可用时才使用GPU
        if TORCH_AVAILABLE and use_cuda:
            actual_use_cuda = torch.cuda.is_available()
        else:
            actual_use_cuda = False
        
        # 初始化LTP，默认使用base模型
        # LTP 4.x 版本的正确用法
        device = "cuda:0" if actual_use_cuda else "cpu"
        self.ltp = LTP("LTP/base")
        if actual_use_cuda:
            self.ltp.to(device)
        device_info = "GPU" if actual_use_cuda else "CPU"
        logger.info(f"SVOExtractor initialized with LTP on {device_info}")
    
    def extract(self, text: str) -> Tuple[List[str], List[str]]:
        """
        提取句子的SVO成分
        
        Args:
            text: 输入句子
        
        Returns:
            (tokens, svo_labels)
            例如：(['我', '们', '学', '习'], ['B-SUB', 'I-SUB', 'B-PRED', 'O'])
        """
        # 使用LTP分析
        # cws：中文分词,dep：依存句法分析
        result = self.ltp.pipeline([text], tasks=["cws", "dep"])
        
        if not result or not result.cws or len(result.cws) == 0:
            # 解析失败，返回全O标签
            tokens = list(text)
            labels = ['O'] * len(tokens)
            return tokens, labels
        
        # 获取第一个句子的解析结果
        words = result.cws[0]  # 分词结果
        dep_info = result.dep[0]  # 依存分析结果 {'head': [2, 0], 'label': ['SBV', 'HED']}
        deprels = dep_info['label']  # 依存关系标签列表
        heads = dep_info['head']  # 中心词索引列表
        
        # 初始化标签（词级别）
        word_labels = ['O'] * len(words)
        
        # 找到谓语（核心动词，HED）
        predicate_indices = []
        for i, deprel in enumerate(deprels):
            if deprel == 'HED':
                predicate_indices.append(i)
        
        # 找到主语（SBV）
        subject_indices = []
        for i, deprel in enumerate(deprels):
            if deprel == 'SBV':
                subject_indices.append(i)
        
        # 找到宾语（VOB）
        object_indices = []
        for i, deprel in enumerate(deprels):
            if deprel == 'VOB':
                object_indices.append(i)
        
        # 标注主语
        for idx in subject_indices:
            word_labels[idx] = 'B-SUB'
        
        # 标注谓语
        for idx in predicate_indices:
            word_labels[idx] = 'B-PRED'
        
        # 标注宾语
        for idx in object_indices:
            word_labels[idx] = 'B-OBJ'
        
        # 将词级别的标签映射到字符级别
        char_labels = []
        char_tokens = []
        
        for word, label in zip(words, word_labels):
            chars = list(word)
            char_tokens.extend(chars)
            
            if label == 'O':
                char_labels.extend(['O'] * len(chars))
            else:
                # 第一个字符用B-，其余用I-
                tag_type = label.split('-')[1]  # SUB, PRED, OBJ
                char_labels.append(f'B-{tag_type}')
                char_labels.extend([f'I-{tag_type}'] * (len(chars) - 1))
        
        return char_tokens, char_labels
    
    def extract_batch(self, texts: List[str]) -> List[Tuple[List[str], List[str]]]:
        """批量提取"""
        return [self.extract(text) for text in texts]


def generate_svo_labels_for_dataset(
    sentences: List[str],
    output_path: str = None,
    use_cuda: bool = True
) -> List[Dict]:
    """
    为数据集生成SVO标签
    
    Args:
        sentences: 句子列表
        output_path: 输出路径（可选）
        use_cuda: 是否使用GPU加速LTP（默认True）
    
    Returns:
        带有SVO标签的数据列表
    """
    if not LTP_AVAILABLE:
        logger.error("LTP not available, cannot generate SVO labels")
        return []
    
    extractor = SVOExtractor(use_cuda=use_cuda)
    results = []
    
    for i, sent in enumerate(sentences):
        tokens, svo_labels = extractor.extract(sent)
        results.append({
            'uid': f'sent_{i}',
            'text': sent,
            'tokens': tokens,
            'svo_labels': svo_labels
        })
    
    logger.info(f"Generated SVO labels for {len(results)} sentences")
    
    # 保存到文件
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved to {output_path}")
    
    return results


if __name__ == "__main__":
    # 测试SVO提取
    if not LTP_AVAILABLE:
        print("Please install ltp first: pip install ltp")
    else:
        # 测试时默认使用CPU（避免GPU问题）
        extractor = SVOExtractor(use_cuda=False)
        
        test_sentences = [
            "我们学习新知识。",
            "人民群众的权利得到保障。",
            "通过这次活动，使我们认识到了错误。"
        ]
        
        print("=== SVO提取测试 ===\n")
        for sent in test_sentences:
            tokens, labels = extractor.extract(sent)
            print(f"句子: {sent}")
            print(f"Token: {tokens}")
            print(f"SVO: {labels}")
            print()
