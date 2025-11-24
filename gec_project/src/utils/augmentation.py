"""
错误生成模块（数据增强）
根据公文特点生成各类错误样本
"""
import random
import jieba
from typing import List, Dict, Tuple, Optional
import logging

from config import (
    PREPOSITIONS, 
    POLITICAL_CONFUSIONS,
    AUG_PREPOSITION_ERROR_RATE,
    AUG_WORD_ORDER_ERROR_RATE,
    AUG_CONFUSION_ERROR_RATE,
    AUG_DELETION_ERROR_RATE,
    AUG_INSERTION_ERROR_RATE
)

# 尝试导入LTP用于精准删除
try:
    from ltp import LTP
    LTP_AVAILABLE = True
except ImportError:
    LTP_AVAILABLE = False
    logging.warning("LTP not installed. Precise component deletion will not work.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ErrorGenerator:
    """
    错误生成器
    
    针对公文领域的常见错误类型：
    1. 介词滥用（如句首多余的"通过"）
    2. 混淆词错误（权利/权力、制定/制订）
    3. 精准成分删除（删除主语/谓语/宾语）
    4. 虚词删除（删除介词/助词/连词）
    5. 插入错误（多余成分或重复字）
    """
    
    def __init__(self, confusion_dict: Dict[str, str] = None, use_ltp: bool = True):
        """
        Args:
            confusion_dict: 混淆词对，如 {"权利": "权力", "制定": "制订"}
            use_ltp: 是否使用LTP进行精准成分删除（默认False，避免性能问题）
        """
        self.confusion_dict = confusion_dict or POLITICAL_CONFUSIONS
        self.prepositions = PREPOSITIONS
        
        # 构建反向混淆字典（双向）
        self.confusion_pairs = {}
        for k, v in self.confusion_dict.items():
            self.confusion_pairs[k] = v
            if v not in self.confusion_pairs:
                self.confusion_pairs[v] = k
        
        # 虚词候选集（用于删除）
        self.function_words = ['的', '地', '得', '了', '是', '在', '对', '于', '和', '与', '及', '或', '而', '以', '为', '把', '被', '将', '给']
        
        # LTP初始化（仅在需要时）
        self.ltp = None
        if use_ltp and LTP_AVAILABLE:
            try:
                self.ltp = LTP("LTP/base")
                logger.info("LTP initialized for precise component deletion")
            except Exception as e:
                logger.warning(f"Failed to initialize LTP: {e}")
    
    def generate_errors(
        self,
        text: str,
        error_types: List[str] = None,
        max_errors: int = 3
    ) -> Tuple[str, List[Dict]]:
        """
        为正确文本生成错误
        
        Args:
            text: 原始正确文本
            error_types: 要生成的错误类型列表
            max_errors: 最大错误数
        
        Returns:
            (错误文本, 错误位置信息列表)
        """
        if error_types is None:
            error_types = [
                'preposition',
                'confusion',
                'component_deletion',  # 替代word_order
                'preposition_deletion',  # 新增虚词删除
                'insertion'
            ]
        
        # 转为token列表
        tokens = list(text)
        errors_applied = []
        
        # 随机选择要应用的错误类型
        num_errors = random.randint(1, max_errors)
        selected_types = random.choices(error_types, k=num_errors)
        
        for error_type in selected_types:
            if error_type == 'preposition':
                tokens, error_info = self._add_preposition_error(tokens)
            elif error_type == 'confusion':
                tokens, error_info = self._add_confusion_error(tokens)
            elif error_type == 'component_deletion':
                tokens, error_info = self._add_component_deletion_error(text, tokens)
            elif error_type == 'preposition_deletion':
                tokens, error_info = self._add_preposition_deletion(tokens)
            elif error_type == 'insertion':
                tokens, error_info = self._add_insertion_error(tokens)
            else:
                continue
            
            if error_info:
                errors_applied.append(error_info)
        
        error_text = ''.join(tokens)
        return error_text, errors_applied # 返回错误文本和错误信息列表
    
    def _add_preposition_error(
        self,
        tokens: List[str]
    ) -> Tuple[List[str], Dict]:
        """
        添加介词滥用错误
        策略：在句首插入多余的介词短语
        """
        if random.random() > AUG_PREPOSITION_ERROR_RATE:
            return tokens, None
        
        # 随机选择一个介词
        prep = random.choice(self.prepositions)
        
        # 在句首插入
        if len(tokens) > 0:
            tokens = list(prep) + tokens
            return tokens, {
                'type': 'preposition_insertion',
                'position': 0,
                'inserted': prep
            }
        
        return tokens, None
    
    def _add_confusion_error(
        self,
        tokens: List[str]
    ) -> Tuple[List[str], Dict]:
        """
        添加混淆词错误
        策略：查找混淆词对并替换
        """
        if random.random() > AUG_CONFUSION_ERROR_RATE:
            return tokens, None
        
        text = ''.join(tokens)
        
        # 查找可替换的混淆词
        for word, confuse_word in self.confusion_pairs.items():
            if word in text:
                # 找到位置并替换
                pos = text.find(word)
                new_tokens = list(text.replace(word, confuse_word, 1))
                return new_tokens, {
                    'type': 'confusion',
                    'position': pos,
                    'original': word,
                    'replaced': confuse_word
                }
        
        return tokens, None
    
    def _add_component_deletion_error(
        self,
        original_text: str,
        tokens: List[str]
    ) -> Tuple[List[str], Optional[Dict]]:
        """
        添加成分删除错误（精准删除主语/谓语/宾语）
        策略：使用LTP识别句法成分，随机删除主语或谓语
        """
        if random.random() > AUG_WORD_ORDER_ERROR_RATE:
            return tokens, None
        
        # 如果没有LTP，退化为随机删除词
        if not self.ltp:
            text = ''.join(tokens)
            words = list(jieba.cut(text))
            if len(words) > 3:
                del_idx = random.randint(0, len(words) - 1)
                deleted_word = words[del_idx]
                words.pop(del_idx)
                new_text = ''.join(words)
                return list(new_text), {
                    'type': 'component_deletion',
                    'deleted': deleted_word,
                    'method': 'random'
                }
            return tokens, None
        
        # 使用LTP进行精准删除
        try:
            result = self.ltp.pipeline([original_text], tasks=["cws", "dep"])
            if not result or not result.cws or len(result.cws) == 0:
                return tokens, None
            
            words = result.cws[0]
            dep_info = result.dep[0]
            deprels = dep_info['label']
            
            # 找到主语(SBV)和谓语(HED)
            subject_indices = [i for i, rel in enumerate(deprels) if rel == 'SBV']
            predicate_indices = [i for i, rel in enumerate(deprels) if rel == 'HED']
            
            # 随机选择删除主语或谓语
            candidates = []
            if subject_indices:
                candidates.extend([('subject', idx) for idx in subject_indices])
            if predicate_indices:
                candidates.extend([('predicate', idx) for idx in predicate_indices])
            
            if not candidates:
                return tokens, None
            
            comp_type, word_idx = random.choice(candidates)
            deleted_word = words[word_idx]
            
            # 删除该词
            new_words = words[:word_idx] + words[word_idx + 1:]
            new_text = ''.join(new_words)
            
            return list(new_text), {
                'type': 'component_deletion',
                'component': comp_type,
                'deleted': deleted_word,
                'position': word_idx,
                'method': 'ltp'
            }
        except Exception as e:
            logger.debug(f"LTP component deletion failed: {e}")
            return tokens, None
    
    def _add_preposition_deletion(
        self,
        tokens: List[str]
    ) -> Tuple[List[str], Optional[Dict]]:
        """
        添加虚词/介词/助词删除错误
        策略：优先删除虚词候选集中的字符
        """
        if random.random() > AUG_DELETION_ERROR_RATE:
            return tokens, None
        
        # 找到所有属于虚词候选集的位置
        function_word_indices = [
            i for i, t in enumerate(tokens)
            if t in self.function_words
        ]
        
        if function_word_indices:
            # 优先删除虚词
            del_idx = random.choice(function_word_indices)
            deleted_char = tokens[del_idx]
            new_tokens = tokens[:del_idx] + tokens[del_idx+1:]
            return new_tokens, {
                'type': 'preposition_deletion',
                'position': del_idx,
                'deleted': deleted_char
            }
        
        # 如果没有虚词，不做修改
        return tokens, None
    
    def _add_insertion_error(
        self,
        tokens: List[str]
    ) -> Tuple[List[str], Optional[Dict]]:
        """
        添加插入错误
        策略：在随机位置插入常见字符或重复字
        """
        if random.random() > AUG_INSERTION_ERROR_RATE:
            return tokens, None
        
        if len(tokens) <= 3:
            return tokens, None
        
        # 50%概率插入重复字，50%概率插入常见虚词
        if random.random() < 0.5:
            # 插入重复字：随机选择一个位置，重复该位置的字符
            ins_idx = random.randint(1, len(tokens) - 2)
            repeat_char = tokens[ins_idx]
            new_tokens = tokens[:ins_idx] + [repeat_char] + tokens[ins_idx:]
            return new_tokens, {
                'type': 'insertion',
                'subtype': 'repeat',
                'position': ins_idx,
                'inserted': repeat_char
            }
        else:
            # 插入常见虚词
            common_chars = ['的', '了', '和', '与', '及', '等', '而', '或']
            ins_idx = random.randint(1, len(tokens) - 2)
            ins_char = random.choice(common_chars)
            new_tokens = tokens[:ins_idx] + [ins_char] + tokens[ins_idx:]
            return new_tokens, {
                'type': 'insertion',
                'subtype': 'common_word',
                'position': ins_idx,
                'inserted': ins_char
            }
        
        return tokens, None


def generate_training_samples(
    correct_sentences: List[str],
    num_samples_per_sentence: int = 3
) -> List[Dict]:
    """
    批量生成训练样本
    
    Args:
        correct_sentences: 正确句子列表
        num_samples_per_sentence: 每个句子生成的错误样本数
    
    Returns:
        样本列表，每个样本包含 source, target, errors
    """
    generator = ErrorGenerator()
    samples = []
    
    for sent_id, correct_sent in enumerate(correct_sentences):
        for sample_id in range(num_samples_per_sentence):
            # 生成错误
            error_sent, errors = generator.generate_errors(correct_sent)
            
            # 如果成功生成错误（至少有一个错误）
            if errors:
                samples.append({
                    'uid': f'sent_{sent_id}_sample_{sample_id}',
                    'source': error_sent,
                    'target': correct_sent,
                    'errors': errors
                })
    
    logger.info(f"Generated {len(samples)} training samples from {len(correct_sentences)} sentences")
    return samples


if __name__ == "__main__":
    # 测试错误生成
    generator = ErrorGenerator()
    
    test_sentences = [
        "我们要认真学习贯彻会议精神。",
        "这是一项重要的工作任务。",
        "人民群众的权利得到保障。"
    ]
    
    print("=== 错误生成测试 ===\n")
    for sent in test_sentences:
        error_sent, errors = generator.generate_errors(sent, max_errors=2)
        print(f"原句: {sent}")
        print(f"错句: {error_sent}")
        print(f"错误: {errors}")
        print()
