"""
数据预处理脚本 (最终融合版)
功能：
1. 清洗原始公文语料 (高精度去噪 + 强力过滤)
2. 生成错误样本
3. 生成SVO标签
4. 构建GECToR标签映射
5. 划分训练集/验证集
"""
import json
import random
import re
from pathlib import Path
from tqdm import tqdm
import logging
from typing import List, Dict, Iterable, Tuple, Optional
import argparse
import difflib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import default_config as cfg
from utils.augmentation import ErrorGenerator
from utils.svo_extract import SVOExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 核心清洗工具模块 (Fusion Edition)
# ==========================================

# 1. 常量定义
CONTACT_KEYWORDS = (
    '联系人', '联系电话', '邮箱', '电子邮箱', '邮箱地址',
    '邮寄地址', '传真', 'QQ', '微信', '咨询电话', '监督电话'
)

ADDRESS_KEYWORDS = (
    '省', '市', '区', '县', '乡', '镇', '村',
    '街道', '路', '巷', '号', '栋', '室',
    '大厦', '园', '楼', '单元', '层'
)

# 2. 正则预编译
PHONE_PAT  = re.compile(r'(1[3-9]\d{9})|(\d{3,4}-\d{7,8})')
EMAIL_PAT  = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,6}')
# 使用 [URL] 替换，避免 <URL> 被 BERT 分词器切碎
URL_PAT    = re.compile(r'https?://[^\s)）]+')
# 去除起止的引号（中文或英文）
QUOTES_PAT = re.compile(r'^[\'"“”]+|[\'"“”]+$')
# 汉字匹配
CHINESE_CHAR_PAT = re.compile(r'[\u4e00-\u9fa5]')


def _strip_leading_meta(text: str) -> str:
    """
    剥离句子起始的元数据（版式、署名、时间、枚举）
    采用 While 循环模式 + Copilot 的高精度正则
    解决嵌套前缀问题，例如："(责任单位：xxx) 22. 加强..." -> "22. 加强..." -> "加强..."
    """
    prev_text = None
    curr_text = text.strip()
    
    # 限制最大循环次数，防止死循环
    loop_count = 0
    while prev_text != curr_text and loop_count < 6:
        prev_text = curr_text
        t = curr_text
        
        # 1) 起始括注：全角（）/[]/【】块
        # 使用非贪婪匹配，防止吃掉正文
        t = re.sub(r'^(（[^）]{1,60}）\s*)', '', t)
        t = re.sub(r'^(\[[^\]]{1,60}\]\s*)', '', t)
        t = re.sub(r'^【[^】]{1,30}】\s*', '', t)
        
        # 2) 起始日期时间（支持 2 位或 4 位年份）
        # 匹配: 2024-03-29 15:54:07 郭启文
        t = re.sub(
            r'^\s*\d{2,4}[-/年]\d{1,2}[-/月]\d{1,2}'
            r'(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?\s*',
            '', 
            t
        )
        
        # 3) 记者 / 报纸署名 / 图片署名
        # 匹配: "□记者...", "本报...讯", "新华社记者.../摄"
        t = re.sub(r'^□?记者[\u4e00-\u9fa5\s]{0,10}(报道|报导)\s*', '', t)
        t = re.sub(r'^(?:新华社记者|中国质量报[^：:]{0,20}记者站记者|本报[^：:]{0,20}讯)[:：]?\s*', '', t)
        t = re.sub(r'^[^，。；：:]{0,12}?报道[：:\s]*', '', t) # 开头的「xxx报道」
        
        # 4) 责任单位 / 来源前缀
        t = re.sub(r'^(责任单位|发布单位|来源|作者)[:：]\s*[^，。；]{1,30}[，。；]?\s*', '', t)
        
        # 5) 起始枚举标记（第X、（一）、1. 等）
        t = re.sub(r'^((第?[一二三四五六七八九十百千万]+[、.，,])|（[一二三四五六七八九十]+）|[(（]?\d{1,3}[)）]?[\.、])\s*', '', t)
        
        curr_text = t.strip()
        loop_count += 1
        
    # 额外清理：文中可能残留的图片署名 "xxx/摄"
    curr_text = re.sub(r'\s*[\u4e00-\u9fa5]{1,6}/摄\s*', '', curr_text)
    curr_text = re.sub(r'\s*新华社记者[\u4e00-\u9fa5\s]{0,10}/摄', '', curr_text)
    
    return curr_text


def normalize_sentence(text: str) -> str:
    """句子规范化主入口：去壳保句"""
    if not text:
        return ""
    
    # 1. HTML 清洗 (防止漏网之鱼)
    t = clean_html_tags(text)
    
    # 2. 剥离前缀噪声 (洋葱剥皮)
    t = _strip_leading_meta(t)
    
    # 3. 去掉起止引号
    t = QUOTES_PAT.sub('', t)
    
    # 4. URL 归一化 (使用 [URL] 更安全)
    t = URL_PAT.sub('[URL]', t)
    
    # 5. 压缩空白
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def should_filter_sentence_core(text: str) -> bool:
    """
    在已经 normalize 后的句子上做噪声过滤
    返回 True 表示需要丢弃
    """
    if not text:
        return True

    # 预计算统计量
    total_len = max(1, len(text))
    digit_count = sum(ch.isdigit() for ch in text)
    digit_ratio = digit_count / total_len
    chinese_chars = len(CHINESE_CHAR_PAT.findall(text))
    
    contact_hits = sum(1 for k in CONTACT_KEYWORDS if k in text)
    has_phone = bool(PHONE_PAT.search(text))
    has_email = bool(EMAIL_PAT.search(text))

    # 1) 强联系方式 / 通联信息过滤
    if contact_hits >= 2:
        return True
    if text.startswith(('联系人', '联系电话', '电话：', '邮寄地址', '地址：', '通信地址')):
        return True
    if has_phone and has_email:
        return True
    if (has_phone or has_email) and contact_hits >= 1:
        return True
    # QQ / 微信 + 明显数字 => 多半是联系方式
    if ('QQ' in text or '微信' in text) and digit_ratio > 0.2:
        return True

    # 2) 地址型噪声：地址关键词密集 + 数字占比适中
    addr_kw_count = sum(1 for k in ADDRESS_KEYWORDS if k in text)
    if addr_kw_count >= 3 and digit_ratio > 0.15:
        return True

    # 3) 高度数字化（表格编号、代码片段等）
    if digit_ratio > 0.45:
        return True

    # 4) 有效汉字过少 / 汉字比例过低
    if chinese_chars < 6: # 绝对数量检查
        return True
    if chinese_chars / total_len < 0.5: # 密度检查
        return True

    return False


def clean_and_filter_sentence(text: str) -> Optional[str]:
    """
    统一对外接口：
      - 返回规范化后的句子字符串：可用于 clean 文件 / 训练
      - 返回 None：认为是垃圾样本，丢弃
    """
    # 1. 规范化 (去皮)
    t = normalize_sentence(text)
    if not t:
        return None

    # 2. 长度过滤 (去尾)
    # 设定为 10，保留 "加强公共法律服务建设。" 但丢弃 "鼓励探索创新。"
    if not (10 <= len(t) <= 250):
        return None
        
    # 3. 噪声规则过滤 (去核)
    if should_filter_sentence_core(t):
        return None

    return t

# ==========================================
# 工具模块结束
# ==========================================


def _yield_json_documents(file_path: Path) -> Iterable[Dict]:
    """Yield documents from either JSONL or JSON files."""
    suffix = file_path.suffix.lower()

    if suffix == '.jsonl':
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if isinstance(doc, dict):
                        yield doc
                except json.JSONDecodeError:
                    pass
        return

    # Handle standard JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except: pass
            return

    def _emit_from_container(container):
        if isinstance(container, list):
            for item in container:
                if isinstance(item, dict):
                    yield item
        elif isinstance(container, dict):
            candidate_keys = ['data', 'documents', 'items', 'results']
            for key in candidate_keys:
                value = container.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield container

    yield from _emit_from_container(data)


def clean_html_tags(text: str) -> str:
    """清理HTML标签和特殊字符"""
    block_tags = r'</?(p|div|br|h[1-6]|li|ul|ol|table|tr|td|th)[^>]*>'
    text = re.sub(block_tags, ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    zero_width_chars = '\u200b\u200c\u200d\ufeff\u00ad'
    for char in zero_width_chars:
        text = text.replace(char, '')
    return text


def clean_raw_data(raw_dir: Path, output_path: Path):
    """
    清洗原始公文数据
    """
    logger.info(f"Cleaning raw data from {raw_dir}")
    
    clean_sentences = set() # 使用set自动去重
    
    data_files = sorted(raw_dir.glob("*.jsonl")) + sorted(raw_dir.glob("*.json"))
    if not data_files:
        logger.warning(f"No .jsonl or .json files found in {raw_dir}")
        return []

    for data_file in data_files:
        # logger.info(f"Processing file: {data_file.name}") # 减少日志输出
        for doc in _yield_json_documents(data_file):
            try:
                content = doc.get('contentText', '')
                if not content:
                    continue

                # 基础切分 (按标点粗分)
                sentences = content.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')

                for sent in sentences:
                    # === 核心调用：清洗与过滤 ===
                    processed = clean_and_filter_sentence(sent)
                    if processed is not None:
                        clean_sentences.add(processed)
                    
            except Exception as e:
                continue
    
    # 转回列表并排序
    final_sentences = sorted(list(clean_sentences))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in final_sentences:
            f.write(sent + '\n')
    
    logger.info(f"Cleaned {len(final_sentences)} sentences, saved to {output_path}")
    return final_sentences


def align_tokens_with_difflib(source_tokens: List[str], target_tokens: List[str], target_svo_labels: List[str]) -> Tuple[List[str], List[str]]:
    """使用difflib对齐，生成 GEC 标签，并将 Target 的 SVO 标签投射到 Source 上"""
    matcher = difflib.SequenceMatcher(None, source_tokens, target_tokens, autojunk=False)
    gec_labels = []
    source_svo_labels = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for k in range(i2 - i1):
                gec_labels.append(cfg.GEC_KEEP_LABEL)
                source_svo_labels.append(target_svo_labels[j1 + k])
        elif tag == 'replace':
            src_len = i2 - i1
            tgt_len = j2 - j1
            for k in range(src_len):
                if k < tgt_len:
                    replace_char = target_tokens[j1 + k]
                    if k == src_len - 1 and src_len < tgt_len:
                        remaining = "".join(target_tokens[j1 + k + 1 : j2])
                        full_replacement = replace_char + remaining
                        if cfg.ENABLE_LABEL_COMPRESSION and full_replacement not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}MASK")
                        else:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}{full_replacement}")
                    else:
                        if cfg.ENABLE_LABEL_COMPRESSION and replace_char not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}MASK")
                        else:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}{replace_char}")
                    source_svo_labels.append(target_svo_labels[j1 + k])
                else:
                    gec_labels.append(cfg.GEC_DELETE_LABEL)
                    source_svo_labels.append('O')
        elif tag == 'delete':
            for k in range(i2 - i1):
                gec_labels.append(cfg.GEC_DELETE_LABEL)
                source_svo_labels.append('O')
        elif tag == 'insert':
            insert_content = "".join(target_tokens[j1:j2])
            if i1 > 0 and len(gec_labels) > 0:
                last_label = gec_labels[-1]
                if last_label == cfg.GEC_KEEP_LABEL:
                    if cfg.ENABLE_LABEL_COMPRESSION and insert_content not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                        gec_labels[-1] = f"{cfg.GEC_APPEND_PREFIX}MASK"
                    else:
                        gec_labels[-1] = f"{cfg.GEC_APPEND_PREFIX}{insert_content}"
            pass

    return gec_labels, source_svo_labels


def build_gec_label_vocab(samples: List[Dict], output_path: Path):
    """构建GEC标签词表"""
    label_set = set()
    label_set.add(cfg.GEC_KEEP_LABEL)
    label_set.add(cfg.GEC_DELETE_LABEL)
    for sample in samples:
        for label in sample['gec_labels']:
            label_set.add(label)
    
    label_set.discard(cfg.GEC_KEEP_LABEL)
    labels = [cfg.GEC_KEEP_LABEL] + sorted(list(label_set))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
    logger.info(f"Built GEC label vocab with {len(labels)} labels")


def build_svo_label_vocab(output_path: Path):
    """构建SVO标签词表"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in cfg.SVO_LABELS:
            f.write(label + '\n')


def generate_training_data(clean_sentences: List[str], output_dir: Path, train_ratio: float = 0.9, num_samples_per_sentence: int = 2, use_cuda: bool = False):
    """生成训练数据"""
    error_gen = ErrorGenerator()
    svo_extractor = SVOExtractor(use_cuda=use_cuda)
    all_samples = []
    
    for sent_id, correct_sent in enumerate(tqdm(clean_sentences, desc="Generating samples")):
        # === 兜底清洗 (Double Check) ===
        # 即使从文件中读取，也再过一遍过滤器，防止旧数据污染
        correct_sent = clean_and_filter_sentence(correct_sent)
        if not correct_sent:
            continue
            
        # 去除零宽字符
        zero_width_chars = '\u200b\u200c\u200d\ufeff\u00ad'
        for char in zero_width_chars:
            correct_sent = correct_sent.replace(char, '')
        
        try:
            # 1. 获取 Target (正确句)
            target_tokens, target_svo_labels = svo_extractor.extract(correct_sent)
            
            if ''.join(target_tokens) != correct_sent:
                continue

            # 加入正确样本
            clean_sample = {
                'uid': f'sent_{sent_id}_clean',
                'text': correct_sent,
                'tokens': list(correct_sent),
                'gec_labels': [cfg.GEC_KEEP_LABEL] * len(correct_sent),
                'svo_labels': target_svo_labels,
                'sent_has_error': 0
            }
            all_samples.append(clean_sample)
            
            # 2. 生成错误样本
            for sample_id in range(num_samples_per_sentence):
                error_sent, errors = error_gen.generate_errors(correct_sent, max_errors=2)
                
                if not errors or error_sent == correct_sent:
                    continue
                
                source_tokens = list(error_sent)
                gec_labels, source_svo_labels = align_tokens_with_difflib(source_tokens, target_tokens, target_svo_labels)
                
                if len(gec_labels) != len(source_tokens):
                    continue

                sample = {
                    'uid': f'sent_{sent_id}_sample_{sample_id}',
                    'text': error_sent,
                    'tokens': source_tokens,
                    'gec_labels': gec_labels,
                    'svo_labels': source_svo_labels,
                    'sent_has_error': 1
                }
                all_samples.append(sample)
            
        except Exception as e:
            continue
    
    # 划分与保存
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    dev_samples = all_samples[split_idx:]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    with open(output_dir / 'dev.json', 'w', encoding='utf-8') as f:
        json.dump(dev_samples, f, ensure_ascii=False, indent=2)
    
    build_gec_label_vocab(all_samples, cfg.VOCAB_DIR / 'label_map.txt')
    build_svo_label_vocab(cfg.VOCAB_DIR / 'svo_labels.txt')
    
    logger.info(f"Generated {len(all_samples)} samples (Train: {len(train_samples)}, Dev: {len(dev_samples)})")


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument('--raw_dir', type=str, default=str(cfg.RAW_DATA_DIR), help="Raw data directory")
    parser.add_argument('--output_dir', type=str, default=str(cfg.SYNTHETIC_DATA_DIR), help="Output directory")
    parser.add_argument('--num_samples', type=int, default=2, help="Samples per sentence")
    parser.add_argument('--use_cuda', action='store_true', default=False, help="Use GPU")
    parser.add_argument('--max_sentences', type=int, default=None, help="Max sentences")
    
    args = parser.parse_args()
    
    # GPU check
    use_cuda = args.use_cuda
    try:
        import torch
        if use_cuda and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")
            use_cuda = False
    except ImportError:
        use_cuda = False

    # 1. Clean Data
    clean_path = cfg.CLEAN_DATA_DIR / 'clean_sentences.txt'
    # 总是重新清洗，确保使用最新规则
    clean_sentences = clean_raw_data(Path(args.raw_dir), clean_path)
    
    # 2. Generate Data
    if args.max_sentences:
        clean_sentences = clean_sentences[:args.max_sentences]
        
    generate_training_data(
        clean_sentences,
        output_dir=Path(args.output_dir),
        num_samples_per_sentence=args.num_samples,
        use_cuda=use_cuda
    )

if __name__ == "__main__":
    main()