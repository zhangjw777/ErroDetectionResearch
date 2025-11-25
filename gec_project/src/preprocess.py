"""
数据预处理脚本
功能：
1. 清洗原始公文语料
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
from typing import List, Dict, Iterable, Tuple
import argparse
import difflib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import default_config as cfg
from utils.augmentation import ErrorGenerator, generate_training_samples
from utils.svo_extract import SVOExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
                    else:
                        logger.warning(f"Skipping non-dict entry in {file_path} line {line_no}")
                except json.JSONDecodeError as exc:
                    logger.warning(f"JSON decode error in {file_path} line {line_no}: {exc}")
        return

    # Default handler assumes standard JSON structure
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback: treat as JSONL content despite extension name
            f.seek(0)
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    if isinstance(doc, dict):
                        yield doc
                    else:
                        logger.warning(f"Skipping non-dict entry in {file_path} line {line_no}")
                except json.JSONDecodeError as exc:
                    logger.warning(f"JSON decode error in {file_path} line {line_no}: {exc}")
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
    """
    清理HTML标签和特殊字符
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    # 将块级标签转换为空格（避免内容粘连）
    block_tags = r'</?(p|div|br|h[1-6]|li|ul|ol|table|tr|td|th)[^>]*>'
    text = re.sub(block_tags, ' ', text, flags=re.IGNORECASE)
    
    # 删除其他HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 将HTML实体转换为空格 (如&nbsp;, &lt;等)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = re.sub(r'&#\d+;', ' ', text)
    
    # 删除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def clean_raw_data(raw_dir: Path, output_path: Path):
    """
    清洗原始公文数据
    
    输入：data/raw/ 目录下的jsonl文件
    输出：data/clean/ 目录下每行一个正确句子的txt文件
    """
    logger.info(f"Cleaning raw data from {raw_dir}")
    
    clean_sentences = []
    
    data_files = sorted(raw_dir.glob("*.jsonl")) + sorted(raw_dir.glob("*.json"))
    if not data_files:
        logger.warning(f"No .jsonl or .json files found in {raw_dir}")
        return []

    for data_file in data_files:
        for doc in _yield_json_documents(data_file):
            try:
                content = doc.get('contentText', '')

                if not content:
                    continue

                content = clean_html_tags(content)

                if not content:
                    continue

                sentences = content.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')

                for sent in sentences:
                    sent = sent.strip()
                    if 20 <= len(sent) <= 200:
                        clean_sentences.append(sent)
            except Exception as e:
                logger.warning(f"Error processing document in {data_file}: {e}")
                continue
    
    # 去重
    clean_sentences = list(set(clean_sentences))
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sent in clean_sentences:
            f.write(sent + '\n')
    
    logger.info(f"Cleaned {len(clean_sentences)} sentences, saved to {output_path}")
    return clean_sentences


def align_tokens_with_difflib(
    source_tokens: List[str],
    target_tokens: List[str],
    target_svo_labels: List[str]  # 新增：传入 Target 的 SVO 标签
) -> Tuple[List[str], List[str]]:
    """
    使用difflib对齐，生成 GEC 标签，并将 Target 的 SVO 标签投射到 Source 上。
    这是实现“认知冲突”的关键步骤。
    
    Returns:
        gec_labels: Source 的编辑标签
        source_svo_labels: Source 的句法标签（来自 Target 的投影）
    """
    matcher = difflib.SequenceMatcher(None, source_tokens, target_tokens,autojunk=False)
    
    gec_labels = []
    source_svo_labels = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # source[i1:i2] vs target[j1:j2]
        
        if tag == 'equal':
            # === 情况1: 完全匹配 ===
            # GEC: KEEP
            # SVO: 直接继承 Target 对应位置的标签
            for k in range(i2 - i1):
                gec_labels.append(cfg.GEC_KEEP_LABEL)
                source_svo_labels.append(target_svo_labels[j1 + k])
        
        elif tag == 'replace':
            # === 情况2: 错别字/词混淆 ===
            # Source: "权力" (i1:i2) -> Target: "权利" (j1:j2) (假设 target_svo 为 OBJ)
            # GEC: REPLACE
            # SVO: 即使字错了，句法成分依然存在，应该继承 Target 的标签！
            
            src_len = i2 - i1
            tgt_len = j2 - j1
            
            for k in range(src_len):
                # 1. 生成 GEC 标签 (REPLACE)
                if k < tgt_len:
                    # 还有对应的 Target 字符
                    replace_char = target_tokens[j1 + k]
                    # 如果是 Source 的最后一个字，且 Target 还有剩余，把剩余的拼进来
                    if k == src_len - 1 and src_len < tgt_len:
                        remaining = "".join(target_tokens[j1 + k + 1 : j2])
                        full_replacement = replace_char + remaining
                        # **标签压缩策略**：如果启用了压缩且不在高频词表中，使用 MASK
                        if cfg.ENABLE_LABEL_COMPRESSION and full_replacement not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}MASK")
                        else:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}{full_replacement}")
                    else:
                        # **标签压缩策略**
                        if cfg.ENABLE_LABEL_COMPRESSION and replace_char not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}MASK")
                        else:
                            gec_labels.append(f"{cfg.GEC_REPLACE_PREFIX}{replace_char}")
                    
                    # 2. 生成 SVO 标签 (继承 Target)
                    # 逻辑：错字也是句子骨架的一部分
                    source_svo_labels.append(target_svo_labels[j1 + k])
                    
                else:
                    # Source 比 Target 长，多出来的部分标记为 DELETE
                    gec_labels.append(cfg.GEC_DELETE_LABEL)
                    # 多出来的错字没有对应的 Target SVO，标记为 O
                    source_svo_labels.append('O')

        elif tag == 'delete':
            # === 情况3: 成分赘余 (Source 多了) ===
            # 例子: Source="通过会议" -> Target="会议" (SUB)
            # "通过" 是 delete。
            # GEC: DELETE
            # SVO: 多出来的介词/废话，肯定不是骨架，标记为 O
            for k in range(i2 - i1):
                gec_labels.append(cfg.GEC_DELETE_LABEL)
                source_svo_labels.append('O')
        
        elif tag == 'insert':
            # === 情况4: 成分缺失 (Source 少了) ===
            # 例子: Source="学习" -> Target="我们学习"
            # "我们" 是 insert。
            # GEC: 前一个 token 标记 APPEND
            # SVO: Source 里根本没有这个字，所以没法给它贴 SVO 标签。
            #      这就导致 Source 序列里缺失了 B-SUB。
            #      这是符合预期的：SVO Head 看到开头是 PRED，GEC Head 看到 APPEND。
            
            insert_content = "".join(target_tokens[j1:j2])
            
            # 处理 GEC 的 APPEND 逻辑
            if i1 > 0 and len(gec_labels) > 0:
                # 尝试挂载到前一个 token
                last_label = gec_labels[-1]
                if last_label == cfg.GEC_KEEP_LABEL:
                    # **标签压缩策略**：只为高频词生成专门的 APPEND 标签
                    if cfg.ENABLE_LABEL_COMPRESSION and insert_content not in cfg.HIGH_FREQ_FUNCTION_WORDS:
                        # 对于非高频词，统一使用 APPEND_MASK
                        gec_labels[-1] = f"{cfg.GEC_APPEND_PREFIX}MASK"
                    else:
                        gec_labels[-1] = f"{cfg.GEC_APPEND_PREFIX}{insert_content}"
                # 如果前一个是 Replace/Delete，简化处理忽略 Append，或者不做处理
            else:
                # 句首 Insert，受限于 GECToR 机制可能丢失，暂忽略
                pass
            
            # SVO: 没有任何 Source Token 产生，所以不需要 append 任何 SVO label
            pass

    return gec_labels, source_svo_labels



def build_gec_label_vocab(samples: List[Dict], output_path: Path):
    """
    构建GEC标签词表
    
    收集所有出现过的GEC标签
    
    **重要**：强制将 $KEEP 放在第 0 位，确保其 ID 为 0
    这对于 FocalLoss 和评估指标的正确性至关重要
    """
    label_set = set()
    label_set.add(cfg.GEC_KEEP_LABEL)
    label_set.add(cfg.GEC_DELETE_LABEL)
    
    for sample in samples:
        for label in sample['gec_labels']:
            label_set.add(label)
    
    # **关键修改**：确保 $KEEP 在第一位，其余按字母顺序排列
    # 先移除 $KEEP，排序其他标签，然后将 $KEEP 放在开头
    label_set.discard(cfg.GEC_KEEP_LABEL)
    labels = [cfg.GEC_KEEP_LABEL] + sorted(list(label_set))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(label + '\n')
    
    logger.info(f"Built GEC label vocab with {len(labels)} labels, saved to {output_path}")
    logger.info(f"$KEEP label is at index 0: {labels[0] == cfg.GEC_KEEP_LABEL}")


def build_svo_label_vocab(output_path: Path):
    """
    构建SVO标签词表（固定的7个标签）
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in cfg.SVO_LABELS:
            f.write(label + '\n')
    
    logger.info(f"Built SVO label vocab, saved to {output_path}")


def generate_training_data(
    clean_sentences: List[str],
    output_dir: Path,
    train_ratio: float = 0.9,
    num_samples_per_sentence: int = 2,
    use_cuda: bool = False
):
    """
    生成训练数据
    
    步骤：
    1. 为每个正确句子生成错误样本
    2. 生成SVO标签
    3. 生成GECToR标签
    4. 划分训练集和验证集
    
    Args:
        clean_sentences: 清洗后的正确句子列表
        output_dir: 输出目录
        train_ratio: 训练集比例
        num_samples_per_sentence: 每个句子生成的错误样本数
        use_cuda: 是否使用GPU加速DDParser（默认False，使用CPU）
    """
    error_gen = ErrorGenerator()
    svo_extractor = SVOExtractor(use_cuda=use_cuda)
    
    all_samples = []
    
    # 生成错误样本
    for sent_id, correct_sent in enumerate(tqdm(clean_sentences, desc="Generating samples")):
        # 1. 获取 Target (正确句) 的 Token 和 SVO 骨架
        target_tokens, target_svo_labels = svo_extractor.extract(correct_sent)
        
        # 先加一条“无错句”样本（句级负样本）
        assert ''.join(target_tokens) == correct_sent
        assert len(target_svo_labels) == len(correct_sent)
        clean_sample = {
        'uid': f'sent_{sent_id}_clean',
        'text': correct_sent,
        'tokens': list(correct_sent),
        'gec_labels': [cfg.GEC_KEEP_LABEL] * len(correct_sent),
        'svo_labels': target_svo_labels,
        'sent_has_error': 0
        }
        all_samples.append(clean_sample)
        for sample_id in range(num_samples_per_sentence):
            # 2. 生成 Source (错句)
            error_sent, errors = error_gen.generate_errors(correct_sent, max_errors=2)
            
            if not errors or error_sent == correct_sent:
                continue
            
            source_tokens = list(error_sent)
            
            # 3. 对齐并投射标签
            # 关键点：直接把 target_svo_labels 传进去进行投射
            gec_labels, source_svo_labels = align_tokens_with_difflib(
                source_tokens, 
                target_tokens, 
                target_svo_labels
            )
            
            # 4. 长度对齐检查 (Double Check)
            assert len(gec_labels) == len(source_tokens), f"GEC label len mismatch: {len(gec_labels)} vs {len(source_tokens)}"
            assert len(source_svo_labels) == len(source_tokens), f"SVO label len mismatch: {len(source_svo_labels)} vs {len(source_tokens)}"
            
            sample = {
                'uid': f'sent_{sent_id}_sample_{sample_id}',
                'text': error_sent,
                'tokens': source_tokens,
                'gec_labels': gec_labels,
                'svo_labels': source_svo_labels,
                'sent_has_error': 1
            }
            
            all_samples.append(sample)
    
    logger.info(f"Generated {len(all_samples)} samples")
    
    # 打乱并划分
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_idx]
    dev_samples = all_samples[split_idx:]
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.json'
    dev_path = output_dir / 'dev.json'
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    
    with open(dev_path, 'w', encoding='utf-8') as f:
        json.dump(dev_samples, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Train: {len(train_samples)} samples -> {train_path}")
    logger.info(f"Dev: {len(dev_samples)} samples -> {dev_path}")
    
    # 构建标签词表
    build_gec_label_vocab(all_samples, cfg.VOCAB_DIR / 'label_map.txt')
    build_svo_label_vocab(cfg.VOCAB_DIR / 'svo_labels.txt')


def main():
    parser = argparse.ArgumentParser(description="Data Preprocessing")
    parser.add_argument('--raw_dir', type=str, default=str(cfg.RAW_DATA_DIR), 
                       help="Raw data directory")
    parser.add_argument('--output_dir', type=str, default=str(cfg.SYNTHETIC_DATA_DIR),
                       help="Output directory for training data")
    parser.add_argument('--num_samples', type=int, default=2,
                       help="Number of error samples per sentence")
    parser.add_argument('--use_cuda', action='store_true', default=False,
                       help="Use GPU for SVO extraction (requires CUDA)")
    parser.add_argument('--max_sentences', type=int, default=None,
                       help="Maximum number of sentences to process (default: None, process all)")
    
    args = parser.parse_args()
    
    # 检测GPU可用性
    try:
        import torch
        if args.use_cuda and torch.cuda.is_available():
            use_cuda = True
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        elif args.use_cuda and not torch.cuda.is_available():
            use_cuda = False
            logger.warning("CUDA requested but not available, falling back to CPU")
        else:
            use_cuda = False
            logger.info("Using CPU for SVO extraction")
    except ImportError:
        use_cuda = False
        logger.warning("PyTorch not found, using CPU")
    
    # 1. 清洗数据 
    clean_path = cfg.CLEAN_DATA_DIR / 'clean_sentences.txt'
    needs_refresh = (not clean_path.exists()) or clean_path.stat().st_size == 0
    if needs_refresh:
        clean_sentences = clean_raw_data(Path(args.raw_dir), clean_path)
    else:
        logger.info(f"Loading clean data from {clean_path}")
        with open(clean_path, 'r', encoding='utf-8') as f:
            clean_sentences = [line.strip() for line in f if line.strip()]
        if not clean_sentences:
            logger.info("Existing clean data file is empty, regenerating from raw data")
            clean_sentences = clean_raw_data(Path(args.raw_dir), clean_path)
    
    logger.info(f"Loaded {len(clean_sentences)} clean sentences")
    
    # 2. 生成训练数据
    if args.max_sentences is None:
        num_to_process = len(clean_sentences)
        logger.info(f"Processing all {num_to_process} sentences with GPU={'enabled' if use_cuda else 'disabled'}")
    else:
        num_to_process = min(len(clean_sentences), args.max_sentences)
        logger.info(f"Processing {num_to_process} / {len(clean_sentences)} sentences with GPU={'enabled' if use_cuda else 'disabled'}")
    
    generate_training_data(
        clean_sentences[:num_to_process],
        output_dir=Path(args.output_dir),
        num_samples_per_sentence=args.num_samples,
        use_cuda=use_cuda  # 使用命令行参数控制
    )
    
    logger.info("Data preprocessing completed!")


if __name__ == "__main__":
    main()
