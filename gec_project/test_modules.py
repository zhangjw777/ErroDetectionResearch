"""
快速测试脚本
用于验证项目各模块是否正常工作
"""
import sys
from pathlib import Path

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

print("=" * 60)
print("GEC Project Module Test")
print("=" * 60)

# 1. 测试配置模块
print("\n1. Testing config module...")
try:
    from src.config import default_config as cfg
    print(f"   ✓ Config loaded")
    print(f"   - BERT Model: {cfg.BERT_MODEL}")
    print(f"   - Max Length: {cfg.MAX_SEQ_LENGTH}")
    print(f"   - Focal Gamma: {cfg.FOCAL_LOSS_GAMMA}")
except Exception as e:
    print(f"   ✗ Config failed: {e}")

# 2. 测试错误生成器
print("\n2. Testing error generator...")
try:
    from src.utils.augmentation import ErrorGenerator
    generator = ErrorGenerator()
    test_text = "我们要认真学习会议精神。"
    error_text, errors = generator.generate_errors(test_text, max_errors=1)
    print(f"   ✓ Error generator works")
    print(f"   - Original: {test_text}")
    print(f"   - Error: {error_text}")
    print(f"   - Errors: {errors}")
except Exception as e:
    print(f"   ✗ Error generator failed: {e}")

# 3. 测试SVO提取器
print("\n3. Testing SVO extractor...")
try:
    from src.utils.svo_extract import DDPARSER_AVAILABLE, SVOExtractor
    if DDPARSER_AVAILABLE:
        extractor = SVOExtractor()
        test_text = "我们学习知识。"
        tokens, labels = extractor.extract(test_text)
        print(f"   ✓ SVO extractor works")
        print(f"   - Text: {test_text}")
        print(f"   - Tokens: {tokens}")
        print(f"   - Labels: {labels}")
    else:
        print(f"   ! DDParser not installed, skipping SVO test")
except Exception as e:
    print(f"   ✗ SVO extractor failed: {e}")

# 4. 检查数据目录
print("\n4. Checking data directories...")
from src.config import DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, SYNTHETIC_DATA_DIR, VOCAB_DIR
for dir_path in [DATA_DIR, RAW_DATA_DIR, CLEAN_DATA_DIR, SYNTHETIC_DATA_DIR, VOCAB_DIR]:
    if dir_path.exists():
        print(f"   ✓ {dir_path.name}/ exists")
    else:
        print(f"   ! {dir_path.name}/ not found (will be created when needed)")

# 5. 检查依赖
print("\n5. Checking dependencies...")
dependencies = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'ddparser': 'DDParser',
    'jieba': 'Jieba',
    'numpy': 'NumPy',
    'tqdm': 'tqdm'
}

for module, name in dependencies.items():
    try:
        __import__(module)
        print(f"   ✓ {name} installed")
    except ImportError:
        print(f"   ✗ {name} not installed (required)")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
print("\nNext steps:")
print("1. Install missing dependencies: pip install -r requirements.txt")
print("2. Put raw data (JSONL) into data/raw/")
print("3. Run preprocessing: python src/preprocess.py")
print("4. Start training: python src/trainer.py")
