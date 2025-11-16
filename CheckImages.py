import os

def clean_dataset(folder):
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', 'webp'}
    removed = 0
    for root, _, files in os.walk(folder):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in valid_exts:
                path = os.path.join(root, f)
                print(f"❌ Removing non-image file: {path}")
                os.remove(path)
                removed += 1
    print(f"✅ 清理完成，共移除 {removed} 个非标准文件。")

# 检查两个数据源
for d in ['KaggleImages', 'SupplementaryImages']:
    clean_dataset(d)