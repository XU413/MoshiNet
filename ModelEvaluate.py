# =====================================
# 书法字体分类 - 模型加载与指标计算（Windows兼容版）
# =====================================
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, ConcatDataset
from PIL import UnidentifiedImageError
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import platform

# --------------------------
# 🔴 核心参数（必须与训练时一致！）
# --------------------------
MODEL_NAME = 'resnet18'  # 'resnet18' 或 'efficientnet_b0'
MODEL_PATH = f'MoShi_{MODEL_NAME}_trained_all_params.pth'
DATA_DIRS = ['KaggleImages', 'SupplementaryImages']
batch_size = 32
val_ratio = 0.2
seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 🔴 Windows多进程修复（关键！）
# --------------------------
if platform.system() == "Windows":
    # Windows强制使用单进程加载数据，避免多进程冲突
    num_workers = 0
else:
    num_workers = 4

# --------------------------
# 🔴 数据预处理（与训练时完全一致）
# --------------------------
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------------
# 🔴 安全数据加载类
# --------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (OSError, UnidentifiedImageError):
            return None
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return torch.utils.data.dataloader.default_collate(batch) if batch else None

# --------------------------
# 🔴 模型加载函数
# --------------------------
def load_model(num_classes):
    if MODEL_NAME == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif MODEL_NAME == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError("不支持的模型类型")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device).eval()
    print(f"✅ 成功加载模型: {MODEL_PATH}")
    return model

# --------------------------
# 🔴 主执行逻辑（Windows必须放在if __name__ == '__main__'中！）
# --------------------------
if __name__ == '__main__':
    # 1. 加载并划分测试数据（仅用验证集）
    datasets_list = [SafeImageFolder(d, transform=test_transforms) for d in DATA_DIRS]
    concat_dataset = ConcatDataset(datasets_list)
    val_size = int(len(concat_dataset) * val_ratio)
    train_size = len(concat_dataset) - val_size
    
    _, test_dataset = random_split(
        concat_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,  # Windows下为0，避免多进程冲突
        pin_memory=True,
        collate_fn=safe_collate_fn
    )
    
    num_classes = len(datasets_list[0].classes)
    class_names = datasets_list[0].classes
    print(f"📊 测试集规模: {len(test_dataset)} 张图片 | 类别: {class_names}")
    
    # 2. 加载模型
    model = load_model(num_classes)
    
    # 3. 推理计算
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="推理中"):
            if batch is None:
                continue
            images, labels = batch
            preds = model(images.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 4. 计算并输出指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print(f"📋 最终指标（{MODEL_NAME.upper()}）")
    print("="*60)
    print(f"Accuracy（准确率）:  {accuracy:.4f}")
    print(f"F1-Score (Micro):   {f1_micro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    print(f"F1-Score (Weighted):{f1_weighted:.4f}")
    print("="*60)
    
    # 5. 混淆矩阵（可选）
    plt.figure(figsize=(8,6))
    sns.heatmap(
        confusion_matrix(all_labels, all_preds),
        annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.title(f'混淆矩阵 - {MODEL_NAME.upper()} (Acc: {accuracy:.4f})')
    plt.tight_layout()
    plt.show()