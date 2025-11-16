# =====================================
# 书法字体分类 - 增强版 "墨识 MoShi"（全模型微调版）
# =====================================
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, ConcatDataset, random_split
from PIL import UnidentifiedImageError
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --------------------------
# 1️⃣ 基本参数
# --------------------------
data_dirs = ['KaggleImages', 'SupplementaryImages']
batch_size = 32
num_epochs = 25
learning_rate = 1e-5  # ✅ 全模型微调建议更小学习率
val_ratio = 0.2
num_workers = 4
MODEL_NAME = 'efficientnet_b0'  # 'resnet18' 或 'efficientnet_b0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 2️⃣ 数据增强与预处理
# --------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomHorizontalFlip(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------------
# ✅ 安全版 ImageFolder
# --------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (OSError, UnidentifiedImageError):
            print(f"[警告] 跳过损坏图片: {path}")
            return None
        if self.transform:
            sample = self.transform(sample)
        return sample, target

# --------------------------
# ✅ 自定义 collate_fn
# --------------------------
def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# --------------------------
# 🔹 主运行区
# --------------------------
if __name__ == "__main__":
    print("Using device:", device)

    # --------------------------
    # 🌱 固定随机种子（保证可复现性）
    # --------------------------
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # --------------------------
    # 3️⃣ 加载多个数据源并合并
    # --------------------------
    datasets_list = []
    for d in data_dirs:
        ds = SafeImageFolder(d, transform=train_transforms)
        datasets_list.append(ds)

    full_dataset = ConcatDataset(datasets_list)
    num_classes = len(datasets_list[0].classes)
    print(f"Classes: {datasets_list[0].classes}")

    # --------------------------
    # 4️⃣ 划分训练集和验证集
    # --------------------------
    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              collate_fn=safe_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            collate_fn=safe_collate_fn)

    print(f"📊 Total: {len(full_dataset)} images | Train: {train_size} | Val: {val_size}")

    # --------------------------
    # 5️⃣ 模型选择（全模型微调）
    # --------------------------
    if MODEL_NAME == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # ✅ 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        print("✅ 使用 ResNet18（全模型微调）")

    elif MODEL_NAME == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

        # ✅ 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True

        print("✅ 使用 EfficientNet_B0（全模型微调）")

    else:
        raise ValueError("Unsupported MODEL_NAME")

    model = model.to(device)

    # --------------------------
    # 6️⃣ 损失函数与优化器
    # --------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数量: {total_params:,}")
    print(f"🟢 可训练参数量: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
    print(f"✅ Using model: {MODEL_NAME}")

    # --------------------------
    # 7️⃣ 训练循环
    # --------------------------
    train_acc_list, val_acc_list = [], []
    print("🚀 开始训练循环...")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        for batch in pbar:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - start_time
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_acc_list.append(train_acc)

        # --------------------------
        # 验证阶段
        # --------------------------
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_acc_list.append(val_acc)

        print(f"\n🧾 Epoch [{epoch+1}/{num_epochs}] 完成 | "
              f"🕒 {epoch_time:.1f}s | ✅ Train Acc: {train_acc:.3f} | 🧪 Val Acc: {val_acc:.3f}")
        if torch.cuda.is_available():
            print(f"💾 GPU 显存占用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

    # --------------------------
    # 8️⃣ 混淆矩阵可视化
    # --------------------------
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=datasets_list[0].classes,
                yticklabels=datasets_list[0].classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {MODEL_NAME.upper()} (MoShi)')
    plt.show()

    # --------------------------
    # 9️⃣ 训练曲线
    # --------------------------
    plt.figure(figsize=(8,4))
    plt.plot(train_acc_list, label='Train Acc')
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Training Curve - {MODEL_NAME.upper()} (MoShi)')
    plt.show()

    # --------------------------
    # 🔟 保存模型
    # --------------------------
    torch.save(model.state_dict(), f'ModelCheckpoints/MoShi_{MODEL_NAME}_trained_all_params.pth')
    print(f"✅ 模型已保存为 MoShi_{MODEL_NAME}_trained_all_params.pth")
