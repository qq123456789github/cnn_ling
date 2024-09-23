import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from tqdm import tqdm
import os
from PIL import Image
import random

# ==============================
# 可调参数部分
# ==============================

# 数据集路径，包含多个类别文件夹（class1, class2, ..., class40）
DATA_DIR = 'images'

# 批次大小
BATCH_SIZE = 32

# 训练的总轮数
NUM_EPOCHS = 50

# 学习率
LEARNING_RATE = 0.001

# 模型保存路径
MODEL_SAVE_PATH = 'yolo_like_classification_model.pth'

# 验证集比例
VAL_SPLIT = 0.2

# 检查点路径（如果需要从检查点继续训练，请设置为相应的路径，否则设置为 None）
CHECKPOINT_PATH = 'yolo_like_classification_model.pth'  # 例如: 'yolo_like_classification_model_epoch_10.pth'


# ==============================
# 1. 自定义数据集类
# ==============================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 数据集根目录，包含多个类别文件夹。
            transform (callable, optional): 应用于样本的可选转换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = self._make_dataset()

    def _make_dataset(self):
        images = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一张全黑图片和标签为0，避免程序中断
            image = Image.new('RGB', (224, 224))
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================
# 2. 数据预处理与加载
# ==============================
def get_data_loaders(root_dir, batch_size=32, val_split=0.2):
    """
    创建训练和验证的数据加载器。

    Args:
        root_dir (str): 数据集根目录，包含多个类别文件夹。
        batch_size (int): 每个批次的样本数量。
        val_split (float): 验证集所占比例。

    Returns:
        train_loader, val_loader, class_names: 训练和验证的数据加载器及类别名称列表。
    """
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 实例化自定义数据集
    dataset = CustomDataset(root_dir, transform)

    # 数据集划分
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, dataset.classes


# ==============================
# 3. 模型定义（YOLO-like backbone）
# ==============================
class YOLOLikeClassifier(nn.Module):
    def __init__(self, num_classes=40):
        super(YOLOLikeClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ==============================
# 4. 模型训练与验证
# ==============================
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, checkpoint_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 动态学习率调整
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 从检查点加载模型
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    else:
        start_epoch = 0

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        train_acc = 100 * correct_train / total_train
        val_acc = validate_model(model, val_loader, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, MODEL_SAVE_PATH)


# 验证函数
def validate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# ==============================
# 5. 主程序入口
# ==============================
if __name__ == '__main__':
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE, VAL_SPLIT)
    model = YOLOLikeClassifier(num_classes=40)
    train_model(model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, checkpoint_path=CHECKPOINT_PATH)
