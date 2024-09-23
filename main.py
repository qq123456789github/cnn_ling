import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
import os
from PIL import Image
import random

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
    dataset = CustomDataset(root_dir=root_dir, transform=transform)

    # 确认类别数为40
    num_classes = len(dataset.classes)
    if num_classes != 40:
        print(f"警告: 数据集中检测到 {num_classes} 个类别，但需要40个类别。请检查数据集。")

    # 设置随机种子以确保可复现性
    random_seed = 42
    torch.manual_seed(random_seed)

    # 计算验证集大小
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # 划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"总样本数: {total_size}")
    print(f"训练集样本数: {train_size}")
    print(f"验证集样本数: {val_size}")
    print(f"类别数: {num_classes}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=os.cpu_count(), pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=os.cpu_count(), pin_memory=False)

    return train_loader, val_loader, dataset.classes

# ==============================
# 3. YOLO 风格的分类模型
# ==============================
class SimpleYOLOLikeModel(nn.Module):
    def __init__(self, num_classes=40):  # 将默认类别数改为40
        super(SimpleYOLOLikeModel, self).__init__()
        self.features = nn.Sequential(
            # 第一组卷积层
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第二组卷积层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第三组卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第四组卷积层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 输出层类别数为40
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================
# 4. 训练与验证函数
# ==============================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc='训练', leave=False)
    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(dataloader, desc='验证', leave=False)
    with torch.no_grad():
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# ==============================
# 5. 推理函数
# ==============================
def predict(model, image_path, transform, device, class_names):
    """
    对单张图片进行预测。

    Args:
        model (nn.Module): 已训练的模型。
        image_path (str): 图片路径。
        transform (callable): 图像预处理函数。
        device (torch.device): 设备。
        class_names (list): 类别名称列表。

    Returns:
        str: 预测的类别名称。
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# ==============================
# 6. 主训练函数
# ==============================
def main():
    # 配置参数
    root_dir = 'images'  # 数据集根目录
    batch_size = 32
    num_epochs = 90
    learning_rate = 0.00001
    checkpoint_path = 'checkpoint.pth'  # 检查点路径
    model_save_path = 'yolo_like_classification_model.pth'
    val_split = 0.2  # 20% 作为验证集

    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 优先使用 GPU
    print(f'使用设备: {device}')

    # 获取数据加载器
    train_loader, val_loader, class_names = get_data_loaders(root_dir, batch_size, val_split)

    # 确认类别数为40
    num_classes = len(class_names)
    if num_classes != 40:
        print(f"错误: 数据集中检测到 {num_classes} 个类别，但需要40个类别。请检查数据集。")
        return

    # 实例化模型
    model = SimpleYOLOLikeModel(num_classes=num_classes).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化起始epoch和最佳验证准确率
    start_epoch = 0
    best_val_acc = 0.0

    # 检查是否存在检查点
    if os.path.exists(checkpoint_path):
        print(f'加载检查点 {checkpoint_path}...')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f'已加载 epoch {checkpoint["epoch"] + 1}，最佳验证准确率 {best_val_acc:.4f}')
    else:
        print('未找到检查点，开始新训练。')

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)

        # 训练一个 epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'训练 Loss: {train_loss:.4f} | 训练 Accuracy: {train_acc:.4f}')

        # 验证一个 epoch
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        print(f'验证 Loss: {val_loss:.4f} | 验证 Accuracy: {val_acc:.4f}')

        # 检查是否为最佳模型
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            print(f'新最佳验证准确率: {best_val_acc:.4f}')

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'class_names': class_names,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'已保存检查点至 {checkpoint_path}')

        # 如果是最佳模型，则额外保存模型权重
        if is_best:
            torch.save(model.state_dict(), model_save_path)
            print(f'新最佳模型已保存至 {model_save_path}')

    print(f'\n训练完成。最佳验证准确率: {best_val_acc:.4f}')

    # 示例推理
    sample_image_path = 'path_to_some_image.jpg'  # 替换为实际图片路径
    if os.path.exists(sample_image_path):
        # 定义与训练时相同的转换
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        predicted_class = predict(model, sample_image_path, transform, device, class_names)
        print(f'预测类别: {predicted_class}')
    else:
        print(f'示例图片路径不存在: {sample_image_path}')

# ==============================
# 7. 运行脚本
# ==============================
if __name__ == '__main__':
    main()
