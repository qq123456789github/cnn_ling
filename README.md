# Ling: YOLO风格的40类图像分类模型

## 目录

- [项目概述](#项目概述)
- [网络架构详解](#网络架构详解)
- [数据集结构](#数据集结构)
- [安装](#安装)
- [使用说明](#使用说明)
  - [训练](#训练)
  - [继续训练](#继续训练)
  - [推理](#推理)
- [检查点保存](#检查点保存)
- [安全注意事项](#安全注意事项)
- [定制化](#定制化)
- [问题排查](#问题排查)
- [贡献](#贡献)
- [许可证](#许可证)

## 项目概述

本项目基于PyTorch实现了一个名为**Ling**的YOLO风格卷积神经网络（CNN），用于40类图像分类任务。模型采用自定义数据集，支持训练、验证、检查点保存及推理功能，适用于需要高效图像分类的应用场景。

## 网络架构详解

### Ling模型

**Ling** 是一个受YOLO（You Only Look Once）启发的简化版CNN架构，专为图像分类任务设计。以下是其详细结构说明：

```python
class Ling(nn.Module):
    def __init__(self, num_classes=40):
        super(Ling, self).__init__()
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
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

### 详细层级解析

1. **第一组卷积层**
   - **Conv2d**: 输入通道数为3（RGB图像），输出通道数为64，卷积核大小为7x7，步幅为2，填充为3。
   - **BatchNorm2d**: 对卷积输出进行批量归一化，加速训练并稳定模型。
   - **ReLU**: 引入非线性激活函数，增加模型表达能力。
   - **MaxPool2d**: 最大池化层，池化核大小为2x2，步幅为2，用于下采样，减少特征图尺寸。

2. **第二组卷积层**
   - **Conv2d**: 输入通道数为64，输出通道数为128，卷积核大小为3x3，填充为1。
   - **BatchNorm2d**
   - **ReLU**
   - **MaxPool2d**

3. **第三组卷积层**
   - **Conv2d**: 输入通道数为128，输出通道数为256，卷积核大小为3x3，填充为1。
   - **BatchNorm2d**
   - **ReLU**
   - **MaxPool2d**

4. **第四组卷积层**
   - **Conv2d**: 输入通道数为256，输出通道数为512，卷积核大小为3x3，填充为1。
   - **BatchNorm2d**
   - **ReLU**
   - **AdaptiveAvgPool2d**: 自适应平均池化，将特征图大小调整为1x1，无论输入尺寸如何。
   - **Flatten**: 将多维特征图展平成一维向量，便于输入到全连接层。

5. **分类器**
   - **Linear**: 全连接层，将512维特征向量映射到512维。
   - **ReLU**
   - **Dropout**: 随机丢弃部分神经元，防止过拟合，丢弃率为50%。
   - **Linear**: 最终全连接层，将512维特征映射到40个类别。

### 架构总结

**Ling** 模型通过连续的卷积和池化操作，逐步提取并压缩图像特征，最终通过全连接层进行分类。自适应平均池化确保了特征向量的固定尺寸，使模型能够适应不同输入图像尺寸。Dropout层有效防止了模型的过拟合，提高了泛化能力。

## 数据集结构

数据集应按照以下目录结构组织：

```
images/
├── class_1/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── class_2/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
...
├── class_40/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
```

- **images/**: 根目录，包含所有类别的子文件夹。
- **class_1/** 到 **class_40/**: 每个文件夹对应一个类别，文件夹名称即为类别名称，内部包含该类别的所有图像。

**注意**: 确保每个类别文件夹内有足够数量的图像，以保证模型的训练效果。

## 安装

### 前提条件

- Python 3.7或更高版本
- [PyTorch](https://pytorch.org/)（根据是否使用GPU选择合适的版本）
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- 其他Python包

### 安装步骤

1. **克隆仓库**

   ```bash
   git clone https://github.com/yourusername/ling-classification.git
   cd ling-classification
   ```

2. **创建虚拟环境（可选）**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows用户使用: venv\Scripts\activate
   ```

3. **安装依赖**

   ```bash
   pip install torch torchvision tqdm pillow
   ```

   *或者使用`requirements.txt`文件安装所有依赖:*

   ```bash
   pip install -r requirements.txt
   ```

## 使用说明

### 训练

1. **准备数据集**

   按照[数据集结构](#数据集结构)组织您的数据集。

2. **运行训练脚本**

   ```bash
   python main.py
   ```

   **默认参数:**

   - **数据集根目录**: `images`
   - **批次大小**: 32
   - **训练轮数**: 20
   - **学习率**: 0.001
   - **验证集比例**: 20%
   - **检查点路径**: `checkpoint.pth`
   - **模型保存路径**: `ling_classification_model.pth`

   *可根据需要修改`main()`函数中的参数。*

### 继续训练

如果训练中断或需要从上次停止的地方继续训练：

1. **确保存在检查点**

   脚本会在每个epoch结束时保存`checkpoint.pth`。

2. **重新运行训练脚本**

   ```bash
   python main.py
   ```

   脚本会自动检测到已有的检查点，并从中断处继续训练。

### 推理

对单张图片进行分类预测：

1. **指定图片路径**

   修改`main()`函数中的`sample_image_path`变量：

   ```python
   sample_image_path = 'path_to_some_image.jpg'  # 替换为您的图片路径
   ```

2. **运行脚本**

   训练完成后，脚本会尝试预测指定图片的类别。

   ```bash
   python main.py
   ```

   **输出示例:**

   ```
   预测类别: class_name
   ```

## 检查点保存

### 保存检查点

每个epoch结束时，脚本会保存一个包含以下内容的检查点字典：

- `epoch`: 当前的epoch数
- `model_state_dict`: 模型权重
- `optimizer_state_dict`: 优化器状态
- `best_val_acc`: 当前最佳验证准确率
- `class_names`: 类别名称列表

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'best_val_acc': best_val_acc,
    'class_names': class_names,
}
torch.save(checkpoint, checkpoint_path)
```

此外，如果当前epoch的验证准确率为最佳，将模型权重单独保存：

```python
if is_best:
    torch.save(model.state_dict(), model_save_path)
```

### 加载检查点

训练开始前，脚本会检查`checkpoint.pth`是否存在，如果存在则加载：

```python
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
```

## 安全注意事项

### 关于 `torch.load` 的 FutureWarning

您可能会看到如下警告：

```
FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly...
```

**解释**: 使用`torch.load`时，默认的`weights_only=False`可能存在安全风险。未来PyTorch将默认`weights_only=True`，限制在加载过程中执行任意代码。

### 建议

1. **仅加载可信来源的检查点**: 确保加载的模型文件来自可信来源，避免潜在的安全风险。
2. **仅加载模型权重**: 如果只需要加载模型权重，设置`weights_only=True`，以提高安全性：

   ```python
   model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
   ```

   **注意**: 这种方式不会加载优化器状态和epoch信息。
3. **关注PyTorch更新**: 留意PyTorch的[安全指南](https://github.com/pytorch/pytorch/blob/main/SECURITY.md)，及时更新代码以适应新的安全措施。

## 定制化

### 超参数

可以通过修改`main()`函数中的配置参数来定制训练过程：

```python
def main():
    # 配置参数
    root_dir = 'images'
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    checkpoint_path = 'checkpoint.pth'
    model_save_path = 'ling_classification_model.pth'
    val_split = 0.2
    # ...
```

### 网络架构

如需修改网络架构，调整`Ling`类。例如，可以添加更多卷积层、调整卷积核大小或修改分类器部分。

### 数据增强

增强数据预处理流程，添加更多的图像变换：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),      # 随机旋转±10度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

## 问题排查

### 常见问题

1. **GPU内存不足**
   - **症状**: 训练过程中出现OOM错误。
   - **解决方案**: 减小`batch_size`或使用梯度累积。

2. **数据加载错误**
   - **症状**: 与图像加载相关的错误。
   - **解决方案**: 确保所有图像格式支持（`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`）且未损坏。

3. **检查点加载问题**
   - **症状**: 加载检查点时报错。
   - **解决方案**: 验证`checkpoint.pth`的完整性，确保正确保存。

### 调试建议

- **详细日志**: 添加打印语句或使用日志工具跟踪数据流。
- **健全性检查**: 确认类别数正确设置为40。
- **可视化**: 可视化部分样本图像及其标签，确保数据加载正确。

## 贡献

欢迎贡献！请按照以下步骤操作：

1. **Fork 仓库**
2. **创建新分支**
   
   ```bash
   git checkout -b feature/YourFeature
   ```

3. **提交更改**
   
   ```bash
   git commit -m "添加 YourFeature"
   ```

4. **推送到分支**
   
   ```bash
   git push origin feature/YourFeature
   ```

5. **打开Pull Request**

请确保代码符合项目编码规范，并包含适当的文档说明。

## 许可证

本项目采用 [MIT 许可证](LICENSE)。

---

*如有任何问题或需要进一步的帮助，请在仓库中提交Issue或联系维护者。*# cnn_ling
