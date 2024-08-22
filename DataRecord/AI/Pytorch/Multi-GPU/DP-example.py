################## DataParallel ##################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import time

# 创建一些随机数据
features = torch.randn(1000, 100)
labels = torch.randint(0, 2, (1000,))

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 创建数据集和数据加载器
dataset = TensorDataset(features, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 检查可用的GPU数量
device_ids = list(range(torch.cuda.device_count()))

# 如果有多个GPU可用，则使用DataParallel
if len(device_ids) > 1:
    model = SimpleModel().cuda()
    model = nn.DataParallel(model, device_ids=device_ids)
else:
    model = SimpleModel().cuda()

# 获取 GPU 数量
num_gpus = torch.cuda.device_count()

# 列出所有 GPU 设备的信息
for i in range(num_gpus):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")

# 输出总的 GPU 数量
print(f"Total number of GPUs: {num_gpus}")

# 也可以打印其他信息，例如：
if num_gpus > 0:
    current_device = torch.cuda.current_device()
    print(f"Current device: {current_device}")
    print(f"Current device name: {torch.cuda.get_device_name(current_device)}")
    print(f"Current device memory allocated: {torch.cuda.memory_allocated(current_device)}")
    print(f"Current device max memory: {torch.cuda.max_memory_allocated(current_device)}")
    
# 开始训练

criterion = nn.BCELoss()                                # 定义损失函数和优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)     # 定义损失函数和优化器

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 移动数据到 GPU
        inputs, targets = inputs.cuda(), targets.float().cuda()
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs.squeeze(), targets)
        
        # 反向传播
        loss.backward()
        
        # 更新权重
        optimizer.step()
        
        # 监控 GPU 使用情况
        for i in range(len(device_ids)):
            print(f"Device {i}: Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
            print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 2:.2f} MB")
        
        # 睡眠一小段时间，以便观察变化
        time.sleep(1)

# 训练结束后再次显示 GPU 信息
for i in range(len(device_ids)):
    print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    print(f"Final Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024 ** 2:.2f} MB")